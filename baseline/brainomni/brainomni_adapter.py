"""BrainOmni adapter using persisted EEG sensor coordinates."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from baseline.abstract.adapter import (
    AbstractDataLoaderFactory,
    AbstractDatasetAdapter,
)
from common.distributed.loader import DistributedGroupBatchSampler
from data.processor.wrapper import load_concat_eeg_datasets
from common.utils import ElectrodeSet


_EEG_SENSOR_TYPE_ID = 0
_SIGNAL_NORMALIZE_EPS_DEFAULT = 1.0e-5
_POSITION_NORMALIZE_EPS_DEFAULT = 1.0e-8


logger = logging.getLogger("baseline")


class BrainOmniFilteredDataset(Dataset):
    """Read-only view of samples that BrainOmni can normalize."""

    def __init__(self, dataset: HFDataset, sample_indices: List[int]) -> None:
        if not sample_indices:
            raise ValueError(
                "BrainOmni filtered dataset requires at least one sample."
            )
        self.dataset = dataset
        self.sample_indices = sample_indices

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: Union[int, str]) -> Any:
        if isinstance(index, str):
            return [
                self.dataset[sample_index][index]
                for sample_index in self.sample_indices
            ]
        return self.dataset[self.sample_indices[index]]

    @property
    def column_names(self) -> List[str]:
        """Return the underlying Hugging Face dataset column names."""
        return list(self.dataset.column_names)


class BrainOmniDatasetAdapter(AbstractDatasetAdapter):
    """Adapter for persisted XYZ, converted to BrainOmni's six coordinates."""

    def __init__(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
        normalize_input: bool = True,
        normalize_position: bool = True,
        signal_normalize_eps: float = _SIGNAL_NORMALIZE_EPS_DEFAULT,
        position_normalize_eps: float = _POSITION_NORMALIZE_EPS_DEFAULT,
    ) -> None:
        self.normalize_input = normalize_input
        self.normalize_position = normalize_position
        self.signal_normalize_eps = float(signal_normalize_eps)
        self.position_normalize_eps = float(position_normalize_eps)
        if self.signal_normalize_eps <= 0.0:
            raise ValueError(
                "BrainOmni adapter expected signal_normalize_eps > 0."
            )
        if self.position_normalize_eps <= 0.0:
            raise ValueError(
                "BrainOmni adapter expected position_normalize_eps > 0."
            )
        self.electrode_set = ElectrodeSet()
        super().__init__(dataset, dataset_names, dataset_configs)

    def _setup_adapter(self) -> None:
        self.model_name = "brainomni"
        self.scale = 1.0
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        return list(self.electrode_set.Electrodes)

    def get_cross_channel_std(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Return the standard deviation after BrainOmni channel centering.

        Parameters
        ----------
        sample : Dict[str, Any]
            Raw sample with ``data`` of shape ``(C, T)`` and a montage name.

        Returns
        -------
        torch.Tensor
            Scalar population standard deviation of the selected channels after
            subtracting their per-timepoint channel mean.
        """
        montage = str(sample["montage"])
        if montage not in self.montage_mappings:
            raise ValueError(f"Montage {montage} not found in mappings")
        data = torch.as_tensor(sample["data"], dtype=torch.float32)
        if data.ndim != 2:
            raise ValueError(
                "BrainOmni expected sample data shape (C, T), but got "
                f"{tuple(data.shape)}."
            )
        selector = torch.as_tensor(
            self.montage_mappings[montage]["sel"], dtype=torch.bool
        )
        if data.shape[0] != selector.numel():
            raise ValueError(
                "BrainOmni sample channel count does not align with its "
                f"montage: expected {selector.numel()}, got {data.shape[0]}."
            )
        selected_data = data[selector] * self.scale
        centered_data = selected_data - selected_data.mean(
            dim=0, keepdim=True
        )
        return centered_data.std(unbiased=False)

    def _apply_model_specific_processing(
        self, data: torch.Tensor, montage_info: Dict[str, Any]
    ) -> torch.Tensor:
        del montage_info
        if data.shape[0] == 0:
            raise ValueError(
                "BrainOmni cannot normalize a sample with no non-constant "
                "channels."
            )
        if not self.normalize_input:
            return data
        data = data - data.mean(dim=0, keepdim=True)
        global_std = data.std(unbiased=False)
        if (
            torch.isnan(global_std)
            or torch.isinf(global_std)
            or global_std.item() < self.signal_normalize_eps
        ):
            raise ValueError(
                "BrainOmni normalization failed: expected finite std >= "
                f"{self.signal_normalize_eps}, got {global_std.item():.6e}."
            )
        return data / (global_std + self.signal_normalize_eps)

    def _process_sample(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, str, List[str], int]]:
        result = super()._process_sample(sample)
        pos = self._get_persisted_positions(sample, result)
        result["pos"] = pos
        result["sensor_type"] = torch.full(
            (pos.shape[0],), _EEG_SENSOR_TYPE_ID, dtype=torch.long
        )
        return result

    def _get_persisted_positions(
        self,
        sample: Dict[str, Any],
        result: Dict[str, Union[torch.Tensor, str, List[str], int]],
    ) -> torch.Tensor:
        """Select persisted XYZ, append zero EEG orientations, and normalize."""
        raw_positions = sample.get("pos")
        if raw_positions is None or len(raw_positions) == 0:
            raise ValueError(
                "BrainOmni requires persisted XYZ electrode positions; "
                "sample has none. Rebuild the dataset with native coordinates "
                "or its configured position_montage."
            )
        positions = torch.as_tensor(raw_positions, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                "BrainOmni adapter expected persisted pos shape (C, 3), "
                f"but got {tuple(positions.shape)}. Rebuild the dataset."
            )
        positions_are_invalid = (
            not torch.isfinite(positions).all()
            or torch.any(torch.all(positions == 0.0, dim=1))
        )
        if positions_are_invalid:
            raise ValueError(
                "BrainOmni requires complete finite persisted XYZ electrode "
                "positions. Rebuild the dataset with a valid native layout or "
                "position_montage."
            )
        selector = torch.as_tensor(
            self.montage_mappings[str(result["montage"])]["sel"],
            dtype=torch.bool,
        )
        if selector.numel() != positions.shape[0]:
            raise ValueError(
                "BrainOmni persisted positions do not align with unselected "
                "EEG channels. Rebuild the dataset."
            )
        positions = positions[selector]
        if positions.shape[0] != len(result["chs"]):
            raise ValueError(
                "BrainOmni persisted positions do not align with selected "
                "EEG channels."
            )
        return self._normalize_eeg_positions(
            torch.cat((positions, torch.zeros_like(positions)), dim=1)
        )

    def _normalize_eeg_positions(self, pos: torch.Tensor) -> torch.Tensor:
        if not self.normalize_position:
            return pos
        if pos.ndim != 2 or pos.shape[1] != 6:
            raise ValueError(
                "BrainOmni adapter expected pos shape (C, 6), "
                f"but got {tuple(pos.shape)}."
            )
        xyz = pos[:, :3]
        xyz_centered = xyz - xyz.mean(dim=0, keepdim=True)
        xyz_squared = torch.sum(xyz_centered**2, dim=1)
        xyz_scale = torch.sqrt(3.0 * torch.mean(xyz_squared))
        if (
            torch.isnan(xyz_scale)
            or torch.isinf(xyz_scale)
            or xyz_scale.item() < self.position_normalize_eps
        ):
            raise ValueError(
                "BrainOmni position normalization failed: expected finite "
                f"scale >= {self.position_normalize_eps}, got "
                f"{xyz_scale.item():.6e}."
            )
        normalized = pos.clone()
        normalized[:, :3] = xyz_centered / (
            xyz_scale + self.position_normalize_eps
        )
        return normalized


class BrainOmniDataLoaderFactory(AbstractDataLoaderFactory):
    """DataLoader factory for BrainOmni's persisted-position adapter."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        normalize_input: bool = True,
        normalize_position: bool = True,
        signal_normalize_eps: float = _SIGNAL_NORMALIZE_EPS_DEFAULT,
        position_normalize_eps: float = _POSITION_NORMALIZE_EPS_DEFAULT,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )
        self.normalize_input = normalize_input
        self.normalize_position = normalize_position
        self.signal_normalize_eps = signal_normalize_eps
        self.position_normalize_eps = position_normalize_eps

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> AbstractDatasetAdapter:
        return BrainOmniDatasetAdapter(
            dataset=dataset,
            dataset_names=dataset_names,
            dataset_configs=dataset_configs,
            normalize_input=self.normalize_input,
            normalize_position=self.normalize_position,
            signal_normalize_eps=self.signal_normalize_eps,
            position_normalize_eps=self.position_normalize_eps,
        )

    def _filter_zero_cross_channel_variation_samples(
        self, dataset: HFDataset, adapter: BrainOmniDatasetAdapter
    ) -> BrainOmniFilteredDataset:
        """Build a read-only view without unnormalizable BrainOmni samples."""
        if not adapter.normalize_input:
            return BrainOmniFilteredDataset(
                dataset=dataset,
                sample_indices=list(range(len(dataset))),
            )
        sample_indices: List[int] = []
        skipped_count = 0
        for sample_index in range(len(dataset)):
            cross_channel_std = adapter.get_cross_channel_std(
                dataset[sample_index]
            )
            if cross_channel_std.item() == 0.0:
                skipped_count += 1
                continue
            sample_indices.append(sample_index)
        if skipped_count > 0:
            logger.warning(
                "BrainOmni skipped %d samples because their selected channels "
                "had zero cross-channel variation after per-timepoint channel "
                "centering, so input normalization would divide by zero.",
                skipped_count,
            )
        if not sample_indices:
            raise ValueError(
                "BrainOmni cannot create a data loader because every "
                "sample has zero cross-channel variation after "
                "per-timepoint channel centering."
            )
        return BrainOmniFilteredDataset(
            dataset=dataset,
            sample_indices=sample_indices,
        )

    def loading_dataset(
        self,
        datasets_config: Dict[str, str],
        split: Any,
        fs: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> tuple[torch.utils.data.DataLoader, DistributedGroupBatchSampler]:
        """Create a BrainOmni loader without unnormalizable samples."""
        dataset_names = list(datasets_config)
        dataset_configs = list(datasets_config.values())
        combined_dataset, _ = load_concat_eeg_datasets(
            dataset_names=dataset_names,
            builder_configs=dataset_configs,
            split=split,
            cast_label=True,
            fs=fs,
        )
        adapter = self.create_adapter(
            dataset=combined_dataset,
            dataset_names=dataset_names,
            dataset_configs=dataset_configs,
        )
        filtered_dataset = (
            self._filter_zero_cross_channel_variation_samples(
                combined_dataset, adapter
            )
        )
        adapter.dataset = filtered_dataset
        sampler = DistributedGroupBatchSampler(
            dataset=filtered_dataset,
            batch_size=self.batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=self.seed,
        )
        dataloader_kwargs: Dict[str, Any] = {
            "batch_sampler": sampler,
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0,
            "prefetch_factor": 2 if self.num_workers > 0 else None,
        }
        if self.num_workers > 0:
            dataloader_kwargs["multiprocessing_context"] = "spawn"
        dataloader = torch.utils.data.DataLoader(adapter, **dataloader_kwargs)
        return dataloader, sampler

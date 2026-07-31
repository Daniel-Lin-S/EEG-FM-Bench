"""BrainOmni adapter using persisted EEG sensor coordinates."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Union

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from baseline.abstract.adapter import (
    AbstractDataLoaderFactory,
    AbstractDatasetAdapter,
)
from common.distributed.env import get_is_master
from common.distributed.loader import DistributedGroupBatchSampler
from data.processor.wrapper import load_concat_eeg_datasets
from common.utils import ElectrodeSet


_EEG_SENSOR_TYPE_ID = 0
_SIGNAL_NORMALIZE_EPS_DEFAULT = 1.0e-5
_POSITION_NORMALIZE_EPS_DEFAULT = 1.0e-8


logger = logging.getLogger("baseline")
_WARNED_SAMPLE_FILTER_SPLITS: set[tuple[str, str]] = set()


class _BrainOmniSampleError(ValueError):
    """Expected sample-level incompatibility with BrainOmni."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BrainOmniFilteredDataset(Dataset):
    """Read-only view of samples that satisfy BrainOmni eligibility."""

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
        if not math.isfinite(self.signal_normalize_eps) or (
            self.signal_normalize_eps <= 0.0
        ):
            raise ValueError(
                "BrainOmni adapter expected finite signal_normalize_eps > 0."
            )
        if not math.isfinite(self.position_normalize_eps) or (
            self.position_normalize_eps <= 0.0
        ):
            raise ValueError(
                "BrainOmni adapter expected finite position_normalize_eps > 0."
            )
        self.electrode_set = ElectrodeSet()
        super().__init__(dataset, dataset_names, dataset_configs)

    def _setup_adapter(self) -> None:
        self.model_name = "brainomni"
        self.scale = 1.0
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        return list(self.electrode_set.Electrodes)

    def _select_signal_data(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Select one sample's channels before signal normalization."""
        montage_value = sample.get("montage")
        if montage_value is None:
            raise _BrainOmniSampleError(
                "montage_missing",
                "sample has no montage identifier",
            )
        montage = str(montage_value)
        if montage not in self.montage_mappings:
            raise _BrainOmniSampleError(
                "montage_not_supported",
                f"montage {montage} is not supported",
            )
        if "data" not in sample:
            raise _BrainOmniSampleError(
                "signal_missing",
                "sample has no signal data",
            )
        try:
            data = torch.as_tensor(sample["data"], dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _BrainOmniSampleError(
                "signal_invalid",
                "sample signal cannot be converted to a float32 tensor",
            ) from exc
        if data.ndim != 2:
            raise _BrainOmniSampleError(
                "signal_shape_invalid",
                "expected sample data shape (C, T), but got "
                f"{tuple(data.shape)}."
            )
        selector = torch.as_tensor(
            self.montage_mappings[montage]["sel"], dtype=torch.bool
        )
        if data.shape[0] != selector.numel():
            raise _BrainOmniSampleError(
                "signal_channel_mismatch",
                "sample channel count does not align with its "
                f"montage: expected {selector.numel()}, got {data.shape[0]}."
            )
        if not math.isfinite(float(self.scale)):
            raise ValueError("BrainOmni signal scale must be finite")
        selected_data = data[selector] * self.scale
        if selected_data.shape[0] == 0:
            raise _BrainOmniSampleError(
                "signal_no_selected_channels",
                "sample has no selected EEG channels",
            )
        if selected_data.shape[1] == 0:
            raise _BrainOmniSampleError(
                "signal_no_timepoints",
                "sample has no time points after channel selection",
            )
        return selected_data

    def _select_persisted_positions(
        self, sample: Dict[str, Any], montage: Union[str, None] = None
    ) -> torch.Tensor:
        """Select and validate persisted XYZ coordinates for one sample."""
        raw_positions = sample.get("pos")
        if raw_positions is None:
            raise _BrainOmniSampleError(
                "position_missing",
                "sample has no persisted XYZ electrode positions",
            )
        try:
            positions = torch.as_tensor(
                raw_positions,
                dtype=torch.float32,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _BrainOmniSampleError(
                "position_invalid",
                "persisted positions cannot be converted to float32",
            ) from exc
        if positions.numel() == 0:
            raise _BrainOmniSampleError(
                "position_missing",
                "sample has no persisted XYZ electrode positions",
            )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise _BrainOmniSampleError(
                "position_shape_invalid",
                "expected persisted pos shape (C, 3), "
                f"but got {tuple(positions.shape)}."
            )
        montage_value = sample.get("montage", montage)
        if montage_value is None:
            raise _BrainOmniSampleError(
                "montage_missing",
                "sample has no montage identifier",
            )
        montage_name = str(montage_value)
        if montage_name not in self.montage_mappings:
            raise _BrainOmniSampleError(
                "montage_not_supported",
                f"montage {montage_name} is not supported",
            )
        selector = torch.as_tensor(
            self.montage_mappings[montage_name]["sel"],
            dtype=torch.bool,
        )
        if selector.numel() != positions.shape[0]:
            raise _BrainOmniSampleError(
                "position_channel_mismatch",
                "persisted positions do not align with unselected EEG "
                "channels",
            )
        positions = positions[selector]
        if positions.shape[0] == 0:
            raise _BrainOmniSampleError(
                "position_no_selected_channels",
                "sample has no selected electrode positions",
            )
        if not torch.isfinite(positions).all():
            raise _BrainOmniSampleError(
                "position_nonfinite",
                "selected XYZ electrode positions contain non-finite values",
            )
        if torch.any(torch.all(positions == 0.0, dim=1)):
            raise _BrainOmniSampleError(
                "position_zero",
                "selected XYZ electrode positions contain a zero coordinate",
            )
        return positions

    def _get_position_normalization_scale(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return a validated spatial scale for position normalization."""
        xyz_centered = positions - positions.mean(dim=0, keepdim=True)
        xyz_squared = torch.sum(xyz_centered**2, dim=1)
        xyz_scale = torch.sqrt(3.0 * torch.mean(xyz_squared))
        if not torch.isfinite(xyz_scale):
            raise _BrainOmniSampleError(
                "position_scale_nonfinite",
                "position normalization scale is non-finite",
            )
        if xyz_scale.item() < self.position_normalize_eps:
            raise _BrainOmniSampleError(
                "position_scale_below_threshold",
                "position normalization scale is below "
                f"{self.position_normalize_eps}",
            )
        return xyz_scale

    def get_sample_rejection(
        self, sample: Dict[str, Any]
    ) -> Union[Dict[str, str], None]:
        """Return an expected preflight rejection, or ``None`` when usable."""
        try:
            selected_data = self._select_signal_data(sample)
            if not torch.isfinite(selected_data).all():
                raise _BrainOmniSampleError(
                    "signal_nonfinite",
                    "selected signal contains non-finite values",
                )
            if self.normalize_input:
                centered_data = selected_data - selected_data.mean(
                    dim=0, keepdim=True
                )
                global_std = centered_data.std(unbiased=False)
                if not torch.isfinite(global_std):
                    raise _BrainOmniSampleError(
                        "signal_std_nonfinite",
                        "centered signal standard deviation is non-finite",
                    )
                if global_std.item() < self.signal_normalize_eps:
                    raise _BrainOmniSampleError(
                        "signal_std_below_threshold",
                        "centered signal standard deviation is below "
                        f"{self.signal_normalize_eps}",
                    )
            positions = self._select_persisted_positions(sample)
            if self.normalize_position:
                self._get_position_normalization_scale(positions)
        except _BrainOmniSampleError as exc:
            return {"code": exc.code, "message": str(exc)}
        return None

    def _apply_model_specific_processing(
        self, data: torch.Tensor, montage_info: Dict[str, Any]
    ) -> torch.Tensor:
        del montage_info
        if data.ndim != 2:
            raise ValueError(
                "BrainOmni expected mapped data shape (C, T), but got "
                f"{tuple(data.shape)}."
            )
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(
                "BrainOmni cannot process a sample with no selected channels "
                "or time points."
            )
        if not torch.isfinite(data).all():
            raise ValueError(
                "BrainOmni input contains non-finite values before "
                "normalization."
            )
        if not self.normalize_input:
            return data
        data = data - data.mean(dim=0, keepdim=True)
        global_std = data.std(unbiased=False)
        if (
            not torch.isfinite(global_std)
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
        positions = self._select_persisted_positions(
            sample,
            montage=str(result["montage"]),
        )
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
        xyz_scale = self._get_position_normalization_scale(xyz)
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
        self._data_diagnostics: Dict[
            tuple[str, str],
            Dict[str, Any],
        ] = {}

    def get_data_diagnostics(
        self,
        dataset_name: str,
    ) -> Mapping[str, Any]:
        """Return BrainOmni sample-filtering diagnostics for one dataset."""
        by_split = [
            {
                key: value
                for key, value in diagnostic.items()
                if key != "dataset"
            }
            for (name, _), diagnostic in sorted(
                self._data_diagnostics.items()
            )
            if name == dataset_name
        ]
        if not by_split:
            return {}
        return {
            "sample_filtering": {
                "skipped_samples": sum(
                    int(item["skipped_samples"])
                    for item in by_split
                ),
                "by_split": by_split,
            }
        }

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

    def _filter_invalid_samples(
        self,
        dataset: HFDataset,
        adapter: BrainOmniDatasetAdapter,
        split: Any,
        dataset_names: List[str],
    ) -> BrainOmniFilteredDataset:
        """Build a view excluding samples that cannot reach the model."""
        sample_indices: List[int] = []
        split_name = str(split)
        counts: Dict[str, Dict[str, Any]] = {}
        for sample_index in range(len(dataset)):
            sample = dataset[sample_index]
            montage_name = str(sample.get("montage", "unknown"))
            dataset_name = montage_name.split("/", maxsplit=1)[0]
            if (
                dataset_name not in dataset_names
                and len(dataset_names) == 1
            ):
                dataset_name = dataset_names[0]
            stats = counts.setdefault(
                dataset_name,
                {"total_samples": 0, "skipped_samples": 0, "reasons": {}},
            )
            stats["total_samples"] += 1
            rejection = adapter.get_sample_rejection(sample)
            if rejection is not None:
                stats["skipped_samples"] += 1
                reasons = stats["reasons"]
                code = rejection["code"]
                reason = reasons.setdefault(
                    code,
                    {
                        "code": code,
                        "message": rejection["message"],
                        "count": 0,
                    },
                )
                reason["count"] += 1
                continue
            sample_indices.append(sample_index)
        for dataset_name in dataset_names:
            counts.setdefault(
                dataset_name,
                {"total_samples": 0, "skipped_samples": 0, "reasons": {}},
            )
        for dataset_name, stats in sorted(counts.items()):
            diagnostic = {
                "dataset": dataset_name,
                "split": split_name,
                "total_samples": stats["total_samples"],
                "retained_samples": (
                    stats["total_samples"] - stats["skipped_samples"]
                ),
                "skipped_samples": stats["skipped_samples"],
                "reasons": [
                    stats["reasons"][code]
                    for code in sorted(stats["reasons"])
                ],
            }
            self._data_diagnostics[
                (dataset_name, split_name)
            ] = diagnostic
            warning_key = (dataset_name, split_name)
            if (
                diagnostic["skipped_samples"] > 0
                and get_is_master()
                and warning_key not in _WARNED_SAMPLE_FILTER_SPLITS
            ):
                _WARNED_SAMPLE_FILTER_SPLITS.add(warning_key)
                logger.warning(
                    "BrainOmni skipped %d/%d samples for dataset %s split %s; "
                    "reasons: %s.",
                    diagnostic["skipped_samples"],
                    diagnostic["total_samples"],
                    dataset_name,
                    split_name,
                    {
                        item["code"]: item["count"]
                        for item in diagnostic["reasons"]
                    },
                )
        empty_datasets = [
            dataset_name
            for dataset_name, stats in sorted(counts.items())
            if dataset_name in dataset_names
            and stats["total_samples"] > 0
            and stats["skipped_samples"] == stats["total_samples"]
        ]
        if empty_datasets:
            raise ValueError(
                "BrainOmni cannot create a data loader because no eligible "
                f"samples remain for split {split_name} in datasets "
                f"{empty_datasets}."
            )
        if not sample_indices:
            raise ValueError(
                "BrainOmni cannot create a data loader because no eligible "
                f"samples remain for split {split_name}."
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
        filtered_dataset = self._filter_invalid_samples(
            combined_dataset,
            adapter,
            split,
            dataset_names,
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

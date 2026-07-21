"""BrainOmni adapter using persisted EEG sensor coordinates."""

from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory, AbstractDatasetAdapter
from common.utils import ElectrodeSet


_EEG_SENSOR_TYPE_ID = 0
_SIGNAL_NORMALIZE_EPS_DEFAULT = 1.0e-5
_POSITION_NORMALIZE_EPS_DEFAULT = 1.0e-8


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
            raise ValueError("BrainOmni adapter expected signal_normalize_eps > 0.")
        if self.position_normalize_eps <= 0.0:
            raise ValueError("BrainOmni adapter expected position_normalize_eps > 0.")
        self.electrode_set = ElectrodeSet()
        super().__init__(dataset, dataset_names, dataset_configs)

    def _setup_adapter(self) -> None:
        self.model_name = "brainomni"
        self.scale = 1.0
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        return list(self.electrode_set.Electrodes)

    def _apply_model_specific_processing(
        self, data: torch.Tensor, montage_info: Dict[str, Any]
    ) -> torch.Tensor:
        del montage_info
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
                "BrainOmni requires persisted XYZ electrode positions, but this sample has none. "
                "Rebuild the dataset with native coordinates or its configured position_montage."
            )
        positions = torch.as_tensor(raw_positions, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                "BrainOmni adapter expected persisted pos shape (C, 3), "
                f"but got {tuple(positions.shape)}. Rebuild the dataset."
            )
        if not torch.isfinite(positions).all() or torch.any(torch.all(positions == 0.0, dim=1)):
            raise ValueError(
                "BrainOmni requires complete finite persisted XYZ electrode positions. "
                "Rebuild the dataset with a valid native layout or position_montage."
            )
        selector = torch.as_tensor(
            self.montage_mappings[str(result["montage"])]["sel"], dtype=torch.bool
        )
        if selector.numel() != positions.shape[0]:
            raise ValueError(
                "BrainOmni persisted positions do not align with unselected EEG channels. "
                "Rebuild the dataset."
            )
        positions = positions[selector]
        if positions.shape[0] != len(result["chs"]):
            raise ValueError(
                "BrainOmni persisted positions do not align with selected EEG channels."
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
        xyz_scale = torch.sqrt(3.0 * torch.mean(torch.sum(xyz_centered**2, dim=1)))
        if (
            torch.isnan(xyz_scale)
            or torch.isinf(xyz_scale)
            or xyz_scale.item() < self.position_normalize_eps
        ):
            raise ValueError(
                "BrainOmni position normalization failed: expected finite scale >= "
                f"{self.position_normalize_eps}, got {xyz_scale.item():.6e}."
            )
        normalized = pos.clone()
        normalized[:, :3] = xyz_centered / (xyz_scale + self.position_normalize_eps)
        return normalized


class BrainOmniDataLoaderFactory(AbstractDataLoaderFactory):
    """DataLoader factory for BrainOmni's persisted-position adapter."""

    def __init__(
        self, batch_size: int = 32, num_workers: int = 2, seed: int = 42,
        normalize_input: bool = True, normalize_position: bool = True,
        signal_normalize_eps: float = _SIGNAL_NORMALIZE_EPS_DEFAULT,
        position_normalize_eps: float = _POSITION_NORMALIZE_EPS_DEFAULT,
    ) -> None:
        super().__init__(batch_size=batch_size, num_workers=num_workers, seed=seed)
        self.normalize_input = normalize_input
        self.normalize_position = normalize_position
        self.signal_normalize_eps = signal_normalize_eps
        self.position_normalize_eps = position_normalize_eps

    def create_adapter(self, dataset: HFDataset, dataset_names: List[str], dataset_configs: List[str]) -> AbstractDatasetAdapter:
        return BrainOmniDatasetAdapter(
            dataset=dataset, dataset_names=dataset_names, dataset_configs=dataset_configs,
            normalize_input=self.normalize_input, normalize_position=self.normalize_position,
            signal_normalize_eps=self.signal_normalize_eps,
            position_normalize_eps=self.position_normalize_eps,
        )

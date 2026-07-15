"""BrainOmni dataset adapter and dataloader factory.

BrainOmni requires two additional channel-level inputs:
1. ``pos`` with shape ``(C, 6)`` for sensor position/orientation.
2. ``sensor_type`` with shape ``(C,)`` for sensor modality IDs.

This adapter derives both from EEG-FM-Bench channel indices.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple, Union, cast

import mne
import numpy as np
import torch
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory, AbstractDatasetAdapter
from common.utils import ElectrodeSet
from data.processor.wrapper import resolve_mne_montage_name, split_dataset_montage_key


logger = logging.getLogger("baseline")

_EEG_SENSOR_TYPE_ID = 0
_MAG_SENSOR_TYPE_ID = 1
_GRAD_SENSOR_TYPE_ID = 2
_SIGNAL_NORMALIZE_EPS_DEFAULT = 1.0e-5
_POSITION_NORMALIZE_EPS_DEFAULT = 1.0e-8
_DEFAULT_MNE_SFREQ = 256.0
_AUTO_MONTAGE_TOKEN = "auto"

_CHANNEL_ALIAS_MAP: Dict[str, str] = {
    "A1": "M1",
    "A2": "M2",
    "T1": "T9",
    "T2": "T10",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

# Matches BrainOmni's sensor_type semantics.
_PLANAR_COIL_TYPE_IDS = {3012}
_MAG_COIL_TYPE_IDS = {4001, 3022, 3024}


class BrainOmniDatasetAdapter(AbstractDatasetAdapter):
    """Dataset adapter for BrainOmni.

    Parameters
    ----------
    dataset : HFDataset
        HuggingFace dataset split.
    dataset_names : list[str]
        Dataset names participating in loading.
    dataset_configs : list[str]
        Dataset config names corresponding to ``dataset_names``.
    position_montage : str, optional
        MNE standard montage used to obtain sensor positions.
    normalize_input : bool, optional
        Whether to apply BrainOmni-style per-sample EEG normalization.
    allow_missing_positions : bool, optional
        Whether channels missing from montage coordinates are allowed.
    """

    def __init__(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
        position_montage: str = _AUTO_MONTAGE_TOKEN,
        normalize_input: bool = True,
        normalize_position: bool = True,
        allow_missing_positions: bool = False,
        signal_normalize_eps: float = _SIGNAL_NORMALIZE_EPS_DEFAULT,
        position_normalize_eps: float = _POSITION_NORMALIZE_EPS_DEFAULT,
    ) -> None:
        self.position_montage = position_montage
        self.normalize_input = normalize_input
        self.normalize_position = normalize_position
        self.allow_missing_positions = allow_missing_positions
        self.signal_normalize_eps = float(signal_normalize_eps)
        self.position_normalize_eps = float(position_normalize_eps)

        if not self.position_montage or not self.position_montage.strip():
            raise ValueError("BrainOmni adapter expected a non-empty position_montage value.")

        if self.signal_normalize_eps <= 0.0:
            raise ValueError(
                "BrainOmni adapter expected signal_normalize_eps > 0, "
                f"but got {self.signal_normalize_eps}."
            )
        if self.position_normalize_eps <= 0.0:
            raise ValueError(
                "BrainOmni adapter expected position_normalize_eps > 0, "
                f"but got {self.position_normalize_eps}."
            )

        self.electrode_set = ElectrodeSet()
        self._warned_missing_channels: set[str] = set()
        self._warned_all_zero_positions = False

        self._sensor_metadata_cache: Dict[
            Tuple[str, Tuple[str, ...]], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._resolved_montage_cache: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        self._montage_cache: Dict[str, mne.channels.DigMontage] = {}
        self._logged_auto_montage_keys: set[str] = set()

        super().__init__(dataset, dataset_names, dataset_configs)

    def _setup_adapter(self) -> None:
        """Initialize BrainOmni adapter options and montage mappings."""
        self.model_name = "brainomni"
        self.scale = 1.0
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        """Return channels supported by BrainOmni adapter."""
        return list(self.electrode_set.Electrodes)

    def _apply_model_specific_processing(self, data: torch.Tensor, montage_info: Dict[str, Any]) -> torch.Tensor:
        """Apply BrainOmni-style EEG normalization.

        Parameters
        ----------
        data : torch.Tensor
            Input EEG signal with shape ``(C, T)``.
        montage_info : Dict[str, Any]
            Montage metadata from abstract adapter, unused for now.

        Returns
        -------
        torch.Tensor
            Normalized EEG signal with shape ``(C, T)``.

        Raises
        ------
        ValueError
            If normalization variance is near zero or non-finite.
        """
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
                f"{self.signal_normalize_eps}, got {global_std.item():.6e} "
                f"for sample shape {tuple(data.shape)}."
            )

        return data / (global_std + self.signal_normalize_eps)

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, str, List[str], int]]:
        """Process one sample and append BrainOmni sensor metadata."""
        result = super()._process_sample(sample)

        chs_value = result["chs"]
        if isinstance(chs_value, torch.Tensor):
            chs_for_extract: Union[torch.Tensor, np.ndarray, List[int]] = chs_value
        elif isinstance(chs_value, np.ndarray):
            chs_for_extract = chs_value
        elif isinstance(chs_value, list):
            if any(not isinstance(idx, (int, np.integer)) for idx in chs_value):
                raise TypeError(
                    "BrainOmni adapter expected 'chs' list values to be integer channel indices."
                )
            chs_for_extract = [int(idx) for idx in chs_value]
        else:
            raise TypeError(
                "BrainOmni adapter expected 'chs' to be Tensor/ndarray/list, "
                f"but got {type(chs_value).__name__}."
            )

        ch_indices = self._extract_channel_indices(chs_for_extract)
        channel_names = self.electrode_set.get_electrodes_name(ch_indices)
        montage_key = str(result["montage"])

        pos, sensor_type = self._get_or_build_sensor_metadata(
            montage_key=montage_key,
            channel_names=channel_names,
        )

        result["pos"] = pos
        result["sensor_type"] = sensor_type
        return result

    def _get_or_build_sensor_metadata(
        self,
        montage_key: str,
        channel_names: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build or fetch cached BrainOmni sensor metadata from MNE ``raw.info``."""
        cache_key = (montage_key, tuple(channel_names))
        cached = self._sensor_metadata_cache.get(cache_key)
        if cached is not None:
            return cached[0].clone(), cached[1].clone()

        info, resolved_montage, missing_channels = self._build_mne_info(
            montage_key=montage_key,
            channel_names=channel_names,
        )
        self._handle_missing_positions(
            missing_channels=missing_channels,
            montage_key=montage_key,
            resolved_montage=resolved_montage,
        )

        pos_np, sensor_type_np = self._extract_pos_sensor_type(info)
        pos = torch.as_tensor(pos_np, dtype=torch.float32)
        pos = self._normalize_eeg_positions(pos)
        sensor_type = torch.as_tensor(sensor_type_np, dtype=torch.long)

        self._sensor_metadata_cache[cache_key] = (pos, sensor_type)
        return pos.clone(), sensor_type.clone()

    def _build_mne_info(
        self,
        montage_key: str,
        channel_names: Sequence[str],
    ) -> tuple[mne.Info, str, list[str]]:
        """Construct ``raw.info`` by setting a montage over sample channel names."""
        if len(channel_names) == 0:
            raise ValueError(
                "BrainOmni adapter expected at least one channel name to build raw.info, but got empty list."
            )

        resolved_montage = self._resolve_montage_name(
            montage_key=montage_key,
            channel_names=channel_names,
        )
        dig_montage = self._get_montage(resolved_montage)
        mapped_names, missing_channels = self._map_channels_to_montage_names(
            channel_names=channel_names,
            dig_montage=dig_montage,
        )

        info = mne.create_info(
            ch_names=list(mapped_names),
            sfreq=_DEFAULT_MNE_SFREQ,
            ch_types=cast(Any, self._infer_channel_types(channel_names)),
        )
        raw = mne.io.RawArray(
            np.zeros((len(mapped_names), 1), dtype=np.float32),
            info,
            verbose=False,
        )
        raw.set_montage(dig_montage, on_missing="ignore", verbose=False)
        return raw.info, resolved_montage, missing_channels

    def _resolve_montage_name(self, montage_key: str, channel_names: Sequence[str]) -> str:
        """Resolve montage name either explicitly or via benchmark-wide mapping."""
        if self.position_montage.lower() != _AUTO_MONTAGE_TOKEN:
            return self.position_montage

        cache_key = (montage_key, tuple(channel_names))
        cached = self._resolved_montage_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            _, benchmark_montage_name = split_dataset_montage_key(montage_key)
        except ValueError:
            if montage_key not in self._logged_auto_montage_keys:
                logger.warning(
                    "BrainOmni adapter could not split montage key '%s'. "
                    "Defaulting benchmark montage name to 'standard_1020'.",
                    montage_key,
                )
            benchmark_montage_name = "standard_1020"

        best_name = resolve_mne_montage_name(benchmark_montage_name)

        if montage_key not in self._logged_auto_montage_keys:
            logger.info(
                "BrainOmni adapter resolved montage '%s' for '%s' via benchmark mapping.",
                best_name,
                montage_key,
            )
            self._logged_auto_montage_keys.add(montage_key)

        self._resolved_montage_cache[cache_key] = best_name
        return best_name

    def _get_montage(self, montage_name: str) -> mne.channels.DigMontage:
        """Load and cache an MNE standard montage."""
        cached = self._montage_cache.get(montage_name)
        if cached is not None:
            return cached

        try:
            montage = mne.channels.make_standard_montage(montage_name)
        except Exception as exc:  # pragma: no cover - depends on MNE runtime catalog.
            raise ValueError(
                f"Failed to build MNE standard montage '{montage_name}' for BrainOmni adapter: {exc}"
            ) from exc

        self._montage_cache[montage_name] = montage
        return montage

    def _map_channels_to_montage_names(
        self,
        channel_names: Sequence[str],
        dig_montage: mne.channels.DigMontage,
    ) -> tuple[list[str], list[str]]:
        """Map channel names to montage-native names with alias support."""
        montage_lookup = {
            self._canonicalize_channel_name(ch_name): ch_name for ch_name in dig_montage.ch_names
        }

        mapped_names: list[str] = []
        missing_channels: list[str] = []
        for channel_name in channel_names:
            canonical_name = self._canonicalize_channel_name(channel_name)
            mapped_name = montage_lookup.get(canonical_name)
            if mapped_name is None:
                mapped_names.append(str(channel_name))
                missing_channels.append(str(channel_name))
            else:
                mapped_names.append(mapped_name)

        return mapped_names, missing_channels

    @staticmethod
    def _infer_channel_types(channel_names: Sequence[str]) -> list[str]:
        """Infer MNE channel types from names for info construction."""
        channel_types: list[str] = []
        for channel_name in channel_names:
            upper_name = str(channel_name).upper()
            if upper_name.startswith("MEG"):
                if "GRAD" in upper_name:
                    channel_types.append("grad")
                elif "MAG" in upper_name:
                    channel_types.append("mag")
                else:
                    logger.warning(
                        "BrainOmni adapter found unknown MEG channel type in '%s'. "
                        "Defaulting to 'mag'.",
                        channel_name,
                    )
                    channel_types.append("mag")
            else:
                channel_types.append("eeg")
        return channel_types

    @staticmethod
    def _canonicalize_channel_name(channel_name: str) -> str:
        """Canonicalize channel names for robust montage matching."""
        upper_name = str(channel_name).upper()
        return _CHANNEL_ALIAS_MAP.get(upper_name, upper_name)

    @staticmethod
    def _extract_pos_sensor_type(info: mne.Info) -> tuple[np.ndarray, np.ndarray]:
        """Extract ``pos`` and ``sensor_type`` from MNE ``info['chs']`` like BrainOmni."""
        pos_rows: list[np.ndarray] = []
        sensor_type_rows: list[int] = []

        for channel_info in info["chs"]:
            kind = int(channel_info["kind"])
            if kind not in [1, 2]:
                raise ValueError(f"BrainOmni adapter encountered unknown sensor kind: {kind}.")

            coil_type = int(channel_info["coil_type"])
            if kind == 2:
                xyz = np.asarray(channel_info["loc"][:3], dtype=np.float32)
                pos_rows.append(np.concatenate((xyz, np.zeros(3, dtype=np.float32)), axis=0))
                sensor_type_rows.append(_EEG_SENSOR_TYPE_ID)
                continue

            xyz = np.asarray(channel_info["loc"][:3], dtype=np.float32)
            dir_idx = 1 if coil_type in _PLANAR_COIL_TYPE_IDS else 3
            direction = np.asarray(
                channel_info["loc"][3 * dir_idx : 3 * (dir_idx + 1)],
                dtype=np.float32,
            )
            pos_rows.append(np.concatenate((xyz, direction), axis=0))

            if coil_type in _MAG_COIL_TYPE_IDS:
                sensor_type_rows.append(_MAG_SENSOR_TYPE_ID)
            else:
                if coil_type not in _PLANAR_COIL_TYPE_IDS:
                    logger.warning(
                        "BrainOmni adapter found unknown MEG coil type %d for channel '%s'. "
                        "Defaulting to gradiometer sensor type.",
                        coil_type,
                        channel_info["ch_name"],
                    )
                sensor_type_rows.append(_GRAD_SENSOR_TYPE_ID)

        if len(pos_rows) == 0:
            raise ValueError("BrainOmni adapter could not extract sensor metadata from empty MNE info['chs'].")

        pos = np.stack(pos_rows, axis=0).astype(np.float32)
        sensor_type = np.asarray(sensor_type_rows, dtype=np.int32)
        return pos, sensor_type

    def _normalize_eeg_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """Normalize EEG sensor xyz coordinates like BrainOmni ``normalize_pos``.

        Parameters
        ----------
        pos : torch.Tensor
            Sensor positions with shape ``(C, 6)``.

        Returns
        -------
        torch.Tensor
            Position tensor where xyz coordinates are centered and scaled.

        Raises
        ------
        ValueError
            If ``pos`` shape is invalid or normalization scale is ill-conditioned.
        """
        if not self.normalize_position:
            return pos

        if pos.ndim != 2 or pos.shape[1] != 6:
            raise ValueError(
                "BrainOmni adapter expected pos shape (C, 6), "
                f"but got {tuple(pos.shape)}."
            )

        xyz = pos[:, :3]
        known_mask = torch.any(xyz != 0.0, dim=1)
        if not bool(torch.any(known_mask)):
            if not self._warned_all_zero_positions:
                logger.warning(
                    "BrainOmni adapter skipped position normalization because all "
                    "channels currently have zero xyz coordinates."
                )
                self._warned_all_zero_positions = True
            return pos

        xyz_known = xyz[known_mask]
        xyz_mean = xyz_known.mean(dim=0, keepdim=True)
        xyz_centered = xyz_known - xyz_mean
        xyz_scale = torch.sqrt(3.0 * torch.mean(torch.sum(xyz_centered**2, dim=1)))

        if (
            torch.isnan(xyz_scale)
            or torch.isinf(xyz_scale)
            or xyz_scale.item() < self.position_normalize_eps
        ):
            raise ValueError(
                "BrainOmni position normalization failed: expected finite scale >= "
                f"{self.position_normalize_eps}, got {xyz_scale.item():.6e} "
                f"for pos shape {tuple(pos.shape)}."
            )

        normalized = pos.clone()
        normalized_xyz = xyz.clone()
        normalized_xyz[known_mask] = xyz_centered / (xyz_scale + self.position_normalize_eps)
        normalized[:, :3] = normalized_xyz
        return normalized

    @staticmethod
    def _extract_channel_indices(chs: Union[torch.Tensor, np.ndarray, List[int]]) -> List[int]:
        """Convert channel index container to python list of ints."""
        if isinstance(chs, torch.Tensor):
            return [int(idx) for idx in chs.detach().cpu().tolist()]
        if isinstance(chs, np.ndarray):
            return [int(idx) for idx in chs.tolist()]
        return [int(idx) for idx in chs]

    def _handle_missing_positions(
        self,
        missing_channels: Sequence[str],
        montage_key: str,
        resolved_montage: str,
    ) -> None:
        """Handle channels whose coordinates cannot be resolved for a montage."""
        missing_unique = sorted(set(str(ch) for ch in missing_channels))
        if len(missing_unique) == 0:
            return

        if not self.allow_missing_positions:
            raise ValueError(
                "BrainOmni adapter could not resolve positions for channels "
                f"{missing_unique} in sample montage '{montage_key}' using standard montage "
                f"'{resolved_montage}'. Set allow_missing_positions=true to use zero vectors."
            )

        unseen_channels: list[str] = []
        for channel_name in missing_unique:
            warn_key = f"{montage_key}|{resolved_montage}|{channel_name}"
            if warn_key not in self._warned_missing_channels:
                unseen_channels.append(channel_name)
                self._warned_missing_channels.add(warn_key)

        if unseen_channels:
            logger.warning(
                "BrainOmni adapter is using zero position vectors for channels %s "
                "in sample montage '%s' with resolved montage '%s'.",
                unseen_channels,
                montage_key,
                resolved_montage,
            )


class BrainOmniDataLoaderFactory(AbstractDataLoaderFactory):
    """DataLoader factory for BrainOmni adapter."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        position_montage: str = _AUTO_MONTAGE_TOKEN,
        normalize_input: bool = True,
        normalize_position: bool = True,
        allow_missing_positions: bool = False,
        signal_normalize_eps: float = _SIGNAL_NORMALIZE_EPS_DEFAULT,
        position_normalize_eps: float = _POSITION_NORMALIZE_EPS_DEFAULT,
    ) -> None:
        super().__init__(batch_size=batch_size, num_workers=num_workers, seed=seed)
        self.position_montage = position_montage
        self.normalize_input = normalize_input
        self.normalize_position = normalize_position
        self.allow_missing_positions = allow_missing_positions
        self.signal_normalize_eps = signal_normalize_eps
        self.position_normalize_eps = position_normalize_eps

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> AbstractDatasetAdapter:
        """Create BrainOmni dataset adapter instance."""
        return BrainOmniDatasetAdapter(
            dataset=dataset,
            dataset_names=dataset_names,
            dataset_configs=dataset_configs,
            position_montage=self.position_montage,
            normalize_input=self.normalize_input,
            normalize_position=self.normalize_position,
            allow_missing_positions=self.allow_missing_positions,
            signal_normalize_eps=self.signal_normalize_eps,
            position_normalize_eps=self.position_normalize_eps,
        )

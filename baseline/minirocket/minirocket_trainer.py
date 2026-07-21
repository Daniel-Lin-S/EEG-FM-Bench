"""External-source multivariate miniROCKET EEG feature extraction.

The implementation intentionally loads the upstream GPL-3.0 source at runtime
from a user-provided clone instead of copying it into this Apache-2.0 project.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from baseline.feature_extractor.trainer import (
    MINIROCKET_MIN_TIMEPOINTS,
    FeatureExtractorTrainer,
)
from baseline.minirocket.minirocket_config import MiniRocketConfig


class MiniRocketTrainer(FeatureExtractorTrainer):
    """Fit the upstream multivariate miniROCKET transform on training EEG."""

    def __init__(self, cfg: MiniRocketConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.parameters = None
        self._minirocket_module: ModuleType | None = None

    def _load_minirocket_module(self) -> ModuleType:
        """Load one upstream multivariate implementation from its clone path.

        Returns
        -------
        types.ModuleType
            Loaded ``minirocket_multivariate`` module.
        """
        source_path = self.cfg.model.minirocket_source_path
        if not source_path:
            raise ValueError(
                "model.minirocket_source_path is required. Clone "
                "https://github.com/angus924/minirocket and set this path to "
                "the clone root."
            )
        module_path = Path(source_path, "code", "minirocket_multivariate.py")
        if not module_path.is_file():
            raise FileNotFoundError(
                "Expected the upstream multivariate miniROCKET module at "
                f"{module_path.resolve()}. Clone "
                "https://github.com/angus924/minirocket and set "
                "model.minirocket_source_path to its root."
            )

        module_hash = hashlib.sha256(
            str(module_path.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        module_name = f"eeg_fm_bench_minirocket_{module_hash}"
        module = sys.modules.get(module_name)
        if module is not None:
            return module
        module_spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(
                f"Unable to load miniROCKET module from {module_path.resolve()}."
            )
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            sys.modules.pop(module_name, None)
            raise ImportError(
                "miniROCKET requires its runtime dependencies. Install them "
                "with: pip install -r requirements/feature_extractors.txt"
            ) from exc
        return module

    def fit_extractor(self, train_data: np.ndarray) -> None:
        """Fit miniROCKET kernel parameters on raw training EEG.

        Parameters
        ----------
        train_data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.
        """
        if train_data.dtype != np.float32:
            raise ValueError(
                "miniROCKET expected float32 EEG data, but got "
                f"{train_data.dtype}."
            )
        if train_data.shape[-1] < MINIROCKET_MIN_TIMEPOINTS:
            raise ValueError(
                "miniROCKET requires at least "
                f"{MINIROCKET_MIN_TIMEPOINTS} timepoints, but got "
                f"{train_data.shape[-1]}."
            )
        np.random.seed(self.cfg.seed)
        self._minirocket_module = self._load_minirocket_module()
        self.parameters = self._minirocket_module.fit(
            train_data,
            num_features=self.cfg.model.minirocket_num_features,
            max_dilations_per_kernel=(
                self.cfg.model.minirocket_max_dilations_per_kernel
            ),
        )

    def transform_features(self, data: np.ndarray) -> np.ndarray:
        """Transform EEG trials with fitted upstream miniROCKET parameters.

        Parameters
        ----------
        data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.

        Returns
        -------
        numpy.ndarray
            Feature matrix with shape ``(n_trials, n_features)``.
        """
        if self.parameters is None or self._minirocket_module is None:
            raise RuntimeError(
                "miniROCKET must be fitted on training data before transform."
            )
        return self._minirocket_module.transform(data, self.parameters)

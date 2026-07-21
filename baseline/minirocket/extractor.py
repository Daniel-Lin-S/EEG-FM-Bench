"""External-source multivariate miniROCKET feature extraction for EEG."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from baseline.feature_extractor.extractor import EEGFeatureExtractor
from baseline.minirocket.minirocket_config import MiniRocketExtractorArgs


MINIROCKET_MIN_TIMEPOINTS = 9


class MiniRocketFeatureExtractor(EEGFeatureExtractor):
    """Fit the upstream multivariate miniROCKET transform on training EEG."""

    def __init__(self, args: MiniRocketExtractorArgs, seed: int):
        self.args = args
        self.seed = seed
        self.parameters = None
        self._module: ModuleType | None = None

    def _fit(self, train_data: np.ndarray) -> None:
        """Fit upstream miniROCKET parameters from training EEG only."""
        if train_data.shape[-1] < MINIROCKET_MIN_TIMEPOINTS:
            raise ValueError(
                "miniROCKET requires at least "
                f"{MINIROCKET_MIN_TIMEPOINTS} timepoints, but got "
                f"{train_data.shape[-1]}."
            )
        np.random.seed(self.seed)
        self._module = self._load_module()
        self.parameters = self._module.fit(
            train_data,
            num_features=self.args.num_features,
            max_dilations_per_kernel=self.args.max_dilations_per_kernel,
        )

    def _transform(self, data: np.ndarray) -> np.ndarray:
        """Transform validated EEG with fitted upstream parameters."""
        if self.parameters is None or self._module is None:
            raise RuntimeError(
                "miniROCKET must be fitted on training data before transform."
            )
        return self._module.transform(data, self.parameters)

    def _load_module(self) -> ModuleType:
        """Load upstream ``minirocket_multivariate`` from its clone path."""
        source_path = self.args.source_path
        if not source_path:
            raise ValueError(
                "model.extractor.source_path is required. Clone "
                "https://github.com/angus924/minirocket and set this path to "
                "the clone root."
            )
        module_path = Path(source_path, "code", "minirocket_multivariate.py")
        if not module_path.is_file():
            raise FileNotFoundError(
                "Expected the upstream multivariate miniROCKET module at "
                f"{module_path.resolve()}. Clone "
                "https://github.com/angus924/minirocket and set "
                "model.extractor.source_path to its root."
            )
        module_name = self._module_name(module_path)
        module = sys.modules.get(module_name)
        if module is not None:
            return module
        module_spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(
                "Unable to load miniROCKET module from "
                f"{module_path.resolve()}."
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

    @staticmethod
    def _module_name(module_path: Path) -> str:
        """Create a stable module name for one external source path."""
        module_hash = hashlib.sha256(
            str(module_path.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        return f"eeg_fm_bench_minirocket_{module_hash}"

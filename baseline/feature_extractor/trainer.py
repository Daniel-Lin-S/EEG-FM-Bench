"""CPU-only training workflow for deterministic EEG feature extractors.

Each configured dataset is streamed from its fixed train, validation, and
test splits. Extractors receive bounded raw ``float32`` arrays shaped
``(batch, channels, timepoints)``. Ridge classifiers operate on disk-backed
features and are selected by validation performance.
"""

import csv
import datetime
import gc
import json
import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import datasets
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from baseline.abstract.trainer import AbstractTrainer, format_console_log_dict
from baseline.feature_extractor.classifier import (
    FeatureClassifier,
    ValidationSelectedRidgeClassifier,
)
from baseline.feature_extractor.artifacts import (
    require_single_seed,
    resolve_feature_extractor_log_path,
)
from baseline.feature_extractor.config import FeatureExtractorConfig
from baseline.feature_extractor.data import AlignedEEGBatch, FeatureSplitReader
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.feature_extractor.runtime import (
    AddressSpaceGuard,
    ModelRunLock,
    peak_resident_memory_bytes,
)
from baseline.feature_extractor.storage import ScratchArray, ScratchSpace
from baseline.feature_extractor.summary import write_feature_extractor_summary
from baseline.hpo.artifacts import check_completion_compatibility
from baseline.utils.identity import IDENTITY_VERSION
from baseline.utils.run_artifacts import get_config_hash, save_resolved_config
from common.distributed.env import get_is_master, get_world_size
from common.log import setup_log
from common.path import OUTPUT_ROOT
from data.processor.wrapper import (
    get_dataset_montage,
    get_dataset_n_class,
    resolve_common_montage_layout,
)


logger = logging.getLogger("baseline")

METRIC_NAME_TO_KEY = {
    "balanced_accuracy": "balanced_acc",
    "accuracy": "acc",
    "f1_weighted": "f1",
}


class FeatureExtractorTrainer(AbstractTrainer, ABC):
    """Shared per-dataset sklearn workflow for feature extractors.

    Parameters
    ----------
    cfg : FeatureExtractorConfig
        Feature-extractor baseline configuration.
    """

    def __init__(
        self,
        cfg: FeatureExtractorConfig,
        pipeline: Optional[FeatureExtractionPipeline] = None,
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.pipeline = pipeline
        self.final_validation_metrics: Dict[str, Dict[str, float]] = {}
        self._runtime_diagnostics: Dict[str, Dict[str, int]] = {}

    def setup_model(self):
        """Return no model because feature extractors run outside PyTorch."""
        return None

    def load_checkpoint(self, checkpoint_path: str):
        """Reject checkpoint loading for non-transferable feature extractors."""
        raise NotImplementedError(
            f"{self.model_type} does not save checkpoints and cannot load "
            f"{checkpoint_path}."
        )

    @abstractmethod
    def fit_extractor(self, train_data: np.ndarray) -> None:
        """Fit extractor state using training EEG data.

        Parameters
        ----------
        train_data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.
        """

    @abstractmethod
    def transform_features(self, data: np.ndarray) -> np.ndarray:
        """Extract one feature vector from every EEG trial.

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

    def get_train_io_path(self, args) -> Tuple[str, str]:
        """Create a seed log root without an empty checkpoint tree."""
        if not get_is_master():
            return "", ""

        require_single_seed(self.cfg)
        config = self.cfg.model_dump(mode="json")
        self.execution_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.execution_id = f"{self.execution_id}-{os.getpid()}"
        if self.log_dir_override is not None:
            log_path = self.log_dir_override
            reused_artifact_root = False
        else:
            log_path, reused_artifact_root = (
                resolve_feature_extractor_log_path(self.cfg)
            )
        log_path.mkdir(parents=True, exist_ok=True)
        if not reused_artifact_root:
            save_resolved_config(
                config,
                log_path / "configs" / f"{self.execution_id}.yaml",
            )
        return str(log_path.resolve()), ""

    def setup_logging(self):
        """Initialize local logging without checkpoint-directory reporting."""
        log_dir, ckpt_dir = self.get_train_io_path(self.cfg.logging)
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        if get_is_master():
            file_path = None
            if self._has_output("log"):
                file_path = str(
                    Path(log_dir, "logs", f"{self.execution_id}.log")
                )
            setup_log(
                file_path=file_path,
                start_time=self.start_time.timestamp(),
                name="baseline",
                level=self.cfg.logging.level.upper(),
            )
            logger.debug("Log directory: %s", Path(self.log_dir).resolve())

    def _open_csv_writer(self, ds_name: str) -> None:
        """Open a CSV trace without neural-training coordinates."""
        if not self._has_output("csv") or not get_is_master():
            return
        self._close_csv_writer()
        csv_path = Path(self.log_dir, "csv", f"{ds_name}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["timestamp", "dataset", "split", "metric", "value"],
        )
        self.csv_writer.writeheader()

    def _write_csv_metrics(self, log_data: dict) -> None:
        """Write final extractor metrics without epochs or optimizer steps."""
        if self.csv_writer is None:
            return
        for key, value in log_data.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            dataset, split, metric = self._parse_csv_metric_key(key)
            self.csv_writer.writerow(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "dataset": dataset,
                    "split": split,
                    "metric": metric,
                    "value": value,
                }
            )
        self.csv_file.flush()

    def _parse_csv_metric_key(self, key: str) -> tuple[str, str, str]:
        """Split one metric key into extractor CSV dimensions."""
        parts = key.split("/")
        dataset = self.current_dataset or ""
        if len(parts) >= 3 and parts[0] in self.ds_conf:
            return parts[0], parts[1], "/".join(parts[2:])
        if len(parts) >= 2:
            return dataset, parts[0], "/".join(parts[1:])
        return dataset, "", key

    def _dataset_is_complete(
        self,
        ds_name: str,
        ds_config: str,
    ) -> bool:
        """Return whether matching no-checkpoint metadata exists."""
        completion_path = self._completion_path(ds_name)
        if not completion_path.is_file():
            return False
        try:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            logger.warning(
                "Ignoring invalid completion metadata: %s",
                completion_path,
            )
            return False
        compatible = (
            completion.get("status") == "completed"
            and completion.get("dataset_config") == ds_config
            and completion.get("model_type") == self.model_type
            and completion.get("checkpoint_path") is None
            and completion.get("has_checkpoint") is False
        )
        if self.campaign_hash is None:
            return compatible
        return (
            compatible
            and check_completion_compatibility(
                completion_path,
                self.campaign_hash,
                self.cfg.seed,
                self.cfg.model_dump(mode="json"),
                campaign_aliases=self.campaign_aliases,
            ).compatible
        )

    def _write_completion(self, ds_name: str, ds_config: str) -> None:
        """Persist extractor metrics without a checkpoint reference."""
        test_metrics = self.final_test_metrics.get(ds_name)
        validation_metrics = self.final_validation_metrics.get(ds_name)
        if test_metrics is None or validation_metrics is None:
            raise RuntimeError(
                f"Cannot mark {ds_name} complete without validation and test "
                "metrics."
            )
        completion_path = self._completion_path(ds_name)
        completion_path.parent.mkdir(parents=True, exist_ok=True)
        completion = {
            "status": "completed",
            "campaign_hash": self.campaign_hash,
            "campaign_identity_version": IDENTITY_VERSION,
            "config_hash": self._resolved_config_hash(),
            "config_hash_version": IDENTITY_VERSION,
            "seed": self.cfg.seed,
            "dataset_config": ds_config,
            "execution_id": self.execution_id,
            "invocation_id": self.campaign_invocation_id,
            "checkpoint_retention_requested": False,
            "selection_provenance": self.selection_provenance,
            "model_type": self.model_type,
            "has_checkpoint": False,
            "checkpoint_path": None,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "completed_at": datetime.datetime.now().isoformat(),
        }
        diagnostics = self._build_completion_diagnostics(ds_name)
        if diagnostics:
            completion["diagnostics"] = diagnostics
        temporary_path = completion_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(completion, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(completion_path)

    def _fixed_channel_layout(
        self,
        ds_name: str,
        ds_config: str,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Return a shared fixed-width layout and every source montage."""
        montages = get_dataset_montage(ds_name, ds_config, self.cfg.fs)
        layout = resolve_common_montage_layout(
            montages, self.model_type, ds_name
        )
        return layout, montages

    def _validate_montage(self, ds_name: str, ds_config: str) -> None:
        """Validate that the dataset has a nonempty shared channel layout."""
        self._fixed_channel_layout(ds_name, ds_config)

    def _open_split_reader(
        self,
        ds_name: str,
        ds_config: str,
        split: datasets.NamedSplit,
        expected_trial_shape: Optional[tuple[int, int]] = None,
    ) -> FeatureSplitReader:
        """Open one classical Arrow reader without neural transformations."""
        channel_layout, montages = self._fixed_channel_layout(
            ds_name,
            ds_config,
        )
        return FeatureSplitReader(
            dataset_name=ds_name,
            dataset_config=ds_config,
            split=split,
            fs=self.cfg.fs,
            batch_size=self.cfg.data.load_batch_size,
            n_class=get_dataset_n_class(ds_name, ds_config),
            channel_layout=channel_layout,
            montages=montages,
            expected_trial_shape=expected_trial_shape,
        )

    def _scratch_root(self) -> Path:
        """Return the local scratch root without persisting it in results."""
        configured = self.cfg.data.scratch_dir
        if configured is not None:
            return Path(configured).expanduser().resolve()
        return Path(
            OUTPUT_ROOT,
            "scratch",
            "feature_extractors",
        ).resolve()

    @staticmethod
    def _advance_split_coverage(
        reader: FeatureSplitReader,
        batch: AlignedEEGBatch,
        expected_start: int,
    ) -> int:
        """Validate one contiguous batch and return its exclusive end."""
        batch_rows = batch.stop - batch.start
        if (
            batch.start != expected_start
            or batch_rows <= 0
            or batch.stop > len(reader)
            or len(batch.data) != batch_rows
            or len(batch.labels) != batch_rows
        ):
            raise ValueError(
                f"Expected contiguous {reader.dataset_name} {reader.split} "
                f"rows beginning at {expected_start}, but got interval "
                f"[{batch.start}, {batch.stop}) with {len(batch.data)} EEG "
                f"rows and {len(batch.labels)} labels."
            )
        return batch.stop

    @staticmethod
    def _require_complete_split_coverage(
        reader: FeatureSplitReader,
        covered_rows: int,
    ) -> None:
        """Require every declared split row to have been consumed once."""
        if covered_rows != len(reader):
            raise ValueError(
                f"Expected all {len(reader)} {reader.dataset_name} "
                f"{reader.split} rows, but consumed {covered_rows}."
            )

    def _copy_reader_to_raw_store(
        self,
        reader: FeatureSplitReader,
        scratch: ScratchSpace,
        store_name: str,
    ) -> tuple[ScratchArray, np.ndarray, tuple[int, int]]:
        """Stream one training split into a C-contiguous float32 memmap."""
        labels = np.empty(len(reader), dtype=np.int64)
        raw_store: Optional[ScratchArray] = None
        covered_rows = 0
        for batch in reader.batches():
            covered_rows = self._advance_split_coverage(
                reader,
                batch,
                covered_rows,
            )
            if raw_store is None:
                raw_store = scratch.create_array(
                    store_name,
                    (len(reader), *batch.data.shape[1:]),
                    np.float32,
                )
            raw_store.array[batch.start:batch.stop] = batch.data
            labels[batch.start:batch.stop] = batch.labels
        self._require_complete_split_coverage(reader, covered_rows)
        if raw_store is None or reader.trial_shape is None:
            raise ValueError(
                f"{reader.dataset_name} {reader.split} produced no usable "
                "training batches."
            )
        raw_store.array.flush()
        return raw_store, labels, reader.trial_shape

    def _extract_reader_to_feature_store(
        self,
        reader: FeatureSplitReader,
        scratch: ScratchSpace,
        store_name: str,
    ) -> tuple[ScratchArray, np.ndarray, tuple[int, int]]:
        """Stream Arrow batches directly into one feature memmap."""
        labels = np.empty(len(reader), dtype=np.int64)
        feature_store: Optional[ScratchArray] = None
        feature_width: Optional[int] = None
        feature_dtype: Optional[np.dtype] = None
        covered_rows = 0
        for batch in reader.batches():
            covered_rows = self._advance_split_coverage(
                reader,
                batch,
                covered_rows,
            )
            features = self.pipeline.transform(batch.data)
            if feature_store is None:
                feature_width = features.shape[1]
                feature_dtype = features.dtype
                feature_store = scratch.create_array(
                    store_name,
                    (len(reader), feature_width),
                    feature_dtype,
                )
            if (
                features.shape[1] != feature_width
                or features.dtype != feature_dtype
            ):
                raise ValueError(
                    f"Expected stable {reader.dataset_name} {reader.split} "
                    f"feature width and dtype, but got {features.shape[1]} "
                    f"and {features.dtype}."
                )
            feature_store.array[batch.start:batch.stop] = features
            labels[batch.start:batch.stop] = batch.labels
        self._require_complete_split_coverage(reader, covered_rows)
        if feature_store is None or reader.trial_shape is None:
            raise ValueError(
                f"{reader.dataset_name} {reader.split} produced no usable "
                "feature batches."
            )
        feature_store.array.flush()
        return feature_store, labels, reader.trial_shape

    def _extract_array_to_feature_store(
        self,
        data: np.ndarray,
        scratch: ScratchSpace,
        store_name: str,
    ) -> ScratchArray:
        """Transform a random-access EEG array in bounded row chunks."""
        feature_store: Optional[ScratchArray] = None
        feature_width: Optional[int] = None
        feature_dtype: Optional[np.dtype] = None
        batch_size = self.cfg.data.feature_batch_size
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            features = self.pipeline.transform(data[start:stop])
            if feature_store is None:
                feature_width = features.shape[1]
                feature_dtype = features.dtype
                feature_store = scratch.create_array(
                    store_name,
                    (len(data), feature_width),
                    feature_dtype,
                )
            if (
                features.shape[1] != feature_width
                or features.dtype != feature_dtype
            ):
                raise ValueError(
                    f"Expected stable feature width and dtype, but got "
                    f"{features.shape[1]} and {features.dtype}."
                )
            feature_store.array[start:stop] = features
        if feature_store is None:
            raise ValueError("Cannot extract features from an empty EEG array.")
        feature_store.array.flush()
        return feature_store

    @staticmethod
    def _validate_training_classes(
        train_labels: np.ndarray,
        n_class: int,
        ds_name: str,
    ) -> None:
        """Require every configured class to be represented during fitting."""
        train_classes = np.unique(train_labels)
        expected_classes = np.arange(n_class)
        if not np.array_equal(train_classes, expected_classes):
            raise ValueError(
                f"Expected {ds_name} training labels "
                f"{expected_classes.tolist()}, "
                f"but got {train_classes.tolist()}."
            )

    def _extract_features(
        self,
        data: np.ndarray,
        split_name: str,
    ) -> np.ndarray:
        """Transform EEG data and validate the resulting finite matrix."""
        features = np.asarray(self.transform_features(data), dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(
                f"Expected {split_name} feature shape (trials, features), but "
                f"got {features.shape}."
            )
        if features.shape[0] != data.shape[0] or features.shape[1] == 0:
            raise ValueError(
                f"Expected {split_name} features with shape "
                f"({data.shape[0]}, n_features > 0), but got {features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError(
                f"{split_name} extracted features contain NaN or inf."
            )
        return features

    def _fit_classifier(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> Tuple[Pipeline, float]:
        """Fit the shared validation-selected Ridge classifier.

        This compatibility helper delegates to the downstream classifier so
        scaler fitting remains independent of the alpha candidate count.
        """
        classifier = ValidationSelectedRidgeClassifier(
            self.cfg.model.classifier
        )
        classifier.fit(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
        )
        if classifier.pipeline is None or classifier.selected_alpha is None:
            raise RuntimeError(
                "Ridge classifier did not retain a fitted pipeline."
            )
        return classifier.pipeline, classifier.selected_alpha

    @staticmethod
    def _selection_score(
        labels: np.ndarray,
        predictions: np.ndarray,
        selection_metric: str,
    ) -> float:
        """Calculate one configured validation selection metric."""
        if selection_metric == "balanced_accuracy":
            return float(balanced_accuracy_score(labels, predictions))
        if selection_metric == "accuracy":
            return float(accuracy_score(labels, predictions))
        if selection_metric == "f1_weighted":
            return float(
                f1_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )
        raise ValueError(
            f"Unsupported Ridge selection metric: {selection_metric}."
        )

    def _evaluate(
        self,
        classifier: FeatureClassifier,
        features: np.ndarray,
        labels: np.ndarray,
        ds_name: str,
        prefix: str,
    ) -> Dict[str, float]:
        """Calculate benchmark classification metrics without neural loss."""
        predictions = classifier.predict(features)
        scores = None
        if self.ds_info[ds_name]["n_class"] == 2:
            scores = np.asarray(classifier.decision_function(features))
        return self._metrics_from_outputs(
            labels,
            predictions,
            scores,
            classifier.classes_,
            ds_name,
            prefix,
        )

    def _metrics_from_outputs(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        scores: Optional[np.ndarray],
        classes: np.ndarray,
        ds_name: str,
        prefix: str,
    ) -> Dict[str, float]:
        """Calculate metrics from bounded-prediction outputs."""
        if labels.shape != predictions.shape:
            raise ValueError(
                f"Expected matching {ds_name} {prefix} labels and "
                f"predictions, but got {labels.shape} and "
                f"{predictions.shape}."
            )
        metrics = {
            f"{ds_name}/{prefix}/acc": float(
                accuracy_score(labels, predictions)
            ),
            f"{ds_name}/{prefix}/balanced_acc": float(
                balanced_accuracy_score(labels, predictions)
            ),
        }
        n_class = self.ds_info[ds_name]["n_class"]
        if n_class == 2:
            if scores is None or scores.shape != labels.shape:
                actual_shape = None if scores is None else scores.shape
                raise ValueError(
                    f"Expected {ds_name} {prefix} binary scores with shape "
                    f"{labels.shape}, but got {actual_shape}."
                )
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"{ds_name} {prefix} decision scores contain NaN or inf."
                )
            positive_label = classes[1]
            binary_labels = labels == positive_label
            if np.unique(binary_labels).size < 2:
                warnings.warn(
                    f"Cannot calculate {ds_name} {prefix} AUROC or AUC-PR "
                    "because the split contains one binary class.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                metrics[f"{ds_name}/{prefix}/auroc"] = float(
                    roc_auc_score(binary_labels, scores)
                )
                metrics[f"{ds_name}/{prefix}/auc_pr"] = float(
                    average_precision_score(binary_labels, scores)
                )
        else:
            metrics[f"{ds_name}/{prefix}/cohen_kappa"] = float(
                cohen_kappa_score(labels, predictions)
            )
            metrics[f"{ds_name}/{prefix}/f1"] = float(
                f1_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )
        return metrics

    def _evaluate_stream(
        self,
        reader: FeatureSplitReader,
        classifier: FeatureClassifier,
        ds_name: str,
        prefix: str,
    ) -> Dict[str, float]:
        """Extract and evaluate a split without storing all test features."""
        labels = np.empty(len(reader), dtype=np.int64)
        predictions = np.empty(len(reader), dtype=classifier.classes_.dtype)
        scores: Optional[np.ndarray] = None
        if self.ds_info[ds_name]["n_class"] == 2:
            scores = np.empty(len(reader), dtype=np.float64)
        covered_rows = 0
        for batch in reader.batches():
            covered_rows = self._advance_split_coverage(
                reader,
                batch,
                covered_rows,
            )
            features = self.pipeline.transform(batch.data)
            batch_predictions = np.asarray(classifier.predict(features))
            expected_shape = (batch.stop - batch.start,)
            if batch_predictions.shape != expected_shape:
                raise ValueError(
                    f"Expected {ds_name} {prefix} predictions with shape "
                    f"{expected_shape}, but got {batch_predictions.shape}."
                )
            labels[batch.start:batch.stop] = batch.labels
            predictions[batch.start:batch.stop] = batch_predictions
            if scores is not None:
                batch_scores = np.asarray(
                    classifier.decision_function(features),
                    dtype=np.float64,
                )
                if batch_scores.shape != expected_shape:
                    raise ValueError(
                        f"Expected {ds_name} {prefix} scores with shape "
                        f"{expected_shape}, but got {batch_scores.shape}."
                    )
                scores[batch.start:batch.stop] = batch_scores
        self._require_complete_split_coverage(reader, covered_rows)
        return self._metrics_from_outputs(
            labels,
            predictions,
            scores,
            classifier.classes_,
            ds_name,
            prefix,
        )

    def get_data_diagnostics(self, ds_name: str) -> Dict[str, object]:
        """Add path-free classical runtime diagnostics to completions."""
        diagnostics = dict(super().get_data_diagnostics(ds_name))
        runtime = self._runtime_diagnostics.get(ds_name)
        if runtime is not None:
            diagnostics["feature_extractor_runtime"] = dict(runtime)
        return diagnostics

    def _run_dataset_classical(
        self,
        ds_name: str,
        ds_config: str,
        scratch: ScratchSpace,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Fit and evaluate one dataset through the classical data path."""
        train_reader = self._open_split_reader(
            ds_name,
            ds_config,
            datasets.Split.TRAIN,
        )
        requires_training = (
            self.pipeline.extractor.requires_random_access_training_data
        )
        if requires_training:
            raw_train, train_labels, trial_shape = (
                self._copy_reader_to_raw_store(
                    train_reader,
                    scratch,
                    "training_eeg",
                )
            )
            del train_reader
            gc.collect()
            self._validate_training_classes(
                train_labels,
                get_dataset_n_class(ds_name, ds_config),
                ds_name,
            )
            self.fit_extractor(raw_train.array)
            train_features = self._extract_array_to_feature_store(
                raw_train.array,
                scratch,
                "training_features",
            )
            raw_train.close()
        else:
            train_features, train_labels, trial_shape = (
                self._extract_reader_to_feature_store(
                    train_reader,
                    scratch,
                    "training_features",
                )
            )
            self._validate_training_classes(
                train_labels,
                get_dataset_n_class(ds_name, ds_config),
                ds_name,
            )
            del train_reader
            gc.collect()

        validation_reader = self._open_split_reader(
            ds_name,
            ds_config,
            datasets.Split.VALIDATION,
            expected_trial_shape=trial_shape,
        )
        validation_features, validation_labels, _ = (
            self._extract_reader_to_feature_store(
                validation_reader,
                scratch,
                "validation_features",
            )
        )
        del validation_reader
        gc.collect()

        classifier = self.pipeline.classifier
        classifier.fit(
            train_features.array,
            train_labels,
            validation_features.array,
            validation_labels,
        )
        validation_metrics = self._evaluate(
            classifier,
            validation_features.array,
            validation_labels,
            ds_name,
            "eval",
        )
        train_features.close()
        validation_features.close()
        gc.collect()

        test_reader = self._open_split_reader(
            ds_name,
            ds_config,
            datasets.Split.TEST,
            expected_trial_shape=trial_shape,
        )
        test_metrics = self._evaluate_stream(
            test_reader,
            classifier,
            ds_name,
            "test",
        )
        del test_reader
        gc.collect()
        return validation_metrics, test_metrics

    def run(self):
        """Run fixed-split feature extraction and validation-selected Ridge."""
        if get_world_size() != 1:
            raise RuntimeError(
                f"{self.model_type} supports one CPU process, but "
                "WORLD_SIZE is "
                f"{get_world_size()}."
            )
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.setup_device("cpu")

        lock_root = Path(OUTPUT_ROOT, "scratch", "feature_extractors")
        with ModelRunLock(
            lock_root,
            self.model_type,
        ), AddressSpaceGuard(self.cfg.data.memory_limit_gib) as memory_guard:
            if self.pipeline is not None:
                classifier = self.pipeline.classifier
                configure_memory = getattr(
                    classifier,
                    "configure_memory_check",
                    None,
                )
                if configure_memory is not None:
                    configure_memory(memory_guard.require_additional)
            self.setup_logging()
            self.init_tensorboard_logging()
            self.init_cloud_logging()
            try:
                for ds_name, ds_config in self.ds_conf.items():
                    if self._dataset_is_complete(ds_name, ds_config):
                        logger.info(
                            "Skipping completed dataset: %s",
                            ds_name,
                        )
                        continue
                    if self.pipeline is None:
                        raise RuntimeError(
                            "Feature-extractor trainer requires an extraction "
                            "pipeline for an incomplete dataset."
                        )
                    self.current_dataset = ds_name
                    self._reset_dataset_outputs(ds_name)
                    self._open_csv_writer(ds_name)
                    self._open_dataset_tensorboard(ds_name)
                    self._validate_montage(ds_name, ds_config)
                    self.collect_dataset_info(mixed=False, ds_name=ds_name)

                    scratch = ScratchSpace(
                        self._scratch_root(),
                        f"{self.model_type}-{ds_name}-{self.execution_id}",
                        memory_guard,
                    )
                    try:
                        with scratch:
                            validation_metrics, test_metrics = (
                                self._run_dataset_classical(
                                    ds_name,
                                    ds_config,
                                    scratch,
                                )
                            )
                    except MemoryError as exc:
                        raise RuntimeError(
                            f"{self.model_type} cannot process {ds_name} "
                            "within the configured classical memory bound: "
                            f"{exc}"
                        ) from exc

                    self._runtime_diagnostics[ds_name] = {
                        "memory_limit_bytes": (
                            memory_guard.effective_limit_bytes
                        ),
                        "peak_resident_memory_bytes": (
                            peak_resident_memory_bytes()
                        ),
                        "peak_scratch_bytes": scratch.peak_bytes,
                        "total_scratch_allocated_bytes": (
                            scratch.total_allocated_bytes
                        ),
                    }
                    self._log_completed_dataset(
                        ds_name,
                        ds_config,
                        validation_metrics,
                        test_metrics,
                    )
                    self._close_csv_writer()

                if get_is_master():
                    self._write_summary()
                logger.info(
                    "%s evaluation completed successfully.",
                    self.model_type,
                )
            finally:
                self._close_csv_writer()
                if self.pipeline is not None:
                    self.pipeline.close()
                self.finish_cloud_logging()
                self.finish_tensorboard_logging()

    def _log_completed_dataset(
        self,
        ds_name: str,
        ds_config: str,
        validation_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ) -> None:
        """Log final metrics and atomically mark one dataset complete."""
        classifier = self.pipeline.classifier
        selected_alpha = getattr(classifier, "selected_alpha", None)
        selection_metric = getattr(
            classifier.args,
            "selection_metric",
            None,
        )
        if selected_alpha is None or selection_metric is None:
            raise RuntimeError(
                "Feature classifier must expose selected alpha and selection "
                "metric for benchmark logging."
            )
        selected_metric_key = METRIC_NAME_TO_KEY[selection_metric]
        selection_metrics = {
            f"{ds_name}/selection/selected_alpha": selected_alpha,
            f"{ds_name}/selection/{selected_metric_key}": (
                validation_metrics[
                    f"{ds_name}/eval/{selected_metric_key}"
                ]
            ),
        }
        self._require_finite_metrics(
            selection_metrics,
            validation_metrics,
            test_metrics,
        )
        for metrics in (
            selection_metrics,
            validation_metrics,
            test_metrics,
        ):
            logger.info(format_console_log_dict(metrics, prefix=ds_name))
            self._log_to_tensorboard(metrics, self.current_step)
            self._write_csv_metrics(metrics)
            if self.cfg.logging.use_cloud:
                self._log_to_cloud(metrics)
        self.final_validation_metrics[ds_name] = validation_metrics
        self.final_test_metrics[ds_name] = test_metrics
        self._write_completion(ds_name, ds_config)

    @staticmethod
    def _require_finite_metrics(
        *metric_groups: Dict[str, float],
    ) -> None:
        """Reject completion when any reported metric is non-finite."""
        for metrics in metric_groups:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, bool) or not np.isfinite(
                    metric_value
                ):
                    raise ValueError(
                        f"Cannot complete feature extraction because metric "
                        f"{metric_name!r} is not finite: {metric_value!r}."
                    )

    def _write_summary(self) -> None:
        """Write one-seed campaign-compatible extractor summary tables."""
        require_single_seed(self.cfg)
        if self.campaign_invocation_id is not None:
            return
        write_feature_extractor_summary(
            Path(self.log_dir),
            self.model_type,
            self.cfg.seed,
            self.ds_conf,
            self._resolved_config_hash(),
        )

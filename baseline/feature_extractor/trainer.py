"""CPU-only training workflow for deterministic EEG feature extractors.

Each configured dataset is loaded in its fixed train, validation, and test
splits. Extractors receive raw ``float32`` arrays shaped
``(n_trials, n_channels, n_timepoints)``. Ridge classifiers operate on
standardized extracted features and are selected by validation performance.
"""

import csv
import datetime
import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import datasets
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

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
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.feature_extractor.summary import write_feature_extractor_summary
from baseline.hpo.artifacts import check_completion_compatibility
from baseline.utils.common import seed_torch
from baseline.utils.identity import IDENTITY_VERSION
from baseline.utils.run_artifacts import get_config_hash, save_resolved_config
from common.distributed.env import get_is_master, get_world_size
from common.log import setup_log
from data.processor.wrapper import (
    get_dataset_montage,
    get_dataset_n_class,
    load_concat_eeg_datasets,
)


logger = logging.getLogger("baseline")

MINIROCKET_MIN_TIMEPOINTS = 9
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

    def _validate_montage(self, ds_name: str, ds_config: str) -> None:
        """Reject datasets that cannot produce one fixed feature width."""
        montages = get_dataset_montage(ds_name, ds_config)
        if len(montages) != 1:
            raise ValueError(
                f"{self.model_type} requires exactly one montage for "
                f"{ds_name}, "
                f"but found {len(montages)} montages."
            )

    def _load_split(
        self,
        ds_name: str,
        ds_config: str,
        split: datasets.NamedSplit,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load one fixed benchmark split into dense EEG arrays."""
        dataset, _ = load_concat_eeg_datasets(
            dataset_names=[ds_name],
            builder_configs=[ds_config],
            split=split,
            cast_label=True,
            fs=self.cfg.fs,
        )
        if len(dataset) == 0:
            raise ValueError(f"{ds_name} {split} split contains no trials.")

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.data.load_batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )
        data_batches = []
        label_batches = []
        for batch in dataloader:
            data = torch.as_tensor(batch["data"], dtype=torch.float32)
            labels = torch.as_tensor(batch["label"], dtype=torch.long)
            if data.ndim != 3:
                raise ValueError(
                    f"Expected {ds_name} {split} data shape "
                    "(batch, channels, timepoints), but got "
                    f"{tuple(data.shape)}."
                )
            if data.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{ds_name} {split} has {data.shape[0]} EEG trials but "
                    f"{labels.shape[0]} labels in one batch."
                )
            if not torch.isfinite(data).all():
                raise ValueError(
                    f"{ds_name} {split} EEG data contains NaN or inf."
                )
            data_batches.append(data.numpy())
            label_batches.append(labels.numpy())

        data_all = np.concatenate(data_batches, axis=0)
        labels_all = np.concatenate(label_batches, axis=0)
        if data_all.shape[0] == 0:
            raise ValueError(
                f"{ds_name} {split} split contains no usable trials."
            )
        return data_all, labels_all

    @staticmethod
    def _validate_split_shapes(
        train_data: np.ndarray,
        validation_data: np.ndarray,
        test_data: np.ndarray,
        ds_name: str,
    ) -> None:
        """Require every split to have the training channel/time shape."""
        expected_shape = train_data.shape[1:]
        for split_name, data in (
            ("validation", validation_data),
            ("test", test_data),
        ):
            if data.shape[1:] != expected_shape:
                raise ValueError(
                    f"Expected {ds_name} {split_name} EEG shape "
                    f"(trials, {expected_shape[0]}, {expected_shape[1]}), "
                    f"but got {tuple(data.shape)}."
                )

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
            scores = np.asarray(classifier.decision_function(features))
            classes = classifier.classes_
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

    def run(self):
        """Run fixed-split feature extraction and validation-selected Ridge."""
        if get_world_size() != 1:
            raise RuntimeError(
                f"{self.model_type} supports one CPU process, but "
                "WORLD_SIZE is "
                f"{get_world_size()}."
            )
        seed_torch(self.cfg.seed)
        self.setup_device("cpu")
        self.setup_logging()
        self.init_tensorboard_logging()
        self.init_cloud_logging()

        for ds_name, ds_config in self.ds_conf.items():
            if self._dataset_is_complete(ds_name, ds_config):
                logger.info("Skipping completed dataset: %s", ds_name)
                continue
            self.current_dataset = ds_name
            self._reset_dataset_outputs(ds_name)
            self._open_csv_writer(ds_name)
            self._open_dataset_tensorboard(ds_name)
            self._validate_montage(ds_name, ds_config)
            self.collect_dataset_info(mixed=False, ds_name=ds_name)

            train_data, train_labels = self._load_split(
                ds_name,
                ds_config,
                datasets.Split.TRAIN,
            )
            validation_data, validation_labels = self._load_split(
                ds_name,
                ds_config,
                datasets.Split.VALIDATION,
            )
            test_data, test_labels = self._load_split(
                ds_name,
                ds_config,
                datasets.Split.TEST,
            )
            self._validate_split_shapes(
                train_data,
                validation_data,
                test_data,
                ds_name,
            )
            self._validate_training_classes(
                train_labels,
                get_dataset_n_class(ds_name, ds_config),
                ds_name,
            )

            fit_result = self.pipeline.fit(
                train_data,
                train_labels,
                validation_data,
                validation_labels,
            )
            validation_features = fit_result.validation_features
            test_features = self.pipeline.transform(test_data)
            classifier = self.pipeline.classifier
            validation_metrics = self._evaluate(
                classifier,
                validation_features,
                validation_labels,
                ds_name,
                "eval",
            )
            test_metrics = self._evaluate(
                classifier,
                test_features,
                test_labels,
                ds_name,
                "test",
            )
            selected_alpha = getattr(classifier, "selected_alpha", None)
            selection_metric = getattr(
                classifier.args,
                "selection_metric",
                None,
            )
            if selected_alpha is None or selection_metric is None:
                raise RuntimeError(
                    "Feature classifier must expose selected alpha and "
                    "selection metric for benchmark logging."
                )
            selected_metric_key = METRIC_NAME_TO_KEY[
                selection_metric
            ]
            selection_metrics = {
                f"{ds_name}/selection/selected_alpha": selected_alpha,
                f"{ds_name}/selection/{selected_metric_key}": (
                    validation_metrics[f"{ds_name}/eval/{selected_metric_key}"]
                ),
            }
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
            self._close_csv_writer()

        self.finish_cloud_logging()
        self.finish_tensorboard_logging()
        if get_is_master():
            self._write_summary()
        logger.info("%s evaluation completed successfully.", self.model_type)

    def _write_summary(self) -> None:
        """Write one-seed campaign-compatible extractor summary tables."""
        require_single_seed(self.cfg)
        write_feature_extractor_summary(
            Path(self.log_dir),
            self.model_type,
            self.cfg.seed,
            self.ds_conf,
            self._resolved_config_hash(),
        )
import os

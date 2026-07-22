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
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from baseline.abstract.trainer import AbstractTrainer, format_console_log_dict
from baseline.feature_extractor.classifier import FeatureClassifier
from baseline.feature_extractor.config import FeatureExtractorConfig
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.utils.common import seed_torch
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
        """Create a log directory without creating an empty checkpoint tree."""
        if not get_is_master():
            return "", ""

        config = self.cfg.model_dump(mode="json")
        config_hash = get_config_hash(config, multitask=False)
        experiment_name = f"{args.experiment_name}-{config_hash}"
        self.execution_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.execution_id = f"{self.execution_id}-{os.getpid()}"
        log_path = Path(args.run_dir, "log", "baseline", self.model_type)
        log_path = log_path / experiment_name
        log_path.mkdir(parents=True, exist_ok=True)
        save_resolved_config(
            config,
            log_path / "configs" / f"{self.execution_id}.yaml",
        )
        return str(log_path), ""

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
                level="INFO",
            )
            logger.info("log dir: %s", self.log_dir)

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

    def _dataset_is_complete(self, ds_name: str, ds_config: str) -> bool:
        """Return whether matching no-checkpoint metadata marks completion."""
        completion_path = self._completion_path(ds_name)
        if not completion_path.is_file():
            return False
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "Ignoring invalid completion metadata: %s",
                completion_path,
            )
            return False
        return (
            completion.get("status") == "completed"
            and completion.get("dataset_config") == ds_config
            and completion.get("model_type") == self.model_type
            and completion.get("checkpoint_path") is None
            and completion.get("has_checkpoint") is False
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
            "dataset_config": ds_config,
            "execution_id": self.execution_id,
            "model_type": self.model_type,
            "has_checkpoint": False,
            "checkpoint_path": None,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "completed_at": datetime.datetime.now().isoformat(),
        }
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
        """Fit and select Ridge candidates with the configured metric."""
        selection_metric = self.cfg.model.classifier.selection_metric
        best_classifier = None
        best_alpha = None
        best_score = -np.inf
        for alpha in sorted(self.cfg.model.classifier.alphas):
            classifier = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeClassifier(alpha=alpha)),
                ]
            )
            classifier.fit(train_features, train_labels)
            predictions = classifier.predict(validation_features)
            score = self._selection_score(
                validation_labels,
                predictions,
                selection_metric,
            )
            if not np.isfinite(score):
                raise ValueError(
                    f"Validation {selection_metric} is NaN for Ridge alpha "
                    f"{alpha}."
                )
            if score > best_score:
                best_classifier = classifier
                best_alpha = alpha
                best_score = score

        if best_classifier is None or best_alpha is None:
            raise RuntimeError("No Ridge classifier candidate was fitted.")
        logger.info(
            "Selected Ridge alpha %.6g with validation %s %.6f.",
            best_alpha,
            selection_metric,
            best_score,
        )
        return best_classifier, float(best_alpha)

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

            self.pipeline.fit(
                train_data,
                train_labels,
                validation_data,
                validation_labels,
            )
            validation_features = self.pipeline.transform(validation_data)
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
        logger.info("%s evaluation completed successfully.", self.model_type)
import os

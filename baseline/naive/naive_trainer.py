"""Run a signal-free majority-class baseline on fixed benchmark splits.

The trainer reads only each processed dataset's ``label`` column. It never
constructs batches or accesses the EEG ``data`` column. Its output mirrors the
flat deterministic artifact format used by the feature-extractor baselines.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path

import datasets
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import f1_score, roc_auc_score

from baseline.feature_extractor.summary import write_feature_extractor_summary
from baseline.feature_extractor.trainer import FeatureExtractorTrainer
from baseline.naive.classifier import MajorityVoteClassifier
from baseline.naive.classifier import _validate_labels
from baseline.naive.naive_config import NaiveConfig
from baseline.utils.common import seed_torch
from common.distributed.env import get_is_master, get_world_size
from data.processor.wrapper import get_dataset_n_class, load_concat_eeg_datasets


logger = logging.getLogger("baseline")
BINARY_CLASS_COUNT = 2


class NaiveTrainer(FeatureExtractorTrainer):
    """Evaluate per-dataset majority predictions without EEG input."""

    def __init__(self, cfg: NaiveConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.classifiers: dict[str, MajorityVoteClassifier] = {}

    def fit_extractor(self, train_data: np.ndarray) -> None:
        """Reject feature extraction because this baseline has no EEG input."""
        raise RuntimeError("naive must not fit an EEG feature extractor.")

    def transform_features(self, data: np.ndarray) -> np.ndarray:
        """Reject feature extraction because this baseline has no EEG input."""
        raise RuntimeError("naive must not transform EEG features.")

    def _load_labels(
        self,
        dataset_name: str,
        dataset_config: str,
        split: datasets.NamedSplit,
    ) -> np.ndarray:
        """Load only one processed dataset split's label column.

        Parameters
        ----------
        dataset_name : str
            Registered dataset identifier.
        dataset_config : str
            Registered dataset-builder configuration.
        split : datasets.NamedSplit
            Fixed benchmark split to load.

        Returns
        -------
        numpy.ndarray
            Integer labels with shape ``(n_trials,)``.
        """
        dataset, _ = load_concat_eeg_datasets(
            dataset_names=[dataset_name],
            builder_configs=[dataset_config],
            split=split,
            cast_label=True,
            fs=self.cfg.fs,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"{dataset_name} {split} split contains no trials."
            )
        if "label" not in dataset.column_names:
            raise ValueError(
                f"{dataset_name} {split} split does not contain labels."
            )
        labels = np.asarray(dataset["label"])
        _validate_labels(
            labels,
            get_dataset_n_class(dataset_name, dataset_config),
            dataset_name,
            _split_name(split),
        )
        return labels.astype(np.int64, copy=False)

    def _evaluate(
        self,
        classifier: MajorityVoteClassifier,
        labels: np.ndarray,
        dataset_name: str,
        split_name: str,
        n_class: int,
    ) -> dict[str, float]:
        """Evaluate constant predictions against one held-out label vector."""
        predictions = classifier.predict(labels.size)
        metrics = {
            f"{dataset_name}/{split_name}/acc": float(
                accuracy_score(labels, predictions)
            ),
            f"{dataset_name}/{split_name}/balanced_acc": float(
                balanced_accuracy_score(labels, predictions)
            ),
        }
        if n_class == BINARY_CLASS_COUNT:
            self._add_binary_metrics(
                metrics,
                classifier.positive_scores(labels.size),
                labels,
                dataset_name,
                split_name,
            )
        else:
            metrics[f"{dataset_name}/{split_name}/cohen_kappa"] = float(
                cohen_kappa_score(labels, predictions)
            )
            metrics[f"{dataset_name}/{split_name}/f1"] = float(
                f1_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )
        self._validate_metrics(metrics, dataset_name, split_name)
        return metrics

    @staticmethod
    def _add_binary_metrics(
        metrics: dict[str, float],
        scores: np.ndarray,
        labels: np.ndarray,
        dataset_name: str,
        split_name: str,
    ) -> None:
        """Add binary ranking metrics when the split has both classes."""
        if np.unique(labels).size < BINARY_CLASS_COUNT:
            warnings.warn(
                f"Cannot calculate {dataset_name} {split_name} AUROC or "
                "AUC-PR because the split contains one binary class.",
                UserWarning,
                stacklevel=2,
            )
            return
        metrics[f"{dataset_name}/{split_name}/auroc"] = float(
            roc_auc_score(labels, scores)
        )
        metrics[f"{dataset_name}/{split_name}/auc_pr"] = float(
            average_precision_score(labels, scores)
        )

    @staticmethod
    def _validate_metrics(
        metrics: dict[str, float],
        dataset_name: str,
        split_name: str,
    ) -> None:
        """Reject undefined metrics instead of persisting NaN results."""
        invalid_metrics = {
            name: value
            for name, value in metrics.items()
            if not math.isfinite(value)
        }
        if invalid_metrics:
            raise ValueError(
                f"{dataset_name} {split_name} produced non-finite metrics: "
                f"{invalid_metrics}."
            )

    def _reset_dataset_outputs(self, dataset_name: str) -> None:
        """Delete only incomplete label-baseline artifacts for one dataset."""
        if not get_is_master():
            return
        self._completion_path(dataset_name).unlink(missing_ok=True)
        Path(self.log_dir, "csv", f"{dataset_name}.csv").unlink(
            missing_ok=True
        )

    def _write_completion(
        self,
        dataset_name: str,
        dataset_config: str,
    ) -> None:
        """Persist deterministic metrics and the fitted label distribution."""
        super()._write_completion(dataset_name, dataset_config)
        completion_path = self._completion_path(dataset_name)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["majority_vote"] = self.classifiers[dataset_name].metadata()
        temporary_path = completion_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(completion, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(completion_path)

    def run(self) -> None:
        """Fit training-label modes and evaluate fixed held-out splits."""
        if get_world_size() != 1:
            raise RuntimeError(
                "naive supports one CPU process, but WORLD_SIZE is "
                f"{get_world_size()}."
            )
        seed_torch(self.cfg.seed)
        self.setup_device("cpu")
        self.setup_logging()
        self.init_tensorboard_logging()
        self.init_cloud_logging()

        for dataset_name, dataset_config in self.ds_conf.items():
            if self._dataset_is_complete(dataset_name, dataset_config):
                logger.info("Skipping completed dataset: %s", dataset_name)
                continue
            self.current_dataset = dataset_name
            self._reset_dataset_outputs(dataset_name)
            self._open_csv_writer(dataset_name)
            try:
                n_class = get_dataset_n_class(dataset_name, dataset_config)
                train_labels = self._load_labels(
                    dataset_name,
                    dataset_config,
                    datasets.Split.TRAIN,
                )
                validation_labels = self._load_labels(
                    dataset_name,
                    dataset_config,
                    datasets.Split.VALIDATION,
                )
                test_labels = self._load_labels(
                    dataset_name,
                    dataset_config,
                    datasets.Split.TEST,
                )
                classifier = MajorityVoteClassifier(
                    self.cfg.seed,
                    dataset_name,
                ).fit(train_labels, n_class)
                self.classifiers[dataset_name] = classifier
                validation_metrics = self._evaluate(
                    classifier,
                    validation_labels,
                    dataset_name,
                    "eval",
                    n_class,
                )
                test_metrics = self._evaluate(
                    classifier,
                    test_labels,
                    dataset_name,
                    "test",
                    n_class,
                )
                for metrics in (validation_metrics, test_metrics):
                    logger.info(
                        self._format_metrics(metrics, dataset_name)
                    )
                    self._log_to_tensorboard(metrics, self.current_step)
                    self._write_csv_metrics(metrics)
                    if self.cfg.logging.use_cloud:
                        self._log_to_cloud(metrics)
                self.final_validation_metrics[dataset_name] = validation_metrics
                self.final_test_metrics[dataset_name] = test_metrics
                self._write_completion(dataset_name, dataset_config)
            finally:
                self._close_csv_writer()

        self.finish_cloud_logging()
        self.finish_tensorboard_logging()
        if get_is_master():
            self._write_summary()
        logger.info("naive evaluation completed successfully.")

    def _write_summary(self) -> None:
        """Write one-seed summaries in the shared deterministic format."""
        if self.campaign_invocation_id is not None:
            return
        write_feature_extractor_summary(
            Path(self.log_dir),
            self.model_type,
            self.cfg.seed,
            self.ds_conf,
            self._resolved_config_hash(),
        )

    @staticmethod
    def _format_metrics(metrics: dict[str, float], dataset_name: str) -> str:
        """Format final metrics without training-step coordinates."""
        values = [
            f"{key.removeprefix(dataset_name + '/')}: {value:.5f}"
            for key, value in metrics.items()
        ]
        return f"{dataset_name} " + ", ".join(values)


def _split_name(split: datasets.NamedSplit) -> str:
    """Map the fixed dataset split object to its user-facing label."""
    if split == datasets.Split.TRAIN:
        return "training"
    if split == datasets.Split.VALIDATION:
        return "validation"
    if split == datasets.Split.TEST:
        return "test"
    raise ValueError(f"Unsupported naive dataset split: {split}.")

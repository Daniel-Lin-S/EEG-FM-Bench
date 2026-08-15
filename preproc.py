"""Build preprocessed datasets from a YAML configuration.

Configuration parameters
------------------------
conf_file : str (CLI only)
    Path, name, or repository-relative name of the YAML to load. It must
    define the target sampling rate.
fs : int
    Sampling rate in Hz supplied to every dataset builder. It determines the
    resampled data written to disk and must match training.
clean_middle_cache : bool, optional, default=False
    Clears a selected builder's intermediate disk cache before rebuilding it.
clean_shared_info : bool, optional, default=False
    Also clears shared builder metadata during cache cleanup; ignored unless
    ``clean_middle_cache`` is true.
refresh_fields : list[str], optional, default=[]
    Field flows to force-refresh when ``refresh_arrow`` is true. ``pos`` is
    currently supported.
refresh_arrow : bool, optional, default=False
    Explicitly rebuild final Arrow artifacts from valid intermediate and field
    caches. Existing completed Arrow artifacts are never changed when false.
num_preproc_arrow_writers : int, optional, default=4
    Worker processes used by ``download_and_prepare`` to materialize the final
    Arrow dataset.
num_preproc_mid_workers : int, optional, default=6
    Worker processes used by the builder's ``preproc`` call to create
    intermediate processed records.
pretrain_datasets : list[str], optional, default=[]
    Registry names built with the builder configuration named ``pretrain``.
finetune_datasets : dict[str, str], optional, default={}
    Dataset registry name to builder-configuration mapping for fine-tuning,
    e.g. ``tuab: finetune``.
    For 'pretrain' configuration, no label is stored and test ratio is 0.
    For 'finetune' or related configurations (e.g., `finetune-reach`), segments are labelled and test-split is preserved.

Values passed after ``conf_file`` on the command line override YAML values,
except ``fs``, which must remain defined by the YAML.
Every dataset and configuration is checked against ``DATASET_SELECTOR`` before
it runs. ``common.conf.BasePreprocArgs`` documents the validation schema.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from dataclasses import dataclass
from typing import Any, Literal, Type

from omegaconf import DictConfig, OmegaConf

from common.conf import BasePreprocArgs
from common.log import setup_log
from common.path import get_conf_file_path
from data.processor.builder import EEGDatasetBuilder
from data.processor.wrapper import DATASET_SELECTOR, resolve_dataset_builder


logger = logging.getLogger('preproc')


@dataclass(frozen=True)
class DatasetPreparationResult:
    """Final status for one requested dataset/configuration pair."""

    dataset_name: str
    config_name: str
    status: Literal['success', 'failed']
    reason: str = ''
    warning_count: int = 0
    warning_messages: tuple[str, ...] = ()
    sample_count: int = 0
    output_dir: str = ''

    @property
    def succeeded(self) -> bool:
        return self.status == 'success'


def _dataset_splits(dataset: Any) -> list[Any]:
    if isinstance(dataset, Mapping):
        return list(dataset.values())
    return [dataset]


def _validate_prepared_dataset(dataset: Any, builder: EEGDatasetBuilder) -> tuple[int, str]:
    """Verify that a loaded dataset is nonempty and backed by real artifacts."""
    splits = _dataset_splits(dataset)
    is_remote = bool(getattr(builder.config, 'is_remote_fs', False))
    sample_count = sum(len(split) for split in splits)
    if sample_count < 1:
        raise RuntimeError('prepared dataset contains no samples')

    artifact_files = []
    for split in splits:
        for cache_file in getattr(split, 'cache_files', []):
            filename = cache_file.get('filename') if isinstance(cache_file, Mapping) else None
            if filename:
                artifact_files.append(
                    str(filename) if is_remote else os.path.abspath(filename)
                )

    if not is_remote:
        if not artifact_files:
            raise RuntimeError('loaded dataset does not reference any Arrow artifact files')
        missing_files = [path for path in artifact_files if not os.path.isfile(path)]
        if missing_files:
            preview = ', '.join(missing_files[:3])
            suffix = '' if len(missing_files) <= 3 else f' (+{len(missing_files) - 3} more)'
            raise FileNotFoundError(f'processed Arrow artifacts are missing: {preview}{suffix}')

    if artifact_files and not is_remote:
        output_dir = os.path.commonpath([os.path.dirname(path) for path in artifact_files])
    else:
        output_dir = str(builder.cache_dir)
    return sample_count, output_dir


def _failure_reason(error: Exception) -> str:
    message = str(error).strip()
    return f'{type(error).__name__}: {message}' if message else type(error).__name__


def _processed_arrow_root(builder: EEGDatasetBuilder) -> str | None:
    """Return the final Arrow root for a builder with local output paths.

    Parameters
    ----------
    builder : EEGDatasetBuilder
        Dataset builder whose final output location is inspected.

    Returns
    -------
    str, optional
        Absolute final Arrow root, or ``None`` when the builder does not expose
        a local processed-output contract.
    """
    config = getattr(builder, 'config', None)
    if config is None or bool(getattr(config, 'is_remote_fs', False)):
        return None
    data_path = getattr(config, 'data_path', None)
    dataset_name = getattr(config, 'dataset_name', None)
    config_name = getattr(config, 'name', None)
    if not all(isinstance(value, str) and value for value in (
            data_path,
            dataset_name,
            config_name,
    )):
        return None
    return os.path.join(data_path, dataset_name, config_name)


def _load_existing_processed_dataset(
        builder: EEGDatasetBuilder,
) -> tuple[int, str] | None:
    """Validate an existing local Arrow dataset without altering it.

    Parameters
    ----------
    builder : EEGDatasetBuilder
        Builder used only to load and validate final Arrow artifacts.

    Returns
    -------
    tuple[int, str], optional
        Sample count and output directory when a valid final dataset exists.
        ``None`` when no final Arrow artifact exists.

    Raises
    ------
    RuntimeError
        If Arrow artifacts exist but are not a valid completed dataset.
    """
    arrow_root = _processed_arrow_root(builder)
    if arrow_root is None:
        return None
    arrow_files = []
    if os.path.isdir(arrow_root):
        for root, _, files in os.walk(arrow_root):
            arrow_files.extend(
                os.path.join(root, file_name)
                for file_name in files
                if file_name.endswith('.arrow')
            )
    if not arrow_files:
        return None
    try:
        return _validate_prepared_dataset(builder.as_dataset(), builder)
    except Exception as error:
        raise RuntimeError(
            'Existing processed Arrow artifacts are incomplete or unreadable; '
            'refusing to replace them while refresh_arrow is false.'
        ) from error


def prepare_dataset(
        conf: BasePreprocArgs,
        builder_cls: Type[EEGDatasetBuilder],
        dataset_name: str,
        config_name: str
) -> DatasetPreparationResult:
    builder = None
    try:
        logger.info(f"Preparing dataset {dataset_name} {config_name} at fs={conf.fs}Hz...")
        builder = builder_cls(config_name, fs=conf.fs)
        if not conf.refresh_arrow:
            existing = _load_existing_processed_dataset(builder)
            if existing is not None:
                sample_count, output_dir = existing
                logger.info(
                    f'Using completed Arrow dataset for {dataset_name} '
                    f'{config_name} at {output_dir}.'
                )
                return DatasetPreparationResult(
                    dataset_name=dataset_name,
                    config_name=config_name,
                    status='success',
                    sample_count=sample_count,
                    output_dir=output_dir,
                )
        if conf.clean_middle_cache:
            builder.clean_all_cache(clean_shared_info=conf.clean_shared_info)
        builder.preproc(n_proc=conf.num_preproc_mid_workers)
        if conf.refresh_arrow:
            materialize_fields = getattr(builder, 'materialize_fields', None)
            if materialize_fields is not None:
                materialize_fields(conf.refresh_fields)
            builder.clean_arrow_set()
        builder.download_and_prepare(num_proc=conf.num_preproc_arrow_writers)
        dataset = builder.as_dataset()
        sample_count, output_dir = _validate_prepared_dataset(dataset, builder)
        warning_messages = tuple(getattr(builder, 'preproc_warning_messages', ()))
        warning_count = int(getattr(builder, 'preproc_warning_count', len(warning_messages)))
        logger.info(
            f'Dataset {dataset_name} {config_name} at fs={conf.fs}Hz is prepared '
            f'with {sample_count} samples at {output_dir}.'
        )
        return DatasetPreparationResult(
            dataset_name=dataset_name,
            config_name=config_name,
            status='success',
            warning_count=warning_count,
            warning_messages=warning_messages,
            sample_count=sample_count,
            output_dir=output_dir,
        )
    except Exception as error:
        reason = _failure_reason(error)
        logger.exception(f'Preparation of dataset {dataset_name} {config_name} failed: {reason}')
        output_dir = str(builder.cache_dir) if builder is not None else ''
        return DatasetPreparationResult(
            dataset_name=dataset_name,
            config_name=config_name,
            status='failed',
            reason=reason,
            warning_count=int(getattr(builder, 'preproc_warning_count', 0)),
            warning_messages=tuple(getattr(builder, 'preproc_warning_messages', ())),
            output_dir=output_dir,
        )


def preproc(conf: BasePreprocArgs) -> list[DatasetPreparationResult]:
    dataset_names = list(conf.pretrain_datasets)
    dataset_configs = ['pretrain' for _ in dataset_names]
    dataset_names.extend(conf.finetune_datasets.keys())
    dataset_configs.extend(conf.finetune_datasets.values())

    results = []
    for dataset_name, config_name in zip(dataset_names, dataset_configs):
        try:
            builder_cls = resolve_dataset_builder(
                dataset_name,
                dataset_selector=DATASET_SELECTOR,
            )
            if config_name not in builder_cls.builder_configs:
                raise ValueError(
                    f'Config {config_name} is not supported for dataset {dataset_name}'
                )
        except Exception as error:
            reason = _failure_reason(error)
            logger.error(f'Cannot prepare dataset {dataset_name} {config_name}: {reason}')
            results.append(DatasetPreparationResult(
                dataset_name=dataset_name,
                config_name=config_name,
                status='failed',
                reason=reason,
            ))
            continue

        results.append(prepare_dataset(
            conf, builder_cls, dataset_name, config_name,
        ))
    return results


def print_preprocessing_summary(results: list[DatasetPreparationResult]) -> None:
    print('======= PREPROCESSING SUMMARY =======')
    for result in results:
        identity = f'{result.dataset_name}/{result.config_name}'
        if not result.succeeded:
            print(f'FAILED: {identity} — {result.reason}')
            continue

        label = 'SUCCESS WITH WARNINGS' if result.warning_count else 'SUCCESS'
        warning_text = (
            f' — {result.warning_count} recording(s) skipped'
            if result.warning_count else ''
        )
        print(
            f'{label}: {identity} — {result.sample_count:,} samples'
            f'{warning_text} — {result.output_dir}'
        )

    succeeded = sum(result.succeeded for result in results)
    failed = len(results) - succeeded
    warned = sum(result.succeeded and result.warning_count > 0 for result in results)
    overall = 'FAILED' if failed else 'SUCCESS'
    print(
        f'OVERALL: {overall} — {succeeded} succeeded, {failed} failed, '
        f'{warned} with warnings'
    )


def _load_config() -> BasePreprocArgs:
    cli_args: DictConfig = OmegaConf.from_cli()
    logger.info(cli_args)
    if "conf_file" not in cli_args:
        raise ValueError(
            "Preprocessing requires conf_file YAML with a top-level fs value."
        )
    logger.info(cli_args.conf_file)
    file_cfg = OmegaConf.load(get_conf_file_path(cli_args.conf_file))
    cli_args.pop("conf_file")
    if "fs" in cli_args:
        raise ValueError(
            "Preprocessing fs must be defined by the YAML, not the CLI."
        )
    if "fs" not in file_cfg:
        raise ValueError(
            "Preprocessing YAML must define a top-level fs value."
        )

    cfg = OmegaConf.merge(file_cfg, cli_args)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.info(cfg)
    return BasePreprocArgs.model_validate(cfg)


def main() -> int:
    setup_log(name='preproc')
    try:
        conf = _load_config()
    except Exception as error:
        reason = _failure_reason(error)
        logger.exception(f'Preprocessing configuration failed: {reason}')
        print('======= PREPROCESSING SUMMARY =======')
        print(f'FAILED: run configuration — {reason}')
        print('OVERALL: FAILED — preprocessing did not start')
        return 2

    results = preproc(conf)
    print_preprocessing_summary(results)
    return 1 if any(not result.succeeded for result in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Build preprocessed datasets from a YAML configuration.

Configuration parameters
------------------------
conf_file : str, optional (CLI only), default=None
    Path, name, or repository-relative name of the YAML to load. When omitted,
    no file is loaded and ``BasePreprocArgs`` defaults are used.
fs : int, optional, default=256
    Sampling rate in Hz supplied to every dataset builder. It determines the
    resampled data written to disk and must match training.
clean_middle_cache : bool, optional, default=False
    Clears a selected builder's intermediate disk cache before rebuilding it.
clean_shared_info : bool, optional, default=False
    Also clears shared builder metadata during cache cleanup; ignored unless
    ``clean_middle_cache`` is true.
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

Values passed after ``conf_file`` on the command line override YAML values.
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
from data.processor.wrapper import DATASET_SELECTOR


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
        if conf.clean_middle_cache:
            builder.clean_all_cache(clean_shared_info=conf.clean_shared_info)
        builder.preproc(n_proc=conf.num_preproc_mid_workers)
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
            if dataset_name not in DATASET_SELECTOR:
                raise ValueError(f'Dataset {dataset_name} is not supported')

            builder_cls = DATASET_SELECTOR[dataset_name]
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
    if 'conf_file' in cli_args:
        logger.info(cli_args.conf_file)
        file_cfg = OmegaConf.load(get_conf_file_path(cli_args.conf_file))
        cli_args.pop("conf_file")
    else:
        file_cfg = OmegaConf.create({})

    code_cfg = OmegaConf.create(BasePreprocArgs().model_dump())
    cfg = OmegaConf.merge(code_cfg, file_cfg, cli_args)
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

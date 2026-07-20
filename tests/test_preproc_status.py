import io
import json
import logging
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import pandas as pd
    import preproc as preproc_module
    from common.conf import BasePreprocArgs
    from common.log import setup_log
    from data.processor.builder import EEGDatasetBuilder
except ModuleNotFoundError as error:
    pd = None
    preproc_module = None
    BasePreprocArgs = None
    setup_log = None
    EEGDatasetBuilder = None
    DEPENDENCY_ERROR = error
else:
    DEPENDENCY_ERROR = None


class FakeSplit:
    def __init__(self, sample_count, artifact_path):
        self.sample_count = sample_count
        self.cache_files = (
            [{'filename': str(artifact_path)}] if artifact_path is not None else []
        )

    def __len__(self):
        return self.sample_count


def make_builder(name, root, events, *, fail=False, warnings=0, create_artifact=True,
                 sample_count=3):
    artifact_path = Path(root) / name / f'{name}-train.arrow'

    class FakeBuilder:
        builder_configs = {'finetune': object()}

        def __init__(self, config_name, fs):
            self.config = SimpleNamespace(is_remote_fs=False)
            self.cache_dir = str(Path(root) / name)
            self.preproc_warning_count = warnings
            self.preproc_warning_messages = (
                [f'{warnings} recording(s) skipped'] if warnings else []
            )

        def clean_all_cache(self, clean_shared_info=False):
            events.append((name, 'clean_all', clean_shared_info))

        def preproc(self, n_proc=None):
            events.append((name, 'preproc'))
            if fail:
                raise KeyError('path')

        def download_and_prepare(self, num_proc=None):
            events.append((name, 'download'))
            if create_artifact:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.touch()

        def as_dataset(self):
            events.append((name, 'load'))
            return {'train': FakeSplit(sample_count, artifact_path)}

    return FakeBuilder


@unittest.skipIf(preproc_module is None, f'Preprocessing dependencies unavailable: {DEPENDENCY_ERROR}')
class PreprocessingStatusTests(unittest.TestCase):
    def _conf(self, dataset_names):
        return BasePreprocArgs(
            num_preproc_arrow_writers=1,
            num_preproc_mid_workers=1,
            finetune_datasets={name: 'finetune' for name in dataset_names},
        )

    def test_mixed_run_continues_and_aggregates_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            selectors = {
                'first': make_builder('first', temp_dir, events),
                'broken': make_builder('broken', temp_dir, events, fail=True),
                'last': make_builder('last', temp_dir, events),
            }
            with patch.dict(preproc_module.DATASET_SELECTOR, selectors, clear=True):
                results = preproc_module.preproc(self._conf(selectors))

        self.assertEqual([result.status for result in results], [
            'success', 'failed', 'success',
        ])
        self.assertIn("KeyError: 'path'", results[1].reason)
        self.assertIn(('last', 'load'), events)

        output = io.StringIO()
        with redirect_stdout(output):
            preproc_module.print_preprocessing_summary(results)
        summary = output.getvalue()
        self.assertIn('SUCCESS: first/finetune', summary)
        self.assertIn("FAILED: broken/finetune — KeyError: 'path'", summary)
        self.assertIn('SUCCESS: last/finetune', summary)
        self.assertIn('OVERALL: FAILED — 2 succeeded, 1 failed', summary)

    def test_success_with_recording_warnings_remains_successful(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            builder = make_builder('warned', temp_dir, events, warnings=2)
            result = preproc_module.prepare_dataset(
                self._conf(['warned']), builder, 'warned', 'finetune',
            )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.warning_count, 2)
        output = io.StringIO()
        with redirect_stdout(output):
            preproc_module.print_preprocessing_summary([result])
        self.assertIn('SUCCESS WITH WARNINGS: warned/finetune', output.getvalue())
        self.assertIn('2 recording(s) skipped', output.getvalue())
        self.assertIn('OVERALL: SUCCESS', output.getvalue())

    def test_clean_middle_cache_rebuilds_processed_arrow_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            builder = make_builder('rebuilt', temp_dir, events)
            conf = self._conf(['rebuilt'])
            conf.clean_middle_cache = True
            conf.clean_shared_info = True
            result = preproc_module.prepare_dataset(conf, builder, 'rebuilt', 'finetune')

        self.assertTrue(result.succeeded)
        self.assertEqual(events[0], ('rebuilt', 'clean_all', True))

    def test_non_clean_run_does_not_clear_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            builder = make_builder('reused', temp_dir, events)
            result = preproc_module.prepare_dataset(
                self._conf(['reused']), builder, 'reused', 'finetune',
            )

        self.assertTrue(result.succeeded)
        self.assertFalse(any(event[1] == 'clean_all' for event in events))

    def test_missing_arrow_artifact_is_a_dataset_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            builder = make_builder(
                'missing', temp_dir, events, create_artifact=False,
            )
            result = preproc_module.prepare_dataset(
                self._conf(['missing']), builder, 'missing', 'finetune',
            )

        self.assertFalse(result.succeeded)
        self.assertIn('processed Arrow artifacts are missing', result.reason)

    def test_empty_loaded_dataset_is_a_dataset_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            builder = make_builder(
                'empty', temp_dir, events, sample_count=0,
            )
            result = preproc_module.prepare_dataset(
                self._conf(['empty']), builder, 'empty', 'finetune',
            )

        self.assertFalse(result.succeeded)
        self.assertIn('contains no samples', result.reason)

    def test_invalid_dataset_is_reported_without_stopping_later_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = []
            selectors = {
                'valid': make_builder('valid', temp_dir, events),
            }
            conf = self._conf(['unknown', 'valid'])
            with patch.dict(preproc_module.DATASET_SELECTOR, selectors, clear=True):
                results = preproc_module.preproc(conf)

        self.assertFalse(results[0].succeeded)
        self.assertIn('not supported', results[0].reason)
        self.assertTrue(results[1].succeeded)

    def test_main_returns_nonzero_for_aggregated_dataset_failure(self):
        failed = preproc_module.DatasetPreparationResult(
            'broken', 'finetune', 'failed', reason='RuntimeError: broken',
        )
        output = io.StringIO()
        with (
            patch.object(preproc_module, '_load_config', return_value=self._conf([])),
            patch.object(preproc_module, 'preproc', return_value=[failed]),
            redirect_stdout(output),
        ):
            exit_code = preproc_module.main()

        self.assertEqual(exit_code, 1)
        self.assertIn('OVERALL: FAILED', output.getvalue())


@unittest.skipIf(EEGDatasetBuilder is None, f'Builder dependencies unavailable: {DEPENDENCY_ERROR}')
class BuilderCacheStatusTests(unittest.TestCase):
    def _cache_builder(self, temp_dir):
        root = Path(temp_dir)
        summary = root / 'summary'
        middle = root / 'middle'
        summary.mkdir()
        middle_file = middle / 'finetune' / 'train' / 'sample.parquet'
        middle_file.parent.mkdir(parents=True)
        middle_file.touch()

        builder = object.__new__(EEGDatasetBuilder)
        builder.summary_path = str(summary)
        builder.info_csv_path = str(summary / 'dataset_finetune_info.csv')
        builder.mid_file_csv_path = str(summary / 'dataset_finetune_fs_256_cache_files.csv')
        builder.preproc_warning_count = 0
        builder.preproc_warning_messages = []
        builder.config = SimpleNamespace(
            name='finetune',
            get_fs_id=lambda: 'fs_256',
            is_remote_fs=False,
            mid_path=str(middle),
        )
        pd.DataFrame([{'path': '/raw/sample'}]).to_csv(
            builder.info_csv_path, index=False,
        )
        pd.DataFrame([{
            'key': 'sample.parquet',
            'split': 'train',
            'cnt': 3,
        }]).to_csv(builder.mid_file_csv_path, index=False)
        Path(builder._done_marker_path()).write_text(json.dumps({
            'warning_count': 2,
            'warning_messages': ['2 recording(s) skipped'],
        }))
        return builder, middle_file

    def test_valid_cache_restores_persisted_warning_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder, _ = self._cache_builder(temp_dir)
            cached = builder._is_preproc_cached()

        self.assertTrue(cached)
        self.assertEqual(builder.preproc_warning_count, 2)
        self.assertEqual(builder.preproc_warning_messages, ['2 recording(s) skipped'])

    def test_stale_done_marker_with_missing_middle_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder, middle_file = self._cache_builder(temp_dir)
            middle_file.unlink()
            cached = builder._is_preproc_cached()

        self.assertFalse(cached)

    def test_all_metadata_failures_raise_clear_dataset_error(self):
        builder = object.__new__(EEGDatasetBuilder)
        builder.preproc_warning_count = 0
        builder.preproc_warning_messages = []
        builder._run_func_parallel = lambda *args, **kwargs: [None, None]

        with self.assertRaisesRegex(RuntimeError, 'No recordings yielded usable metadata'):
            builder._gather_data_info(['one', 'two'], n_proc=1)
        self.assertEqual(builder.preproc_warning_count, 2)

    def test_empty_test_selection_does_not_create_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_files = Path(temp_dir) / 'cache_files.csv'
            pd.DataFrame([
                {'key': 'train.parquet', 'split': 'train', 'cnt': 1},
                {'key': 'valid.parquet', 'split': 'valid', 'cnt': 1},
            ]).to_csv(cache_files, index=False)
            builder = object.__new__(EEGDatasetBuilder)
            builder.mid_file_csv_path = str(cache_files)
            builder.config = SimpleNamespace(is_finetune=True)
            splits = builder._split_generators(None)

        self.assertEqual([str(split.name) for split in splits], ['train', 'validation'])

    def test_clearing_arrow_cache_resets_loaded_split_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            arrow_path = root / 'dataset' / 'finetune'
            arrow_path.mkdir(parents=True)
            (arrow_path / 'dataset_info.json').touch()
            builder = object.__new__(EEGDatasetBuilder)
            builder.info = SimpleNamespace(splits={'test': 'stale'})
            builder.config = SimpleNamespace(
                data_path=str(root),
                dataset_name='dataset',
                name='finetune',
                database_proc_root=str(root),
                get_fs_id=lambda: 'fs_256',
            )
            builder.clean_arrow_set()

        self.assertFalse(arrow_path.exists())
        self.assertIsNone(builder.info.splits)


class LoggingAndShellStatusTests(unittest.TestCase):
    @unittest.skipIf(setup_log is None, f'Logging dependencies unavailable: {DEPENDENCY_ERROR}')
    def test_named_logger_emits_each_record_to_only_one_stream(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        logger_name = 'preproc-status-test'
        with redirect_stdout(stdout), redirect_stderr(stderr):
            test_logger = setup_log(name=logger_name)
            test_logger.info('one info')
            test_logger.warning('one warning')

        self.assertEqual(stdout.getvalue().count('one info'), 1)
        self.assertNotIn('one warning', stdout.getvalue())
        self.assertEqual(stderr.getvalue().count('one warning'), 1)
        self.assertFalse(test_logger.propagate)
        test_logger.handlers.clear()

    def test_shell_ends_with_python_summary_and_preserves_failure_code(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_python = root / 'fake-python'
            fake_python.write_text(
                '#!/usr/bin/env bash\n'
                'echo "======= PREPROCESSING SUMMARY ======="\n'
                'echo "FAILED: broken/finetune — RuntimeError: broken"\n'
                'echo "OVERALL: FAILED — 0 succeeded, 1 failed, 0 with warnings"\n'
                'exit 1\n'
            )
            fake_python.chmod(0o755)
            environment = os.environ.copy()
            environment.update({
                'PYTHON': str(fake_python),
                'LOG_DIR': str(root / 'logs'),
            })
            completed = subprocess.run(
                ['bash', 'scripts/preproc.sh'],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertNotIn('EXIT STATUS: SUCCESS', completed.stdout)
        self.assertNotIn('EXIT STATUS: FAILED', completed.stdout)
        self.assertTrue(completed.stdout.rstrip().endswith(
            'OVERALL: FAILED — 0 succeeded, 1 failed, 0 with warnings'
        ))


if __name__ == '__main__':
    unittest.main()

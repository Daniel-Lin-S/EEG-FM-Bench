import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import path
from common.summary import build_summary_path, migrate_legacy_summary_artifacts


class DataPathTests(unittest.TestCase):
    def test_inria_bci_discovers_only_recording_csvs_from_root_layout(self):
        source = Path('data/dataset/inria_bci.py').read_text()
        self.assertIn('scan_sub_dir: str = ""', source)
        self.assertIn('return self._filter_raw_data_file_candidates(raw_data_files)', source)
        self.assertIn("path_parts[0] in {'train', 'test'}", source)
        self.assertIn("re.fullmatch(r'Data_S\\d{2}_Sess\\d{2}\\.csv', path_parts[1])", source)

    def test_inria_bci_uses_shared_candidate_validation_guard(self):
        source = Path('data/processor/builder.py').read_text()
        self.assertIn('def _is_valid_raw_data_file(self, file_path: str) -> bool:', source)
        self.assertIn('def _filter_raw_data_file_candidates(self, file_paths: list[str]) -> list[str]:', source)
        self.assertIn("logger.warning(f'Skipping non-recording data file: {file_path}')", source)
        self.assertIn('if not self._is_valid_raw_data_file(data):', source)

    def test_runtime_uses_ignored_local_data_paths_file(self):
        self.assertTrue(path.DATA_PATHS_FILE.endswith('data_paths.local.yaml'))
        self.assertTrue(path.DATA_PATHS_TEMPLATE_FILE.endswith('data_paths.yaml'))

    def test_shared_yaml_parses_output_and_raw_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'data_paths.yaml'
            config_path.write_text(
                'output_root: /tmp/eegfm-output\n'
                'raw_roots:\n'
                '  adftd: /tmp/eegfm-adftd-raw\n'
            )
            config = path._load_data_paths(str(config_path))

        self.assertEqual(config['output_root'], '/tmp/eegfm-output')
        self.assertEqual(config['raw_roots']['adftd'], '/tmp/eegfm-adftd-raw')

    def test_configured_raw_root_is_used_verbatim(self):
        raw_root = '/tmp/eegfm-adftd-raw'
        with patch.dict(path.DATASET_RAW_ROOTS, {'adftd': raw_root}, clear=True):
            resolved, configured = path.get_dataset_raw_path('adftd', 'ADFTD')

        self.assertTrue(configured)
        self.assertEqual(resolved, raw_root)
        self.assertNotIn('ADFTD', resolved)

    def test_unmapped_dataset_uses_legacy_suffix_path(self):
        with patch.dict(path.DATASET_RAW_ROOTS, {}, clear=True):
            resolved, configured = path.get_dataset_raw_path('adftd', 'ADFTD')

        self.assertFalse(configured)
        self.assertEqual(resolved, os.path.join(path.DATABASE_RAW_ROOT, 'ADFTD'))

    def test_writable_roots_are_derived_from_output_root(self):
        self.assertEqual(path.DATABASE_PROC_ROOT, os.path.join(path.OUTPUT_ROOT, 'processed'))
        self.assertEqual(path.DATABASE_CACHE_ROOT, os.path.join(path.OUTPUT_ROOT, 'cache'))
        self.assertEqual(path.DATABASE_SUMMARY_ROOT, os.path.join(path.OUTPUT_ROOT, 'summary'))
        self.assertEqual(path.LOG_ROOT, os.path.join(path.OUTPUT_ROOT, 'logs'))
        self.assertEqual(path.RUN_ROOT, os.path.join(path.OUTPUT_ROOT, 'run'))

    def test_missing_configured_raw_root_fails_before_processing(self):
        missing = '/tmp/eegfm-missing-raw-root'
        with self.assertRaises(FileNotFoundError):
            path.validate_configured_raw_path('adftd', missing, configured=True)

    def test_hbn_has_no_raw_tsv_writer(self):
        source = Path('data/dataset/hbn.py').read_text()
        self.assertNotIn('_fix_channel_tsv', source)
        self.assertNotIn("to_csv(str(path)", source)
        self.assertIn('set_channel_types', source)

    def test_summary_path_and_legacy_migration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / 'processed' / 'ADFTD' / 'summary' / 'pretrain'
            summary = Path(build_summary_path(str(root / 'summary'), 'adftd', 'ADFTD', 'pretrain'))
            legacy.mkdir(parents=True)
            summary.mkdir(parents=True)
            (legacy / 'moved.csv').write_text('legacy-only')
            (legacy / 'conflict.csv').write_text('legacy-wins')
            (summary / 'conflict.csv').write_text('new-loses')

            self.assertTrue(migrate_legacy_summary_artifacts(
                str(root / 'processed'), 'ADFTD', 'pretrain', str(summary)))

            self.assertEqual((summary / 'moved.csv').read_text(), 'legacy-only')
            self.assertEqual((summary / 'conflict.csv').read_text(), 'legacy-wins')
            self.assertFalse(legacy.exists())
            self.assertFalse((root / 'processed' / 'ADFTD').exists())


if __name__ == '__main__':
    unittest.main()

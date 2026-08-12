import unittest
from types import SimpleNamespace

try:
    from data.dataset.tue.tuep import TuepBuilder, TuepV200Builder
    from data.processor.wrapper import DATASET_SELECTOR
except ModuleNotFoundError as error:
    TuepBuilder = None
    TuepV200Builder = None
    DATASET_SELECTOR = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    TuepBuilder is None,
    f'TUEP dependencies unavailable: {IMPORT_ERROR}',
)
class TuepBuilderTests(unittest.TestCase):
    categories = ['epilepsy', 'no_epilepsy']

    @staticmethod
    def _builder(builder_class, raw_path):
        builder = object.__new__(builder_class)
        builder.config = SimpleNamespace(
            category=TuepBuilderTests.categories,
            raw_path=raw_path,
        )
        return builder

    def test_official_v201_labels_follow_numbered_class_directories(self):
        builder = self._builder(TuepBuilder, '/raw/tuep')
        epilepsy_path = (
            '/raw/tuep/data/00_epilepsy/subject/s001/01_tcp_ar/'
            'subject_s001_t000.edf'
        )
        no_epilepsy_path = (
            '/raw/tuep/data/01_no_epilepsy/subject/s001/01_tcp_ar/'
            'subject_s001_t000.edf'
        )

        self.assertEqual(
            builder._resolve_exp_events(epilepsy_path, {}),
            [('epilepsy', 0, -1)],
        )
        self.assertEqual(
            builder._resolve_exp_events(no_epilepsy_path, {}),
            [('no_epilepsy', 0, -1)],
        )

    def test_legacy_v200_labels_follow_class_roots(self):
        builder = self._builder(TuepV200Builder, '/raw/tuep_v200')
        epilepsy_path = (
            '/raw/tuep_v200/epilepsy_edf/subject/s001/01_tcp_ar/'
            'subject_s001_t000.edf'
        )
        no_epilepsy_path = (
            '/raw/tuep_v200/no_epilepsy_edf/subject/s001/01_tcp_ar/'
            'subject_s001_t000.edf'
        )

        self.assertEqual(
            builder._resolve_exp_events(epilepsy_path, {}),
            [('epilepsy', 0, -1)],
        )
        self.assertEqual(
            builder._resolve_exp_events(no_epilepsy_path, {}),
            [('no_epilepsy', 0, -1)],
        )

    def test_legacy_v200_rejects_unknown_class_root(self):
        builder = self._builder(TuepV200Builder, '/raw/tuep_v200')
        file_path = (
            '/raw/tuep_v200/unknown/subject/s001/01_tcp_ar/'
            'subject_s001_t000.edf'
        )

        with self.assertRaisesRegex(ValueError, 'legacy TUEP class directory'):
            builder._resolve_exp_events(file_path, {})

    def test_legacy_v200_builder_is_registered(self):
        self.assertIs(DATASET_SELECTOR['tuep_v2_0_0'], TuepV200Builder)

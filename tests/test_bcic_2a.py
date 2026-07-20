import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    from scipy.io import savemat
    from data.dataset.bcic.bcic_2a import BCIC2ABuilder, BCIC2AConfig
except ModuleNotFoundError as error:
    BCIC2ABuilder = None
    BCIC2AConfig = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


class FakeRaw:
    def __init__(self, onsets, descriptions):
        self.annotations = SimpleNamespace(
            onset=np.asarray(onsets),
            description=np.asarray(descriptions),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@unittest.skipIf(BCIC2ABuilder is None, f'BCIC dependencies unavailable: {IMPORT_ERROR}')
class BCIC2AGDFTests(unittest.TestCase):
    def _builder(self, raw_path, *, finetune=True):
        builder = object.__new__(BCIC2ABuilder)
        builder.config = SimpleNamespace(
            raw_path=raw_path,
            scan_sub_dir='',
            file_ext='gdf',
            is_finetune=finetune,
            wnd_div_sec=4,
            category=['left', 'right', 'foot', 'tongue'],
        )
        return builder

    def test_config_targets_original_gdf_channels(self):
        config = BCIC2AConfig(name='finetune', is_finetune=True)
        self.assertEqual(config.file_ext, 'gdf')
        self.assertEqual(len(config.montage['10_20']), 22)
        self.assertEqual(config.montage['10_20'][0], 'EEG-Fz')
        self.assertEqual(config.montage['10_20'][-1], 'EEG-16')

    def test_walk_discovers_only_gdf_recordings_from_direct_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'A01T.gdf').touch()
            (root / 'A01E.gdf').touch()
            (root / 'true_labels').mkdir()
            (root / 'true_labels' / 'A01E.mat').touch()

            files = self._builder(str(root))._walk_raw_data_files()

        self.assertEqual(
            {Path(path).name for path in files},
            {'A01T.gdf', 'A01E.gdf'},
        )

    def test_training_events_map_only_motor_imagery_cues(self):
        builder = self._builder('/raw')
        builder._read_raw_data = lambda *args, **kwargs: FakeRaw(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            ['32766', '768', '769', '771', '1023'],
        )

        events = builder._resolve_exp_events(
            '/raw/A01T.gdf',
            {'session_type': 'T'},
        )

        self.assertEqual(events, [
            ('left', 2000, 6000),
            ('foot', 3000, 7000),
        ])

    def test_evaluation_events_use_released_classlabel_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels = root / 'true_labels'
            labels.mkdir()
            savemat(labels / 'A01E.mat', {
                'classlabel': np.asarray([[1], [4]], dtype=np.uint8),
            })
            builder = self._builder(str(root))
            builder._read_raw_data = lambda *args, **kwargs: FakeRaw(
                [0.0, 1.5, 3.5],
                ['768', '783', '783'],
            )

            events = builder._resolve_exp_events(
                str(root / 'A01E.gdf'),
                {'session_type': 'E'},
            )

        self.assertEqual(events, [
            ('left', 1500, 5500),
            ('tongue', 3500, 7500),
        ])

    def test_recording_name_resolves_subject_and_session(self):
        builder = self._builder('/raw')
        self.assertEqual(builder._resolve_file_name('/raw/A09T.gdf'), {
            'subject': 9,
            'session': 1,
            'session_type': 'T',
        })
        self.assertEqual(builder._resolve_file_name('/raw/A09E.gdf'), {
            'subject': 9,
            'session': 2,
            'session_type': 'E',
        })


if __name__ == '__main__':
    unittest.main()

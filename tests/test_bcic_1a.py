import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    from data.dataset.bcic.bcic_1a import BCIC1ABuilder
except ModuleNotFoundError as error:
    BCIC1ABuilder = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(BCIC1ABuilder is None, f'BCIC dependencies unavailable: {IMPORT_ERROR}')
class BCIC1AEventsTests(unittest.TestCase):
    def _builder(self, raw_path):
        builder = object.__new__(BCIC1ABuilder)
        builder.config = SimpleNamespace(
            raw_path=raw_path,
            scan_sub_dir='BCICIV_1calib_1000Hz_mat',
            scan_eval_sub_dir='BCICIV_1eval_1000Hz_mat',
            true_labels_sub_dir='true_labels',
            file_ext='mat',
            is_finetune=True,
            category=['left', 'right', 'foot'],
        )
        return builder

    @staticmethod
    def _recording_data(n_samples):
        nfo = np.empty((1, 1), dtype=[('fs', object), ('classes', object)])
        nfo['fs'][0, 0] = np.array([[1000]])
        nfo['classes'][0, 0] = np.array([['left', 'right']], dtype=object)
        return {'cnt': np.empty((n_samples, 59)), 'nfo': nfo}

    def test_finetune_walks_eval_when_matching_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration = root / 'BCICIV_1calib_1000Hz_mat'
            evaluation = root / 'BCICIV_1eval_1000Hz_mat'
            labels = root / 'true_labels'
            calibration.mkdir()
            evaluation.mkdir()
            labels.mkdir()
            (calibration / 'BCICIV_calib_ds1a_1000Hz.mat').touch()
            (evaluation / 'BCICIV_eval_ds1a_1000Hz.mat').touch()
            (labels / 'BCICIV_eval_ds1a_1000Hz_true_y.mat').touch()

            files = self._builder(str(root))._walk_raw_data_files()

        self.assertEqual(len(files), 2)

    def test_configured_root_does_not_discover_true_label_mat_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration = root / 'BCICIV_1calib_1000Hz_mat'
            evaluation = root / 'BCICIV_1eval_1000Hz_mat'
            labels = root / 'true_labels'
            calibration.mkdir()
            evaluation.mkdir()
            labels.mkdir()
            (calibration / 'BCICIV_calib_ds1a_1000Hz.mat').touch()
            (evaluation / 'BCICIV_eval_ds1a_1000Hz.mat').touch()
            (labels / 'BCICIV_eval_ds1a_1000Hz_true_y.mat').touch()

            builder = self._builder(str(root))
            builder.config.scan_sub_dir = ''
            files = builder._walk_raw_data_files()

        self.assertEqual(
            files,
            [
                str(calibration / 'BCICIV_calib_ds1a_1000Hz.mat'),
                str(evaluation / 'BCICIV_eval_ds1a_1000Hz.mat'),
            ],
        )

    def test_finetune_fails_when_eval_sidecar_is_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'BCICIV_1calib_1000Hz_mat').mkdir()
            evaluation = root / 'BCICIV_1eval_1000Hz_mat'
            evaluation.mkdir()
            (evaluation / 'BCICIV_eval_ds1a_1000Hz.mat').touch()

            with self.assertRaisesRegex(FileNotFoundError, 'true_labels'):
                self._builder(str(root))._walk_raw_data_files()

    def test_eval_true_y_emits_only_active_label_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels = root / 'true_labels'
            evaluation = root / 'BCICIV_1eval_1000Hz_mat'
            labels.mkdir()
            evaluation.mkdir()
            eval_path = evaluation / 'BCICIV_eval_ds1a_1000Hz.mat'
            label_path = labels / 'BCICIV_eval_ds1a_1000Hz_true_y.mat'
            eval_path.touch()
            label_path.touch()
            recording = self._recording_data(7)
            true_y = {'true_y': np.array([[np.nan, 0, -1, -1, np.nan, 1, 1, -1]])}

            with patch('data.dataset.bcic.bcic_1a.loadmat') as loadmat:
                loadmat.side_effect = lambda path: true_y if path == str(label_path) else recording
                annotations = self._builder(str(root))._resolve_exp_events(str(eval_path), {})

        self.assertEqual(annotations, [('left', 2, 4), ('right', 5, 7)])

    def test_eval_true_y_shorter_than_signal_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels = root / 'true_labels'
            evaluation = root / 'BCICIV_1eval_1000Hz_mat'
            labels.mkdir()
            evaluation.mkdir()
            eval_path = evaluation / 'BCICIV_eval_ds1a_1000Hz.mat'
            label_path = labels / 'BCICIV_eval_ds1a_1000Hz_true_y.mat'
            recording = self._recording_data(7)
            true_y = {'true_y': np.array([[-1, -1, 0, 1]])}

            with patch('data.dataset.bcic.bcic_1a.loadmat') as loadmat:
                loadmat.side_effect = lambda path: true_y if path == str(label_path) else recording
                annotations = self._builder(str(root))._resolve_exp_events(str(eval_path), {})

        self.assertEqual(annotations, [('left', 0, 2), ('right', 3, 4)])

    def test_calibration_events_still_use_mrk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration = root / 'BCICIV_1calib_1000Hz_mat'
            calibration.mkdir()
            calibration_path = calibration / 'BCICIV_calib_ds1a_1000Hz.mat'
            calibration_path.touch()
            recording = self._recording_data(10_000)
            mrk = np.empty((1, 1), dtype=[('pos', object), ('y', object)])
            mrk['pos'][0, 0] = np.array([[1000], [3000]])
            mrk['y'][0, 0] = np.array([[-1], [1]])
            recording['mrk'] = mrk

            with patch('data.dataset.bcic.bcic_1a.loadmat', return_value=recording):
                annotations = self._builder(str(root))._resolve_exp_events(str(calibration_path), {})

        self.assertEqual(annotations, [('left', 1000, 7000), ('right', 3000, 9000)])


if __name__ == '__main__':
    unittest.main()

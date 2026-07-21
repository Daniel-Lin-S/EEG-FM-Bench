import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import h5py
    import numpy as np
    from data.dataset.bcic.bcic_2020_3 import BCIC2020ImagineBuilder
except ModuleNotFoundError as error:
    h5py = None
    np = None
    BCIC2020ImagineBuilder = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(BCIC2020ImagineBuilder is None, f'BCIC dependencies unavailable: {IMPORT_ERROR}')
class BCIC2020ReaderTests(unittest.TestCase):
    categories = ['hello', 'help me', 'stop', 'thank you', 'yes']

    def _builder(self, raw_path=''):
        builder = object.__new__(BCIC2020ImagineBuilder)
        builder.config = SimpleNamespace(
            category=self.categories,
            montage={'10_20': [f'C{index}' for index in range(64)]},
            raw_path=raw_path,
            scan_sub_dir='',
            file_ext='mat',
        )
        return builder

    def _record(self, trials=2):
        x = np.zeros((4, 64, trials), dtype=np.float32)
        for trial in range(trials):
            x[:, 0, trial] = np.arange(1 + 4 * trial, 5 + 4 * trial)
        y = np.zeros((5, trials), dtype=np.uint8)
        y[[0, 4][:trials], np.arange(trials)] = 1
        return SimpleNamespace(
            mnt=SimpleNamespace(
                clab=np.array([f'C{index}' for index in range(64)], dtype=object),
                pos_3d=np.arange(64 * 3, dtype=np.float32).reshape(64, 3) + 1,
            ),
            x=x,
            clab=np.array([f'C{index}' for index in range(64)], dtype=object),
            fs=256,
            t=np.arange(4),
            y=y,
            className=np.array(self.categories, dtype=object),
        )

    @staticmethod
    def _write_test_sheet(path, sample_name, labels):
        rows = [
            f'<row r="2"><c r="B2" t="inlineStr"><is><t>{sample_name}</t></is></c></row>',
            '<row r="3"><c r="B3" t="inlineStr"><is><t>Trial #</t></is></c>'
            '<c r="C3" t="inlineStr"><is><t>True Label</t></is></c></row>',
        ]
        for row, label in enumerate(labels, start=4):
            rows.append(f'<row r="{row}"><c r="B{row}"><v>{row - 3}</v></c>'
                        f'<c r="C{row}"><v>{label}</v></c></row>')
        worksheet = (
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData>' + ''.join(rows) + '</sheetData></worksheet>'
        )
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('xl/worksheets/sheet1.xml', worksheet)

    def test_v5_reader_concatenates_epochs_and_annotations(self):
        record = self._record()
        with patch('data.dataset.bcic.bcic_2020_3.loadmat', return_value={'epo_train': record, 'mnt': record.mnt}):
            raw = self._builder()._read_raw_data('/tmp/Training set/Data_Sample01.mat')

        np.testing.assert_allclose(raw.get_data()[0], np.arange(1, 9) * 1e-6)
        self.assertEqual(raw.info['sfreq'], 256.0)
        self.assertEqual(raw.annotations.description.tolist(), ['0', '4'])
        np.testing.assert_allclose(raw.annotations.onset, [0.0, 4 / 256])
        np.testing.assert_allclose(raw.info['chs'][0]['loc'][:3], [1, 2, 3])

    def test_v5_reader_uses_the_source_sampling_rate(self):
        record = self._record()
        record.fs = 128
        with patch('data.dataset.bcic.bcic_2020_3.loadmat', return_value={'epo_train': record, 'mnt': record.mnt}):
            raw = self._builder()._read_raw_data('/tmp/Training set/Data_Sample01.mat')

        self.assertEqual(raw.info['sfreq'], 128.0)

    def test_v73_reader_uses_true_label_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            test_dir = root / 'Test set'
            test_dir.mkdir()
            mat_path = test_dir / 'Data_Sample01.mat'
            self._write_test_sheet(test_dir / 'Track3_Answer Sheet_Test.xlsx', 'Data_Sample01', [1, 5, 3])
            with h5py.File(mat_path, 'w') as mat_file:
                record = mat_file.create_group('epo_test')
                record.create_dataset('x', data=np.zeros((4, 64, 3), dtype=np.float32))
                record.create_dataset('clab', data=np.array([f'C{index}'.encode() for index in range(64)]))
                record.create_dataset('fs', data=256)
                record.create_dataset('t', data=np.arange(4))
                montage = mat_file.create_group('mnt')
                montage.create_dataset('clab', data=np.array([f'C{index}'.encode() for index in range(64)]))
                montage.create_dataset(
                    'pos_3d', data=np.arange(64 * 3, dtype=np.float32).reshape(64, 3) + 1
                )

            raw = self._builder()._read_raw_data(str(mat_path))

        self.assertEqual(raw.annotations.description.tolist(), ['0', '4', '2'])
        np.testing.assert_allclose(raw.info['chs'][0]['loc'][:3], [1, 2, 3])

    def test_invalid_one_hot_labels_raise_clear_error(self):
        labels = np.zeros((5, 2), dtype=np.uint8)
        labels[0, 0] = 1
        labels[1, 0] = 1
        with self.assertRaisesRegex(ValueError, 'Invalid one-hot'):
            self._builder()._one_hot_labels(labels, 2, 'broken.mat')

    def test_walk_discovers_mat_files_from_raw_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            training = raw_root / 'Training set'
            training.mkdir()
            (training / 'Data_Sample01.mat').touch()
            (training / 'ignored.fif').touch()
            files = self._builder(str(raw_root))._walk_raw_data_files()

        self.assertEqual(files, [str(training / 'Data_Sample01.mat')])


if __name__ == '__main__':
    unittest.main()

import json
import logging
import os
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import h5py
import mne
import numpy as np
import pandas as pd
import s3fs
from scipy.io import loadmat
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class BCIC2020ImagineSpeechConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "EEG of five-class imagined speech words/phrases were recorded. 70 trials per class (70× 5 = 350 "
        "trials) are released for training (60 trials per class) and validation (10 trials per class) purpose. "
        "Using the given validation set is not obligated. Validation for the training data can be performed "
        "not only by the given validation set but also with the competitors’ choice (example: N-fold cross validation over the whole data). "
        "The test data (10 trials per class) will be released later. The dataset was divided into epochs based on cue information (event codes).")
    citation: Optional[str] = "https://osf.io/pq7vb/"  # download from this link.

    filter_notch: float = 50.0
    is_notched: bool = True

    dataset_name: Optional[str] = 'bcic_2020_3'
    task_type: DatasetTaskType = DatasetTaskType.LINGUAL

    file_ext: str = 'mat'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                                'Fp1',     'Fp2',
                          'AF7','AF3',     'AF4','AF8',
                  'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
        'FT9','FT7','FC5','FC3','FC1',     'FC2','FC4','FC6','FT8','FT10',
                  'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
        'TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10',
                  'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
                    'PO9','PO7','PO3','POz','PO4','PO8','PO10',
                                 'O1','Oz','O2',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 3
    suffix_path: str = os.path.join('BCIC_2020_3')
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: [
        'hello', 'help me', 'stop', 'thank you', 'yes'
    ])


class BCIC2020ImagineBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = BCIC2020ImagineSpeechConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain', **kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        # noinspection PyTypeChecker
        scan_path: str = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext):
                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject = int(file_name.split('_')[1][-2:])
        session = 1

        return {
            'subject': subject,
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_20',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            events, event_id = mne.events_from_annotations(raw)
            sf = raw.info['sfreq']
            events[:, 0] = events[:, 0] / sf

        mapping = {}
        for k, v in event_id.items():
            mapping[str(v)] = int(k)

        annotations = []
        for ev in events:
            label = 'default'
            if self.config.is_finetune:
                c = ev[2]
                c = mapping[str(c)]
                label = self.config.category[c]
            t_start = ev[0].item() * 1000
            annotations.append((
                label,
                round(t_start),
                round(t_start + self.config.wnd_div_sec * 1000)
            ))

        return annotations

    def _persist_example_file(self, sample: dict):
        # pretrain datasets have no ground truth will be assigned a label item which indicates all signal array
        path, montage, label, split, subject = (
            sample['path'], sample['montage'], json.loads(sample['label']), sample['split'], sample['subject'])
        try:
            with self._read_raw_data(path, preload=True, verbose=False) as data:
                data = self._select_data_channels(data, path, montage)
                raw = self._fetch_signal_ndarray(data)
                chs_idx = self._fetch_chs_index(montage)

                examples = self._generate_window_sample(raw, montage, chs_idx, label, self.config.persist_drop_last)
                if len(examples) < 1:
                    return None

                df = pd.DataFrame(data=examples)
                df['subject'] = str(subject)
                filename = f"{self._encode_path(path)}.parquet"
                output_path = self._build_output_dir(split, filename)

                if self.config.is_remote_fs:
                    fs = s3fs.S3FileSystem(**self.s3_conf)
                    with fs.open(output_path, 'wb') as f:
                        df.to_parquet(
                            f,
                            compression=self.config.mid_compress_algo,
                            engine='pyarrow',
                            index=False)
                    fs.invalidate_cache()
                else:
                    df.to_parquet(
                        output_path,
                        compression=self.config.mid_compress_algo,
                        engine='pyarrow',
                        index=False)
        except Exception as e:
            logger.error(f"Error persisting example file {path}: {str(e)}")
            return None

        mid_df = pd.DataFrame(data={
            'key': [filename],
            'split': [split],
            'cnt': [len(examples)],})
        return mid_df

    def _divide_split(self, df: DataFrame) -> DataFrame:

        if self.config.is_finetune:
            df.loc[df['path'].str.contains('Training set'), 'split'] = 'train'
            df.loc[df['path'].str.contains('Validation set'), 'split'] = 'valid'
            df.loc[df['path'].str.contains('Test set'), 'split'] = 'test'
        else:
            df.loc[:, 'split'] = 'train'
            df.loc[df['path'].str.contains('Test set'), 'split'] = 'valid'

        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        if h5py.is_hdf5(file_path):
            epochs, ch_names, fs, labels = self._read_v73_recording(file_path)
        else:
            epochs, ch_names, fs, labels = self._read_v5_recording(file_path)
        return self._epochs_to_raw(epochs, ch_names, fs, labels, verbose)

    def _read_v5_recording(self, file_path: str):
        record_name = 'epo_train' if 'Training set' in file_path else 'epo_validation'
        contents = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        if record_name not in contents:
            raise ValueError(f'Missing {record_name!r} in {file_path}.')

        record = contents[record_name]
        epochs, ch_names, fs = self._normalise_recording(
            self._mat_field(record, 'x', file_path),
            self._mat_field(record, 'clab', file_path),
            self._mat_field(record, 'fs', file_path),
            self._mat_field(record, 't', file_path),
            file_path,
        )
        class_names = self._matlab_strings(self._mat_field(record, 'className', file_path))
        if len(class_names) != len(self.config.category):
            raise ValueError(
                f'Expected {len(self.config.category)} class names in {file_path}, found {len(class_names)}.'
            )
        labels = self._one_hot_labels(self._mat_field(record, 'y', file_path), epochs.shape[2], file_path)
        return epochs, ch_names, fs, labels

    def _read_v73_recording(self, file_path: str):
        with h5py.File(file_path, 'r') as mat_file:
            if 'epo_test' not in mat_file or not isinstance(mat_file['epo_test'], h5py.Group):
                raise ValueError(f"Missing HDF5 struct 'epo_test' in {file_path}.")
            record = mat_file['epo_test']
            epochs, ch_names, fs = self._normalise_recording(
                self._hdf5_field(mat_file, record, 'x', file_path),
                self._hdf5_field(mat_file, record, 'clab', file_path),
                self._hdf5_field(mat_file, record, 'fs', file_path),
                self._hdf5_field(mat_file, record, 't', file_path),
                file_path,
            )
        labels = self._test_labels(file_path)
        if len(labels) != epochs.shape[2]:
            raise ValueError(
                f'Test label count ({len(labels)}) does not match trial count ({epochs.shape[2]}) in {file_path}.'
            )
        return epochs, ch_names, fs, labels

    @staticmethod
    def _mat_field(record: Any, field_name: str, file_path: str):
        if isinstance(record, dict):
            if field_name in record:
                return record[field_name]
        elif isinstance(record, np.ndarray) and record.dtype.names:
            record = record.squeeze()
            if field_name in record.dtype.names:
                return record[field_name]
        elif hasattr(record, field_name):
            return getattr(record, field_name)
        raise ValueError(f'Missing field {field_name!r} in {file_path}.')

    def _hdf5_field(self, mat_file: h5py.File, record: h5py.Group, field_name: str, file_path: str):
        if field_name not in record:
            raise ValueError(f'Missing field {field_name!r} in HDF5 file {file_path}.')
        return self._hdf5_value(mat_file, record[field_name])

    def _hdf5_value(self, mat_file: h5py.File, node):
        if isinstance(node, h5py.Group):
            return {name: self._hdf5_value(mat_file, child) for name, child in node.items()}

        value = node[()]
        if h5py.check_dtype(ref=value.dtype) is not None:
            refs = np.asarray(value).reshape(-1, order='F')
            values = [self._hdf5_value(mat_file, mat_file[ref]) for ref in refs]
            return values[0] if len(values) == 1 else values

        matlab_class = node.attrs.get('MATLAB_class', b'')
        if isinstance(matlab_class, bytes) and matlab_class.decode() == 'char':
            chars = np.asarray(value).reshape(-1, order='F')
            return ''.join(chr(int(char)) for char in chars if int(char) != 0)
        return value

    def _normalise_recording(self, x, clab, fs, t, file_path: str):
        ch_names = self._matlab_strings(clab)
        expected_channels = len(self.config.montage['10_20'])
        if len(ch_names) != expected_channels:
            raise ValueError(f'Expected {expected_channels} channel labels in {file_path}, found {len(ch_names)}.')

        fs_values = np.asarray(fs).reshape(-1)
        if fs_values.size != 1 or not np.isfinite(fs_values[0]) or fs_values[0] <= 0:
            raise ValueError(f'Expected one positive sampling-rate value in {file_path}, found {fs_values}.')
        fs = float(fs_values[0])

        data = np.asarray(x, dtype=np.float32)
        times = np.asarray(t).reshape(-1)
        if data.ndim != 3:
            raise ValueError(f'Expected a 3D epoch array in {file_path}, found shape {data.shape}.')

        sample_axes = [axis for axis, size in enumerate(data.shape) if size == len(times)]
        channel_axes = [axis for axis, size in enumerate(data.shape) if size == len(ch_names)]
        if len(sample_axes) != 1 or len(channel_axes) != 1 or sample_axes[0] == channel_axes[0]:
            raise ValueError(
                f'Cannot identify sample and channel axes for shape {data.shape} in {file_path}.'
            )
        trial_axis = next(axis for axis in range(3) if axis not in {sample_axes[0], channel_axes[0]})
        epochs = np.moveaxis(data, (sample_axes[0], channel_axes[0], trial_axis), (0, 1, 2))
        return epochs, ch_names, fs

    def _one_hot_labels(self, y, n_trials: int, file_path: str) -> np.ndarray:
        labels = np.asarray(y)
        expected_classes = len(self.config.category)
        if labels.shape != (expected_classes, n_trials):
            raise ValueError(
                f'Expected one-hot labels with shape ({expected_classes}, {n_trials}) in {file_path}, '
                f'found {labels.shape}.'
            )
        if not np.all((labels == 0) | (labels == 1)) or not np.all(labels.sum(axis=0) == 1):
            raise ValueError(f'Invalid one-hot labels in {file_path}.')
        return labels.argmax(axis=0).astype(np.int64)

    @staticmethod
    def _matlab_strings(values) -> list[str]:
        if isinstance(values, str):
            return [values.rstrip('\x00')]
        if isinstance(values, bytes):
            return [values.decode().rstrip('\x00')]

        strings = []
        for value in np.asarray(values, dtype=object).reshape(-1):
            if isinstance(value, bytes):
                strings.append(value.decode().rstrip('\x00'))
            elif isinstance(value, str):
                strings.append(value.rstrip('\x00'))
            else:
                array = np.asarray(value)
                if array.dtype.kind in {'U', 'S'}:
                    strings.append(''.join(array.reshape(-1).tolist()).rstrip('\x00'))
                else:
                    strings.append(''.join(chr(int(char)) for char in array.reshape(-1) if int(char) != 0))
        return strings

    def _test_labels(self, file_path: str) -> np.ndarray:
        spreadsheet_path = os.path.join(os.path.dirname(file_path), 'Track3_Answer Sheet_Test.xlsx')
        if not os.path.isfile(spreadsheet_path):
            raise FileNotFoundError(f'Missing test-label spreadsheet: {spreadsheet_path}')

        namespace = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        with zipfile.ZipFile(spreadsheet_path) as archive:
            shared_strings = []
            if 'xl/sharedStrings.xml' in archive.namelist():
                shared_root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
                shared_strings = [''.join(item.itertext()) for item in shared_root.findall('x:si', namespace)]
            sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))

        cells = {}
        for cell in sheet.findall('.//x:c', namespace):
            reference = cell.attrib['r']
            value = cell.findtext('x:v', default='', namespaces=namespace)
            if cell.attrib.get('t') == 's' and value:
                value = shared_strings[int(value)]
            elif cell.attrib.get('t') == 'inlineStr':
                value = ''.join(cell.itertext())
            cells[reference] = value

        sample_name = self._extract_file_name(file_path)
        header_reference = next(
            (reference for reference, value in cells.items() if value == sample_name),
            None,
        )
        if header_reference is None:
            raise ValueError(f'No test-label column for {sample_name} in {spreadsheet_path}.')

        header_column = self._xlsx_column(header_reference)
        header_row = self._xlsx_row(header_reference)
        trial_column = self._xlsx_column_name(header_column)
        label_column = self._xlsx_column_name(header_column + 1)
        header_values = (
            cells.get(f'{trial_column}{header_row + 1}', ''),
            cells.get(f'{label_column}{header_row + 1}', ''),
        )
        if header_values != ('Trial #', 'True Label'):
            raise ValueError(
                f'Expected Trial # and True Label headers below {sample_name} in {spreadsheet_path}, '
                f'found {header_values!r}.'
            )

        labels = []
        for row in range(header_row + 2, 10_000):
            trial = cells.get(f'{trial_column}{row}', '')
            value = cells.get(f'{label_column}{row}', '')
            if not trial and not value:
                break
            if not trial or not value:
                raise ValueError(
                    f'Incomplete trial/label pair at row {row} for {sample_name} in {spreadsheet_path}.'
                )
            expected_trial = len(labels) + 1
            if int(trial) != expected_trial:
                raise ValueError(
                    f'Expected trial {expected_trial} at row {row} for {sample_name} in {spreadsheet_path}, '
                    f'found {trial!r}.'
                )
            label = int(value) - 1
            if label < 0 or label >= len(self.config.category):
                raise ValueError(f'Invalid test label {value!r} for {sample_name} in {spreadsheet_path}.')
            labels.append(label)
        return np.asarray(labels, dtype=np.int64)

    @staticmethod
    def _xlsx_column(reference: str) -> int:
        column = ''.join(char for char in reference if char.isalpha())
        value = 0
        for char in column:
            value = value * 26 + ord(char) - ord('A') + 1
        return value

    @staticmethod
    def _xlsx_row(reference: str) -> int:
        row = ''.join(char for char in reference if char.isdigit())
        if not row:
            raise ValueError(f'Invalid spreadsheet cell reference {reference!r}.')
        return int(row)

    @staticmethod
    def _xlsx_column_name(index: int) -> str:
        chars = []
        while index:
            index, remainder = divmod(index - 1, 26)
            chars.append(chr(ord('A') + remainder))
        return ''.join(reversed(chars))

    def _epochs_to_raw(self, epochs: np.ndarray, ch_names: list[str], fs: float, labels: np.ndarray, verbose: bool):
        n_samples, n_channels, n_trials = epochs.shape
        if len(labels) != n_trials:
            raise ValueError(f'Label count ({len(labels)}) does not match trial count ({n_trials}).')

        signal = np.transpose(epochs, (1, 2, 0)).reshape(n_channels, n_trials * n_samples)
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(signal * 1e-6, info, verbose=verbose)
        raw.set_annotations(mne.Annotations(
            onset=np.arange(n_trials) * n_samples / fs,
            duration=np.full(n_trials, n_samples / fs),
            description=[str(int(label)) for label in labels],
        ))
        return raw


if __name__ == "__main__":
    builder = BCIC2020ImagineBuilder('finetune')
    # builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
    #
    # labels = torch.tensor(dataset['train']['label'], dtype=torch.int32)
    # labels = torch.bincount(labels, minlength=4)
    # print(labels)
    #
    # labels = torch.tensor(dataset['validation']['label'], dtype=torch.int32)
    # labels = torch.bincount(labels, minlength=4)
    # print(labels)
    #
    # labels = torch.tensor(dataset['test']['label'], dtype=torch.int32)
    # labels = torch.bincount(labels, minlength=4)
    # print(labels)


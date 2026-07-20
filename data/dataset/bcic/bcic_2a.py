import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne.io
import numpy as np
from mne.io import BaseRaw
from pandas import DataFrame
from scipy.io import loadmat

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class BCIC2AConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "This data set consists of EEG data from 9 subjects. The cue-based BCI paradigm consisted of "
        "four different motor imagery tasks, namely the imagination of movement of the "
        "left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4). "
        "Two sessions on different days were recorded for each subject. Each session is comprised "
        "of 6 runs separated by short breaks. One run consists of 48 trials (12 for each of the "
        "four possible classes), yielding a total of 288 trials per session.")
    citation: Optional[str] = "https://www.bbci.de/competition/iv/desc_2a.pdf"

    filter_low: float = 0.5
    filter_high: float = 45.0
    filter_notch: float = 50.0
    is_notched: bool = False

    dataset_name: Optional[str] = 'bcic_2a'
    task_type: DatasetTaskType = DatasetTaskType.MOTOR_IMAGINARY

    file_ext: str = 'gdf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                                'EEG-Fz',
                'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4',
            'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8',
                'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13',
                         'EEG-14', 'EEG-Pz', 'EEG-15',
                                  'EEG-16',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 4
    suffix_path: str = os.path.join('BCI Competition IV', '2a')
    scan_sub_dir: str = ''

    category: list[str] = field(default_factory=lambda: [
        'left', 'right', 'foot', 'tongue'
    ])


class BCIC2ABuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = BCIC2AConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    _TRAIN_EVENT_LABELS = {
        '769': 'left',
        '770': 'right',
        '771': 'foot',
        '772': 'tongue',
    }
    _EVALUATION_EVENT = '783'

    def __init__(self, config_name='pretrain', **kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        # noinspection PyTypeChecker
        scan_path: str = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext):
                    # if self.config.is_finetune and 'E' in file:
                    #     continue
                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        match = re.fullmatch(r'A(\d{2})([TE])', file_name)
        if match is None:
            raise ValueError(f'Unexpected BCIC IV 2a recording name: {file_name}')
        subject, session_type = match.groups()

        return {
            'subject': int(subject),
            'session': 1 if session_type == 'T' else 2,
            'session_type': session_type,
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
            event_pairs = [
                (onset, str(description))
                for onset, description in zip(
                    raw.annotations.onset,
                    raw.annotations.description,
                )
            ]

        session_type = info.get('session_type') or self._resolve_file_name(file_path)['session_type']
        if session_type == 'T':
            cue_events = [
                (onset, self._TRAIN_EVENT_LABELS[description])
                for onset, description in event_pairs
                if description in self._TRAIN_EVENT_LABELS
            ]
        else:
            cue_onsets = [
                onset for onset, description in event_pairs
                if description == self._EVALUATION_EVENT
            ]
            if self.config.is_finetune:
                label_path = os.path.join(
                    self.config.raw_path,
                    'true_labels',
                    f'{self._extract_file_name(file_path)}.mat',
                )
                class_ids = loadmat(label_path)['classlabel'].reshape(-1)
                if len(cue_onsets) != len(class_ids):
                    raise ValueError(
                        f'BCIC IV 2a evaluation cue/label count mismatch for {file_path}: '
                        f'{len(cue_onsets)} cues, {len(class_ids)} labels'
                    )
                cue_events = [
                    (onset, self.config.category[int(class_id) - 1])
                    for onset, class_id in zip(cue_onsets, class_ids)
                ]
            else:
                cue_events = [(onset, 'default') for onset in cue_onsets]

        return [
            (
                label if self.config.is_finetune else 'default',
                round(onset * 1000),
                round((onset + self.config.wnd_div_sec) * 1000),
            )
            for onset, label in cue_events
        ]

    def _divide_split(self, df: DataFrame) -> DataFrame:

        if self.config.is_finetune:
            df.loc[df['subject'].isin(np.array([1, 2, 3, 4, 5])), 'split'] = 'train'
            df.loc[df['subject'].isin(np.array([6, 7])), 'split'] = 'valid'
            df.loc[df['subject'].isin(np.array([8, 9])), 'split'] = 'test'
        else:
            df.loc[df['subject'].isin(np.array([1, 2, 3, 4, 5, 6, 7])), 'split'] = 'train'
            df.loc[df['subject'].isin(np.array([8, 9])), 'split'] = 'valid'

        return df

    def standardize_chs_names(self, montage: str):
        if montage == '10_20':
            return [
                                'FZ',
                  'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
              'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                  'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
                          'P1', 'PZ', 'P2',
                                'POZ',
            ]
        else:
            raise ValueError('No such montage in bcic_2a dataset')

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_gdf(file_path, preload=preload, verbose=verbose)
            return raw


if __name__ == "__main__":
    builder = BCIC2ABuilder('finetune')
    # builder.clean_disk_cache()
    builder.preproc(n_proc=2)
    builder.download_and_prepare(num_proc=2)
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


import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGDatasetBuilder, EEGConfig

OFFICIAL_LABEL_PREFIX_LENGTH = 3
OFFICIAL_CLASS_DIRS = {'00_epilepsy', '01_no_epilepsy'}
LEGACY_CLASS_LABELS = {
    'epilepsy_edf': 'epilepsy',
    'no_epilepsy_edf': 'no_epilepsy',
}


@dataclass
class TuepConfig(EEGConfig):
    name: str = "pretrain"
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("2.0.1")
    description: Optional[str] = (
        'TUH EEG Epilepsy Corpus, a corpus developed to motivate the development of new methods '
        'for automatic analysis of EEG files using machine learning. '
        'This corpus is a subset of the TUH EEG Corpus and contains sessions from patients '
        'with epilepsy. To balance the corpus, some sessions are provided from patients that '
        'do not have epilepsy.')
    citation: Optional[str] = """\
    @INPROCEEDINGS{8257044,
    author={Veloso, L. and McHugh, J. and von Weltin, E. and Lopez, S. and Obeid, I. and Picone, J.},
    booktitle={2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)}, 
    title={Big data resources for EEGs: Enabling deep learning research}, 
    year={2017},
    volume={},
    number={},
    pages={1-3},
    keywords={Electroencephalography;Epilepsy;Machine learning;Neural engineering;Hospitals;History;Training data},
    doi={10.1109/SPMB.2017.8257044}}
    """

    filter_notch: float = 60.0

    dataset_name: Optional[str] = 'tuep'
    task_type: DatasetTaskType = DatasetTaskType.SEIZURE
    file_ext: str = 'edf'
    position_montage: Optional[str] = 'standard_1020'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '01_tcp_ar': [
            'EEG FP1-REF',
            'EEG FP2-REF',
            'EEG F7-REF',
            'EEG F3-REF',
            'EEG FZ-REF',
            'EEG F4-REF',
            'EEG F8-REF',
            'EEG A1-REF',
            'EEG T3-REF',
            'EEG C3-REF',
            'EEG CZ-REF',
            'EEG C4-REF',
            'EEG A2-REF',
            'EEG T4-REF',
            'EEG T5-REF',
            'EEG P3-REF',
            'EEG PZ-REF',
            'EEG P4-REF',
            'EEG T6-REF',
            'EEG O1-REF',
            'EEG O2-REF',
        ],
        '02_tcp_le': [
            'EEG FP1-LE',
            'EEG FP2-LE',
            'EEG F7-LE',
            'EEG F3-LE',
            'EEG FZ-LE',
            'EEG F4-LE',
            'EEG F8-LE',
            'EEG A1-LE',
            'EEG T3-LE',
            'EEG C3-LE',
            'EEG CZ-LE',
            'EEG C4-LE',
            'EEG T4-LE',
            'EEG A2-LE',
            'EEG T5-LE',
            'EEG P3-LE',
            'EEG PZ-LE',
            'EEG P4-LE',
            'EEG T6-LE',
            'EEG O1-LE',
            'EEG OZ-LE',
            'EEG O2-LE',
        ],
        '03_tcp_ar_a': [
            'EEG FP1-REF',
            'EEG FP2-REF',
            'EEG F7-REF',
            'EEG F3-REF',
            'EEG FZ-REF',
            'EEG F4-REF',
            'EEG F8-REF',
            'EEG T3-REF',
            'EEG C3-REF',
            'EEG CZ-REF',
            'EEG C4-REF',
            'EEG T4-REF',
            'EEG T5-REF',
            'EEG P3-REF',
            'EEG PZ-REF',
            'EEG P4-REF',
            'EEG T6-REF',
            'EEG O1-REF',
            'EEG O2-REF',
        ],
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 45
    suffix_path: str =  os.path.join('TUE', 'tuep')
    scan_sub_dir: str = ''

    category: list[str] = field(default_factory=lambda: ['epilepsy', 'no_epilepsy'])


class TuepBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TuepConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session, term = file_name.split('_')[-3:]
        session = int(session[1:])
        term = int(term[1:])
        return {
            'subject': subject,
            'session': session,
            'term': term,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        montage = self._extract_middle_path(file_path, -2, -1)[0]
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            sex = raw.info['subject_info']['sex']
            time = raw.duration

        info.update({
            'montage': montage,
            'time': time,
            'sex': sex,
        })
        return info

    def _resolve_exp_events(
            self, file_path: str, info: dict[str, Any]
    ) -> list[tuple[str, int, int]]:
        class_dir = self._extract_middle_path(file_path, -5, -4)[0]
        if class_dir not in OFFICIAL_CLASS_DIRS:
            raise ValueError(
                f'Expected official TUEP class directory in '
                f'{sorted(OFFICIAL_CLASS_DIRS)}, but got {class_dir!r} '
                f'for recording {file_path}.')

        label = class_dir[OFFICIAL_LABEL_PREFIX_LENGTH:]
        if label not in self.config.category:
            raise ValueError(
                f'Expected TUEP label in {self.config.category}, but got '
                f'{label!r} from class directory {class_dir!r}.')
        return [(label, 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        df = self._divide_label_balance_all_split(df, None if self.config.is_finetune else ['train', 'valid'])
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.split(sep=' ')[1].split('-')[0] for ch in chs]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs_std]
        self._std_chs_cache[montage] = chs_std
        return chs_std


@dataclass
class TuepV200Config(TuepConfig):
    """Configuration for the Tianpu TUEP v2.0.0 directory layout."""

    version: Optional[Union[datasets.utils.Version, str]] = (
        datasets.utils.Version("2.0.0")
    )
    dataset_name: Optional[str] = 'tuep_v2_0_0'
    suffix_path: str = os.path.join('TUE', 'tuep_v2_0_0')


class TuepV200Builder(TuepBuilder):
    """Build TUEP v2.0.0 from separate epilepsy and no-epilepsy roots."""

    BUILDER_CONFIG_CLASS = TuepV200Config
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),
    ]

    def _resolve_exp_events(
            self, file_path: str, info: dict[str, Any]
    ) -> list[tuple[str, int, int]]:
        """Return the legacy class label encoded in the top-level directory.

        Parameters
        ----------
        file_path : str
            Absolute EDF path below the configured TUEP v2.0.0 raw root.
        info : dict[str, Any]
            Recording metadata. It is not used because the corpus label is
            encoded by the top-level class directory.

        Returns
        -------
        list[tuple[str, int, int]]
            One full-recording event labelled ``epilepsy`` or ``no_epilepsy``.

        Raises
        ------
        ValueError
            If the recording is not below a recognised legacy class root.
        """
        relative_path = os.path.relpath(file_path, self.config.raw_path)
        class_dir = relative_path.split(os.sep, maxsplit=1)[0]
        try:
            label = LEGACY_CLASS_LABELS[class_dir]
        except KeyError as error:
            raise ValueError(
                f'Expected legacy TUEP class directory in '
                f'{sorted(LEGACY_CLASS_LABELS)}, but got {class_dir!r} '
                f'for recording {file_path}.') from error
        return [(label, 0, -1)]


if __name__ == "__main__":
    builder = TuepBuilder()
    builder.preproc()
    builder.download_and_prepare(num_proc=8)
    dataset = builder.as_dataset()
    print(dataset)

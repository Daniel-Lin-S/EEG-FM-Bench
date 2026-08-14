"""Build the SEED emotion dataset from canonical Neuroscan CNT recordings.

The configured SEED raw root must contain exactly the 15-subject by
three-session Neuroscan Version 3.0 CNT recordings in
``seed_eeg/SEED_RAW_EEG`` and its source ``time.txt``. The builder keeps the
public ``seed`` identity and its existing data schema, but it does not reuse
caches made by the EEGLAB source adapter.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mne
import numpy as np

from data.dataset.seeds.seed import SeedBuilder, SeedConfig


CNT_SOURCE_DIRECTORY = Path('seed_eeg') / 'SEED_RAW_EEG'
LEGACY_SET_DIRECTORY = Path('SEED_EEG') / 'SEED_RAW_EEG' / 'resampled'
TIME_FILE_NAME = 'time.txt'
SOURCE_SAMPLING_FREQUENCY = 1000.0
EXPECTED_SUBJECTS = tuple(range(1, 16))
EXPECTED_SESSIONS = tuple(range(1, 4))
EXPECTED_RECORDING_COUNT = len(EXPECTED_SUBJECTS) * len(EXPECTED_SESSIONS)
READER_TIMEOUT_SECONDS = 60.0
CONTRACT_FILE_NAME = 'seed_cnt_source_contract.json'
CONTRACT_VERSION = 2
SOURCE_LAYOUT_CNT = 'canonical_cnt_v1'
SOURCE_LAYOUT_SET = 'legacy_set_v1'
CNT_NON_BENCHMARK_CHANNELS = ('M1', 'M2', 'VEO', 'HEO')
NEUROSCAN_CNT_SIGNATURE = b'Version 3.0\x00'
NEUROSCAN_CNT_DATA_FORMAT = 'int16'
NEUROSCAN_CNT_EOG_CHANNELS = ('VEO', 'HEO')
PREFLIGHT_SAMPLE_COUNT = 10
NEUROSCAN_CNT_READER = 'mne.io.read_raw_cnt'


@dataclass(frozen=True)
class SeedSourceLayout:
    """Validated, source-layout-specific SEED raw-data locations.

    Attributes
    ----------
    kind : str
        Identifier for the supported CNT or EEGLAB layout.
    raw_root : Path
        Configured SEED raw root.
    recordings : tuple[Path, ...]
        Recording paths ordered by subject and session.
    time_path : Path, optional
        CNT trial-boundary file.  EEGLAB layouts do not require this file.
    """

    kind: str
    raw_root: Path
    recordings: tuple[Path, ...]
    time_path: Optional[Path] = None


def _expected_recording_names(extension: str) -> set[str]:
    """Return the canonical SEED recording names for one file extension."""
    return {
        f'{subject}_{session}.{extension}'
        for subject in EXPECTED_SUBJECTS
        for session in EXPECTED_SESSIONS
    }


def expected_seed_cnt_channels() -> list[str]:
    """Return the exact canonical SEED CNT source-channel order.

    The CNT source has 62 EEG channels used by the legacy SEED converter plus
    the two mastoid and two ocular channels. The benchmark montage later drops
    these four non-benchmark channels along with the two CB channels.

    Returns
    -------
    list[str]
        Ordered source channel names with shape ``(66,)``.
    """
    channels = SeedBuilder._orig_ch_names()
    channels.insert(channels.index('TP7'), 'M1')
    channels.insert(channels.index('P7'), 'M2')
    return channels + list(CNT_NON_BENCHMARK_CHANNELS[2:])


def _reader_crash_message(file_path: Path, exit_code: int) -> str:
    """Describe a contained CNT-reader crash without blaming source layout."""
    if exit_code < 0:
        signal_number = -exit_code
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f'signal {signal_number}'
        failure = f'crashed with {signal_name} (exit code {exit_code})'
    else:
        failure = f'exited unexpectedly with code {exit_code}'
    return (
        f"MNE's Neuroscan CNT reader {failure} while opening "
        f'{file_path.name}. '
        'This is a reader-process failure, not a SEED filename, directory, '
        'or label rejection. The source uses the legacy Neuroscan Version 3.0 '
        'format and must be read with explicit int16 samples and recomputed '
        'sample counts. Preprocessing stopped without loading CNT data.'
    )


def _validate_neuroscan_signatures(recordings: tuple[Path, ...]) -> None:
    """Require the legacy Neuroscan signature before selecting its reader.

    Parameters
    ----------
    recordings : tuple[Path, ...]
        Canonically ordered CNT recordings with shape ``(45,)``.

    Raises
    ------
    ValueError
        If any recording is not a Neuroscan Version 3.0 CNT file.
    """
    for file_path in recordings:
        with file_path.open('rb') as file:
            signature = file.read(len(NEUROSCAN_CNT_SIGNATURE))
        if signature != NEUROSCAN_CNT_SIGNATURE:
            raise ValueError(
                f'Expected the legacy Neuroscan Version 3.0 CNT signature '
                f'for {file_path.name}; refusing to use a different CNT '
                'reader or source format.'
            )


def _read_seed_neuroscan_cnt(
        file_path: str | Path,
        preload: bool = False,
        verbose: bool = False,
) -> mne.io.BaseRaw:
    """Open a SEED Neuroscan CNT recording with its proven sample contract.

    The source ``numsamples`` header field is invalid. MNE therefore needs the
    known int16 payload format and event-table-derived sample count.

    Parameters
    ----------
    file_path : str or Path
        Neuroscan Version 3.0 CNT recording.
    preload : bool, optional, default=False
        Whether MNE should load all samples immediately.
    verbose : bool, optional, default=False
        Whether MNE should emit reader logs.

    Returns
    -------
    mne.io.BaseRaw
        CNT recording with 66 source channels and a correct sample count.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'.*Could not parse meas date from the header.*',
            category=RuntimeWarning,
        )
        return mne.io.read_raw_cnt(
            file_path,
            data_format=NEUROSCAN_CNT_DATA_FORMAT,
            recompute_n_samples=True,
            eog=NEUROSCAN_CNT_EOG_CHANNELS,
            preload=preload,
            verbose=verbose,
        )


def _validate_recording_directory(
        directory: Path,
        extension: str,
) -> tuple[Path, ...]:
    """Validate and order one canonical SEED recording directory.

    Parameters
    ----------
    directory : Path
        Directory expected to contain the 45 canonical recordings.
    extension : str
        File extension without a leading period.

    Returns
    -------
    tuple[Path, ...]
        Recordings ordered by subject then session.

    Raises
    ------
    ValueError
        If the directory contains missing, unexpected, or duplicate recordings.
    """
    expected_names = _expected_recording_names(extension)
    recordings = tuple(sorted(directory.glob(f'*.{extension}')))
    actual_names = {path.name for path in recordings}
    if (
            actual_names != expected_names
            or len(recordings) != EXPECTED_RECORDING_COUNT
    ):
        missing = sorted(expected_names.difference(actual_names))
        unexpected = sorted(actual_names.difference(expected_names))
        details = []
        if missing:
            details.append(f'missing {missing[:3]}')
        if unexpected:
            details.append(f'unexpected {unexpected[:3]}')
        if len(recordings) != EXPECTED_RECORDING_COUNT:
            details.append(
                f'expected {EXPECTED_RECORDING_COUNT} recordings, got '
                f'{len(recordings)}'
            )
        raise ValueError(
            f'Invalid SEED {extension.upper()} recording layout in '
            f'{directory}: '
            + '; '.join(details)
        )
    return tuple(
        directory / f'{subject}_{session}.{extension}'
        for subject in EXPECTED_SUBJECTS
        for session in EXPECTED_SESSIONS
    )


def detect_seed_source_layout(raw_root: str | Path) -> SeedSourceLayout:
    """Detect exactly one supported SEED raw layout.

    Parameters
    ----------
    raw_root : str or Path
        Configured SEED dataset root.

    Returns
    -------
    SeedSourceLayout
        The complete canonical CNT or legacy EEGLAB source layout.

    Raises
    ------
    FileNotFoundError
        If the configured root does not exist.
    ValueError
        If source data are incomplete, mixed, or unrecognised.
    """
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f'Configured SEED raw root is not a directory: {root}'
        )

    cnt_directory = root / CNT_SOURCE_DIRECTORY
    set_directory = root / LEGACY_SET_DIRECTORY
    cnt_files = (
        tuple(cnt_directory.glob('*.cnt')) if cnt_directory.is_dir() else ()
    )
    set_files = (
        tuple(set_directory.glob('*.set')) if set_directory.is_dir() else ()
    )
    all_cnt_files = tuple(root.rglob('*.cnt'))
    all_set_files = tuple(root.rglob('*.set'))
    time_path = cnt_directory / TIME_FILE_NAME

    if cnt_files and all_set_files:
        raise ValueError(
            'SEED raw root contains both canonical CNT and legacy EEGLAB '
            'recordings; select one source layout before preprocessing.'
        )
    if cnt_files or time_path.exists():
        recordings = _validate_recording_directory(cnt_directory, 'cnt')
        if not time_path.is_file():
            raise ValueError(
                f'Canonical SEED CNT layout requires {time_path}.'
            )
        return SeedSourceLayout(
            kind=SOURCE_LAYOUT_CNT,
            raw_root=root,
            recordings=recordings,
            time_path=time_path,
        )
    if set_files:
        if all_cnt_files:
            raise ValueError(
                'SEED raw root contains both canonical CNT and legacy EEGLAB '
                'recordings; select one source layout before preprocessing.'
            )
        recordings = _validate_recording_directory(set_directory, 'set')
        return SeedSourceLayout(
            kind=SOURCE_LAYOUT_SET,
            raw_root=root,
            recordings=recordings,
        )

    known_sources = list(all_cnt_files) + list(all_set_files)
    if known_sources:
        raise ValueError(
            'SEED raw root contains recordings outside a supported canonical '
            'CNT or legacy EEGLAB layout.'
        )
    raise ValueError(
        'SEED raw root contains neither canonical CNT recordings nor legacy '
        'EEGLAB recordings.'
    )


def parse_seed_cnt_time_file(
        time_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse the source-provided 1000 Hz SEED trial boundaries.

    Parameters
    ----------
    time_path : str or Path
        ``time.txt`` containing ``start_point_list`` and ``end_point_list``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Integer start and end sample arrays, each with shape ``(15,)``.

    Raises
    ------
    ValueError
        If the file is malformed, lacks the 1000 Hz declaration, or has invalid
        trial boundaries.
    """
    path = Path(time_path)
    text = path.read_text(encoding='utf-8')
    if '#1000Hz' not in text.replace(' ', ''):
        raise ValueError(
            f'Expected a 1000 Hz declaration in SEED timing file: {path}'
        )
    lists: dict[str, np.ndarray] = {}
    for name in ('start_point_list', 'end_point_list'):
        match = re.search(rf'{name}\s*=\s*\[([^]]*)\]', text)
        if match is None:
            raise ValueError(f'Missing {name} in SEED timing file: {path}')
        values = [value.strip() for value in match.group(1).split(',')]
        try:
            lists[name] = np.asarray(
                [int(value) for value in values],
                dtype=np.int64,
            )
        except ValueError as error:
            raise ValueError(
                f'Expected integer {name} values in SEED timing file: {path}'
            ) from error
    starts = lists['start_point_list']
    ends = lists['end_point_list']
    expected_shape = (15,)
    if starts.shape != expected_shape or ends.shape != expected_shape:
        raise ValueError(
            'Expected 15 SEED trial boundaries in each timing list, but got '
            f'{starts.shape} starts and {ends.shape} ends.'
        )
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError(
            'SEED timing boundaries must be non-negative and ordered.'
        )
    if np.any(starts[1:] <= starts[:-1]) or np.any(ends[1:] <= ends[:-1]):
        raise ValueError('SEED timing boundaries must be strictly increasing.')
    return starts, ends


def _read_neuroscan_cnt_metadata_worker(
        file_path: str,
        result_queue: Any,
) -> None:
    """Read CNT metadata and samples in a child process.

    Parameters
    ----------
    file_path : str
        Neuroscan CNT recording path.
    result_queue : multiprocessing.Queue
        Queue used to return metadata or a serialised reader error.
    """
    try:
        raw = _read_seed_neuroscan_cnt(file_path, preload=False, verbose=False)
        try:
            samples = raw.get_data(start=0, stop=PREFLIGHT_SAMPLE_COUNT)
            result_queue.put({
                'channels': raw.ch_names,
                'n_times': int(raw.n_times),
                'sfreq': float(raw.info['sfreq']),
                'sample_shape': list(samples.shape),
                'samples_are_finite': bool(np.isfinite(samples).all()),
            })
        finally:
            raw.close()
    except BaseException as error:
        result_queue.put({
            'error': f'{type(error).__name__}: {error}',
        })


@dataclass
class SeedCntConfig(SeedConfig):
    """Configuration for canonical SEED Neuroscan CNT recordings."""

    file_ext: str = 'cnt'
    orig_fs: float = SOURCE_SAMPLING_FREQUENCY


class SeedCntBuilder(SeedBuilder):
    """Build SEED from verified CNT recordings without source fallback."""

    BUILDER_CONFIG_CLASS = SeedCntConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, wnd_div_sec=4),
        BUILDER_CONFIG_CLASS(
            name='finetune_sub_dependent',
            is_finetune=True,
            wnd_div_sec=4,
            is_cross_subject=False,
        ),
    ]

    def __init__(self, config_name: str = 'pretrain', **kwargs: Any) -> None:
        super().__init__(config_name, **kwargs)
        self._cnt_source_layout = detect_seed_source_layout(
            self.config.raw_path
        )
        if self._cnt_source_layout.kind != SOURCE_LAYOUT_CNT:
            raise ValueError(
                'SeedCntBuilder requires the canonical SEED CNT layout, but '
                f'found {self._cnt_source_layout.kind!r}.'
            )
        if self._cnt_source_layout.time_path is None:
            raise ValueError('Canonical SEED CNT layout is missing time.txt.')
        _validate_neuroscan_signatures(self._cnt_source_layout.recordings)
        self._trial_starts, self._trial_ends = parse_seed_cnt_time_file(
            self._cnt_source_layout.time_path
        )

    def _walk_raw_data_files(self) -> list[str]:
        """Return only the validated canonical CNT recordings."""
        return [str(path) for path in self._cnt_source_layout.recordings]

    def _read_raw_data(
            self,
            file_path: str,
            preload: bool = False,
            verbose: bool = False,
    ) -> mne.io.BaseRaw:
        """Read a CNT recording through the Neuroscan reader and no fallback."""
        return _read_seed_neuroscan_cnt(
            file_path,
            preload=preload,
            verbose=verbose,
        )

    def _resolve_exp_events(
            self,
            file_path: str,
            info: dict[str, Any],
    ) -> list[tuple[str, int, int]]:
        """Return source-timed SEED emotion events in milliseconds."""
        if not self.config.is_finetune:
            return [('default', 0, -1)]
        labels = self.label_meta[0]
        return [
            (
                self.config.category[int(label)],
                int(start * 1000 / SOURCE_SAMPLING_FREQUENCY),
                int(end * 1000 / SOURCE_SAMPLING_FREQUENCY),
            )
            for label, start, end in zip(
                labels,
                self._trial_starts,
                self._trial_ends,
            )
        ]

    def _read_cnt_metadata_isolated(self, file_path: Path) -> dict[str, Any]:
        """Read one CNT header in a child process and fail on reader crashes."""
        context = mp.get_context('spawn')
        result_queue = context.Queue()
        process = context.Process(
            target=_read_neuroscan_cnt_metadata_worker,
            args=(str(file_path), result_queue),
        )
        process.start()
        process.join(READER_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError(
                f'Neuroscan CNT reader timed out after '
                f'{READER_TIMEOUT_SECONDS} seconds '
                f'for {file_path.name}.'
            )
        if process.exitcode != 0:
            raise RuntimeError(
                _reader_crash_message(file_path, process.exitcode)
            )
        try:
            result = result_queue.get_nowait()
        except queue.Empty as error:
            raise RuntimeError(
                f'Neuroscan CNT reader returned no metadata for '
                f'{file_path.name}.'
            ) from error
        if not isinstance(result, dict):
            raise RuntimeError(
                'Neuroscan CNT reader returned invalid metadata for '
                f'{file_path.name}.'
            )
        if 'error' in result:
            raise RuntimeError(
                f'Neuroscan CNT reader failed for {file_path.name}: '
                f'{result["error"]}'
            )
        return result

    def _validate_cnt_metadata(
            self,
            file_path: Path,
            metadata: dict[str, Any],
    ) -> None:
        """Validate one isolated CNT header against the SEED source contract."""
        expected_channels = expected_seed_cnt_channels()
        channels = metadata.get('channels')
        if not isinstance(channels, list) or channels != expected_channels:
            raise ValueError(
                f'Expected documented 66 SEED CNT channels for '
                f'{file_path.name}, '
                f'but got {channels!r}.'
            )
        try:
            sfreq = float(metadata['sfreq'])
            n_times = int(metadata['n_times'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f'Expected numeric CNT sampling metadata for {file_path.name}.'
            ) from error
        if not np.isfinite(sfreq) or sfreq != SOURCE_SAMPLING_FREQUENCY:
            raise ValueError(
                f'Expected {SOURCE_SAMPLING_FREQUENCY} Hz CNT sampling for '
                f'{file_path.name}, but got {sfreq} Hz.'
            )
        if n_times <= int(self._trial_ends[-1]):
            raise ValueError(
                f'CNT recording {file_path.name} ends at sample {n_times}, but '
                f'SEED timing requires sample {int(self._trial_ends[-1])}.'
            )
        expected_shape = (
            len(expected_channels),
            PREFLIGHT_SAMPLE_COUNT,
        )
        observed_shape = metadata.get('sample_shape')
        if observed_shape != list(expected_shape):
            raise ValueError(
                f'Expected readable CNT samples with shape {expected_shape} '
                f'for {file_path.name}, but got {observed_shape!r}.'
            )
        if metadata.get('samples_are_finite') is not True:
            raise ValueError(
                f'Expected finite CNT samples for {file_path.name}.'
            )

    def _preflight_cnt_recordings(self) -> None:
        """Prove every CNT recording has compatible metadata before writes."""
        for file_path in self._cnt_source_layout.recordings:
            metadata = self._read_cnt_metadata_isolated(file_path)
            self._validate_cnt_metadata(file_path, metadata)

    def _source_contract_path(self) -> Path:
        """Return the private sidecar used to prevent source-cache mixing."""
        return Path(self.summary_path) / CONTRACT_FILE_NAME

    def _source_contract(self) -> dict[str, Any]:
        """Return the semantic contract required for CNT cache reuse."""
        if self._cnt_source_layout.time_path is None:
            raise ValueError('Canonical SEED CNT layout is missing time.txt.')
        timing_hash = hashlib.sha256(
            self._cnt_source_layout.time_path.read_bytes()
        ).hexdigest()
        return {
            'version': CONTRACT_VERSION,
            'adapter': type(self).__name__,
            'dataset_name': self.config.dataset_name,
            'source_layout': SOURCE_LAYOUT_CNT,
            'container_format': 'neuroscan_cnt_version_3_0',
            'reader': NEUROSCAN_CNT_READER,
            'reader_data_format': NEUROSCAN_CNT_DATA_FORMAT,
            'reader_recompute_n_samples': True,
            'recordings': [
                path.name for path in self._cnt_source_layout.recordings
            ],
            'timing_sha256': timing_hash,
            'source_sampling_frequency': SOURCE_SAMPLING_FREQUENCY,
            'source_channels': expected_seed_cnt_channels(),
        }

    def _source_contract_matches(self) -> bool:
        """Return whether an existing cache has this exact CNT contract."""
        path = self._source_contract_path()
        try:
            observed = json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        return observed == self._source_contract()

    def _has_existing_artifacts(self) -> bool:
        """Return whether a signal or Arrow artifact exists for this run."""
        paths = [
            Path(self.info_csv_path),
            Path(self.mid_file_csv_path),
            Path(self._done_marker_path()),
            Path(self.config.mid_path) / self.config.name,
            (
                Path(self.config.data_path)
                / self.config.dataset_name
                / self.config.name
            ),
        ]
        return any(
            path.is_file() or (path.is_dir() and any(path.iterdir()))
            for path in paths
        )

    def _require_compatible_existing_artifacts(self) -> None:
        """Refuse to clean, reuse, or overwrite unprovenanced SEED artifacts."""
        has_artifacts = self._has_existing_artifacts()
        if has_artifacts and not self._source_contract_matches():
            raise RuntimeError(
                'Existing SEED artifacts lack a matching CNT source contract; '
                'they are preserved and will not be reused or overwritten.'
            )

    def _write_source_contract(self) -> None:
        """Atomically record the CNT contract after successful preprocessing."""
        path = self._source_contract_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix('.tmp')
        temporary_path.write_text(
            json.dumps(
                self._source_contract(),
                indent=2,
                sort_keys=True,
            ) + '\n',
            encoding='utf-8',
        )
        os.replace(temporary_path, path)

    def _is_preproc_cached(self) -> bool:
        """Reuse CNT signal cache only when its source contract matches."""
        return self._source_contract_matches() and super()._is_preproc_cached()

    def preproc(self, n_proc: Optional[int] = None) -> None:
        """Preflight CNT data before creating, cleaning, or reusing output."""
        self._require_compatible_existing_artifacts()
        if self._is_preproc_cached():
            return
        self._preflight_cnt_recordings()
        super().preproc(n_proc=n_proc)
        self._write_source_contract()

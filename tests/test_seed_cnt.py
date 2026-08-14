"""Tests for SEED source selection and fail-closed CNT preprocessing guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data.dataset.seeds.seed import SeedBuilder
from data.dataset.seeds.seed_cnt import (
    CNT_SOURCE_DIRECTORY,
    LEGACY_SET_DIRECTORY,
    NEUROSCAN_CNT_DATA_FORMAT,
    NEUROSCAN_CNT_EOG_CHANNELS,
    NEUROSCAN_CNT_SIGNATURE,
    PREFLIGHT_SAMPLE_COUNT,
    SOURCE_LAYOUT_CNT,
    SOURCE_LAYOUT_SET,
    SeedCntBuilder,
    SeedCntConfig,
    SeedSourceLayout,
    _read_seed_neuroscan_cnt,
    _validate_neuroscan_signatures,
    detect_seed_source_layout,
    expected_seed_cnt_channels,
    parse_seed_cnt_time_file,
)
from data.processor.wrapper import resolve_dataset_builder


def _write_time_file(path: Path) -> None:
    """Write a valid source timing file with 15 short trials."""
    starts = ','.join(str(index * 20) for index in range(15))
    ends = ','.join(str(index * 20 + 10) for index in range(15))
    path.write_text(
        f'start_point_list = [{starts}];#1000Hz\n'
        f'end_point_list = [{ends}];\n',
        encoding='utf-8',
    )


def _write_recordings(directory: Path, extension: str) -> tuple[Path, ...]:
    """Create the canonical 15-subject by three-session recording names."""
    directory.mkdir(parents=True)
    recordings = []
    for subject in range(1, 16):
        for session in range(1, 4):
            path = directory / f'{subject}_{session}.{extension}'
            path.touch()
            recordings.append(path)
    return tuple(recordings)


def _write_cnt_layout(root: Path) -> tuple[Path, ...]:
    """Create one complete canonical CNT layout in a temporary root."""
    directory = root / CNT_SOURCE_DIRECTORY
    recordings = _write_recordings(directory, 'cnt')
    _write_time_file(directory / 'time.txt')
    return recordings


class SeedSourceLayoutTests(unittest.TestCase):
    """Verify strict, source-aware SEED adapter selection."""

    def test_detects_complete_cnt_layout_and_resolves_cnt_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recordings = _write_cnt_layout(root)

            layout = detect_seed_source_layout(root)

            self.assertEqual(layout.kind, SOURCE_LAYOUT_CNT)
            self.assertEqual(layout.recordings, recordings)
            self.assertIs(
                resolve_dataset_builder('seed', raw_path=root),
                SeedCntBuilder,
            )

    def test_detects_complete_legacy_set_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_recordings(root / LEGACY_SET_DIRECTORY, 'set')

            layout = detect_seed_source_layout(root)

            self.assertEqual(layout.kind, SOURCE_LAYOUT_SET)
            self.assertIs(
                resolve_dataset_builder('seed', raw_path=root),
                SeedBuilder,
            )

    def test_rejects_mixed_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cnt_layout(root)
            _write_recordings(root / LEGACY_SET_DIRECTORY, 'set')

            with self.assertRaisesRegex(ValueError, 'both canonical CNT'):
                detect_seed_source_layout(root)

    def test_rejects_incomplete_and_unrecognised_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / CNT_SOURCE_DIRECTORY
            directory.mkdir(parents=True)
            (directory / '1_1.cnt').touch()
            _write_time_file(directory / 'time.txt')

            with self.assertRaisesRegex(ValueError, 'Invalid SEED CNT'):
                detect_seed_source_layout(root)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            unknown = root / 'other'
            unknown.mkdir()
            (unknown / '1_1.cnt').touch()

            with self.assertRaisesRegex(ValueError, 'outside a supported'):
                detect_seed_source_layout(root)


class SeedCntBuilderTests(unittest.TestCase):
    """Test CNT metadata and cache guards without reading real CNT files."""

    def test_expected_channels_include_source_only_recordings(self) -> None:
        """Keep the full CNT header contract separate from EEG selection."""
        channels = expected_seed_cnt_channels()

        self.assertEqual(len(channels), 66)
        self.assertEqual(SeedCntConfig().orig_fs, 1000.0)
        self.assertEqual(NEUROSCAN_CNT_DATA_FORMAT, 'int16')
        self.assertEqual(channels[channels.index('TP7') - 1], 'M1')
        self.assertEqual(channels[channels.index('P7') - 1], 'M2')
        self.assertEqual(channels[-2:], ['VEO', 'HEO'])
        self.assertEqual(
            [
                channel for channel in channels
                if channel not in {'M1', 'M2', 'VEO', 'HEO'}
            ],
            SeedBuilder._orig_ch_names(),
        )

    def test_neuroscan_reader_has_a_fixed_sample_contract(self) -> None:
        """Use the reader settings required by the SEED CNT payload."""
        raw = SimpleNamespace()
        with patch(
                'data.dataset.seeds.seed_cnt.mne.io.read_raw_cnt',
                return_value=raw,
        ) as read_raw_cnt:
            observed = _read_seed_neuroscan_cnt('recording.cnt')

        self.assertIs(observed, raw)
        read_raw_cnt.assert_called_once_with(
            'recording.cnt',
            data_format=NEUROSCAN_CNT_DATA_FORMAT,
            recompute_n_samples=True,
            eog=NEUROSCAN_CNT_EOG_CHANNELS,
            preload=False,
            verbose=False,
        )

    def test_rejects_a_non_neuroscan_cnt_signature(self) -> None:
        """Do not redirect an unknown CNT container to this reader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'unknown.cnt'
            path.write_bytes(b'not a Neuroscan CNT')

            with self.assertRaisesRegex(ValueError, 'Neuroscan Version 3.0'):
                _validate_neuroscan_signatures((path,))

            path.write_bytes(NEUROSCAN_CNT_SIGNATURE)
            _validate_neuroscan_signatures((path,))

    @staticmethod
    def _builder(root: Path) -> SeedCntBuilder:
        recordings = _write_cnt_layout(root)
        layout = SeedSourceLayout(
            kind=SOURCE_LAYOUT_CNT,
            raw_root=root,
            recordings=recordings,
            time_path=root / CNT_SOURCE_DIRECTORY / 'time.txt',
        )
        builder = object.__new__(SeedCntBuilder)
        builder._cnt_source_layout = layout
        builder._trial_starts, builder._trial_ends = parse_seed_cnt_time_file(
            layout.time_path
        )
        builder.label_meta = np.asarray([[2, 1, 0] * 5], dtype=np.int64)
        builder.config = SimpleNamespace(
            dataset_name='seed',
            is_finetune=True,
            category=['sad', 'neutral', 'happy'],
            name='finetune',
            mid_path=str(root / 'cache'),
            data_path=str(root / 'processed'),
            get_fs_id=lambda: 'fs_256',
        )
        builder.summary_path = str(root / 'summary')
        builder.info_csv_path = str(root / 'summary' / 'seed_finetune_info.csv')
        builder.mid_file_csv_path = str(root / 'summary' / 'seed_finetune.csv')
        return builder

    def test_events_preserve_seed_labels_and_source_time_units(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = self._builder(Path(temp_dir))

            events = builder._resolve_exp_events('ignored.cnt', {})

            self.assertEqual(events[0], ('happy', 0, 10))
            self.assertEqual(events[1], ('neutral', 20, 30))
            self.assertEqual(len(events), 15)

    def test_rejects_incompatible_cnt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = self._builder(Path(temp_dir))
            metadata = {
                'channels': expected_seed_cnt_channels(),
                'n_times': int(builder._trial_ends[-1]) + 1,
                'sfreq': 1000.0,
                'sample_shape': [66, PREFLIGHT_SAMPLE_COUNT],
                'samples_are_finite': True,
            }
            builder._validate_cnt_metadata(
                builder._cnt_source_layout.recordings[0],
                metadata,
            )

            metadata['channels'] = metadata['channels'][:-1]
            with self.assertRaisesRegex(ValueError, 'documented 66'):
                builder._validate_cnt_metadata(
                    builder._cnt_source_layout.recordings[0],
                    metadata,
                )

    def test_reader_subprocess_crash_fails_closed(self) -> None:
        class FailedProcess:
            """Minimal crashed-child process fixture."""

            exitcode = -11

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        class FailedContext:
            """Context that returns a process with a crash exit status."""

            @staticmethod
            def Queue() -> object:
                return object()

            @staticmethod
            def Process(**kwargs: object) -> FailedProcess:
                return FailedProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            builder = self._builder(Path(temp_dir))
            with patch(
                    'data.dataset.seeds.seed_cnt.mp.get_context',
                    return_value=FailedContext(),
            ):
                with self.assertRaisesRegex(RuntimeError, 'SIGSEGV'):
                    builder._read_cnt_metadata_isolated(
                        builder._cnt_source_layout.recordings[0]
                    )

    def test_reader_failure_occurs_before_base_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = self._builder(Path(temp_dir))
            reader_error = RuntimeError('reader metadata could not be proven')
            with patch.object(
                    builder,
                    '_preflight_cnt_recordings',
                    side_effect=reader_error,
            ):
                with patch.object(SeedBuilder, 'preproc') as base_preproc:
                    with self.assertRaisesRegex(
                            RuntimeError,
                            'could not be proven',
                    ):
                        builder.preproc()

            base_preproc.assert_not_called()

    def test_legacy_artifacts_are_preserved_without_a_cnt_contract(
            self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            builder = self._builder(root)
            artifact = (
                Path(builder.config.mid_path) / builder.config.name / 'old'
            )
            artifact.mkdir(parents=True)
            (artifact / 'record.parquet').touch()

            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                builder._require_compatible_existing_artifacts()

            builder._write_source_contract()
            builder._require_compatible_existing_artifacts()

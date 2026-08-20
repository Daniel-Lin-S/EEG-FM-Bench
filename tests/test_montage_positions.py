import unittest

import mne
import numpy as np

from data.dataset.seeds.seed import SeedBuilder, SeedConfig
from data.dataset.tue.tuab import TuabConfig
from data.dataset.tue.tuar import TuarConfig
from data.dataset.tue.tuep import TuepV200Config
from data.dataset.tue.tusl import TuslConfig
from data.dataset.tue.tusz import TuszConfig
from data.dataset.tue.tuev import TuevConfig
from data.processor.montage import resolve_electrode_positions


class MontagePositionTests(unittest.TestCase):
    @staticmethod
    def _raw(ch_names):
        info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
        data = np.arange(len(ch_names) * 10, dtype=np.float64).reshape(
            len(ch_names),
            10,
        )
        return mne.io.RawArray(data, info, verbose=False)

    @staticmethod
    def _standardized_tue_channels(channels):
        replacements = {
            'T3': 'T7',
            'T4': 'T8',
            'T5': 'P7',
            'T6': 'P8',
        }
        labels = [channel.split(maxsplit=1)[-1] for channel in channels]
        return [
            replacements.get(label.split('-')[0], label.split('-')[0])
            for label in labels
        ]

    def test_native_coordinates_take_precedence(self):
        raw = self._raw(["Fp1", "Fp2"])
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        for channel, position in zip(raw.info["chs"], expected):
            channel["loc"][:3] = position
        actual = resolve_electrode_positions(raw, "standard_1020", ["FP1", "FP2"])
        np.testing.assert_allclose(actual, expected)

    def test_template_fallback_supports_prefixes_and_aliases(self):
        raw = self._raw(["EEG Fp1", "EEG T3"])
        positions = resolve_electrode_positions(raw, "standard_1020", ["FP1", "T7"])
        self.assertEqual(positions.shape, (2, 3))
        self.assertTrue(np.isfinite(positions).all())
        self.assertFalse(np.any(np.all(positions == 0.0, axis=1)))

    def test_biosemi_channel_labels_are_resolved(self):
        raw = self._raw(["C29", "A1"])
        positions = resolve_electrode_positions(raw, "biosemi128", ["FP1", "CZ"])
        self.assertEqual(positions.shape, (2, 3))
        self.assertTrue(np.isfinite(positions).all())

    def test_unresolved_template_channel_raises(self):
        raw = self._raw(["not-an-electrode"])
        with self.assertRaisesRegex(ValueError, "could not resolve"):
            resolve_electrode_positions(raw, "standard_1020", ["UNKNOWN"])

    def test_seed_selected_channels_use_standard_1020_coordinates(self):
        config = SeedConfig()
        channels = config.montage['10_10']
        raw = self._raw(channels)

        positions = resolve_electrode_positions(
            raw,
            config.position_montage,
            channels,
        )

        self.assertEqual(config.position_montage, 'standard_1020')
        self.assertEqual(positions.shape, (len(channels), 3))
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(set(channels).issubset(SeedBuilder._orig_ch_names()))
        self.assertEqual(
            set(SeedBuilder._orig_ch_names()).difference(channels),
            {'CB1', 'CB2'},
        )

    def test_tue_positions_preserve_reference_channel_names_and_samples(self):
        configs = (
            TuabConfig(),
            TuepV200Config(),
            TuarConfig(),
            TuslConfig(),
            TuszConfig(),
            TuevConfig(),
        )
        for config in configs:
            self.assertEqual(config.position_montage, 'standard_1020')
            for montage_name, channels in config.montage.items():
                with self.subTest(
                        dataset=config.dataset_name,
                        montage=montage_name,
                ):
                    raw = self._raw(channels)
                    expected_samples = raw.get_data().copy()
                    standardized = self._standardized_tue_channels(channels)
                    positions = resolve_electrode_positions(
                        raw,
                        config.position_montage,
                        standardized,
                    )

                    self.assertEqual(raw.ch_names, channels)
                    np.testing.assert_array_equal(
                        raw.get_data(),
                        expected_samples,
                    )
                    self.assertEqual(positions.shape, (len(channels), 3))
                    self.assertTrue(np.isfinite(positions).all())
                    self.assertFalse(
                        np.any(np.all(positions == 0.0, axis=1))
                    )

    def test_tuar_uniform_canonical_layout_is_accepted(self):
        """Equivalent TUAR montage names share one fixed channel layout."""
        config = TuarConfig()
        montages = {
            f"tuar/{name}": self._standardized_tue_channels(channels)
            for name, channels in config.montage.items()
        }
        from data.processor.wrapper import resolve_common_montage_layout

        layout = resolve_common_montage_layout(
            montages,
            "catch22",
            "tuar",
        )

        self.assertEqual(len(montages), 3)
        self.assertEqual(layout, next(iter(montages.values())))

    def test_tuep_montages_use_their_shared_canonical_layout(self):
        """TUEP reference variants align to their common channel sequence."""
        config = TuepV200Config()
        montages = {
            f"{config.dataset_name}/{name}": self._standardized_tue_channels(
                channels
            )
            for name, channels in config.montage.items()
        }
        from data.processor.wrapper import resolve_common_montage_layout

        layout = resolve_common_montage_layout(
            montages, "minirocket", config.dataset_name
        )

        self.assertEqual(
            layout,
            [
                "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
                "T7", "C3", "CZ", "C4", "T8", "P7", "P3",
                "PZ", "P4", "P8", "O1", "O2",
            ],
        )

    def test_unequal_canonical_layout_reports_each_montage(self):
        """Fixed-width validation names the layouts that differ."""
        from data.processor.wrapper import resolve_common_montage_layout

        with self.assertRaisesRegex(
                ValueError,
                "tuar/01_tcp_ar=.*tuar/02_tcp_le=",
        ):
            resolve_common_montage_layout(
                {
                    "tuar/01_tcp_ar": ["FP1", "FP2"],
                    "tuar/02_tcp_le": ["CZ"],
                },
                "minirocket",
                "tuar",
            )

if __name__ == "__main__":
    unittest.main()

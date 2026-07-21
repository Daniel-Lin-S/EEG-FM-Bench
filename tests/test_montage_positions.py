import unittest

import mne
import numpy as np

from data.processor.montage import resolve_electrode_positions


class MontagePositionTests(unittest.TestCase):
    @staticmethod
    def _raw(ch_names):
        info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
        return mne.io.RawArray(np.zeros((len(ch_names), 10)), info, verbose=False)

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


if __name__ == "__main__":
    unittest.main()

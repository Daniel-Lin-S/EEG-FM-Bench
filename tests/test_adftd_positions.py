import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np

from data.dataset.adftd import AdftdBuilder


class AdftdPositionTests(unittest.TestCase):
    def test_bids_reader_preserves_native_locations(self):
        info = mne.create_info(["Fp1", "Fp2"], sfreq=100.0, ch_types="eeg")
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        for channel, position in zip(info["chs"], expected):
            channel["loc"][:3] = position
        raw = mne.io.RawArray(np.zeros((2, 20)), info, verbose=False)

        with patch("data.dataset.adftd.mne_bids.get_bids_path_from_fname", return_value=SimpleNamespace()), patch(
            "data.dataset.adftd.mne_bids.read_raw_bids", return_value=raw
        ):
            loaded = object.__new__(AdftdBuilder)._read_raw_data("subject.set")

        np.testing.assert_allclose(
            [channel["loc"][:3] for channel in loaded.info["chs"]], expected
        )


if __name__ == "__main__":
    unittest.main()

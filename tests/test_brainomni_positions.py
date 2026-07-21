import sys
import types
import unittest

import torch


if "optimi" not in sys.modules:
    optimi = types.ModuleType("optimi")
    optimi.StableAdamW = object
    sys.modules["optimi"] = optimi

from baseline.brainomni.brainomni_adapter import BrainOmniDatasetAdapter


class BrainOmniPositionTests(unittest.TestCase):
    def _adapter(self):
        adapter = object.__new__(BrainOmniDatasetAdapter)
        adapter.normalize_position = False
        adapter.position_normalize_eps = 1e-8
        adapter.montage_mappings = {"demo/10_20": {"sel": [True, False, True]}}
        return adapter

    def test_persisted_xyz_is_selected_and_extended_to_six_dimensions(self):
        result = {"montage": "demo/10_20", "chs": [1, 2]}
        positions = self._adapter()._get_persisted_positions(
            {"pos": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, result
        )
        self.assertEqual(tuple(positions.shape), (2, 6))
        torch.testing.assert_close(positions[:, :3], torch.tensor([[1, 2, 3], [7, 8, 9]], dtype=torch.float32))
        torch.testing.assert_close(positions[:, 3:], torch.zeros((2, 3)))

    def test_missing_positions_require_rebuild(self):
        with self.assertRaisesRegex(ValueError, "requires persisted XYZ"):
            self._adapter()._get_persisted_positions(
                {"pos": []}, {"montage": "demo/10_20", "chs": []}
            )


if __name__ == "__main__":
    unittest.main()

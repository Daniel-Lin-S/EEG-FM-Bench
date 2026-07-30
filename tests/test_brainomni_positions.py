import sys
import types
import unittest

import torch


if "optimi" not in sys.modules:
    optimi = types.ModuleType("optimi")
    optimi.StableAdamW = object
    sys.modules["optimi"] = optimi

from baseline.brainomni.brainomni_adapter import (
    BrainOmniDatasetAdapter,
    BrainOmniFilteredDataset,
)


class BrainOmniPositionTests(unittest.TestCase):
    def _adapter(self):
        adapter = object.__new__(BrainOmniDatasetAdapter)
        adapter.normalize_input = True
        adapter.normalize_position = False
        adapter.scale = 1.0
        adapter.signal_normalize_eps = 1e-5
        adapter.position_normalize_eps = 1e-8
        adapter.montage_mappings = {
            "demo/10_20": {
                "idx": [0, 2],
                "sel": [True, False, True],
            }
        }
        return adapter

    def test_persisted_xyz_is_selected_and_extended_to_six_dimensions(self):
        result = {"montage": "demo/10_20", "chs": [1, 2]}
        positions = self._adapter()._get_persisted_positions(
            {"pos": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, result
        )
        self.assertEqual(tuple(positions.shape), (2, 6))
        expected_xyz = torch.tensor(
            [[1, 2, 3], [7, 8, 9]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(positions[:, :3], expected_xyz)
        torch.testing.assert_close(positions[:, 3:], torch.zeros((2, 3)))

    def test_missing_positions_require_rebuild(self):
        with self.assertRaisesRegex(ValueError, "requires persisted XYZ"):
            self._adapter()._get_persisted_positions(
                {"pos": []}, {"montage": "demo/10_20", "chs": []}
            )


    def test_filtered_dataset_preserves_column_names(self):
        class DatasetWithColumns:
            column_names = ["montage", "data"]

            def __getitem__(self, index):
                return {"montage": "demo/10_20", "data": index}

        filtered_dataset = BrainOmniFilteredDataset(
            dataset=DatasetWithColumns(),
            sample_indices=[1],
        )
        self.assertEqual(filtered_dataset.column_names, ["montage", "data"])
        self.assertEqual(filtered_dataset["montage"], ["demo/10_20"])

    def test_zero_cross_channel_variation_is_identified(self):
        cross_channel_std = self._adapter().get_cross_channel_std(
            {
                "data": torch.tensor(
                    [[2.0, 4.0], [1.0, 3.0], [2.0, 4.0]]
                ),
                "montage": "demo/10_20",
            }
        )
        self.assertEqual(cross_channel_std.item(), 0.0)

    def test_nonzero_cross_channel_variation_is_retained(self):
        cross_channel_std = self._adapter().get_cross_channel_std(
            {
                "data": torch.tensor(
                    [[2.0, 4.0], [1.0, 3.0], [5.0, 6.0]]
                ),
                "montage": "demo/10_20",
            }
        )
        self.assertGreater(cross_channel_std.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

import sys
import types
import unittest
from unittest import mock

import torch


if "optimi" not in sys.modules:
    optimi = types.ModuleType("optimi")
    optimi.StableAdamW = object
    sys.modules["optimi"] = optimi

import baseline.brainomni.brainomni_adapter as brainomni_adapter_module
from baseline.brainomni.brainomni_adapter import (
    BrainOmniDataLoaderFactory,
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
        with self.assertRaisesRegex(ValueError, "persisted XYZ"):
            self._adapter()._get_persisted_positions(
                {"pos": []}, {"montage": "demo/10_20", "chs": []}
            )

    def test_unselected_zero_position_is_ignored(self):
        result = {"montage": "demo/10_20", "chs": [1, 2]}
        positions = self._adapter()._get_persisted_positions(
            {"pos": [[1, 2, 3], [0, 0, 0], [7, 8, 9]]},
            result,
        )
        self.assertEqual(tuple(positions.shape), (2, 6))

    def test_near_zero_signal_std_is_rejected_during_preflight(self):
        sample = {
            "montage": "demo/10_20",
            "data": [[0.0, 0.0], [2.0, 2.0], [1.0e-6, 1.0e-6]],
            "pos": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        }
        rejection = self._adapter().get_sample_rejection(sample)
        self.assertEqual(
            rejection["code"],
            "signal_std_below_threshold",
        )

    def test_nonfinite_signal_is_rejected_during_preflight(self):
        sample = {
            "montage": "demo/10_20",
            "data": [[0.0, float("nan")], [2.0, 2.0], [1.0, 2.0]],
            "pos": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        }

        rejection = self._adapter().get_sample_rejection(sample)

        self.assertEqual(rejection["code"], "signal_nonfinite")

    def test_malformed_positions_are_rejected_during_preflight(self):
        sample = {
            "montage": "demo/10_20",
            "data": [[0.0, 1.0], [2.0, 2.0], [1.0, 3.0]],
            "pos": [[1, 2], [4, 5], [7, 8]],
        }

        rejection = self._adapter().get_sample_rejection(sample)

        self.assertEqual(rejection["code"], "position_shape_invalid")

    def test_degenerate_selected_positions_are_rejected(self):
        adapter = self._adapter()
        adapter.normalize_position = True
        sample = {
            "montage": "demo/10_20",
            "data": [[0.0, 1.0], [2.0, 2.0], [1.0, 3.0]],
            "pos": [[1, 2, 3], [4, 5, 6], [1, 2, 3]],
        }

        rejection = adapter.get_sample_rejection(sample)

        self.assertEqual(
            rejection["code"],
            "position_scale_below_threshold",
        )

    def test_unexpected_preflight_exception_propagates(self):
        adapter = self._adapter()
        with mock.patch.object(
            adapter,
            "_select_signal_data",
            side_effect=RuntimeError("implementation failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "implementation failure",
            ):
                adapter.get_sample_rejection({})

    def test_filter_diagnostics_and_warning_are_stable_by_split(self):
        adapter = self._adapter()
        factory = BrainOmniDataLoaderFactory(num_workers=0)
        valid_sample = {
            "montage": "demo/10_20",
            "data": [[0.0, 1.0], [2.0, 2.0], [1.0, 3.0]],
            "pos": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        }
        invalid_sample = {
            **valid_sample,
            "data": [[0.0, 0.0], [2.0, 2.0], [0.0, 0.0]],
        }
        warning_key = ("demo", "train")
        brainomni_adapter_module._WARNED_SAMPLE_FILTER_SPLITS.discard(
            warning_key
        )

        with self.assertLogs("baseline", level="WARNING") as captured:
            first = factory._filter_invalid_samples(
                [valid_sample, invalid_sample],
                adapter,
                "train",
                ["demo"],
            )
            second = factory._filter_invalid_samples(
                [valid_sample, invalid_sample],
                adapter,
                "train",
                ["demo"],
            )

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(len(captured.output), 1)
        self.assertEqual(
            factory.get_data_diagnostics("demo"),
            {
                "sample_filtering": {
                    "skipped_samples": 1,
                    "by_split": [
                        {
                            "split": "train",
                            "total_samples": 2,
                            "retained_samples": 1,
                            "skipped_samples": 1,
                            "reasons": [
                                {
                                    "code": (
                                        "signal_std_below_threshold"
                                    ),
                                    "message": (
                                        "centered signal standard "
                                        "deviation is below 1e-05"
                                    ),
                                    "count": 1,
                                }
                            ],
                        }
                    ],
                }
            },
        )

    def test_filter_cache_reuses_fingerprinted_preflight_indices(self):
        class FingerprintedSamples(list):
            _fingerprint = "stable-fingerprint"

        adapter = self._adapter()
        factory = BrainOmniDataLoaderFactory(num_workers=0)
        samples = FingerprintedSamples([
            {
                "montage": "demo/10_20",
                "data": [[0.0, 1.0], [2.0, 2.0], [1.0, 3.0]],
                "pos": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            }
        ])
        brainomni_adapter_module._ELIGIBILITY_INDEX_CACHE.clear()
        with mock.patch.object(
            adapter,
            "get_sample_rejection",
            wraps=adapter.get_sample_rejection,
        ) as rejection:
            first = factory._filter_invalid_samples(
                samples,
                adapter,
                "validation",
                ["demo"],
                ["finetune"],
            )
            second = factory._filter_invalid_samples(
                samples,
                adapter,
                "validation",
                ["demo"],
                ["finetune"],
            )
            factory._filter_invalid_samples(
                samples,
                adapter,
                "validation",
                ["demo"],
                ["pretrain"],
            )

        self.assertEqual(len(first), len(second))
        self.assertEqual(rejection.call_count, 2)
        brainomni_adapter_module._ELIGIBILITY_INDEX_CACHE.clear()

    def test_filter_rejects_a_dataset_with_no_eligible_samples(self):
        adapter = self._adapter()
        factory = BrainOmniDataLoaderFactory(num_workers=0)
        invalid_sample = {
            "montage": "demo/10_20",
            "data": [[0.0, 0.0], [2.0, 2.0], [0.0, 0.0]],
            "pos": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        }
        warning_key = ("demo", "validation")
        brainomni_adapter_module._WARNED_SAMPLE_FILTER_SPLITS.discard(
            warning_key
        )

        with mock.patch.object(
            brainomni_adapter_module,
            "get_is_master",
            return_value=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "no eligible samples remain",
            ):
                factory._filter_invalid_samples(
                    [invalid_sample],
                    adapter,
                    "validation",
                    ["demo"],
                )

        diagnostics = factory.get_data_diagnostics("demo")
        split_details = diagnostics["sample_filtering"]["by_split"][0]
        self.assertEqual(split_details["total_samples"], 1)
        self.assertEqual(split_details["retained_samples"], 0)
        self.assertEqual(split_details["skipped_samples"], 1)

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

    def test_tuar_montages_keep_signal_and_position_alignment(self):
        """Each TUAR montage key selects matching persisted XYZ positions."""
        adapter = self._adapter()
        montage_names = (
            "tuar/01_tcp_ar",
            "tuar/02_tcp_le",
            "tuar/03_tcp_ar_a",
        )
        adapter.montage_mappings = {
            montage_name: {"idx": [0, 2], "sel": [True, False, True]}
            for montage_name in montage_names
        }

        for montage_index, montage_name in enumerate(montage_names):
            with self.subTest(montage=montage_name):
                sample = {
                    "montage": montage_name,
                    "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]],
                    "pos": [
                        [montage_index + 1.0, 1.0, 1.0],
                        [9.0, 9.0, 9.0],
                        [montage_index + 2.0, 2.0, 2.0],
                    ],
                }
                rejection = adapter.get_sample_rejection(sample)
                self.assertIsNone(rejection)
                positions = adapter._get_persisted_positions(
                    sample,
                    {"montage": montage_name, "chs": [0, 1]},
                )
                self.assertEqual(tuple(positions.shape), (2, 6))
                self.assertEqual(positions[0, 0].item(), montage_index + 1.0)
                self.assertEqual(positions[1, 0].item(), montage_index + 2.0)

if __name__ == "__main__":
    unittest.main()

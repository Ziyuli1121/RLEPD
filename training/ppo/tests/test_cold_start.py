"""
Unit tests for Stage 2 cold-start utilities.

These tests are not executed automatically in this environment. Run them
manually inside the target training setup once PyTorch and other dependencies
are available, e.g.:

    python -m training.ppo.tests.test_cold_start
"""

import unittest
from pathlib import Path

import numpy as np

from training.ppo import cold_start
from training.ppo.cold_start import (
    EPDTable,
    build_dirichlet_params,
    load_predictor_table,
    table_from_dirichlet,
)

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - torch not bundled in this env
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[3]
SNAPSHOT_PATH = ROOT_DIR / "exps" / "00036-ms_coco-10-36-epd-dpm-1-discrete" / "network-snapshot-000005.pkl"


@unittest.skipUnless(SNAPSHOT_PATH.exists(), "Baseline predictor snapshot not found.")
@unittest.skipIf(torch is None, "PyTorch is required to load predictor snapshots.")
class PredictorTableRoundTripTest(unittest.TestCase):
    def test_round_trip_error_below_threshold(self) -> None:
        table = load_predictor_table(SNAPSHOT_PATH)
        init = build_dirichlet_params(table, concentration=100.0)
        recon_positions, recon_weights = table_from_dirichlet(init)

        self.assertTrue(np.all(np.diff(table.positions, axis=-1) > 0), "Positions must be strictly increasing.")
        self.assertIsInstance(table.metadata.get("sanitized", False), (bool, np.bool_))

        valid_pos = ~init.invalid_pos_rows
        valid_weight = ~init.invalid_weight_rows

        if valid_pos.any():
            diff_pos = np.abs(table.positions[valid_pos] - recon_positions[valid_pos])
            self.assertLess(diff_pos.max(), 1e-4, msg=f"Max position diff {diff_pos.max():.6f}")

        if valid_weight.any():
            diff_weight = np.abs(table.weights[valid_weight] - recon_weights[valid_weight])
            self.assertLess(diff_weight.max(), 1e-4, msg=f"Max weight diff {diff_weight.max():.6f}")


class FallbackBehaviourTest(unittest.TestCase):
    def test_uniform_fallback_triggered_on_degenerate_rows(self) -> None:
        num_steps = 4
        num_points = 2
        # First two segments extremely small to trigger fallback, remaining row is healthy.
        positions = np.array(
            [
                [1e-8, 0.999999],
                [0.25, 0.75],
                [0.3, 0.9],
            ],
            dtype=np.float64,
        )
        weights = np.array(
            [
                [1e-9, 1 - 1e-9],
                [0.6, 0.4],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        )
        table = EPDTable(positions=positions, weights=weights, num_steps=num_steps, num_points=num_points)
        init = build_dirichlet_params(table, concentration=10.0, min_segment=1e-5, min_weight=1e-5)

        self.assertTrue(init.invalid_pos_rows[0], "First row should trigger position fallback.")
        self.assertTrue(init.invalid_weight_rows[0], "First row should trigger weight fallback.")

        uniform_segments = np.full(num_points + 1, 1.0 / (num_points + 1))
        uniform_weights = np.full(num_points, 1.0 / num_points)

        np.testing.assert_allclose(init.mean_pos_segments[0], uniform_segments, atol=1e-12)
        np.testing.assert_allclose(init.mean_weights[0], uniform_weights, atol=1e-12)


class SanitizationTest(unittest.TestCase):
    def test_out_of_order_rows_are_sorted_and_flagged(self) -> None:
        positions = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        weights = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
        scale_dir = np.array([[1.2, 0.9], [1.0, 1.0]], dtype=np.float64)

        (
            sanitized_positions,
            sanitized_weights,
            sanitized_scale_dir,
            sanitized_scale_time,
            reordered_rows,
            adjusted_rows,
        ) = cold_start._sanitize_table_arrays(  # type: ignore[attr-defined]
            positions=positions,
            weights=weights,
            scale_dir=scale_dir,
            scale_time=None,
        )

        # First row should be sorted in ascending order.
        np.testing.assert_array_equal(sanitized_positions[0], np.sort(positions[0]))
        # Weights and scale_dir should follow the same permutation.
        np.testing.assert_array_equal(sanitized_weights[0], weights[0][[1, 0]])
        np.testing.assert_array_equal(sanitized_scale_dir[0], scale_dir[0][[1, 0]])

        self.assertTrue(reordered_rows[0])
        self.assertFalse(reordered_rows[1])
        self.assertFalse(adjusted_rows.any(), "No adjustments expected beyond sorting.")
        self.assertIsNone(sanitized_scale_time)

    def test_duplicate_positions_are_slightly_separated(self) -> None:
        positions = np.array([[0.5, 0.5], [0.2, 0.7]], dtype=np.float64)
        weights = np.array([[0.4, 0.6], [0.3, 0.7]], dtype=np.float64)

        (
            sanitized_positions,
            sanitized_weights,
            _,
            _,
            _,
            adjusted_rows,
        ) = cold_start._sanitize_table_arrays(  # type: ignore[attr-defined]
            positions=positions,
            weights=weights,
            scale_dir=None,
            scale_time=None,
        )

        # First row should now be strictly increasing.
        self.assertGreater(sanitized_positions[0, 1] - sanitized_positions[0, 0], 0.0)
        # Adjusted rows flag must be true for the duplicate row.
        self.assertTrue(adjusted_rows[0])
        self.assertFalse(adjusted_rows[1])
        # Weights should remain untouched aside from potential permutation.
        np.testing.assert_array_equal(sanitized_weights[0], weights[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

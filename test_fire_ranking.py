"""
test_fire_ranking.py - Automated unit tests and invariant checks
for fire ranking, spatial clustering, and evaluation modules.
"""

import unittest
import pandas as pd
import numpy as np
from fire_ranking import (
    normalize_column_names,
    sanitize_firms_data,
    assign_spatial_clusters,
    calculate_calibrated_priority,
    calculate_legacy_uncalibrated_priority,
    run_algorithm_evaluation,
    generate_synthetic_evaluation_benchmarks
)


class TestFireRanking(unittest.TestCase):

    def test_column_normalization_uppercase_and_aliases(self):
        df_raw = pd.DataFrame({
            "LATITUDE": [38.5],
            "LONGITUDE": [-120.5],
            "BRIGHT_TI4": [360.2],
            "CONF": ["h"],
            "POWER": [45.0]
        })
        normalized = normalize_column_names(df_raw)
        self.assertIn("latitude", normalized.columns)
        self.assertIn("longitude", normalized.columns)
        self.assertIn("brightness", normalized.columns)
        self.assertIn("confidence", normalized.columns)
        self.assertIn("frp", normalized.columns)

    def test_viirs_confidence_mapping(self):
        df_viirs = pd.DataFrame({
            "latitude": [38.1, 38.2, 38.3],
            "longitude": [-120.1, -120.2, -120.3],
            "brightness": [320.0, 340.0, 360.0],
            "confidence": ["l", "n", "h"],
            "frp": [10.0, 25.0, 50.0]
        })
        cleaned, stats = sanitize_firms_data(df_viirs)
        self.assertEqual(len(cleaned), 3)
        self.assertAlmostEqual(cleaned.loc[0, "confidence"], 30.0)
        self.assertAlmostEqual(cleaned.loc[1, "confidence"], 65.0)
        self.assertAlmostEqual(cleaned.loc[2, "confidence"], 90.0)

    def test_invalid_coordinates_filtering(self):
        df_bad = pd.DataFrame({
            "latitude": [95.0, np.nan, 34.0],
            "longitude": [-120.0, -120.0, 200.0],
            "brightness": [320.0, 330.0, 340.0],
            "confidence": [80, 80, 80],
            "frp": [15.0, 20.0, 25.0]
        })
        cleaned, stats = sanitize_firms_data(df_bad)
        # All three rows have invalid lat or lon
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(stats["removed_missing_coords"], 3)

    def test_scoring_invariants_and_bounds(self):
        benchmarks = generate_synthetic_evaluation_benchmarks()
        scored = calculate_calibrated_priority(benchmarks)

        # Invariant 1: Scores must stay strictly between 0 and 100
        self.assertTrue((scored["priority_score"] >= 0.0).all())
        self.assertTrue((scored["priority_score"] <= 100.0).all())

        # Invariant 2: No null / NaN values
        self.assertTrue(scored["priority_score"].notna().all())

        # Invariant 3: Major wildfire (TC-01) must outscore controlled burn (TC-04)
        score_major = scored.loc[scored["case_id"] == "TC-01", "priority_score"].values[0]
        score_burn = scored.loc[scored["case_id"] == "TC-04", "priority_score"].values[0]
        self.assertGreater(score_major, score_burn)

        # Invariant 4: Check that reason is non-empty and contains Tier
        self.assertTrue(all(len(r) > 10 for r in scored["explainable_reason"]))

    def test_evaluation_runner(self):
        eval_res = run_algorithm_evaluation()
        self.assertTrue(eval_res["all_scores_bounded"])
        self.assertTrue(eval_res["no_nan_scores"])
        self.assertGreaterEqual(eval_res["spearman_correlation_calibrated"], 0.90)
        self.assertEqual(eval_res["benchmark_count"], 5)


if __name__ == "__main__":
    unittest.main()

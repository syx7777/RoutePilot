from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from mine_badcases import mine_badcases  # noqa: E402
from calculate_metrics import calculate_metric_values, load_prediction_actual  # noqa: E402


class MetricDefinitionTest(unittest.TestCase):
    def test_metric_denominator_uses_absolute_actuals(self) -> None:
        rows = [{"actual": 10.0, "prediction": 12.0}, {"actual": -10.0, "prediction": -12.0}]

        metrics = calculate_metric_values(rows)

        # sum(|actual|) = 20，绝对误差合计 = 4 -> wape 0.2；误差方向相互抵消 -> bias 0.0
        self.assertAlmostEqual(metrics["wape"], 0.2)
        self.assertAlmostEqual(metrics["bias"], 0.0)

    def test_positional_alignment_rejects_mismatched_row_counts(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            prediction = Path(tmp) / "prediction.csv"
            actual = Path(tmp) / "actual.csv"
            prediction.write_text("prediction\n1\n2\n", encoding="utf-8")
            actual.write_text("actual\n1\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_prediction_actual(prediction, actual)


class MineBadcasesTest(unittest.TestCase):
    def test_extreme_badcases_use_ape_and_absolute_error_thresholds(self) -> None:
        rows = [
            {"sku": "A", "date": "2026-01-01", "prediction": 70, "actual": 100},
            {"sku": "B", "date": "2026-01-01", "prediction": 125, "actual": 100},
            {"sku": "C", "date": "2026-01-01", "prediction": 108, "actual": 100},
            {"sku": "D", "date": "2026-01-01", "prediction": 1, "actual": 5},
        ]

        result = mine_badcases(rows, top_n=10)

        extreme = {row["sku"]: row for row in result if row["badcase_type"].startswith("extreme_")}
        self.assertEqual({"A", "B"}, set(extreme))
        self.assertEqual("extreme_underestimate", extreme["A"]["badcase_type"])
        self.assertEqual("extreme_overestimate", extreme["B"]["badcase_type"])

    def test_consecutive_directional_bias_is_grouped_by_key_and_date(self) -> None:
        rows = [
            {"sku": "A", "store": "S1", "date": "2026-01-01", "prediction": 80, "actual": 100},
            {"sku": "A", "store": "S1", "date": "2026-01-02", "prediction": 81, "actual": 100},
            {"sku": "A", "store": "S1", "date": "2026-01-03", "prediction": 82, "actual": 100},
            {"sku": "A", "store": "S1", "date": "2026-01-04", "prediction": 120, "actual": 100},
            {"sku": "B", "store": "S1", "date": "2026-01-01", "prediction": 121, "actual": 100},
            {"sku": "B", "store": "S1", "date": "2026-01-02", "prediction": 122, "actual": 100},
            {"sku": "B", "store": "S1", "date": "2026-01-03", "prediction": 123, "actual": 100},
        ]

        result = mine_badcases(rows, top_n=10)

        streaks = {
            (row["badcase_type"], row["sku"]): row
            for row in result
            if row["badcase_type"].startswith("consecutive_")
        }
        self.assertIn(("consecutive_underestimate", "A"), streaks)
        self.assertIn(("consecutive_overestimate", "B"), streaks)
        self.assertEqual(3, streaks[("consecutive_underestimate", "A")]["streak_length"])
        self.assertEqual("2026-01-01", streaks[("consecutive_underestimate", "A")]["streak_start_date"])
        self.assertEqual("2026-01-03", streaks[("consecutive_underestimate", "A")]["streak_end_date"])


if __name__ == "__main__":
    unittest.main()

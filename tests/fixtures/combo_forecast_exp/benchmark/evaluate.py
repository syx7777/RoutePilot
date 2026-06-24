from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    prediction_rows = _read_csv(run_dir / "prediction.csv")
    actual_rows = _read_csv(Path.cwd() / "benchmark" / "data" / "actual.csv")
    actual_by_key = {(row["date"], row["combo_id"]): float(row["actual"]) for row in actual_rows}
    errors = []
    actual_sum = 0.0
    for row in prediction_rows:
        actual = actual_by_key[(row["date"], row["combo_id"])]
        prediction = float(row["prediction"])
        actual_sum += abs(actual)
        errors.append(prediction - actual)
    metrics = {
        "wape": sum(abs(error) for error in errors) / actual_sum,
        "bias": sum(errors) / actual_sum,
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": (sum(error * error for error in errors) / len(errors)) ** 0.5,
        "sample_count": len(errors),
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

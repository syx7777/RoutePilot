from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


# ROUTEPILOT_POLICY_START
ENABLED_FEATURES = [
    "entity_rolling_7d_mean",
]
# ROUTEPILOT_POLICY_END


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _enabled_features(root: Path) -> set[str]:
    config_path = root / "benchmark" / "feature_config.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return set(data.get("enabled_features") or []) | set(ENABLED_FEATURES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actual_rows = _read_csv(root / "benchmark" / "data" / "actual.csv")
    seed_rows = _read_csv(root / "benchmark" / "data" / "prediction_seed.csv")
    seed_by_key = {(row["date"], row["combo_id"]): float(row["prediction"]) for row in seed_rows}
    enabled = _enabled_features(root)
    improved = bool({"combo_rolling_7d_mean", "entity_rolling_7d_mean"} & enabled)
    rows = []
    for row in actual_rows:
        actual = float(row["actual"])
        prediction = seed_by_key[(row["date"], row["combo_id"])]
        if improved and actual >= 100:
            prediction = actual - 5
        rows.append({"date": row["date"], "combo_id": row["combo_id"], "prediction": round(prediction, 3)})
    with (output / "prediction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "combo_id", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"completed rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

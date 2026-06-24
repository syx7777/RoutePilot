from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_metric_csv(path: str | Path) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metric = str(row.get("metric"))
            value = row.get("value")
            if metric == "rows":
                metrics["sample_count"] = int(float(value or 0))
            else:
                metrics[metric] = float(value or 0)
    return metrics


def metrics_csv_to_json(metrics_csv: str | Path, output_path: str | Path) -> dict[str, float | int]:
    metrics = read_metric_csv(metrics_csv)
    Path(output_path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def summarize_badcases(badcases_csv: str | Path, output_path: str | Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    with Path(badcases_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            badcase_type = str(row.get("badcase_type", "unknown"))
            counts[badcase_type] = counts.get(badcase_type, 0) + 1
            if len(samples) < 5:
                samples.append(row)
    summary = {"counts": counts, "samples": samples}
    Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def read_csv_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def write_json(path: str | Path, value: Any) -> Any:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value

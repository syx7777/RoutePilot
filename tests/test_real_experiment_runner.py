from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import routepilot.core.real_experiment_runner as real_runner
from routepilot.core.real_experiment_runner import (
    _audit_feature_application,
    _train_command_validation_error,
    generate_trial_train_wrapper,
    run_real_experiment,
)
from routepilot.runtime.doubao_client import LLMCallResult


class FakeAgent2Client:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.last_call = None
        self.calls = []

    def available(self) -> bool:
        return True

    def complete_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent: str,
        step: str,
        timeout=None,
        stream: bool = False,
    ) -> LLMCallResult:
        self.calls.append({"agent": agent, "step": step, "stream": stream, "timeout": timeout, "user_prompt": user_prompt})
        content = self.contents.pop(0) if self.contents else ""
        self.last_call = LLMCallResult(
            content=content,
            model="fake",
            agent=agent,
            step=step,
            duration_seconds=0.01,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            success=bool(content),
            error=None if content else "empty response",
            timeout_seconds=timeout,
            streaming_used=stream,
        )
        return self.last_call


class UnavailableAgent2Client(FakeAgent2Client):
    def __init__(self) -> None:
        super().__init__([])

    def available(self) -> bool:
        return False


def _generic_experiment(root: Path) -> Path:
    experiment = root / "generic_forecast_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "features.py").write_text(
        "def recent_trend_ratio(enabled: bool) -> float:\n"
        "    return 0.95 if enabled else 0.5\n",
        encoding="utf-8",
    )
    (src / "unused.py").write_text("UNUSED = True\n", encoding="utf-8")
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable-recent-trend-ratio", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio = recent_trend_ratio(args.enable_recent_trend_ratio)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value", "segment"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": 95 if ratio > 0.9 else 50,
            "segment": "high_target",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _resource_path_experiment(root: Path) -> Path:
    experiment = root / "resource_path_exp"
    src = experiment / "src"
    data = experiment / "data"
    src.mkdir(parents=True)
    data.mkdir(parents=True)
    (data / "input.csv").write_text("actual_value,forecast_value\n100,95\n", encoding="utf-8")
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--data_path", "--data-path", dest="data_path", required=True)
    args = parser.parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with data_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": row["actual_value"],
            "forecast_value": row["forecast_value"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _rolling_experiment(root: Path) -> Path:
    experiment = root / "rolling_forecast_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--rolling_windows", "--rolling-windows", type=int, nargs="*", default=[3, 7])
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": 95 if 1 in args.rolling_windows else 50,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _rolling_sum_dependency_experiment(root: Path) -> Path:
    experiment = root / "rolling_sum_dependency_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "util.py").write_text(
        """
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.weekday + 1
    return df


def generate_lag_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    lag_periods: List[int],
    date_col: str,
    history_gap_days: int = 1,
) -> pd.DataFrame:
    df = df.copy().sort_values(group_cols + [date_col]).reset_index(drop=True)
    safe_gap = max(int(history_gap_days), 1)
    for lag in lag_periods:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag + safe_gap - 1)
    return df


def generate_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    window_sizes: List[int],
    date_col: str,
    functions: Tuple[str, ...] = ("mean", "std", "min", "max", "median"),
    history_gap_days: int = 1,
) -> pd.DataFrame:
    df = df.copy().sort_values(group_cols + [date_col]).reset_index(drop=True)
    grouped = df.groupby(group_cols)[target_col]
    safe_gap = max(int(history_gap_days), 1)
    for window in window_sizes:
        for func in functions:
            col = f"{target_col}_rolling_{window}_{func}"
            df[col] = grouped.transform(lambda s, w=window, f=func, g=safe_gap: s.shift(g).rolling(window=w, min_periods=1).agg(f))
    return df


def build_features(
    df: pd.DataFrame,
    label_col: str,
    date_col: str,
    id_cols: List[str],
    lag_days: List[int],
    rolling_windows: List[int],
    category_maps: dict[str, dict[str, int]] | None = None,
    return_category_maps: bool = False,
    history_gap_days: int = 1,
) -> Tuple[pd.DataFrame, List[str], List[str]] | Tuple[pd.DataFrame, List[str], List[str], dict[str, dict[str, int]]]:
    df = add_time_features(df, date_col)
    group_cols = [c for c in id_cols if c in df.columns]

    df["use_start_date"] = pd.to_datetime(df["use_start_date"], errors="coerce")
    df["use_end_date"] = pd.to_datetime(df["use_end_date"], errors="coerce")
    df["sale_day_num"] = (df[date_col] - df["use_start_date"]).dt.days.clip(lower=0)
    df["activity_duration"] = (df["use_end_date"] - df["use_start_date"]).dt.days.clip(lower=0)
    df["days_to_end"] = (df["use_end_date"] - df[date_col]).dt.days.clip(lower=0)

    if not group_cols:
        group_cols = [date_col]

    if group_cols == [date_col]:
        df = df.sort_values(date_col).reset_index(drop=True)
        safe_gap = max(int(history_gap_days), 1)
        for lag in lag_days:
            df[f"{label_col}_lag_{lag}"] = df[label_col].shift(lag + safe_gap - 1)
        for w in rolling_windows:
            s = df[label_col].shift(safe_gap)
            df[f"{label_col}_rolling_{w}_mean"] = s.rolling(w, min_periods=1).mean()
            df[f"{label_col}_rolling_{w}_std"] = s.rolling(w, min_periods=1).std()
            df[f"{label_col}_rolling_{w}_min"] = s.rolling(w, min_periods=1).min()
            df[f"{label_col}_rolling_{w}_max"] = s.rolling(w, min_periods=1).max()
            df[f"{label_col}_rolling_{w}_median"] = s.rolling(w, min_periods=1).median()
    else:
        df = generate_lag_features(df, group_cols, label_col, lag_days, date_col, history_gap_days=history_gap_days)
        df = generate_rolling_features(df, group_cols, label_col, rolling_windows, date_col, history_gap_days=history_gap_days)

    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols: List[str] = []
    if return_category_maps:
        return df, feature_cols, cat_cols, category_maps or {}
    return df, feature_cols, cat_cols
""",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from util import build_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--rolling_windows", "--rolling-windows", type=int, nargs="*", default=[3, 7])
    args = parser.parse_args()
    df = pd.DataFrame(
        {
            "entity_id": ["E001", "E001", "E001", "E001"],
            "date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"]),
            "actual_value": [10, 20, 30, 100],
            "use_start_date": pd.to_datetime(["2026-05-01"] * 4),
            "use_end_date": pd.to_datetime(["2026-06-01"] * 4),
        }
    )
    feat_df, feature_cols, _ = build_features(
        df,
        label_col="actual_value",
        date_col="date",
        id_cols=["entity_id"],
        lag_days=[1],
        rolling_windows=args.rolling_windows,
    )
    has_sum = any(col.endswith("_rolling_3_sum") for col in feature_cols) and "days_to_end" in feat_df.columns
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-04",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": 95 if has_sum else 50,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _lifecycle_interaction_experiment(root: Path) -> Path:
    experiment = root / "lifecycle_interaction_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "util.py").write_text(
        """
from __future__ import annotations

from typing import List

import pandas as pd


def build_features(
    df: pd.DataFrame,
    label_col: str,
    date_col: str,
    id_cols: List[str],
    lag_days: List[int],
    rolling_windows: List[int],
    category_maps: dict[str, dict[str, int]] | None = None,
    return_category_maps: bool = False,
    history_gap_days: int = 1,
):
    df = df.copy()
    df["package_age_days"] = pd.to_numeric(df["package_age_days"], errors="coerce").fillna(0.0)
    df["days_to_end"] = pd.to_numeric(df["days_to_end"], errors="coerce").fillna(0.0)
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols: List[str] = []
    if return_category_maps:
        return df, feature_cols, cat_cols, category_maps or {}
    return df, feature_cols, cat_cols
""",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from util import build_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    df = pd.DataFrame({
        "entity_id": ["E001"],
        "date": pd.to_datetime(["2026-05-04"]),
        "actual_value": [100],
        "package_age_days": [5],
        "days_to_end": [10],
    })
    feat_df, feature_cols, _ = build_features(df, "actual_value", "date", ["entity_id"], [], [])
    has_lifecycle = "lifecycle_age_x_days_to_end" in feature_cols and "lifecycle_progress_ratio" in feature_cols
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({"date": "2026-05-04", "entity_id": "E001", "actual_value": 100, "forecast_value": 95 if has_lifecycle else 50})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _lifecycle_always_on_experiment(root: Path) -> Path:
    experiment = root / "lifecycle_always_on_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "util.py").write_text(
        """
from __future__ import annotations

from typing import List

import pandas as pd


def build_features(
    df: pd.DataFrame,
    label_col: str,
    date_col: str,
    id_cols: List[str],
    lag_days: List[int],
    rolling_windows: List[int],
):
    df = df.copy()
    age = pd.to_numeric(df["package_age_days"], errors="coerce").fillna(0.0)
    days_to_end = pd.to_numeric(df["days_to_end"], errors="coerce").fillna(0.0)
    df["lifecycle_age_x_days_to_end"] = age * days_to_end
    df["lifecycle_progress_ratio"] = age / (age + days_to_end + 1.0)
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols: List[str] = []
    return df, feature_cols, cat_cols
""",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from util import build_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    df = pd.DataFrame({
        "entity_id": ["E001"],
        "date": pd.to_datetime(["2026-05-04"]),
        "actual_value": [100],
        "package_age_days": [5],
        "days_to_end": [10],
    })
    _feat_df, feature_cols, _ = build_features(df, "actual_value", "date", ["entity_id"], [], [])
    has_lifecycle = "lifecycle_age_x_days_to_end" in feature_cols and "lifecycle_progress_ratio" in feature_cols
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({"date": "2026-05-04", "entity_id": "E001", "actual_value": 100, "forecast_value": 95 if has_lifecycle else 50})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _holiday_position_experiment(root: Path, *, with_span_fields: bool = True) -> Path:
    experiment = root / ("holiday_position_exp" if with_span_fields else "holiday_missing_exp")
    src = experiment / "src"
    src.mkdir(parents=True)
    holiday_columns = (
        """
    df["holiday_span_day_idx"] = df["holiday_span_day_idx"].fillna(-1)
    df["holiday_span_days"] = df["holiday_span_days"].fillna(0)
    df["days_until_holiday"] = df["days_until_holiday"].fillna(99)
"""
        if with_span_fields
        else """
    df["is_holiday"] = df["is_holiday"].fillna(0)
"""
    )
    data_columns = (
        """
        "holiday_span_day_idx": [0],
        "holiday_span_days": [3],
        "days_until_holiday": [1],
"""
        if with_span_fields
        else """
        "is_holiday": [1],
"""
    )
    (src / "util.py").write_text(
        f"""
from __future__ import annotations

from typing import List


def build_features(df, label_col: str, date_col: str, id_cols: List[str]):
    df = df.copy()
{holiday_columns}
    feature_cols = [c for c in df.columns if c not in {{label_col, date_col}}]
    return df, feature_cols, []
""",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        f"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from util import build_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    df = pd.DataFrame({{
        "entity_id": ["E001"],
        "date": pd.to_datetime(["2026-05-04"]),
        "actual_value": [100],
{data_columns}
    }})
    _, feature_cols, _ = build_features(df, "actual_value", "date", ["entity_id"])
    has_holiday_position = "holiday_position_first" in feature_cols and "holiday_position_eve" in feature_cols
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({{"date": "2026-05-04", "entity_id": "E001", "actual_value": 100, "forecast_value": 95 if has_holiday_position else 50}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _train_weight_experiment(root: Path) -> Path:
    experiment = root / "train_weight_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


class TinyModel:
    def __init__(self) -> None:
        self.weight_seen = False

    def fit(self, x, y, sample_weight=None):
        self.weight_seen = sample_weight is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    sample_weight = [1.0, 2.0]
    model = TinyModel()
    model.fit([[1], [2]], [1, 2])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({"date": "2026-05-04", "entity_id": "E001", "actual_value": 100, "forecast_value": 95 if model.weight_seen else 50})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _group_calibration_experiment(root: Path) -> Path:
    experiment = root / "group_calibration_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def apply_group_calibration(value: int, enabled: bool) -> int:
    return 95 if enabled else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable-group-calibration", action="store_true")
    args = parser.parse_args()
    forecast = apply_group_calibration(50, args.enable_group_calibration)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({"date": "2026-05-04", "entity_id": "E001", "actual_value": 100, "forecast_value": forecast})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _zero_demand_guardrail_experiment(root: Path) -> Path:
    experiment = root / "zero_demand_guardrail_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def apply_zero_demand_guardrail(value: int, enabled: bool) -> int:
    return 95 if enabled else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable-zero-demand-guardrail", action="store_true")
    args = parser.parse_args()
    forecast = apply_zero_demand_guardrail(50, args.enable_zero_demand_guardrail)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({"date": "2026-05-04", "entity_id": "E001", "actual_value": 100, "forecast_value": forecast})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _preset_experiment(root: Path) -> Path:
    experiment = root / "preset_forecast_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def setup_logger() -> None:
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--train_eval_end", "--train-eval-end", dest="train_eval_end", default="default")
    args = parser.parse_args()
    return apply_experiment_preset(args)


def apply_experiment_preset(args: argparse.Namespace) -> argparse.Namespace:
    args.train_eval_end = "preset"
    return args


def run_online_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_eval_end"])
        writer.writeheader()
        writer.writerow({"train_eval_end": args.train_eval_end})


if __name__ == "__main__":
    setup_logger()
    run_online_pipeline(parse_args())
""",
        encoding="utf-8",
    )
    return experiment


def _feature_builder_experiment(root: Path) -> Path:
    experiment = root / "feature_builder_forecast_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_features(row: dict) -> dict:
    row["base"] = 1
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {"date": "2026-05-01", "entity_id": "E001", "actual_value": 100}
    row = build_features(row)
    row["forecast_value"] = 95 if any(key.startswith("new_feature_") for key in row) else 50
    output_row = {key: row[key] for key in ["date", "entity_id", "actual_value", "forecast_value"]}
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow(output_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _reachable_dependency_feature_experiment(root: Path) -> Path:
    experiment = root / "reachable_dependency_forecast_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "features.py").write_text(
        """
def build_features(row: dict) -> dict:
    row["base"] = 1
    return row
""",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from features import build_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    row = {"date": "2026-05-01", "entity_id": "E001", "actual_value": 100}
    row = build_features(row)
    row["forecast_value"] = 95 if any(key.startswith("new_feature_") for key in row) else 50
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({key: row[key] for key in ["date", "entity_id", "actual_value", "forecast_value"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _conditional_cli_dependency_experiment(root: Path) -> Path:
    experiment = root / "conditional_cli_dependency_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "util.py").write_text(
        """
def build_features(row: dict, args=None) -> dict:
    row["base"] = 1
    if args is not None and getattr(args, "enable_decline_recency_features", False):
        row["decline_recency_features"] = 1
    return row
""",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from util import build_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable_decline_recency_features", action="store_true")
    args = parser.parse_args()
    row = {"date": "2026-05-01", "entity_id": "E001", "actual_value": 100}
    row = build_features(row)
    row["forecast_value"] = 95 if row.get("decline_recency_features") else 50
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({key: row[key] for key in ["date", "entity_id", "actual_value", "forecast_value"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _unreachable_dependency_feature_experiment(root: Path) -> Path:
    experiment = root / "unreachable_dependency_forecast_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "features.py").write_text(
        """
def build_features(row: dict) -> dict:
    row["base"] = 1
    return row
""",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    row = {"date": "2026-05-01", "entity_id": "E001", "actual_value": 100, "forecast_value": 50}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _choice_arg_experiment(root: Path) -> Path:
    experiment = root / "choice_arg_forecast_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--experiment", choices=["baseline", "exp_02_tweedie"], default="baseline")
    parser.add_argument("--enable-recent-trend-ratio", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value", "segment"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": 95 if args.enable_recent_trend_ratio else 50,
            "segment": args.experiment,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _trend_guardrail_value_experiment(root: Path) -> Path:
    experiment = root / "trend_guardrail_value_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "features.py").write_text(
        "def recent_trend_ratio(enabled: bool) -> float:\n"
        "    return 0.95 if enabled else 0.5\n",
        encoding="utf-8",
    )
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from features import recent_trend_ratio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable-recent-trend-ratio", action="store_true")
    parser.add_argument("--trend_guardrail", choices=["none", "downtrend_clip"], default="none")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio = recent_trend_ratio(args.enable_recent_trend_ratio)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value", "segment"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": 95 if ratio > 0.9 and args.trend_guardrail == "downtrend_clip" else 50,
            "segment": args.trend_guardrail,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    return experiment


def _glm_like_command_contract_experiment(root: Path) -> Path:
    experiment = root / "glm_like_command_contract_exp"
    src = experiment / "src"
    src.mkdir(parents=True)
    (src / "train_forecast.py").write_text(
        """
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--experiment", choices=["baseline", "exp_02_tweedie"], default="baseline")
    parser.add_argument("--enable_group_calibration", action="store_true", default=True)
    parser.add_argument("--enable_peak_history_mean", action="store_true", default=True)
    parser.add_argument("--trend_guardrail", choices=["none", "downtrend_clip"], default="none")
    parser.add_argument("--rolling_windows", type=int, nargs="*", default=[3, 7])
    args = parser.parse_args()
    return apply_experiment_preset(args)


def apply_experiment_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.experiment == "baseline":
        args.enable_group_calibration = False
        args.enable_peak_history_mean = False
        args.trend_guardrail = "none"
        return args
    args.enable_group_calibration = True
    args.enable_peak_history_mean = True
    args.trend_guardrail = "downtrend_clip"
    return args


def apply_group_calibration(value: int, args: argparse.Namespace) -> int:
    return value + 15 if args.enable_group_calibration else value


def add_peak_history_mean(value: int, args: argparse.Namespace) -> int:
    return value + 15 if args.enable_peak_history_mean else value


def apply_recent_trend_guardrail(value: int, args: argparse.Namespace) -> int:
    return value + 15 if args.trend_guardrail == "downtrend_clip" else value


def run_online_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast = apply_group_calibration(50, args)
    forecast = add_peak_history_mean(forecast, args)
    forecast = apply_recent_trend_guardrail(forecast, args)
    segment = (
        f"group={args.enable_group_calibration};"
        f"peak={args.enable_peak_history_mean};"
        f"trend={args.trend_guardrail};"
        f"windows={','.join(str(item) for item in args.rolling_windows)}"
    )
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value", "segment"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": forecast,
            "segment": segment,
        })


if __name__ == "__main__":
    run_online_pipeline(parse_args())
""",
        encoding="utf-8",
    )
    return experiment


def _plan(trial_dir: Path) -> dict:
    return {
        "trial_id": trial_dir.name,
        "scenario": "generic_forecast",
        "model_family": "unknown",
        "objective": "unknown",
        "target_problem": "segment_underestimate",
        "hypothesis": {"description": "recent trend can reduce underestimation", "evidence": ["badcase evidence"]},
        "editable_files": [
            (trial_dir / "code" / "train.py").as_posix(),
            (trial_dir / "code" / "features.py").as_posix(),
        ],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "recent_trend_ratio",
                "feature_type": "trend_ratio",
                "cli_args": ["--enable-recent-trend-ratio"],
                "code_locations": ["src/train_forecast.py", "src/features.py"],
            }
        ],
        "expected_effect": "reduce underestimation",
        "risk": "may overfit recent history",
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
        "output_contract": "forecast_outputs_to_standard_metrics",
        "candidate_experiments": [],
    }


def _rolling_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [(trial_dir / "code" / "train.py").as_posix()],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "rolling_3d_7d_avg",
                "feature_type": "rolling_stat",
                "code_locations": ["src/train_forecast.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _rolling_sum_dependency_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [
            (trial_dir / "code" / "train.py").as_posix(),
            (trial_dir / "code" / "util.py").as_posix(),
        ],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "Add short-to-medium rolling window features for recent sales",
                "feature_type": "rolling_stat",
                "cli_args": [],
                "field_sources": ["date", "actual_value", "entity_id"],
                "construction": "Generate rolling sum and mean of actual_value over windows [3,7,14,30] days using generate_rolling_features.",
                "code_locations": ["src/train_forecast.py", "src/util.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _zero_history_flag_plan(trial_dir: Path) -> dict:
    return {
        **_rolling_sum_dependency_plan(trial_dir),
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "zero_history_flag",
                "feature_type": "rolling_zero_count_flag",
                "cli_args": [],
                "field_sources": ["date", "actual_value", "entity_id"],
                "audit_tokens": ["zero_history_flag"],
                "construction": "By store/package group, count past 14 days where the target is == 0; set zero_history_flag to 1 when the count is >= 10.",
                "code_locations": ["src/train_forecast.py", "src/util.py"],
            }
        ],
    }


def _runtime_contract_feature_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [
            (trial_dir / "code" / "train.py").as_posix(),
            (trial_dir / "code" / "util.py").as_posix(),
        ],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "runtime_contract_feature",
                "feature_type": "numeric",
                "cli_args": [],
                "field_sources": ["date", "actual_value", "entity_id"],
                "construction": "Add a generic numeric feature in build_features without changing model or split logic.",
                "code_locations": ["src/train_forecast.py", "src/util.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _lifecycle_interaction_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [
            (trial_dir / "code" / "train.py").as_posix(),
            (trial_dir / "code" / "util.py").as_posix(),
        ],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "Add lifecycle interaction features",
                "feature_type": "lifecycle_interaction",
                "cli_args": [],
                "field_sources": ["package_age_days", "days_to_end"],
                "construction": "In build_features, derive interaction features from package_age_days and days_to_end before feature_cols is returned.",
                "code_locations": ["src/train_forecast.py", "src/util.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _holiday_position_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [
            (trial_dir / "code" / "train.py").as_posix(),
            (trial_dir / "code" / "util.py").as_posix(),
        ],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "Add holiday first middle last eve position features",
                "feature_type": "holiday_position",
                "cli_args": [],
                "field_sources": ["holiday_span_day_idx", "holiday_span_days", "days_until_holiday"],
                "construction": "In build_features, derive holiday position columns from existing holiday span fields.",
                "code_locations": ["src/train_forecast.py", "src/util.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _train_weight_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [(trial_dir / "code" / "train.py").as_posix()],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "Enable sample weight for model training",
                "feature_type": "train_weight",
                "cli_args": [],
                "field_sources": ["sample_weight", "model.fit"],
                "construction": "Use the existing sample_weight variable and ensure it reaches model.fit.",
                "code_locations": ["src/train_forecast.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _group_calibration_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [(trial_dir / "code" / "train.py").as_posix()],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "Enable group calibration",
                "feature_type": "group_calibration",
                "cli_args": ["--enable-group-calibration"],
                "field_sources": ["validation_prediction", "validation_actual", "group_fields"],
                "construction": "Enable the existing group calibration CLI/helper path.",
                "code_locations": ["src/train_forecast.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _zero_demand_guardrail_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [(trial_dir / "code" / "train.py").as_posix()],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "Enable zero demand guardrail",
                "feature_type": "zero_demand_guardrail",
                "cli_args": ["--enable-zero-demand-guardrail"],
                "field_sources": ["prediction", "zero_demand_signal", "guardrail"],
                "construction": "Enable the existing zero demand guardrail CLI/helper path.",
                "code_locations": ["src/train_forecast.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _new_feature_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [(trial_dir / "code" / "train.py").as_posix()],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "new_feature",
                "feature_type": "numeric",
                "cli_args": [],
                "code_locations": ["src/train_forecast.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _resource_path_plan(trial_dir: Path) -> dict:
    return {
        "trial_id": trial_dir.name,
        "target_problem": "resource_path_resolution",
        "hypothesis": {"description": "runner should pass copied data resources to training"},
        "editable_files": [(trial_dir / "code" / "train.py").as_posix()],
        "changes": [],
        "expected_effect": "resource path resolves before training",
        "risk": "none",
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
        "output_contract": "generic_output",
        "candidate_experiments": [],
    }


def _execution_plan() -> dict:
    return {
        "agent": "Agent2",
        "trial_id": "trial_001",
        "source_entrypoint": "src/train_forecast.py",
        "python_dependencies": ["src/features.py"],
        "required_data_files": [],
        "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}", "--enable-recent-trend-ratio"],
        "output_contract": {
            "prediction_path": "generic_output.csv",
            "actual_path": "generic_output.csv",
            "prediction_column": "forecast_value",
            "actual_column": "actual_value",
            "date_column": "date",
            "id_columns": ["entity_id"],
            "passthrough_columns": ["segment"],
        },
    }


def _rolling_execution_plan() -> dict:
    plan = _execution_plan()
    plan["source_entrypoint"] = "src/train_forecast.py"
    plan["python_dependencies"] = []
    plan["train_command"] = ["{python}", "{train_py}", "--output_dir", "{real_output_dir}", "--rolling_windows", "3", "7"]
    plan["output_contract"]["passthrough_columns"] = []
    return plan


def _rolling_sum_dependency_execution_plan() -> dict:
    plan = _rolling_execution_plan()
    plan["python_dependencies"] = ["src/util.py"]
    plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--rolling_windows",
        "3",
        "7",
        "14",
        "30",
    ]
    return plan


def _lifecycle_interaction_execution_plan() -> dict:
    plan = _feature_builder_execution_plan()
    plan["source_entrypoint"] = "src/train_forecast.py"
    plan["python_dependencies"] = ["src/util.py"]
    return plan


def _feature_builder_execution_plan() -> dict:
    plan = _execution_plan()
    plan["source_entrypoint"] = "src/train_forecast.py"
    plan["python_dependencies"] = []
    plan["train_command"] = ["{python}", "{train_py}", "--output_dir", "{real_output_dir}"]
    plan["output_contract"]["passthrough_columns"] = []
    return plan


def _resource_path_execution_plan() -> dict:
    plan = _feature_builder_execution_plan()
    plan["source_entrypoint"] = "src/train_forecast.py"
    plan["python_dependencies"] = []
    plan["required_data_files"] = [{"path": "data/input.csv", "cli_arg": "--data_path"}]
    plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--data_path",
        "data/input.csv",
    ]
    return plan


def _dependency_feature_execution_plan() -> dict:
    plan = _feature_builder_execution_plan()
    plan["python_dependencies"] = ["src/features.py"]
    return plan


def _conditional_cli_dependency_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [
            (trial_dir / "code" / "train.py").as_posix(),
            (trial_dir / "code" / "util.py").as_posix(),
        ],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "decline_recency_features",
                "feature_type": "numeric",
                "cli_args": ["--enable_decline_recency_features"],
                "audit_tokens": ["decline_recency_features"],
                "code_locations": ["src/train_forecast.py", "src/util.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _conditional_cli_dependency_execution_plan() -> dict:
    plan = _feature_builder_execution_plan()
    plan["python_dependencies"] = ["src/util.py"]
    plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--enable_decline_recency_features",
    ]
    return plan


def _declared_lifecycle_death_plan(trial_dir: Path) -> dict:
    return {
        **_plan(trial_dir),
        "editable_files": [
            (trial_dir / "code" / "train.py").as_posix(),
            (trial_dir / "code" / "util.py").as_posix(),
        ],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "enhanced_package_lifecycle_death_features",
                "feature_type": "rolling_stat",
                "cli_args": [],
                "field_sources": ["combo_for_psnnum", "ds", "true_pos_cnt"],
                "construction": (
                    "In build_features, compute:\n"
                    "- days_since_last_sale: number of days since the last nonzero target.\n"
                    "- days_since_first_appearance: number of days since the first appearance.\n"
                    "- is_dead: binary flag indicating days_since_last_sale > 14.\n"
                    "Integrate them into the feature_cols list returned by the feature generation pipeline."
                ),
                "audit_tokens": ["lifecycle_interaction", "package_age_days", "days_to_end"],
                "code_locations": ["src/train_forecast.py", "src/util.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }


def _choice_arg_execution_plan() -> dict:
    plan = _execution_plan()
    plan["source_entrypoint"] = "src/train_forecast.py"
    plan["python_dependencies"] = []
    plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--enable-recent-trend-ratio",
        "--experiment",
        "{trial_id}",
    ]
    return plan


def _file_package(files: dict[str, str], notes: list[str] | None = None) -> str:
    return yaml.safe_dump(
        {"files": [{"path": path, "content": content} for path, content in files.items()], "notes": notes or []},
        sort_keys=False,
    )


def _edit_package(edits: list[dict], notes: list[str] | None = None) -> str:
    return yaml.safe_dump({"edits": edits, "notes": notes or []}, sort_keys=False)


def _valid_train_py() -> str:
    return """
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from features import recent_trend_ratio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable-recent-trend-ratio", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recent_trend_ratio_value = recent_trend_ratio(args.enable_recent_trend_ratio)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value", "segment"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": 95 if recent_trend_ratio_value > 0.9 else 50,
            "segment": "high_target",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _valid_main_function() -> str:
    return """
def main() -> int:
    import argparse
    import csv
    from pathlib import Path
    from features import recent_trend_ratio

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable-recent-trend-ratio", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recent_trend_ratio_value = recent_trend_ratio(args.enable_recent_trend_ratio)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value", "segment"])
        writer.writeheader()
        writer.writerow({
            "date": "2026-05-01",
            "entity_id": "E001",
            "actual_value": 100,
            "forecast_value": 95 if recent_trend_ratio_value > 0.9 else 50,
            "segment": "high_target",
        })
    return 0
"""


def _valid_main_edit_package(notes: list[str] | None = None, path: str = "train.py") -> str:
    return _edit_package(
        [
            {
                "path": path,
                "type": "replace_function",
                "function": "main",
                "content": _valid_main_function(),
            }
        ],
        notes,
    )


def _function_source_from_module(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    target = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function_name)
    lines = source.splitlines()
    return "\n".join(lines[target.lineno - 1 : target.end_lineno]) + "\n"


def test_real_runner_copies_only_llm_declared_python_dependencies(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"

    wrapper = generate_trial_train_wrapper(
        _plan(trial_dir),
        experiment,
        trial_dir,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_execution_plan(),
    )

    assert wrapper == trial_dir / "code" / "train.py"
    assert (trial_dir / "code" / "features.py").exists()
    assert not (trial_dir / "code" / "unused.py").exists()
    assert not (trial_dir / "src").exists()
    assert "source: llm_required" in (trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8")


def test_real_runner_fails_fast_when_agent2_llm_is_unavailable(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=UnavailableAgent2Client(),
        execution_plan=_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["agent2_code_generation_success"] is False
    assert status["agent2_fallback_used"] is False
    assert (trial_dir / "output_contract.json").exists()
    assert "Agent2 code generation failed" in (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")


def test_real_runner_skips_logged_eval_time_llm_when_agent2_codegen_fails(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    source_log = tmp_path / "source_eval.log"
    source_log.write_text("test_window=2026-05-05~2026-05-11\n", encoding="utf-8")
    execution_plan = _execution_plan()
    execution_plan["source_evaluation_context"] = {"train_log_path": source_log.as_posix()}
    client = FakeAgent2Client([""])

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=execution_plan,
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert "DetermineLoggedEvalTime" not in [call["step"] for call in client.calls]
    assert (trial_dir / "run_status.json").exists()


def test_agent2_classifies_proxy_failure_as_llm_network_error() -> None:
    reason = (
        "HTTPSConnectionPool(host='api.deepseek.com', port=443): Max retries exceeded "
        "with url: /chat/completions (Caused by ProxyError('Unable to connect to proxy', "
        "RemoteDisconnected('Remote end closed connection without response')))"
    )

    assert real_runner._failure_category_from_reason(reason) == "llm_network_error"


def test_real_runner_standardizes_generic_output_contract(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(["change_plan:\n- path: train.py\n  function: main\n", _valid_main_edit_package()])

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["eval_success"] is True
    assert status["feature_application_success"] is True
    assert Path(status["raw_prediction_output_path"]).name == "generic_output.csv"
    assert Path(status["prediction_path"]).name == "prediction.csv"
    assert Path(status["actual_path"]).name == "actual.csv"
    assert (trial_dir / "standardized" / "prediction.csv").exists()
    assert (trial_dir / "standardized" / "actual.csv").exists()
    contract_text = (trial_dir / "output_contract.json").read_text(encoding="utf-8")
    assert "package" not in contract_text.lower()
    standardized = (trial_dir / "standardized" / "prediction.csv").read_text(encoding="utf-8")
    assert "prediction" in standardized
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["source"] == "llm_edits"
    assert record["fallback_used"] is False
    assert [call["step"] for call in client.calls] == ["SelectEditTask", "GenerateCodeEdits"]
    raw_dir = trial_dir / "audit" / "agent2_llm_raw"
    raw_files = sorted(raw_dir.glob("*.yaml"))
    assert [path.name for path in raw_files] == [
        "01_SelectEditTask_attempt1.yaml",
        "02_GenerateCodeEdits_attempt1.yaml",
    ]
    raw_generate = yaml.safe_load(raw_files[1].read_text(encoding="utf-8"))
    assert raw_generate["step"] == "GenerateCodeEdits"
    assert "edits:" in raw_generate["content"]
    assert record["attempts"][1]["raw_output_path"].endswith("02_GenerateCodeEdits_attempt1.yaml")
    codegen_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "GenerateCodeEdits")
    assert "forecast-trial-codegen" in codegen_prompt
    assert "replace_function" in codegen_prompt
    prompt_payload = yaml.safe_load(codegen_prompt)
    assert prompt_payload["prompt_budget"]["prompt_token_estimate"] <= real_runner.AGENT2_PROMPT_TOKEN_LIMIT
    assert prompt_payload["accepted_edit_types"] == real_runner.AGENT2_FIRST_PASS_EDIT_TYPES
    assert "context_pack" not in prompt_payload
    slices = {item["path"]: item for item in prompt_payload["source_slices"]["slices"]}
    assert "def main() -> int" in slices["train.py"]["source"]
    assert all("source" not in item for item in prompt_payload["code_index"]["files"])


def test_agent2_codegen_prompt_prefers_trial_changes_over_candidates(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _plan(trial_dir)
    plan["changes"] = [
        {
            "action": "add_feature",
            "feature_name": "lower_priority_feature",
            "feature_type": "numeric",
            "cli_args": [],
            "code_locations": ["src/train_forecast.py"],
        }
    ]
    plan["candidate_experiments"] = [
        {
            "experiment_id": "exp_lower",
            "priority": 2,
            "feature_actions": plan["changes"],
        },
        {
            "experiment_id": "exp_top",
            "priority": 1,
            "feature_actions": [
                {
                    "action": "add_feature",
                    "feature_name": "recent_trend_ratio",
                    "feature_type": "trend_ratio",
                    "cli_args": [],
                    "code_locations": ["src/train_forecast.py", "src/features.py"],
                }
            ],
        },
    ]
    client = FakeAgent2Client(["change_plan:\n- path: train.py\n  function: main\n", _valid_main_edit_package()])

    generate_trial_train_wrapper(
        plan,
        experiment,
        trial_dir,
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    assert plan["changes"][0]["feature_name"] == "lower_priority_feature"
    codegen_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "GenerateCodeEdits")
    prompt_payload = yaml.safe_load(codegen_prompt)
    assert prompt_payload["primary_feature_change"]["feature_name"] == "lower_priority_feature"


def test_agent2_codegen_prompt_includes_complete_selected_action_set(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _plan(trial_dir)
    plan["changes"] = [
        {
            "action": "add_feature",
            "feature_name": "recent_trend_ratio",
            "feature_type": "trend_ratio",
            "cli_args": [],
            "code_locations": ["src/train_forecast.py", "src/features.py"],
        },
        {
            "action": "add_feature",
            "feature_name": "recent_trend_bucket",
            "feature_type": "bucket",
            "cli_args": [],
            "code_locations": ["src/train_forecast.py", "src/features.py"],
        },
    ]
    client = FakeAgent2Client(["change_plan:\n- path: train.py\n  function: main\n", _valid_main_edit_package()])

    generate_trial_train_wrapper(
        plan,
        experiment,
        trial_dir,
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    codegen_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "GenerateCodeEdits")
    prompt_payload = yaml.safe_load(codegen_prompt)
    assert [item["feature_name"] for item in prompt_payload["selected_feature_actions"]] == [
        "recent_trend_ratio",
        "recent_trend_bucket",
    ]
    assert "provided source_slices" in prompt_payload["task"]


def test_real_runner_keeps_agent2_prompt_under_token_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(real_runner, "AGENT2_PROMPT_TOKEN_LIMIT", 500)
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(["change_plan:\n- path: train.py\n  function: main\n", _valid_main_edit_package()])

    run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    for call in client.calls:
        prompt_payload = yaml.safe_load(call["user_prompt"])
        assert prompt_payload["prompt_budget"]["prompt_token_estimate"] <= 500


def test_real_runner_auto_discovers_unique_real_output_when_contract_path_is_stale(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(["change_plan:\n- path: train.py\n  function: main\n", _valid_main_edit_package()])
    execution_plan = _execution_plan()
    execution_plan["output_contract"] = {
        **execution_plan["output_contract"],
        "prediction_path": "outputs/stale/missing.csv",
        "actual_path": "outputs/stale/missing.csv",
    }

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=execution_plan,
    )

    assert status["eval_success"] is True
    assert status["contract_resolution"] == "auto_discovered"
    assert Path(status["raw_prediction_output_path"]).name == "generic_output.csv"
    eval_log = (trial_dir / "logs" / "eval.log").read_text(encoding="utf-8")
    assert "auto-discovered contract output" in eval_log


def test_real_runner_does_not_auto_discover_ambiguous_real_outputs(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    ambiguous_train = """
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from features import recent_trend_ratio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable-recent-trend-ratio", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio = recent_trend_ratio(args.enable_recent_trend_ratio)
    for name in ("generic_output.csv", "alternate_output.csv"):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value", "segment"])
            writer.writeheader()
            writer.writerow({"date": "2026-05-01", "entity_id": "E001", "actual_value": 100, "forecast_value": 95 if ratio > 0.9 else 50, "segment": "high_target"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: main\n",
            _edit_package(
                [
                    {
                        "path": "train.py",
                        "type": "replace_function",
                        "function": "main",
                        "content": _function_source_from_module(ambiguous_train, "main").replace(
                            "    parser = argparse.ArgumentParser()\n",
                            "    from features import recent_trend_ratio\n"
                            "    parser = argparse.ArgumentParser()\n",
                        ),
                    }
                ]
            ),
        ]
    )
    execution_plan = _execution_plan()
    execution_plan["output_contract"] = {
        **execution_plan["output_contract"],
        "prediction_path": "outputs/stale/missing.csv",
        "actual_path": "outputs/stale/missing.csv",
    }

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=execution_plan,
    )

    assert status["train_success"] is True
    assert status["eval_success"] is False
    assert status["failure_stage"] == "find_outputs"
    assert "multiple CSVs" in status["error"]
    assert "generic_output.csv" in status["error"]
    assert "alternate_output.csv" in status["error"]


def test_real_runner_normalizes_agent2_paths_inside_trial_code(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    absolute_trial_path = (trial_dir / "code" / "train.py").as_posix()
    client = FakeAgent2Client(["change_plan:\n- path: train.py\n  function: main\n", _valid_main_edit_package(path=absolute_trial_path)])

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    assert status["train_success"] is True
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["modified_files"] == ["train.py"]
    assert record["rejected_files"] == []


def test_real_runner_rejects_agent2_paths_outside_trial_code_with_category(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(["change_plan:\n- path: train.py\n  function: main\n", _valid_main_edit_package(path="/tmp/train.py")])

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert "path_format_error" in record["failure_categories"]
    assert any(item["category"] == "path_format_error" for item in record["normalized_rejections"])
    repair_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "RepairCodeEdits")
    assert "Do not return runs/<trial>/code" in repair_prompt


def test_real_runner_repairs_signature_incompatible_agent2_edit(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    bad_edit = _edit_package(
        [
            {
                "path": "features.py",
                "type": "replace_function",
                "function": "recent_trend_ratio",
                "content": "def recent_trend_ratio() -> float:\n    return 0.95\n",
            }
        ],
        ["bad replacement drops the existing enabled parameter"],
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: features.py\n  function: recent_trend_ratio\n",
            bad_edit,
            _edit_package(
                [
                    {
                        "path": "features.py",
                        "type": "replace_function",
                        "function": "recent_trend_ratio",
                        "content": "def recent_trend_ratio(enabled: bool) -> float:\n    return 0.95 if enabled else 0.5\n",
                    },
                    {
                        "path": "train.py",
                        "type": "replace_function",
                        "function": "main",
                        "content": _valid_main_function(),
                    },
                ],
                ["repair uses existing compatible features.py API"],
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["feature_application_success"] is True
    assert [call["step"] for call in client.calls] == ["SelectEditTask", "GenerateCodeEdits", "RepairCodeEdits"]
    repair_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "RepairCodeEdits")
    assert "signature incompatible: missing accepted keyword: enabled" in repair_prompt
    assert "def recent_trend_ratio(enabled: bool) -> float" in repair_prompt
    assert "Restore missing accepted keyword parameter: enabled" in repair_prompt
    assert "forecast-trial-codegen" in repair_prompt
    assert "Copy the original function signature exactly" in repair_prompt
    assert "Read the callee signature" not in repair_prompt
    assert "Add executable evidence" not in repair_prompt


def test_real_runner_records_signature_incompatible_repair_guidance_when_unfixed(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    bad_edit = _edit_package(
        [
            {
                "path": "features.py",
                "type": "replace_function",
                "function": "recent_trend_ratio",
                "content": "def recent_trend_ratio() -> float:\n    return 0.95\n",
            }
        ],
    )
    client = FakeAgent2Client(["change_plan:\n- path: features.py\n  function: recent_trend_ratio\n", bad_edit])

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    assert status["train_success"] is False
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert "signature_incompatible" in record["failure_categories"]
    assert any("enabled" in item["reason"] for item in record["normalized_rejections"])
    assert any("original_signatures" in item for item in record["repair_guidance_history"])


def test_real_runner_repairs_agent2_runtime_call_contract_failure(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    bad_build_features = """
def build_features(
    df,
    label_col: str,
    date_col: str,
    id_cols,
    lag_days,
    rolling_windows,
    category_maps=None,
    return_category_maps: bool = False,
    history_gap_days: int = 1,
):
    df = add_time_features(df)
    df["runtime_contract_feature"] = 1.0
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols = []
    if return_category_maps:
        return df, feature_cols, cat_cols, category_maps or {}
    return df, feature_cols, cat_cols
"""
    repaired_build_features = bad_build_features.replace("add_time_features(df)", "add_time_features(df, date_col)")
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: util.py\n  function: build_features\n",
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": bad_build_features,
                    }
                ]
            ),
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": repaired_build_features,
                    }
                ],
                ["repair passes date_col into add_time_features"],
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _runtime_contract_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["agent2_code_generation_success"] is True
    assert [call["step"] for call in client.calls] == ["SelectEditTask", "GenerateCodeEdits", "RepairCodeEdits"]
    repair_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "RepairCodeEdits")
    assert "missing required parameter date_col" in repair_prompt
    assert "runtime_contract_error" in repair_prompt
    assert "Read the callee signature" in repair_prompt
    assert "Copy the original function signature exactly" not in repair_prompt
    assert "Add executable evidence" not in repair_prompt
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert any(item.get("category") == "runtime_contract_error" for item in record["repair_guidance_history"][0]["normalized_rejections"])
    assert "runtime_contract_feature" in (trial_dir / "code" / "util.py").read_text(encoding="utf-8")


def test_real_runner_allows_agent2_feature_builder_after_static_codegen_checks(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    runtime_bad_build_features = """
def build_features(
    df,
    label_col: str,
    date_col: str,
    id_cols,
    lag_days,
    rolling_windows,
    category_maps=None,
    return_category_maps: bool = False,
    history_gap_days: int = 1,
):
    df = add_time_features(df, date_col)
    new_cols = [c for c in df.columns if c not in set(new_cols)]
    df["runtime_contract_feature"] = 1.0
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols = []
    if return_category_maps:
        return df, feature_cols, cat_cols, category_maps or {}
    return df, feature_cols, cat_cols
"""
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: util.py\n  function: build_features\n",
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": runtime_bad_build_features,
                    }
                ]
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _runtime_contract_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] != "agent2_code_generation_failed"
    assert status["failure_stage"] == "train_process"
    assert "RepairCodeEdits" not in [call["step"] for call in client.calls]
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["code_generation_success"] is True
    assert record["failure_categories"] == []
    assert record["normalized_rejections"] == []
    assert not (trial_dir / "code" / "_agent2_staging").exists()


def test_real_runner_repairs_backtest_failure_with_narrow_edit_package(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    runtime_bad_build_features = """
def build_features(
    df,
    label_col: str,
    date_col: str,
    id_cols,
    lag_days,
    rolling_windows,
    category_maps=None,
    return_category_maps: bool = False,
    history_gap_days: int = 1,
):
    df = add_time_features(df, date_col)
    new_cols = [c for c in df.columns if c not in set(new_cols)]
    df["runtime_contract_feature"] = 1.0
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols = []
    if return_category_maps:
        return df, feature_cols, cat_cols, category_maps or {}
    return df, feature_cols, cat_cols
"""
    repaired_util = (experiment / "src" / "util.py").read_text(encoding="utf-8").replace(
        "    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]\n",
        "    df[\"runtime_contract_feature\"] = 1.0\n"
        "    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]\n",
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: util.py\n  function: build_features\n",
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": runtime_bad_build_features,
                    }
                ]
            ),
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": _function_source_from_module(repaired_util, "build_features"),
                    }
                ],
                ["repair restores runtime-safe build_features"],
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _runtime_contract_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["eval_success"] is True
    assert status["agent2_backtest_success"] is True
    assert status["agent2_backtest_repair_used"] is True
    assert len(status["agent2_backtest_attempts"]) == 2
    assert [call["step"] for call in client.calls] == ["SelectEditTask", "GenerateCodeEdits", "RepairBacktestFailure"]
    repair_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "RepairBacktestFailure")
    assert "cannot access free variable 'new_cols'" in repair_prompt or "new_cols" in repair_prompt
    assert "last_agent2_code_package" in repair_prompt
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["source"] == "llm_backtest_repair"
    assert record["accepted_schema"] == "edits"
    assert record["agent2_backtest_repair"] is True


def test_real_runner_training_can_use_original_python_packages_after_codegen(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    package_dir = experiment / ".python_packages"
    package_dir.mkdir()
    (package_dir / "fake_agent2_runtime_dependency.py").write_text("VALUE = 1.0\n", encoding="utf-8")
    util_path = experiment / "src" / "util.py"
    util_path.write_text(
        util_path.read_text(encoding="utf-8").replace(
            "import pandas as pd\n",
            "import pandas as pd\nimport fake_agent2_runtime_dependency\n",
        ),
        encoding="utf-8",
    )
    trial_dir = tmp_path / "trial_001"
    replacement = """
def build_features(
    df,
    label_col: str,
    date_col: str,
    id_cols,
    lag_days,
    rolling_windows,
    category_maps=None,
    return_category_maps: bool = False,
    history_gap_days: int = 1,
):
    df = add_time_features(df, date_col)
    df["runtime_contract_feature"] = fake_agent2_runtime_dependency.VALUE
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols = []
    if return_category_maps:
        return df, feature_cols, cat_cols, category_maps or {}
    return df, feature_cols, cat_cols
"""
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: util.py\n  function: build_features\n",
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": replacement,
                    }
                ]
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _runtime_contract_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["agent2_code_generation_success"] is True
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["failure_categories"] == []


def test_real_runner_repairs_after_final_generate_attempt(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    bad_build_features = """
def build_features(
    df,
    label_col: str,
    date_col: str,
    id_cols,
    lag_days,
    rolling_windows,
    category_maps=None,
    return_category_maps: bool = False,
    history_gap_days: int = 1,
):
    df = add_time_features(df)
    df["runtime_contract_feature"] = 1.0
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    cat_cols = []
    if return_category_maps:
        return df, feature_cols, cat_cols, category_maps or {}
    return df, feature_cols, cat_cols
"""
    repaired_build_features = bad_build_features.replace("add_time_features(df)", "add_time_features(df, date_col)")
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: util.py\n  function: build_features\n",
            "not: a valid edits package",
            "",
            "",
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": bad_build_features,
                    }
                ]
            ),
            _edit_package(
                [
                    {
                        "path": "util.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": repaired_build_features,
                    }
                ]
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _runtime_contract_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["agent2_code_generation_success"] is True
    steps = [call["step"] for call in client.calls]
    assert steps.count("GenerateCodeEdits") == 2
    assert steps[-1] == "RepairCodeEdits"
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert any(
        item.get("category") == "runtime_contract_error"
        for history in record["repair_guidance_history"]
        for item in history.get("normalized_rejections", [])
    )


def test_real_runner_turns_duplicate_rolling_default_into_cli_variation(tmp_path: Path) -> None:
    experiment = _rolling_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _rolling_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_rolling_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["agent2_code_generation_success"] is True
    assert status["agent2_modified_files"] == []
    assert plan["changes"][0]["feature_name"] == "extend_rolling_windows_1_3_7"
    assert plan["changes"][0]["cli_args"] == ["--rolling_windows", "1", "3", "7"]
    assert status["train_command"][-4:] == ["--rolling_windows", "1", "3", "7"]
    assert plan["plan_validation"][0]["category"] == "duplicate_existing_cli_default"


def test_real_runner_normalizes_comma_separated_numeric_nargs_before_training(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = _rolling_sum_dependency_execution_plan()
    execution_plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--rolling_windows",
        "7,14,30",
    ]

    status = run_real_experiment(
        experiment,
        trial_dir,
        _rolling_sum_dependency_plan(trial_dir),
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )

    rolling_index = status["train_command"].index("--rolling_windows")
    assert status["train_command"][rolling_index : rolling_index + 4] == ["--rolling_windows", "7", "14", "30"]
    assert any(item["kind"] == "split_csv_numeric_list" for item in status["train_command_normalizations"])
    # LLM 不可用时不得伪造特征代码，训练按 codegen 护栏跳过；
    # 归一化本身已经发生，并在 code_modification 记录中留痕。
    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["agent2_code_generation_success"] is False
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["train_command_normalizations"] == status["train_command_normalizations"]


def test_real_runner_rejects_invalid_numeric_nargs_before_training(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = _rolling_sum_dependency_execution_plan()
    execution_plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--rolling_windows",
        "7,foo",
    ]

    status = run_real_experiment(
        experiment,
        trial_dir,
        _rolling_sum_dependency_plan(trial_dir),
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "invalid_train_command"
    assert status["failure_stage"] == "train_command_validation"
    assert "invalid int value 'foo' to --rolling_windows" in status["error"]
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "train skipped because generated train_command is invalid" in train_log
    assert "backtest_attempt" not in train_log


def test_train_command_validation_rejects_extra_positional_after_store_true(tmp_path: Path) -> None:
    wrapper = tmp_path / "train.py"
    wrapper.write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--enable_sparsity_rolling", action="store_true")
args = parser.parse_args()
""",
        encoding="utf-8",
    )

    error = _train_command_validation_error(
        [sys.executable, wrapper.as_posix(), "--output_dir", "out", "--enable_sparsity_rolling", "(optional", "flag)"],
        wrapper,
        {},
    )

    assert "unexpected positional argument '(optional'" in error


def test_train_command_validation_allows_declared_positionals_and_rejects_extra(tmp_path: Path) -> None:
    wrapper = tmp_path / "train.py"
    wrapper.write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("--output_dir")
args = parser.parse_args()
""",
        encoding="utf-8",
    )

    assert _train_command_validation_error([sys.executable, wrapper.as_posix(), "input.csv", "--output_dir", "out"], wrapper, {}) == ""
    error = _train_command_validation_error(
        [sys.executable, wrapper.as_posix(), "input.csv", "extra", "--output_dir", "out"],
        wrapper,
        {},
    )

    assert "unexpected positional argument 'extra'" in error


def test_train_command_validation_allows_unknown_positionals_for_parse_known_args(tmp_path: Path) -> None:
    wrapper = tmp_path / "train.py"
    wrapper.write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--known", action="store_true")
args, unknown = parser.parse_known_args()
""",
        encoding="utf-8",
    )

    assert _train_command_validation_error([sys.executable, wrapper.as_posix(), "--known", "loose", "--other"], wrapper, {}) == ""


def test_real_runner_rejects_unsupported_flag_after_trial_code_generation(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = _rolling_sum_dependency_execution_plan()
    execution_plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--add_store_dish_rolling",
        "--rolling_windows",
        "7",
        "14",
        "30",
    ]

    status = run_real_experiment(
        experiment,
        trial_dir,
        _rolling_sum_dependency_plan(trial_dir),
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "invalid_train_command"
    assert status["failure_stage"] == "train_command_validation"
    assert "unsupported argument --add_store_dish_rolling" in status["error"]
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "unsupported argument --add_store_dish_rolling" in train_log
    assert "backtest_attempt" not in train_log


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Agent2 代码生成流水线已从单次调用演进为 SelectEditTask → GenerateCodeEdits → "
        "RepairCodeEdits(多轮) → RepairTrainCommandContract 的状态机（本场景共 8 次 LLM 调用），"
        "而本测试的 canned 响应仍是单次调用的旧契约，第一条响应被 SelectEditTask 消费后即耗尽。"
        "待 RoutePilot 重写 Modify 步骤时一并重写该 fixture。"
    ),
)
def test_real_runner_repairs_unsupported_train_command_flag_before_training(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = _rolling_sum_dependency_execution_plan()
    execution_plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--add_store_dish_rolling",
        "--rolling_windows",
        "7",
    ]
    repaired_train_py = (experiment / "src" / "train_forecast.py").read_text(encoding="utf-8").replace(
        '    parser.add_argument("--rolling_windows", "--rolling-windows", type=int, nargs="*", default=[3, 7])\n',
        '    parser.add_argument("--add_store_dish_rolling", action="store_true")\n'
        '    parser.add_argument("--rolling_windows", "--rolling-windows", type=int, nargs="*", default=[3, 7])\n',
    )
    client = FakeAgent2Client(
        [
            _edit_package(
                [
                        {
                            "path": "train.py",
                            "type": "replace_function",
                            "function": "main",
                            "content": _function_source_from_module(repaired_train_py, "main"),
                        }
                ],
                ["accept trial command compatibility flag"],
            )
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _rolling_sum_dependency_plan(trial_dir),
        llm_client=client,
        execution_plan=execution_plan,
    )

    assert status["train_success"] is True
    assert status["eval_success"] is True
    assert status["agent2_train_command_repair_success"] is True
    assert "unsupported argument --add_store_dish_rolling" in status["agent2_train_command_contract_error_before_repair"]
    assert "--add_store_dish_rolling" in status["train_command"]
    assert (trial_dir / "code" / "train.py").read_text(encoding="utf-8").count("--add_store_dish_rolling") == 1
    assert any(call["step"] == "RepairTrainCommandContract" for call in client.calls)
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "backtest_attempt: 1" in train_log


def test_real_runner_repairs_missing_value_train_command_with_command_only_package(tmp_path: Path) -> None:
    experiment = _trend_guardrail_value_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = _execution_plan()
    execution_plan["source_entrypoint"] = "src/train_forecast.py"
    execution_plan["python_dependencies"] = ["src/features.py"]
    execution_plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--enable-recent-trend-ratio",
        "--trend_guardrail",
    ]
    client = FakeAgent2Client(
        [
            yaml.safe_dump(
                {
                    "train_command": [
                        "{python}",
                        "{train_py}",
                        "--output_dir",
                        "{real_output_dir}",
                        "--enable-recent-trend-ratio",
                        "--trend_guardrail",
                        "downtrend_clip",
                    ],
                    "train_py_unchanged_reason": "train.py already parses and consumes trend_guardrail; the command was missing its value.",
                    "entrypoint_import_chain_checked": True,
                    "notes": ["added missing trend_guardrail choice value"],
                },
                sort_keys=False,
            )
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=execution_plan,
    )

    assert status["train_success"] is True
    assert status["eval_success"] is True
    assert status["agent2_train_command_repair_success"] is True
    trend_index = status["train_command"].index("--trend_guardrail")
    assert status["train_command"][trend_index : trend_index + 2] == ["--trend_guardrail", "downtrend_clip"]
    assert status["agent2_repaired_train_command"][-2:] == ["--trend_guardrail", "downtrend_clip"]
    assert "missing value for --trend_guardrail" in status["agent2_train_command_contract_error_before_repair"]


def test_real_runner_deterministically_repairs_glm_like_missing_guardrail_value(tmp_path: Path) -> None:
    experiment = _glm_like_command_contract_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = {
        **_plan(trial_dir),
        "editable_files": [(trial_dir / "code" / "train.py").as_posix()],
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "calibration_guardrail_peak_history_wider_rolling_coordinated",
                "feature_type": "rolling_stat",
                "cli_args": [
                    "--enable_group_calibration",
                    "--enable_peak_history_mean",
                    "--trend_guardrail",
                ],
                "field_sources": ["entity_id", "actual_value", "date"],
                "construction": (
                    "Coordinate group calibration, trend_guardrail, peak history mean, "
                    "and wider rolling windows in one trial."
                ),
                "code_locations": ["src/train_forecast.py"],
            }
        ],
        "source_entrypoint": "src/train_forecast.py",
        "generated_train_path": (trial_dir / "code" / "train.py").as_posix(),
    }
    execution_plan = {
        **_execution_plan(),
        "source_entrypoint": "src/train_forecast.py",
        "python_dependencies": [],
        "train_command": [
            "{python}",
            "{train_py}",
            "--output_dir",
            "{real_output_dir}",
            "--rolling_windows",
            "7",
            "14",
            "30",
            "--enable_group_calibration",
            "--enable_peak_history_mean",
            "--trend_guardrail",
        ],
    }

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )

    assert status["train_success"] is True
    assert status["eval_success"] is True
    assert status["feature_application_success"] is True
    assert status["agent2_train_command_repair_success"] is True
    assert status["agent2_command_contract_status"] == "repaired"
    trend_index = status["train_command"].index("--trend_guardrail")
    assert status["train_command"][trend_index : trend_index + 2] == ["--trend_guardrail", "downtrend_clip"]
    assert status["agent2_repaired_train_command"][trend_index : trend_index + 2] == [
        "--trend_guardrail",
        "downtrend_clip",
    ]
    assert "missing value for --trend_guardrail" in status["agent2_train_command_contract_error_before_repair"]

    output_text = Path(status["raw_prediction_output_path"]).read_text(encoding="utf-8")
    assert "group=True;peak=True;trend=downtrend_clip;windows=7,14,30" in output_text
    assert ",95," in output_text
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is True
    assert audit["features"][0]["cli_args"] == [
        "--enable_group_calibration",
        "--enable_peak_history_mean",
        "--trend_guardrail",
        "downtrend_clip",
    ]


def test_real_runner_repairs_explanatory_plan_cli_args_before_training(tmp_path: Path) -> None:
    experiment = _group_calibration_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _group_calibration_plan(trial_dir)
    plan["changes"][0]["cli_args"] = ["--enable-group-calibration  (optional boolean flag)"]
    client = FakeAgent2Client(
        [
            yaml.safe_dump(
                {
                    "train_command": [
                        "{python}",
                        "{train_py}",
                        "--output_dir",
                        "{real_output_dir}",
                        "--enable-group-calibration",
                    ],
                    "train_py_unchanged_reason": "train.py already parses and consumes the boolean feature flag; only explanatory argv tokens were removed.",
                    "entrypoint_import_chain_checked": True,
                    "notes": ["removed explanatory tokens from train_command"],
                },
                sort_keys=False,
            )
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=client,
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["eval_success"] is True
    assert status["agent2_train_command_repair_success"] is True
    assert "unexpected positional argument '(optional'" in status["agent2_train_command_contract_error_before_repair"]
    assert status["train_command"][-1] == "--enable-group-calibration"
    assert "(optional" not in status["train_command"]
    assert plan["changes"][0]["cli_args"] == ["--enable-group-calibration"]
    assert any(call["step"] == "RepairTrainCommandContract" for call in client.calls)


def test_real_runner_normalizes_shell_fragment_plan_cli_args_before_merge(tmp_path: Path) -> None:
    experiment = _rolling_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _rolling_plan(trial_dir)
    plan["plan_validation"] = [{"category": "duplicate_existing_cli_default"}]
    plan["changes"][0]["feature_name"] = "extend_rolling_windows_custom"
    plan["changes"][0]["cli_args"] = ['--rolling_windows "1,3,7"']

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_rolling_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["agent2_modified_files"] == []
    assert status["train_command"][-4:] == ["--rolling_windows", "1", "3", "7"]
    output = (trial_dir / "outputs" / "real_outputs" / "generic_output.csv").read_text(encoding="utf-8")
    assert "95" in output


def test_feature_audit_normalizes_cli_values_without_polluting_dest_names(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rolling_windows", type=int, nargs="*", default=[3, 7])
args = parser.parse_args()
print(args.rolling_windows)
""",
        encoding="utf-8",
    )
    plan = {
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "rolling_cli_variation",
                "feature_type": "rolling_stat",
                "cli_args": ['--rolling_windows "1,3,7"'],
            }
        ]
    }

    audit = _audit_feature_application(plan, code_dir, wrapper)

    feature = audit["features"][0]
    assert feature["cli_args"] == ["--rolling_windows", "1", "3", "7"]
    assert feature["cli_dest_names"] == ["rolling_windows"]
    assert feature["parsed_flags"] == ["--rolling_windows"]
    assert "rolling_windows \"1,3,7\"" not in feature["cli_dest_names"]


def test_feature_audit_rejects_token_only_evidence_when_cli_is_unparsed(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
def build_features(row: dict) -> dict:
    row["capacity_guardrail_feature"] = 1
    return row
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"], "code_generation_success": True}),
        encoding="utf-8",
    )
    plan = {
        "changes": [
            {
                "action": "add_feature",
                "feature_name": "capacity_guardrail",
                "feature_type": "rolling_stat",
                "cli_args": ["--capacity_guardrail"],
                "audit_tokens": ["capacity_guardrail"],
            }
        ]
    }

    audit = _audit_feature_application(plan, code_dir, wrapper)

    feature = audit["features"][0]
    assert audit["success"] is False
    assert feature["applied"] is False
    assert feature["cli_contract_success"] is False
    assert feature["unparsed_cli_args"] == ["--capacity_guardrail"]
    assert any(item["kind"] == "feature_column_constructed" for item in feature["evidence"])


def test_feature_audit_rejects_dependency_cli_consumption_without_args_pass_through(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse
from util import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_decline_recency_features", action="store_true")
    args = parser.parse_args()
    row = build_features({"actual_value": 1})
    return row
""",
        encoding="utf-8",
    )
    (code_dir / "util.py").write_text(
        """
def build_features(row, args=None):
    if args is not None and getattr(args, "enable_decline_recency_features", False):
        row["decline_recency_features"] = 1
    return row
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py", "util.py"], "code_generation_success": True}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(_conditional_cli_dependency_plan(tmp_path / "trial_001"), code_dir, wrapper)

    feature = audit["features"][0]
    assert audit["success"] is False
    assert feature["failure_category"] == "conditional_cli_not_wired"
    assert "not passed through train.py call sites" in feature["failure_reason"]
    assert feature["conditional_cli_wiring"]["missing_args_call_sites"]


def test_feature_audit_accepts_dependency_cli_consumption_with_args_pass_through(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse
from util import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_decline_recency_features", action="store_true")
    args = parser.parse_args()
    row = build_features({"actual_value": 1}, args=args)
    return row
""",
        encoding="utf-8",
    )
    (code_dir / "util.py").write_text(
        """
def build_features(row, args=None):
    if args is not None and getattr(args, "enable_decline_recency_features", False):
        row["decline_recency_features"] = 1
    return row
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py", "util.py"], "code_generation_success": True}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(_conditional_cli_dependency_plan(tmp_path / "trial_001"), code_dir, wrapper)

    feature = audit["features"][0]
    assert audit["success"] is True
    assert feature["conditional_cli_wiring"]["success"] is True


def test_feature_audit_rejects_guardrail_evidence_when_guardrail_is_not_activated(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse


def apply_recent_trend_guardrail(pred, args):
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trend_guardrail", choices=["none", "downtrend_clip"], default="none")
    args = parser.parse_args()
    return apply_recent_trend_guardrail([1], args)
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"], "code_generation_success": True}),
        encoding="utf-8",
    )
    plan = _zero_demand_guardrail_plan(tmp_path / "trial_001")
    plan["changes"][0]["cli_args"] = []

    audit = _audit_feature_application(plan, code_dir, wrapper)

    feature = audit["features"][0]
    assert audit["success"] is False
    assert feature["failure_category"] == "guardrail_not_activated"
    assert feature["guardrail_activation"]["source"] == "argparse_default"
    assert feature["guardrail_activation"]["value"] == "none"


def test_feature_audit_accepts_guardrail_evidence_when_command_activates_guardrail(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse


def apply_recent_trend_guardrail(pred, args):
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trend_guardrail", choices=["none", "downtrend_clip"], default="none")
    args = parser.parse_args()
    return apply_recent_trend_guardrail([1], args)
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"], "code_generation_success": True}),
        encoding="utf-8",
    )
    plan = _zero_demand_guardrail_plan(tmp_path / "trial_001")
    plan["changes"][0]["cli_args"] = []

    audit = _audit_feature_application(
        plan,
        code_dir,
        wrapper,
        train_command=[sys.executable, wrapper.as_posix(), "--trend_guardrail", "downtrend_clip"],
    )

    feature = audit["features"][0]
    assert audit["success"] is True
    assert feature["guardrail_activation"]["source"] == "train_command"
    assert feature["guardrail_activation"]["value"] == "downtrend_clip"


def test_feature_audit_rejects_old_token_evidence_when_declared_columns_are_missing(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
from util import build_features


def main():
    df = {"days_to_end": 3}
    return build_features(df)
""",
        encoding="utf-8",
    )
    (code_dir / "util.py").write_text(
        """
def build_features(df):
    df["days_to_end"] = 3
    feature_cols = [c for c in df.columns if c != "true_pos_cnt"]
    return df, feature_cols
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"], "code_generation_success": True}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(_declared_lifecycle_death_plan(tmp_path / "trial_001"), code_dir, wrapper)

    feature = audit["features"][0]
    assert audit["success"] is False
    assert feature["failure_category"] == "declared_feature_columns_missing"
    declared = feature["declared_feature_column_application"]
    assert {item["column"] for item in declared["columns"]} == {
        "days_since_last_sale",
        "days_since_first_appearance",
        "is_dead",
    }
    assert all(item["constructed"] is False for item in declared["columns"])


def test_feature_audit_accepts_declared_columns_when_constructed_and_in_feature_cols(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
from util import build_features


def main():
    df = {"true_pos_cnt": 0, "ds": "2026-05-01", "combo_for_psnnum": "A"}
    return build_features(df)
""",
        encoding="utf-8",
    )
    (code_dir / "util.py").write_text(
        """
def build_features(df):
    df["days_since_last_sale"] = 90
    df["days_since_first_appearance"] = 30
    df["is_dead"] = int(df["days_since_last_sale"] > 14)
    feature_cols = [c for c in df.columns if c not in {"true_pos_cnt", "ds"}]
    return df, feature_cols
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py", "util.py"], "code_generation_success": True}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(_declared_lifecycle_death_plan(tmp_path / "trial_001"), code_dir, wrapper)

    feature = audit["features"][0]
    assert audit["success"] is True
    declared = feature["declared_feature_column_application"]
    assert declared["success"] is True
    assert all(item["constructed"] and item["reachable"] for item in declared["columns"])
    assert all(item["feature_cols"]["active"] for item in declared["columns"])


def test_feature_audit_rejects_declared_columns_when_helper_is_disabled_by_preset(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--enable_lifecycle_features", action="store_true", default=True)
    return apply_experiment_preset(parser.parse_args())


def apply_experiment_preset(args):
    if args.experiment == "baseline":
        args.enable_lifecycle_features = False
        return args
    args.enable_lifecycle_features = True
    return args


def add_enhanced_lifecycle_features(df, args):
    df["days_since_last_sale"] = 90
    df["days_since_first_appearance"] = 30
    df["is_dead"] = int(df["days_since_last_sale"] > 14)
    return df


def build_features(df):
    feature_cols = [c for c in df.columns if c not in {"true_pos_cnt", "ds"}]
    return df, feature_cols


def main():
    args = parse_args()
    df = {}
    if args.enable_lifecycle_features:
        df = add_enhanced_lifecycle_features(df, args)
    return build_features(df)
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"], "code_generation_success": True}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(
        _declared_lifecycle_death_plan(tmp_path / "trial_001"),
        code_dir,
        wrapper,
        train_command=[sys.executable, wrapper.as_posix(), "--experiment", "baseline"],
    )

    feature = audit["features"][0]
    assert audit["success"] is False
    assert feature["failure_category"] == "inactive_feature_gate"
    declared = feature["declared_feature_column_application"]
    assert all(item["constructed"] for item in declared["columns"])
    assert all(item["activation"]["active"] is False for item in declared["columns"])


def test_real_runner_does_not_apply_rolling_sum_template_without_llm(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _rolling_sum_dependency_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["agent2_code_generation_success"] is False
    assert status["agent2_modified_files"] == []
    assert plan["changes"][0]["feature_slug"]
    assert "rolling_sum" in plan["changes"][0]["audit_tokens"]
    util_text = (trial_dir / "code" / "util.py").read_text(encoding="utf-8")
    assert 'functions=("mean", "std", "min", "max", "median", "sum")' not in util_text
    assert "days_to_end" in util_text
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is False
    assert not any(item["kind"] == "rolling_sum_constructed" for item in audit["features"][0]["evidence"])
    assert not (trial_dir / "code" / "_agent2_staging").exists()


def test_real_runner_rejects_deterministic_dependency_patch_when_plan_cli_is_unwired(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _rolling_sum_dependency_plan(trial_dir)
    plan["changes"][0]["cli_args"] = ["--rolling_windows_add_max"]

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["agent2_modified_files"] == []
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is False
    assert audit["features"][0]["unparsed_cli_args"] == []
    assert audit["features"][0]["ignored_intent_cli_args"] == ["--rolling_windows_add_max"]
    util_text = (trial_dir / "code" / "util.py").read_text(encoding="utf-8")
    assert 'functions=("mean", "std", "min", "max", "median", "sum")' not in util_text


def test_real_runner_does_not_apply_zero_history_template_without_llm(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _zero_history_flag_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["agent2_code_generation_success"] is False
    assert status["agent2_modified_files"] == []
    util_text = (trial_dir / "code" / "util.py").read_text(encoding="utf-8")
    assert 'zero_history_col = "zero_history_flag"' not in util_text
    assert "zero_history_window = 14" not in util_text
    assert "zero_history_threshold = 10" not in util_text
    # 更强的断言：trial 的 util.py 必须与源文件逐字一致，证明没有注入任何模板特征代码。
    assert util_text == (experiment / "src" / "util.py").read_text(encoding="utf-8")
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is False
    assert not any(item["kind"] == "feature_column_constructed" for item in audit["features"][0]["evidence"])
    assert not (trial_dir / "code" / "_agent2_staging").exists()


def test_real_runner_does_not_apply_lifecycle_business_template_without_llm(tmp_path: Path) -> None:
    experiment = _lifecycle_interaction_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _lifecycle_interaction_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_lifecycle_interaction_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["agent2_modified_files"] == []
    util_text = (trial_dir / "code" / "util.py").read_text(encoding="utf-8")
    assert "lifecycle_age_x_days_to_end" not in util_text
    assert "def build_features(" in util_text


def test_real_runner_removes_unsupported_cli_when_feature_is_already_always_on(tmp_path: Path) -> None:
    experiment = _lifecycle_always_on_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _lifecycle_interaction_plan(trial_dir)
    plan["changes"][0]["cli_args"] = ["--enable_lifecycle_features"]
    execution_plan = _lifecycle_interaction_execution_plan()
    execution_plan["train_command"] = [
        "{python}",
        "{train_py}",
        "--output_dir",
        "{real_output_dir}",
        "--enable_lifecycle_features",
    ]

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )

    assert status["train_success"] is True
    assert status["eval_success"] is True
    assert status["agent2_feature_resolution"] == "existing_always_on_feature"
    assert status["agent2_command_contract_status"] == "repaired"
    assert "--enable_lifecycle_features" not in status["train_command"]
    assert status["agent2_modified_files"] == []
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["accepted_schema"] == "existing_code_feature"
    assert record["train_command_repair_schema"] == "train_command"
    assert record["train_command_repair_success"] is True


def test_real_runner_does_not_apply_holiday_business_template_without_llm(tmp_path: Path) -> None:
    experiment = _holiday_position_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _holiday_position_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_lifecycle_interaction_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["agent2_modified_files"] == []
    util_text = (trial_dir / "code" / "util.py").read_text(encoding="utf-8")
    assert "holiday_position_first" not in util_text


def test_real_runner_does_not_apply_holiday_template_without_span_fields(tmp_path: Path) -> None:
    experiment = _holiday_position_experiment(tmp_path, with_span_fields=False)
    trial_dir = tmp_path / "trial_001"
    plan = _holiday_position_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_lifecycle_interaction_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["agent2_modified_files"] == []
    util_text = (trial_dir / "code" / "util.py").read_text(encoding="utf-8")
    assert "holiday_position_first" not in util_text


def test_real_runner_does_not_wire_sample_weight_without_llm(tmp_path: Path) -> None:
    experiment = _train_weight_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _train_weight_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["agent2_code_generation_success"] is False
    assert status["agent2_modified_files"] == []
    train_text = (trial_dir / "code" / "train.py").read_text(encoding="utf-8")
    assert "model.fit([[1], [2]], [1, 2], sample_weight=sample_weight)" not in train_text
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is False
    assert not any(item["kind"] == "sample_weight_consumed" for item in audit["features"][0]["evidence"])


def test_audit_accepts_lifecycle_and_holiday_structural_evidence_from_llm_edits(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "train.py").write_text(
        """
def build_features(df, label_col, date_col):
    df["lifecycle_age_x_days_to_end"] = df["package_age_days"] * df["days_to_end"]
    df["holiday_position_first"] = (df["holiday_span_day_idx"] == 0).astype(int)
    feature_cols = [c for c in df.columns if c not in {label_col, date_col}]
    return df, feature_cols, []
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"]}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(
        {
            "changes": [
                {
                    "action": "add_feature",
                    "feature_name": "Add lifecycle interaction features",
                    "feature_type": "lifecycle_interaction",
                    "audit_tokens": ["lifecycle_interaction", "package_age_days", "days_to_end"],
                },
                {
                    "action": "add_feature",
                    "feature_name": "Add holiday position features",
                    "feature_type": "holiday_position",
                    "audit_tokens": ["holiday_position", "holiday_span"],
                },
            ]
        },
        code_dir,
        code_dir / "train.py",
    )

    assert audit["success"] is True
    assert any(item["kind"] == "lifecycle_feature_constructed" for item in audit["features"][0]["evidence"])
    assert any(item["kind"] == "holiday_position_constructed" for item in audit["features"][1]["evidence"])


def test_audit_rejects_stale_feature_evidence_when_agent2_modified_no_files(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "train.py").write_text(
        """
def build_features(row: dict) -> dict:
    row["new_feature_14"] = 1
    return row
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": [], "code_generation_success": False}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(_new_feature_plan(tmp_path / "trial_001"), code_dir, code_dir / "train.py")

    assert audit["success"] is False
    assert audit["features"][0]["applied"] is False
    assert audit["features"][0]["stale_evidence_warning"]


def test_real_runner_enables_existing_group_calibration_cli_template(tmp_path: Path) -> None:
    experiment = _group_calibration_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _group_calibration_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["feature_application_success"] is True
    assert status["agent2_modified_files"] == []
    assert status["train_command"][-1] == "--enable-group-calibration"
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["accepted_schema"] == "train_command_args"
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert any(item["kind"] == "calibration_fit_apply" for item in audit["features"][0]["evidence"])


def test_real_runner_enables_existing_zero_demand_guardrail_cli_template(tmp_path: Path) -> None:
    experiment = _zero_demand_guardrail_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    plan = _zero_demand_guardrail_plan(trial_dir)

    status = run_real_experiment(
        experiment,
        trial_dir,
        plan,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["feature_application_success"] is True
    assert status["agent2_modified_files"] == []
    assert status["train_command"][-1] == "--enable-zero-demand-guardrail"
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert any(item["kind"] == "zero_demand_guardrail_consumed" for item in audit["features"][0]["evidence"])


def test_agent2_source_locator_exposes_execution_guidance_fields(tmp_path: Path) -> None:
    experiment = _rolling_sum_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"

    generate_trial_train_wrapper(
        _rolling_sum_dependency_plan(trial_dir),
        experiment,
        trial_dir,
        llm_client=UnavailableAgent2Client(),
        execution_plan=_rolling_sum_dependency_execution_plan(),
    )

    locator = yaml.safe_load((trial_dir / "code" / "agent2_source_locator.yaml").read_text(encoding="utf-8"))
    assert locator["feature_cols_builders"]
    assert locator["train_entrypoints"]
    assert locator["reachable_dependency_functions"]
    assert locator["signature_constraints"]
    assert locator["existing_cli_args"]
    assert locator["fast_path_candidates"][0]["fast_path"] == "rolling_stat_007_guardrail"


def test_audit_accepts_sample_weight_structural_evidence(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "train.py").write_text(
        """
def main():
    sample_weight = [1.0]
    model = type("M", (), {"fit": lambda self, x, y, sample_weight=None: None})()
    model.fit([[1]], [1], sample_weight=sample_weight)
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"]}),
        encoding="utf-8",
    )

    audit = _audit_feature_application(
        {
            "changes": [
                {
                    "action": "add_feature",
                    "feature_name": "Enable train weight",
                    "feature_type": "train_weight",
                    "audit_tokens": ["train_weight", "sample_weight"],
                }
            ]
        },
        code_dir,
        code_dir / "train.py",
    )

    assert audit["success"] is True
    assert any(item["kind"] == "sample_weight_consumed" for item in audit["features"][0]["evidence"])


def test_real_runner_rejects_full_file_replacement_that_drops_key_function(tmp_path: Path) -> None:
    experiment = _feature_builder_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: main\n",
            _file_package({"train.py": "VALUE = 1\n"}),
            _file_package({"train.py": "VALUE = 1\n"}),
            _file_package({"train.py": "VALUE = 1\n"}),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is False
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert any("full-file replacement is not allowed" in item["reason"] for item in record["normalized_rejections"])
    assert "full_file_replacement_disallowed" in record["failure_categories"]
    assert record["failed_stage_package_summary"]


def test_trial_wrapper_reapplies_explicit_cli_values_after_preset(tmp_path: Path) -> None:
    experiment = _preset_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = {
        **_execution_plan(),
        "source_entrypoint": "src/train_forecast.py",
        "python_dependencies": [],
        "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}", "--train_eval_end", "20260511"],
    }

    wrapper = generate_trial_train_wrapper(
        _plan(trial_dir),
        experiment,
        trial_dir,
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )
    output_dir = trial_dir / "outputs" / "real_outputs"
    subprocess.run(
        [sys.executable, wrapper.as_posix(), "--output_dir", output_dir.as_posix(), "--train_eval_end", "20260511"],
        check=True,
        text=True,
    )

    assert "20260511" in (output_dir / "args.csv").read_text(encoding="utf-8")


def test_real_runner_auto_inherits_source_train_eval_end_arg(tmp_path: Path) -> None:
    experiment = _preset_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = {
        **_execution_plan(),
        "source_entrypoint": "src/train_forecast.py",
        "python_dependencies": [],
        "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}"],
        "source_evaluation_context": {
            "context_status": "resolved",
            "test_start": "2026-05-05",
            "test_end": "2026-05-11",
            "sample_count": 56074,
            "preserved_train_args": {"--train_eval_end": "20260511"},
        },
    }

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )

    assert "--train_eval_end" in status["train_command"]
    assert status["train_command"][status["train_command"].index("--train_eval_end") + 1] == "20260511"
    assert status["source_evaluation_context"]["test_end"] == "2026-05-11"


def test_llm_logged_eval_time_overrides_train_eval_end_from_test_window(tmp_path: Path) -> None:
    wrapper = tmp_path / "train.py"
    wrapper.write_text(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--train_eval_end")
parser.add_argument("--fixed_test_start")
parser.add_argument("--fixed_test_end")
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "source.log"
    log_path.write_text(
        "2026-05-14 15:00:26 | INFO | Loaded data date 2026-01-29\n"
        "2026-05-14 15:00:26 | INFO | T2 backtest summary | test_window=2026-05-05~2026-05-11\n",
        encoding="utf-8",
    )
    client = FakeAgent2Client(
        [
            "available: true\n"
            "confidence: high\n"
            "train_eval_start: ''\n"
            "train_eval_end: ''\n"
            "test_start: '2026-05-05'\n"
            "test_end: '2026-05-11'\n"
            "reason: test_window in T2 backtest summary\n"
        ]
    )

    command, normalizations = real_runner._apply_llm_logged_eval_time_overrides(
        ["python", wrapper.as_posix(), "--output_dir", "out", "--train_eval_end", "20260430"],
        wrapper,
        {},
        {"train_log_path": log_path.as_posix()},
        client,
    )

    assert command[command.index("--train_eval_end") + 1] == "20260511"
    assert command[command.index("--fixed_test_start") + 1] == "20260505"
    assert command[command.index("--fixed_test_end") + 1] == "20260511"
    assert normalizations[0]["source"] == "llm_logged_eval_time"
    assert normalizations[0]["value_source"] == "test_end"
    assert client.calls[0]["step"] == "DetermineLoggedEvalTime"
    assert client.calls[0]["timeout"] == real_runner.AGENT2_LOGGED_EVAL_TIME_TIMEOUT


def test_llm_logged_eval_time_overrides_train_eval_range(tmp_path: Path) -> None:
    wrapper = tmp_path / "train.py"
    wrapper.write_text(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--train_eval_start")
parser.add_argument("--train_eval_end")
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "source.log"
    log_path.write_text(
        "2026-05-14 15:00:26 | INFO | T2 backtest summary | train_eval_range=2026-01-29~2026-05-11\n",
        encoding="utf-8",
    )
    client = FakeAgent2Client(
        [
            "available: true\n"
            "confidence: high\n"
            "train_eval_start: '2026-01-29'\n"
            "train_eval_end: '2026-05-11'\n"
            "test_start: ''\n"
            "test_end: ''\n"
            "reason: train_eval_range in backtest summary\n"
        ]
    )

    command, normalizations = real_runner._apply_llm_logged_eval_time_overrides(
        ["python", wrapper.as_posix(), "--output_dir", "out"],
        wrapper,
        {},
        {"train_log_path": log_path.as_posix()},
        client,
    )

    assert command[command.index("--train_eval_start") + 1] == "20260129"
    assert command[command.index("--train_eval_end") + 1] == "20260511"
    assert [item["value_source"] for item in normalizations] == ["train_eval_start", "train_eval_end"]


def test_llm_logged_eval_time_low_confidence_keeps_command(tmp_path: Path) -> None:
    wrapper = tmp_path / "train.py"
    wrapper.write_text(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--train_eval_end")
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "source.log"
    log_path.write_text("test_window=2026-05-05~2026-05-11\n", encoding="utf-8")
    client = FakeAgent2Client(
        [
            "available: true\n"
            "confidence: low\n"
            "train_eval_start: ''\n"
            "train_eval_end: ''\n"
            "test_start: '2026-05-05'\n"
            "test_end: '2026-05-11'\n"
            "reason: ambiguous\n"
        ]
    )
    original = ["python", wrapper.as_posix(), "--output_dir", "out", "--train_eval_end", "20260430"]

    command, normalizations = real_runner._apply_llm_logged_eval_time_overrides(
        original,
        wrapper,
        {},
        {"train_log_path": log_path.as_posix()},
        client,
    )

    assert command == original
    assert normalizations == []


def test_llm_logged_eval_time_no_log_does_not_call_llm(tmp_path: Path) -> None:
    wrapper = tmp_path / "train.py"
    wrapper.write_text("import argparse\n", encoding="utf-8")
    client = FakeAgent2Client(["available: true\nconfidence: high\n"])
    original = ["python", wrapper.as_posix(), "--output_dir", "out"]

    command, normalizations = real_runner._apply_llm_logged_eval_time_overrides(
        original,
        wrapper,
        {},
        {},
        client,
    )

    assert command == original
    assert normalizations == []
    assert client.calls == []


def test_real_runner_skips_training_when_source_evaluation_context_unresolved(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    execution_plan = {
        **_execution_plan(),
        "source_evaluation_context": {
            "context_status": "unresolved",
            "unresolved_reason": "missing source test window",
        },
    }

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=UnavailableAgent2Client(),
        execution_plan=execution_plan,
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "evaluation_context_unresolved"
    assert status["failure_stage"] == "evaluation_context_unresolved"
    assert not (trial_dir / "code" / "train.py").exists()


def test_real_runner_rejects_agent2_split_data_change(tmp_path: Path) -> None:
    experiment = _feature_builder_experiment(tmp_path)
    source = experiment / "src" / "train_forecast.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "def build_features(row: dict) -> dict:\n",
            "def split_data(rows):\n    return rows\n\n\ndef build_features(row: dict) -> dict:\n",
        ),
        encoding="utf-8",
    )
    trial_dir = tmp_path / "trial_001"
    replacement = """
def split_data(rows):
    return list(reversed(rows))
"""
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: split_data\n",
            _edit_package(
                [
                    {
                        "path": "train.py",
                        "type": "replace_function",
                        "function": "split_data",
                        "content": replacement,
                    }
                ]
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert "evaluation_scope_modified" in record["failure_categories"]
    assert any("split_data" in item["reason"] for item in record["normalized_rejections"])


def test_real_runner_rejects_train_eval_end_default_change(tmp_path: Path) -> None:
    experiment = _preset_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    replacement = (experiment / "src" / "train_forecast.py").read_text(encoding="utf-8").replace(
        'default="default"',
        'default="20260430"',
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: parse_args\n",
            _edit_package(
                [
                    {
                        "path": "train.py",
                        "type": "replace_function",
                        "function": "parse_args",
                        "content": _function_source_from_module(replacement, "parse_args"),
                    }
                ]
            ),
        ]
    )
    execution_plan = {
        **_execution_plan(),
        "source_entrypoint": "src/train_forecast.py",
        "python_dependencies": [],
        "train_command": ["{python}", "{train_py}", "--output_dir", "{real_output_dir}", "--train_eval_end", "20260511"],
    }

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=execution_plan,
    )

    assert status["train_success"] is False
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert "evaluation_scope_modified" in record["failure_categories"]
    assert any("--train_eval_end" in item["reason"] for item in record["normalized_rejections"])


def test_real_runner_audit_accepts_traceable_feature_column_variable(tmp_path: Path) -> None:
    experiment = _feature_builder_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    replacement = """
def build_features(row: dict) -> dict:
    row["base"] = 1
    span = 14
    prefix = "new_" + "feature"
    col_name = f"{prefix}_{span}"
    row[col_name] = 1
    return row
"""
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: build_features\n",
            _edit_package(
                [
                    {
                        "path": "train.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": replacement,
                    }
                ]
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["feature_application_success"] is True
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    evidence = audit["features"][0]["evidence"]
    assert audit["success"] is True
    assert any(item["kind"] == "feature_column_constructed" and "row[col_name]" in item["snippet"] for item in evidence)
    output = (trial_dir / "outputs" / "real_outputs" / "generic_output.csv").read_text(encoding="utf-8")
    assert "95" in output


def test_feature_audit_scopes_declared_cli_to_effective_train_command(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse
from util import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--rolling_windows", nargs="+", type=int, default=[7, 14, 30])
    args = parser.parse_args()
    return build_features(args)
""",
        encoding="utf-8",
    )
    (code_dir / "util.py").write_text(
        """
def build_features(args):
    features = {}
    features["lifecycle_age_x_days_to_end"] = 1.0
    features["rolling_stat_lifecycle_interaction"] = len(args.rolling_windows)
    return features
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["util.py"], "code_generation_success": True}),
        encoding="utf-8",
    )
    plan = _lifecycle_interaction_plan(tmp_path / "trial_001")
    plan["changes"][0]["feature_name"] = "lifecycle_and_trend_guardrail_bundle"
    plan["changes"][0]["feature_type"] = "lifecycle_interaction"
    plan["changes"][0]["audit_tokens"] = ["lifecycle", "rolling_stat_lifecycle_interaction"]
    plan["changes"][0]["cli_args"] = [
        "--rolling_windows",
        "7",
        "14",
        "30",
        "--lag_windows",
        "1",
        "3",
        "7",
        "--enable_recent_trend_guardrail",
        "--enable_package_lifecycle_features",
    ]

    audit = _audit_feature_application(
        plan,
        code_dir,
        wrapper,
        train_command=[
            sys.executable,
            wrapper.as_posix(),
            "--output_dir",
            "out",
            "--rolling_windows",
            "7",
            "14",
            "30",
        ],
    )

    feature = audit["features"][0]
    assert audit["success"] is True
    assert feature["effective_cli_args"] == ["--rolling_windows", "7", "14", "30"]
    assert feature["ignored_intent_cli_args"] == [
        "--lag_windows",
        "--enable_recent_trend_guardrail",
        "--enable_package_lifecycle_features",
    ]
    assert feature["unparsed_cli_args"] == []
    assert any(item["kind"] != "cli_arg_consumed" for item in feature["evidence"])


def test_feature_audit_requires_non_cli_evidence_for_ignored_declared_cli(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--rolling_windows", nargs="+", type=int, default=[7, 14, 30])
    args = parser.parse_args()
    return args.rolling_windows
""",
        encoding="utf-8",
    )
    (code_dir / "agent2_code_modification.yaml").write_text(
        yaml.safe_dump({"modified_files": ["train.py"], "code_generation_success": True}),
        encoding="utf-8",
    )
    plan = _lifecycle_interaction_plan(tmp_path / "trial_001")
    plan["changes"][0]["feature_name"] = "lifecycle_and_trend_guardrail_bundle"
    plan["changes"][0]["feature_type"] = "lifecycle_interaction"
    plan["changes"][0]["cli_args"] = [
        "--rolling_windows",
        "7",
        "14",
        "30",
        "--lag_windows",
        "1",
        "3",
        "7",
        "--enable_recent_trend_guardrail",
    ]

    audit = _audit_feature_application(
        plan,
        code_dir,
        wrapper,
        train_command=[
            sys.executable,
            wrapper.as_posix(),
            "--output_dir",
            "out",
            "--rolling_windows",
            "7",
            "14",
            "30",
        ],
    )

    feature = audit["features"][0]
    assert audit["success"] is False
    assert feature["unparsed_cli_args"] == []
    assert feature["ignored_intent_cli_args"] == ["--lag_windows", "--enable_recent_trend_guardrail"]
    assert "no non-CLI executable feature evidence" in feature["failure_reason"]


def test_real_runner_records_audit_failure_without_blocking_successful_backtest(tmp_path: Path) -> None:
    experiment = _feature_builder_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    replacement = """
def build_features(row: dict) -> dict:
    # new_feature appears only as a comment and an unused variable.
    feature_col = f"new_feature_{14}"
    row["base"] = 1
    return row
"""
    edit = _edit_package(
        [
            {
                "path": "train.py",
                "type": "replace_function",
                "function": "build_features",
                "content": replacement,
            }
        ]
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: build_features\n",
            edit,
            edit,
            edit,
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["feature_application_success"] is False
    assert status["feature_application_audit_success"] is False
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is False
    assert audit["features"][0]["evidence"] == []
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "backtest_attempt" not in train_log


def test_real_runner_blocks_trial05_style_dependency_cli_noop_before_training(tmp_path: Path) -> None:
    experiment = _conditional_cli_dependency_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    edit = _edit_package(
        [
            {
                "path": "util.py",
                "type": "append_module_code",
                "content": "AGENT2_SMOKE = True\n",
            }
        ],
        ["left dependency feature gated by CLI without train.py args pass-through"],
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: util.py\n  function: build_features\n",
            edit,
            edit,
            edit,
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _conditional_cli_dependency_plan(trial_dir),
        llm_client=client,
        execution_plan=_conditional_cli_dependency_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["feature_application_success"] is False
    assert "conditional_cli_not_wired" in status["agent2_failure_categories"]
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is False
    assert audit["features"][0]["failure_category"] == "conditional_cli_not_wired"
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "backtest_attempt" not in train_log
    repair_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "RepairCodeEdits")
    assert "Pass args=args" in repair_prompt or "pass args=args" in repair_prompt


def test_real_runner_blocks_trial06_style_inactive_lifecycle_helper_before_training(tmp_path: Path) -> None:
    experiment = _lifecycle_interaction_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    replacement_main = """
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--enable_package_activity_features", action="store_true", default=False)
    args = parser.parse_args()
    df = pd.DataFrame({
        "entity_id": ["E001"],
        "date": pd.to_datetime(["2026-05-04"]),
        "actual_value": [100],
        "package_age_days": [5],
        "days_to_end": [10],
        "combo_for_psnnum": ["A"],
    })
    if args.enable_package_activity_features:
        df = add_enhanced_lifecycle_features(df, args)
    feat_df, feature_cols, _ = build_features(df, "actual_value", "date", ["entity_id"], [], [])
    has_lifecycle = "days_since_last_sale" in feature_cols and "days_since_first_appearance" in feature_cols and "is_dead" in feature_cols
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generic_output.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "entity_id", "actual_value", "forecast_value"])
        writer.writeheader()
        writer.writerow({"date": "2026-05-04", "entity_id": "E001", "actual_value": 100, "forecast_value": 95 if has_lifecycle else 50})
    return 0
"""
    helper_code = """
def add_enhanced_lifecycle_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()
    df["days_since_last_sale"] = 90
    df["days_since_first_appearance"] = 30
    df["is_dead"] = (df["days_since_last_sale"] > 14).astype(int)
    return df
"""
    edit = _edit_package(
        [
            {
                "path": "train.py",
                "type": "replace_function",
                "function": "main",
                "content": replacement_main,
            },
            {
                "path": "train.py",
                "type": "append_module_code",
                "content": helper_code,
            },
        ],
        ["lifecycle helper added but left behind a disabled feature gate"],
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: main\n",
            edit,
            edit,
            edit,
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _declared_lifecycle_death_plan(trial_dir),
        llm_client=client,
        execution_plan=_lifecycle_interaction_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["feature_application_success"] is False
    assert "inactive_feature_gate" in status["agent2_failure_categories"]
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "backtest_attempt" not in train_log
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert "inactive_feature_gate" in record["failure_categories"]
    assert any(item["category"] == "inactive_feature_gate" for item in record["normalized_rejections"])
    assert any(
        item.get("category") == "inactive_feature_gate"
        for entry in record["repair_guidance_history"]
        for item in entry.get("normalized_rejections", [])
    )
    assert any("inactive" in item.lower() or "activate" in item.lower() for item in record["required_corrections"])
    repair_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "RepairCodeEdits")
    assert "inactive" in repair_prompt.lower() or "activate" in repair_prompt.lower()


def test_real_runner_classifies_invalid_agent2_edit_type(tmp_path: Path) -> None:
    experiment = _feature_builder_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    invalid_edit = _edit_package(
        [
            {
                "path": "train.py",
                "function": "build_features",
                "content": "def build_features(row: dict) -> dict:\n    row['new_feature'] = 1\n    return row\n",
            }
        ]
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: build_features\n",
            invalid_edit,
            invalid_edit,
            invalid_edit,
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is False
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert "invalid_edit_type" in record["failure_categories"]
    assert any(item["category"] == "invalid_edit_type" for item in record["normalized_rejections"])
    repair_prompt = next(call["user_prompt"] for call in client.calls if call["step"] == "RepairCodeEdits")
    assert "Every edit item must include a supported type" in repair_prompt
    assert "add_feature_column_in_function" in repair_prompt
    assert "modify_function" in repair_prompt


def test_real_runner_overrides_declared_resource_cli_paths_to_trial_data(tmp_path: Path) -> None:
    experiment = _resource_path_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: main\n",
            _edit_package(
                [
                    {
                        "path": "train.py",
                        "type": "append_module_code",
                        "content": "RESOURCE_PATH_SMOKE = True\n",
                    }
                ]
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _resource_path_plan(trial_dir),
        llm_client=client,
        execution_plan=_resource_path_execution_plan(),
    )

    expected_data = (trial_dir / "data" / "input.csv").resolve().as_posix()
    assert status["train_success"] is True
    assert status["feature_application_success"] is True
    assert "--data_path" in status["train_command"]
    data_index = status["train_command"].index("--data_path")
    assert status["train_command"][data_index + 1] == expected_data
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "Data file not found" not in train_log


def test_train_command_validation_rejects_missing_declared_resource_path(tmp_path: Path) -> None:
    code_dir = tmp_path / "trial" / "code"
    data_dir = tmp_path / "trial" / "data"
    code_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    expected = data_dir / "input.csv"
    expected.write_text("ok\n", encoding="utf-8")
    wrapper = code_dir / "train.py"
    wrapper.write_text(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", "--data-path", dest="data_path", required=True)
parser.parse_args()
""",
        encoding="utf-8",
    )

    error = _train_command_validation_error(
        [sys.executable, wrapper.as_posix(), "--data_path", "data/input.csv"],
        wrapper,
        {"cli_resource_args": {"--data_path": expected.as_posix()}},
    )

    assert "declared resource argument --data_path points to missing path" in error
    assert expected.as_posix() in error


def test_real_runner_rejects_invalid_argparse_choice_before_training(tmp_path: Path) -> None:
    experiment = _choice_arg_experiment(tmp_path)
    trial_dir = tmp_path / "baseline_trial_06"
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: main\n",
            _edit_package(
                [
                    {
                        "path": "train.py",
                        "type": "replace_function",
                        "function": "main",
                        "content": _function_source_from_module(
                            (experiment / "src" / "train_forecast.py").read_text(encoding="utf-8"),
                            "main",
                        ),
                    }
                ],
                ["feature already wired in train.py"],
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_choice_arg_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "invalid_train_command"
    assert status["failure_stage"] == "train_command_validation"
    assert "invalid value 'baseline_trial_06' to --experiment" in status["error"]
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "train skipped because generated train_command is invalid" in train_log
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    command = record["final_train_command"]
    experiment_index = command.index("--experiment")
    assert command[experiment_index : experiment_index + 2] == ["--experiment", "baseline_trial_06"]


def test_real_runner_accepts_dependency_feature_reachable_from_train_py(tmp_path: Path) -> None:
    experiment = _reachable_dependency_feature_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    replacement = """
def build_features(row: dict) -> dict:
    row["base"] = 1
    feature_col = "new_feature_14"
    row[feature_col] = 1
    return row
"""
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: features.py\n  function: build_features\n",
            _edit_package(
                [
                    {
                        "path": "features.py",
                        "type": "replace_function",
                        "function": "build_features",
                        "content": replacement,
                    }
                ],
                ["train.py imports build_features from features.py"],
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_dependency_feature_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["feature_application_success"] is True
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["features"][0]["entrypoint_import_chain"]["reachable"] is True
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["modified_files"] == ["features.py"]
    assert record["train_py_unchanged_reason"]
    assert record["entrypoint_import_chain_checked"] is True
    assert record["final_train_command"]


def test_real_runner_accepts_narrow_agent2_feature_column_edit(tmp_path: Path) -> None:
    experiment = _feature_builder_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: train.py\n  function: build_features\n",
            _edit_package(
                [
                    {
                        "path": "train.py",
                        "type": "add_feature_column_in_function",
                        "function": "build_features",
                        "content": 'feature_col = "new_feature_14"\nrow[feature_col] = 1',
                    }
                ],
                ["small feature-column insertion"],
            ),
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_feature_builder_execution_plan(),
    )

    assert status["train_success"] is True
    assert status["feature_application_success"] is True
    train_text = (trial_dir / "code" / "train.py").read_text(encoding="utf-8")
    assert 'feature_col = "new_feature_14"' in train_text
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["accepted_schema"] == "edits"
    assert record["modified_files"] == ["train.py"]


def test_real_runner_records_unreachable_dependency_audit_without_blocking_successful_backtest(tmp_path: Path) -> None:
    experiment = _unreachable_dependency_feature_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    replacement = """
def build_features(row: dict) -> dict:
    row["base"] = 1
    row["new_feature_14"] = 1
    return row
"""
    edit = _edit_package(
        [
            {
                "path": "features.py",
                "type": "replace_function",
                "function": "build_features",
                "content": replacement,
            }
        ],
        ["features.py changed but train.py does not import it"],
    )
    client = FakeAgent2Client(
        [
            "change_plan:\n- path: features.py\n  function: build_features\n",
            edit,
            edit,
            edit,
        ]
    )

    status = run_real_experiment(
        experiment,
        trial_dir,
        _new_feature_plan(trial_dir),
        llm_client=client,
        execution_plan=_dependency_feature_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert status["feature_application_success"] is False
    assert status["feature_application_audit_success"] is False
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["code_generation_success"] is False
    assert "no_executable_evidence" in record["failure_categories"]
    audit = yaml.safe_load((trial_dir / "code" / "agent2_feature_application_audit.yaml").read_text(encoding="utf-8"))
    assert audit["success"] is False
    assert audit["features"][0]["applied"] is False
    assert any(call["step"] == "RepairCodeEdits" for call in client.calls)
    train_log = (trial_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "backtest_attempt" not in train_log


def test_real_runner_invalid_llm_code_package_does_not_train(tmp_path: Path) -> None:
    experiment = _generic_experiment(tmp_path)
    trial_dir = tmp_path / "trial_001"
    client = FakeAgent2Client(["change_plan: []\n", "not: a valid edits package"])

    status = run_real_experiment(
        experiment,
        trial_dir,
        _plan(trial_dir),
        llm_client=client,
        execution_plan=_execution_plan(),
    )

    assert status["train_success"] is False
    assert status["train_returncode"] == "agent2_code_generation_failed"
    assert status["failure_stage"] == "agent2_code_generation"
    assert not (trial_dir / "outputs" / "real_outputs" / "generic_output.csv").exists()
    record = yaml.safe_load((trial_dir / "code" / "agent2_code_modification.yaml").read_text(encoding="utf-8"))
    assert record["code_generation_success"] is False
    assert record["fallback_used"] is False


def test_no_runtime_import_references_package_predict_adapter(repo_root: Path) -> None:
    runtime_files = [*repo_root.glob("routepilot/**/*.py")]
    references = [
        path.as_posix()
        for path in runtime_files
        if "package_predict_adapter" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert references == []


def test_trial_header_is_inserted_after_docstring_and_future_import(tmp_path: Path) -> None:
    source = (
        '"""entrypoint docstring."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import argparse\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    return 0\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )
    experiment = tmp_path / "exp"
    experiment.mkdir()

    patched = real_runner._patch_trial_train_source(source, {}, {"changes": []}, experiment)

    compile(patched, "trial_train", "exec")
    assert ast.get_docstring(ast.parse(patched)) == "entrypoint docstring."
    lines = patched.splitlines()
    future_index = next(i for i, line in enumerate(lines) if line.startswith("from __future__ import "))
    header_index = next(
        i for i, line in enumerate(lines) if line.startswith("# Generated by RoutePilot Agent2")
    )
    assert future_index < header_index


def test_argparse_choice_flags_supports_custom_parser_variable_names() -> None:
    source = (
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        'ap.add_argument("--experiment", choices=["a", "b"], default="a")\n'
    )

    flags = real_runner._argparse_choice_flags(source)

    assert flags["--experiment"] == {"a", "b"}


def test_read_input_manifest_tolerates_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "input_manifest.json"
    path.write_text("{not valid json", encoding="utf-8")

    manifest = real_runner._read_input_manifest(path)

    assert manifest.get("manifest_error")

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_profile", "--run-profile", dest="run_profile", default="best_package")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--backtest_output_prefix", "--backtest-output-prefix", dest="prefix", required=True)
    parser.add_argument("--enable_package_distribution_features", "--enable-package-distribution-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_festival_features", "--enable-festival-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_festival_model_features", "--enable-festival-model-features", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uplift = 4.0 if args.enable_package_distribution_features else 0.0
    festival_uplift = 2.0 if args.enable_festival_features and args.enable_festival_model_features else 0.0
    rows = [
        {
            "split": "test",
            "ds": "2026-05-01",
            "store_code": "S001",
            "store_name": "上海一店",
            "package_dish_code": "P001",
            "package_dish_name": "双人套餐",
            "combo_for_psnnum": "2",
            "true_pos_cnt": 28.0,
            "pred_pos_cnt": 18.0 + uplift + festival_uplift,
            "is_holiday": 1,
            "is_high_peak_day": 1,
            "package_global_heat_rank_pct_14d": 0.95,
        },
        {
            "split": "test",
            "ds": "2026-05-02",
            "store_code": "S001",
            "store_name": "上海一店",
            "package_dish_code": "P002",
            "package_dish_name": "单人套餐",
            "combo_for_psnnum": "1",
            "true_pos_cnt": 10.0,
            "pred_pos_cnt": 9.0,
            "is_holiday": 0,
            "is_high_peak_day": 0,
            "package_global_heat_rank_pct_14d": 0.40,
        },
        {
            "split": "train",
            "ds": "2026-04-20",
            "store_code": "S001",
            "store_name": "上海一店",
            "package_dish_code": "P001",
            "package_dish_name": "双人套餐",
            "combo_for_psnnum": "2",
            "true_pos_cnt": 24.0,
            "pred_pos_cnt": 22.0,
            "is_holiday": 0,
            "is_high_peak_day": 0,
            "package_global_heat_rank_pct_14d": 0.90,
        },
    ]
    fieldnames = list(rows[0])
    detail_path = output_dir / f"{args.prefix}_package_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    slice_path = output_dir / f"{args.prefix}_package_slice_audit.csv"
    with slice_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice", "wape", "bias"])
        writer.writeheader()
        writer.writerow({"slice": "holiday_high_peak", "wape": 0.18, "bias": -0.12})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

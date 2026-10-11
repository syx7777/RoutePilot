"""Demo 预测实验：一个能跑、确定性的最小例子。

它读取 configs/base.yaml 里的特征开关，用最小二乘拟合一个合成目标，
并把预测值和真实值写到 outputs/prediction.csv 与 data/actual.csv。

目标 y 里含 x2*x3 的乘积项，所以：
- 不加交互特征时，线性模型拟合不好，WAPE 偏高；
- 加上 x2*x3 后，WAPE 明显下降 —— 正好用来演示 RoutePilot 的 keep。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="demo forecast experiment")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    features = config.get("features") or {}
    drop_columns = set(features.get("drop_columns") or [])
    use_interactions = bool(features.get("add_interactions"))
    interactions = features.get("interactions") or ["x2", "x3"]

    n_train, n_test = 300, 60
    n = n_train + n_test
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 2.0 * x1 + 3.0 * x2 * x3 + rng.normal(scale=0.05, size=n)

    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    if use_interactions:
        frame["_x2_x3"] = frame[interactions[0]] * frame[interactions[1]]
    frame = frame.drop(columns=[c for c in frame.columns if c in drop_columns])

    ds = np.arange(n)
    train_mask = ds < n_train
    test_mask = ~train_mask

    matrix = frame.to_numpy()
    train_x = np.column_stack([np.ones(int(train_mask.sum())), matrix[train_mask]])
    test_x = np.column_stack([np.ones(int(test_mask.sum())), matrix[test_mask]])
    coef, *_ = np.linalg.lstsq(train_x, y[train_mask], rcond=None)
    yhat = test_x @ coef

    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ds": ds[test_mask], "yhat": yhat}).to_csv(
        "outputs/prediction.csv", index=False
    )
    pd.DataFrame({"ds": ds[test_mask], "y": y[test_mask]}).to_csv(
        "data/actual.csv", index=False
    )
    print(f"wrote outputs/prediction.csv ({int(test_mask.sum())} rows)")


if __name__ == "__main__":
    main()
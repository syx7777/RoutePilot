from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_experiment(project_dir: str | Path, trial_dir: str | Path) -> dict[str, Any]:
    project = Path(project_dir)
    trial = Path(trial_dir)
    trial.mkdir(parents=True, exist_ok=True)

    train = subprocess.run(
        [sys.executable, (project / "benchmark" / "train.py").as_posix(), "--output", trial.as_posix()],
        cwd=project.as_posix(),
        text=True,
        capture_output=True,
    )
    (trial / "train.log").write_text(train.stdout + train.stderr, encoding="utf-8")

    eval_result = subprocess.run(
        [sys.executable, (project / "benchmark" / "evaluate.py").as_posix(), "--run-dir", trial.as_posix()],
        cwd=project.as_posix(),
        text=True,
        capture_output=True,
    )
    (trial / "eval.log").write_text(eval_result.stdout + eval_result.stderr, encoding="utf-8")

    status: dict[str, Any] = {
        "train_success": train.returncode == 0,
        "eval_success": eval_result.returncode == 0,
        "train_returncode": train.returncode,
        "eval_returncode": eval_result.returncode,
        "prediction_path": (trial / "prediction.csv").as_posix(),
    }
    (trial / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status

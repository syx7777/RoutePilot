from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def fixture_exp(tmp_path: Path, repo_root: Path) -> Path:
    source = repo_root / "tests" / "fixtures" / "combo_forecast_exp"
    target = tmp_path / "combo_forecast_exp"
    shutil.copytree(source, target)
    return target


@pytest.fixture()
def package_predict_exp(tmp_path: Path, repo_root: Path) -> Path:
    source = repo_root / "tests" / "fixtures" / "package_predict_exp"
    target = tmp_path / "package_predict_exp"
    shutil.copytree(source, target)
    return target

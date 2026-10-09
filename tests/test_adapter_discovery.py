from __future__ import annotations

from pathlib import Path

import yaml

from routepilot.adapter.discovery import draft_manifest
from routepilot.adapter.manifest import ProjectManifest

TRAIN_SOURCE = '''\
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    print(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _build_project(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / "outputs").mkdir(parents=True)

    (root / "configs" / "base.yaml").write_text("model:\n  n_estimators: 100\n", encoding="utf-8")
    (root / "src" / "train.py").write_text(TRAIN_SOURCE, encoding="utf-8")
    (root / "src" / "features.py").write_text("def build() -> None:\n    return None\n", encoding="utf-8")
    (root / "scripts" / "make_data.py").write_text("print('data')\n", encoding="utf-8")
    (root / "data" / "actual.csv").write_text("unique_id,ds,y\nS001,2024-01-01,10.0\n", encoding="utf-8")
    (root / "outputs" / "prediction.csv").write_text(
        "unique_id,ds,yhat\nS001,2024-01-01,9.5\n", encoding="utf-8"
    )
    return root


def test_draft_manifest_resolves_project_a_shape(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    draft = draft_manifest(root)
    manifest = ProjectManifest.model_validate(draft.manifest)

    assert manifest.project.entrypoint == "src/train.py"
    assert manifest.run.command == ["python", "src/train.py", "--config", "configs/base.yaml"]
    assert manifest.artifacts.prediction == "outputs/prediction.csv"
    assert manifest.artifacts.actual == "data/actual.csv"
    assert manifest.artifacts.columns.prediction == "yhat"
    assert manifest.artifacts.columns.actual == "y"
    assert manifest.artifacts.columns.date == "ds"
    assert manifest.artifacts.columns.id == ["unique_id"]
    assert "configs/*.yaml" in manifest.editable
    assert "src/features.py" in manifest.editable
    assert "data/**" in manifest.protected
    assert "scripts/**" in manifest.protected
    assert "src/train.py" in manifest.protected
    assert draft.confidence == 1.0
    assert draft.questions == []


def test_target_column_named_y_does_not_hijack_prediction_file(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    draft = draft_manifest(root)
    # prediction.csv 的列是 yhat，不能被 ACTUAL_HINTS 里的 "y" 误判为真实值文件。
    assert draft.manifest["artifacts"]["prediction"] == "outputs/prediction.csv"
    assert draft.manifest["artifacts"]["actual"] == "data/actual.csv"


def test_draft_ignores_routepilot_own_artifacts(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    (root / "routepilot.yaml").write_text(
        yaml.safe_dump({"project": {"name": "stale"}}), encoding="utf-8"
    )
    (root / "routepilot.discovery.json").write_text("{}", encoding="utf-8")

    draft = draft_manifest(root)
    # 自己的草稿不能被当成业务配置，否则 --config 会指向 routepilot.yaml。
    assert "routepilot.yaml" not in " ".join(draft.manifest["run"]["command"])
    assert draft.manifest["editable"] == ["configs/*.yaml", "src/features.py"]


def test_draft_prefers_config_dir_over_root_yaml(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    (root / "settings.yaml").write_text("model: {}\n", encoding="utf-8")
    draft = draft_manifest(root)
    assert draft.manifest["run"]["command"] == [
        "python",
        "src/train.py",
        "--config",
        "configs/base.yaml",
    ]


def test_draft_reports_questions_when_artifacts_missing(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text(TRAIN_SOURCE, encoding="utf-8")

    draft = draft_manifest(tmp_path)
    assert draft.confidence < 1.0
    assert any("prediction" in item for item in draft.questions)
    assert any("actual" in item for item in draft.questions)

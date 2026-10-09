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


def test_draft_ignores_runs_dir_written_by_routepilot(tmp_path: Path) -> None:
    """run 会在 <project>/runs/ 下留日志与 editable 存档，不能被当成可编辑面。"""
    root = _build_project(tmp_path)
    revert_dir = root / "runs" / "demo-1" / "revert" / "configs"
    revert_dir.mkdir(parents=True)
    (revert_dir / "base.yaml").write_text("model:\n  n_estimators: 1\n", encoding="utf-8")

    draft = draft_manifest(root)
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


def _build_wide_project(root: Path) -> Path:
    """宽表项目：主键是 row_id，没有时间列；outputs/ 里还有 AutoGluon 的排行榜。"""
    for name in ("configs", "src", "scripts", "data", "outputs"):
        (root / name).mkdir(parents=True, exist_ok=True)

    (root / "configs" / "base.yaml").write_text("model:\n  n_estimators: 100\n", encoding="utf-8")
    (root / "src" / "train.py").write_text(TRAIN_SOURCE, encoding="utf-8")
    (root / "src" / "features.py").write_text("def build() -> None:\n    return None\n", encoding="utf-8")
    (root / "scripts" / "make_data.py").write_text("print('data')\n", encoding="utf-8")
    (root / "data" / "actual.csv").write_text("row_id,y\nR000001,10.0\n", encoding="utf-8")
    (root / "outputs" / "prediction.csv").write_text("row_id,yhat\nR000001,9.5\n", encoding="utf-8")
    # 辅助产物：列名里带 "pred"，且文件名排序在 prediction.csv 之前
    (root / "outputs" / "leaderboard.csv").write_text(
        "model,score_val,pred_time_val\nWeightedEnsemble_L2,-1.0,0.02\n", encoding="utf-8"
    )
    return root


def test_draft_prefers_semantic_filename_over_incidental_column_hit(tmp_path: Path) -> None:
    """leaderboard.csv 不能因为 pred_time_val 命中 "pred" 就抢占真正的预测产物。"""
    root = _build_wide_project(tmp_path)
    draft = draft_manifest(root)
    manifest = ProjectManifest.model_validate(draft.manifest)

    assert manifest.artifacts.prediction == "outputs/prediction.csv"
    assert manifest.artifacts.columns.prediction == "yhat"
    assert manifest.artifacts.actual == "data/actual.csv"


def test_draft_recovers_underscore_id_primary_key(tmp_path: Path) -> None:
    """row_id 这类下划线后缀命名必须能被识别为主键（"id" 是短提示词）。"""
    root = _build_wide_project(tmp_path)
    draft = draft_manifest(root)
    manifest = ProjectManifest.model_validate(draft.manifest)

    assert manifest.artifacts.columns.id == ["row_id"]
    # 对齐键不再缺失 => 不该产生任何待人工确认项
    assert draft.questions == []
    assert draft.confidence == round(5 / 6, 2)


def test_draft_reports_question_when_artifact_candidates_tie(tmp_path: Path) -> None:
    root = _build_wide_project(tmp_path)
    (root / "outputs" / "prediction_backup.csv").write_text(
        "row_id,yhat\nR000001,9.5\n", encoding="utf-8"
    )

    draft = draft_manifest(root)
    assert any("并列" in item for item in draft.questions)
    # 并列时仍按路径确定性择一，不能随机
    assert draft.manifest["artifacts"]["prediction"] == "outputs/prediction.csv"

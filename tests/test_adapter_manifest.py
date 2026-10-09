from __future__ import annotations

from pathlib import Path

import pytest

from routepilot.adapter.manifest import (
    ProjectManifest,
    classify_path,
    load_manifest,
    matches_glob,
    resolve_project_root,
    validate_manifest,
)


def _manifest_dict(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "project": {"name": "demo", "root": ".", "entrypoint": "src/train.py"},
        "run": {"command": ["python", "src/train.py"]},
        "artifacts": {
            "prediction": "outputs/pred.csv",
            "actual": "data/test.csv",
            "columns": {"prediction": "yhat", "actual": "y", "date": "ds"},
        },
        "metrics": {"primary": {"name": "wape", "direction": "minimize", "min_delta": 0.005}},
        "editable": ["src/features.py", "configs/*.yaml"],
        "protected": ["data/**", "src/split.py"],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("rel_path", "pattern", "expected"),
    [
        ("configs/a.yaml", "configs/*.yaml", True),
        ("configs/sub/a.yaml", "configs/*.yaml", False),
        ("src/features.py", "src/features.py", True),
        ("src/other.py", "src/features.py", False),
        ("src/features.py", "*.py", True),
        ("data/test.csv", "data/**", True),
        ("data/nested/deep/test.csv", "data/**", True),
        ("src/split.py", "data/**", False),
    ],
)
def test_matches_glob_follows_filesystem_semantics(
    rel_path: str, pattern: str, expected: bool
) -> None:
    assert matches_glob(rel_path, pattern) is expected


def test_matches_glob_ignores_windows_separators() -> None:
    assert matches_glob("configs\\base.yaml", "configs/*.yaml") is True


def test_classify_path_prefers_protected() -> None:
    verdict = classify_path("data/test.csv", editable=["data/**"], protected=["data/**"])
    assert verdict == "protected"


def test_classify_path_marks_unlisted_files() -> None:
    assert classify_path("src/train.py", ["src/features.py"], ["data/**"]) == "unlisted"


def test_manifest_requires_at_least_one_editable_pattern() -> None:
    with pytest.raises(ValueError):
        ProjectManifest.model_validate(_manifest_dict(editable=[]))


def test_manifest_rejects_editable_shadowed_by_protected() -> None:
    with pytest.raises(ValueError) as excinfo:
        ProjectManifest.model_validate(
            _manifest_dict(editable=["data/**"], protected=["data/**"])
        )
    assert "shadowed" in str(excinfo.value)


def test_manifest_rejects_negative_min_delta() -> None:
    with pytest.raises(ValueError):
        ProjectManifest.model_validate(
            _manifest_dict(
                metrics={"primary": {"name": "wape", "direction": "minimize", "min_delta": -1}}
            )
        )


def test_manifest_rejects_shell_style_command_padding() -> None:
    with pytest.raises(ValueError):
        ProjectManifest.model_validate(
            _manifest_dict(run={"command": ["python src/train.py "]})
        )


def test_load_example_manifest(repo_root: Path) -> None:
    manifest_path = repo_root / "examples" / "routepilot.example.yaml"
    manifest = load_manifest(manifest_path)
    assert manifest.project.name == "m5-demand-forecast"
    assert manifest.metrics.primary.name == "wape"
    assert manifest.metrics.guards[0].name == "bias"
    assert resolve_project_root(manifest, manifest_path) == manifest_path.parent.resolve()


def test_validate_manifest_reports_missing_entrypoint(tmp_path: Path) -> None:
    manifest = ProjectManifest.model_validate(_manifest_dict())
    report = validate_manifest(manifest, tmp_path)
    assert report.ok is False
    assert any("entrypoint" in error for error in report.errors)


def test_validate_manifest_accepts_project_and_warns_on_missing_artifacts(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("print('ok')\n", encoding="utf-8")
    manifest = ProjectManifest.model_validate(_manifest_dict())
    report = validate_manifest(manifest, tmp_path)
    assert report.ok is True
    assert any("artifacts.prediction" in warning for warning in report.warnings)

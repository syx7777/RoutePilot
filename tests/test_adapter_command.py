from __future__ import annotations

import sys
from pathlib import Path

from routepilot.adapter.command import CommandProjectAdapter, build_adapter
from routepilot.adapter.manifest import ProjectManifest

_TRAIN_SCRIPT = (
    "from pathlib import Path\n"
    "Path('outputs').mkdir(exist_ok=True)\n"
    "Path('outputs/pred.csv').write_text('ds,yhat\\n2020-01-01,1.0\\n', encoding='utf-8')\n"
    "print('trained')\n"
)


def _build_project(root: Path, script: str = _TRAIN_SCRIPT) -> ProjectManifest:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "test.csv").write_text("ds,y\n2020-01-01,1.0\n", encoding="utf-8")
    (root / "src" / "features.py").write_text("FEATURE = 1\n", encoding="utf-8")
    (root / "src" / "train.py").write_text(script, encoding="utf-8")
    return ProjectManifest.model_validate(
        {
            "project": {"name": "demo", "root": ".", "entrypoint": "src/train.py"},
            "run": {"command": [sys.executable, "src/train.py"]},
            "artifacts": {
                "prediction": "outputs/pred.csv",
                "actual": "data/test.csv",
                "columns": {"prediction": "yhat", "actual": "y", "date": "ds"},
            },
            "metrics": {
                "primary": {"name": "wape", "direction": "minimize", "min_delta": 0.005}
            },
            "editable": ["src/features.py", "src/train.py", "configs/*.yaml"],
            "protected": ["data/**"],
        }
    )


def test_discover_reports_ready_for_valid_project(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path)
    discovery = build_adapter(manifest, tmp_path).discover()
    assert discovery.confidence == 1.0
    # 产物在 run 之前必然不存在，只能作为 warning 出现。
    assert discovery.warnings
    assert all("not found yet" in warning for warning in discovery.warnings)


def test_discover_reports_zero_confidence_when_entrypoint_missing(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path)
    (tmp_path / "src" / "train.py").unlink()
    discovery = build_adapter(manifest, tmp_path).discover()
    assert discovery.confidence == 0.0
    assert any("entrypoint" in warning for warning in discovery.warnings)


def test_run_executes_command_and_locates_artifacts(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path)
    adapter = CommandProjectAdapter(manifest, tmp_path)
    result = adapter.run()
    assert result.success is True
    assert result.returncode == 0
    assert result.prediction_path == str((tmp_path / "outputs" / "pred.csv").resolve())
    assert result.actual_path == str((tmp_path / "data" / "test.csv").resolve())
    assert "trained" in Path(result.stdout_path or "").read_text(encoding="utf-8")


def test_run_reports_failure_on_nonzero_exit(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path, script="import sys\nsys.exit(3)\n")
    result = CommandProjectAdapter(manifest, tmp_path).run()
    assert result.success is False
    assert result.returncode == 3
    assert result.prediction_path is None


def test_run_reports_timeout(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path, script="import time\ntime.sleep(30)\n")
    manifest.run.timeout_sec = 1
    result = CommandProjectAdapter(manifest, tmp_path).run()
    assert result.success is False
    assert "timed out" in result.error


def test_snapshot_and_rollback_restore_editable_surface(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path)
    adapter = CommandProjectAdapter(manifest, tmp_path)
    snapshot = adapter.snapshot()

    (tmp_path / "src" / "features.py").write_text("FEATURE = 999\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "new.yaml").write_text("added: true\n", encoding="utf-8")

    adapter.rollback(snapshot)

    assert (tmp_path / "src" / "features.py").read_text(encoding="utf-8") == "FEATURE = 1\n"
    assert (tmp_path / "configs" / "new.yaml").exists() is False


def test_snapshot_ignores_protected_files(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path)
    adapter = CommandProjectAdapter(manifest, tmp_path)
    assert "data/test.csv" not in adapter.snapshot().files


def test_resolve_command_uses_project_venv_for_bare_python(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path)
    manifest.run.command = ["python", "src/train.py"]
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    executable = scripts / "python.exe"
    executable.write_text("", encoding="utf-8")

    adapter = CommandProjectAdapter(manifest, tmp_path)
    resolved = adapter._resolve_command(list(manifest.run.command))

    assert resolved[0] == executable.as_posix()
    assert resolved[1:] == ["src/train.py"]


def test_resolve_command_keeps_explicit_interpreter_path(tmp_path: Path) -> None:
    manifest = _build_project(tmp_path)
    adapter = CommandProjectAdapter(manifest, tmp_path)
    resolved = adapter._resolve_command(list(manifest.run.command))
    assert resolved[0] == sys.executable

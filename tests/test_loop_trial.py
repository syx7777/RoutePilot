from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from routepilot.adapter.command import CommandProjectAdapter
from routepilot.adapter.manifest import ProjectManifest
from routepilot.loop.proposer import FileEdit, Proposal, ScriptedProposer
from routepilot.loop.report import write_report
from routepilot.loop.trial import run_optimization
from routepilot.profiler import PROJECT_RUN, Profiler

TRAIN_SOURCE = '''\
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    offset = int(config.get("prediction_offset", 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = ["unique_id,ds,yhat"]
    for day in range(1, 5):
        rows.append(f"S001,2024-01-0{day},{110 + offset}")
    (output / "prediction.csv").write_text("\\n".join(rows) + "\\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

ACTUAL_ROWS = "unique_id,ds,y\n" + "".join(
    f"S001,2024-01-0{day},100\n" for day in range(1, 5)
)


def _manifest(min_delta: float = 0.02) -> ProjectManifest:
    return ProjectManifest.model_validate(
        {
            "project": {"name": "toy", "root": ".", "entrypoint": "src/train.py"},
            "run": {
                "command": [
                    "python",
                    "src/train.py",
                    "--config",
                    "configs/model.yaml",
                    "--output_dir",
                    "outputs",
                ]
            },
            "artifacts": {
                "prediction": "outputs/prediction.csv",
                "actual": "data/actual.csv",
                "columns": {
                    "prediction": "yhat",
                    "actual": "y",
                    "date": "ds",
                    "id": ["unique_id"],
                },
            },
            "metrics": {
                "primary": {"name": "wape", "direction": "minimize", "min_delta": min_delta}
            },
            "editable": ["configs/*.yaml"],
            "protected": ["data/**", "src/train.py"],
        }
    )


def _build_project(root: Path, *, offset: int = 0) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    (root / "configs" / "model.yaml").write_text(
        f"prediction_offset: {offset}\n", encoding="utf-8"
    )
    (root / "data" / "actual.csv").write_text(ACTUAL_ROWS, encoding="utf-8")
    (root / "src" / "train.py").write_text(TRAIN_SOURCE, encoding="utf-8")
    return root


def _proposal(summary: str, offset: int, path: str = "configs/model.yaml") -> Proposal:
    return Proposal(
        summary=summary,
        edits=[FileEdit(path=path, content=f"prediction_offset: {offset}\n")],
    )


def _adapter(root: Path, manifest: ProjectManifest) -> CommandProjectAdapter:
    return CommandProjectAdapter(manifest, root, log_dir=root / ".routepilot_logs")


def _silent(_: str) -> None:
    return None


def test_keeps_improvement_and_persists_the_edit(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    proposer = ScriptedProposer([_proposal("小幅调整", -1), _proposal("明显改善", -6)])

    outcome = run_optimization(
        _adapter(root, manifest), manifest, goal="降低 WAPE", proposer=proposer, log=_silent
    )

    assert outcome.baseline_metrics["wape"] == pytest.approx(0.10)
    assert [record.decision for record in outcome.trials] == ["rollback", "keep"]
    assert outcome.kept_trials == [2]
    assert outcome.final_metrics["wape"] == pytest.approx(0.04)
    # keep 就地生效：配置文件应保留 trial 2 的内容
    assert (root / "configs" / "model.yaml").read_text(encoding="utf-8") == "prediction_offset: -6\n"


def test_rollback_restores_the_original_file(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    proposer = ScriptedProposer([_proposal("改善不足", -1)])

    outcome = run_optimization(
        _adapter(root, manifest), manifest, goal="降低 WAPE", proposer=proposer, log=_silent
    )

    assert [record.decision for record in outcome.trials] == ["rollback"]
    assert outcome.kept_trials == []
    assert outcome.final_metrics["wape"] == pytest.approx(0.10)
    assert (root / "configs" / "model.yaml").read_text(encoding="utf-8") == "prediction_offset: 0\n"


def test_rejects_proposal_outside_editable_surface(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    original_train = (root / "src" / "train.py").read_text(encoding="utf-8")
    proposer = ScriptedProposer([_proposal("改受保护文件", -6, path="src/train.py")])

    outcome = run_optimization(
        _adapter(root, manifest), manifest, goal="降低 WAPE", proposer=proposer, log=_silent
    )

    assert outcome.trials[0].decision == "rollback"
    assert "editable" in outcome.trials[0].reason
    assert (root / "src" / "train.py").read_text(encoding="utf-8") == original_train


def test_stops_when_proposer_is_exhausted(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    proposer = ScriptedProposer([])

    outcome = run_optimization(
        _adapter(root, manifest), manifest, goal="降低 WAPE", proposer=proposer, log=_silent
    )

    assert outcome.trials == []
    assert outcome.baseline_metrics["wape"] == pytest.approx(0.10)


def test_run_optimization_raises_when_project_is_not_ready(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    (root / "src" / "train.py").unlink()
    manifest = _manifest()

    with pytest.raises(RuntimeError, match="未通过接入检查"):
        run_optimization(
            _adapter(root, manifest),
            manifest,
            goal="降低 WAPE",
            proposer=ScriptedProposer([]),
            log=_silent,
        )


def test_write_report_records_revert_snapshot(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    proposer = ScriptedProposer([_proposal("明显改善", -6)])

    outcome = run_optimization(
        _adapter(root, manifest), manifest, goal="降低 WAPE", proposer=proposer, log=_silent
    )
    output_dir = tmp_path / "runs" / "demo"
    json_path, markdown_path = write_report(outcome, output_dir)

    assert json_path.is_file() and markdown_path.is_file()
    assert "trial" in markdown_path.read_text(encoding="utf-8")
    revert = output_dir / "revert" / "configs" / "model.yaml"
    assert revert.read_text(encoding="utf-8") == "prediction_offset: 0\n"


def test_max_trials_overrides_manifest_budget(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    proposer = ScriptedProposer(
        [_proposal("一", -1), _proposal("二", -2), _proposal("三", -3)]
    )

    outcome = run_optimization(
        _adapter(root, manifest),
        manifest,
        goal="降低 WAPE",
        proposer=proposer,
        max_trials=1,
        log=_silent,
    )

    assert len(outcome.trials) == 1


def test_report_json_contains_primary_improvement(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    proposer = ScriptedProposer([_proposal("明显改善", -6)])
    outcome = run_optimization(
        _adapter(root, manifest), manifest, goal="降低 WAPE", proposer=proposer, log=_silent
    )

    payload = yaml.safe_load(
        __import__("json").dumps(outcome.to_dict(), ensure_ascii=False)
    )
    assert payload["primary_improvement"] == pytest.approx(0.06)


def test_run_optimization_records_profile_spans(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()
    profiler = Profiler()

    outcome = run_optimization(
        _adapter(root, manifest),
        manifest,
        goal="降低 WAPE",
        proposer=ScriptedProposer([_proposal("明显改善", -6)]),
        profiler=profiler,
        log=_silent,
    )

    assert [span.name for span in profiler.spans] == ["baseline_run", "trial_1_run"]
    assert all(span.category == PROJECT_RUN for span in profiler.spans)
    assert outcome.profile["attribution"][PROJECT_RUN] > 0
    assert outcome.profile["findings"]


def test_run_optimization_without_profiler_leaves_profile_empty(tmp_path: Path) -> None:
    root = _build_project(tmp_path)
    manifest = _manifest()

    outcome = run_optimization(
        _adapter(root, manifest),
        manifest,
        goal="降低 WAPE",
        proposer=ScriptedProposer([]),
        log=_silent,
    )

    assert outcome.profile == {}

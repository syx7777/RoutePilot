from __future__ import annotations

import json
from pathlib import Path

from comboscope.runtime.trial_archive import archive_trial_files


def test_archive_trial_files_groups_key_artifacts(tmp_path: Path) -> None:
    trial = tmp_path / "trial_001"
    (trial / "data").mkdir(parents=True)
    (trial / "outputs" / "real_outputs").mkdir(parents=True)
    (trial / "logs").mkdir()
    (trial / "data" / "input_manifest.json").write_text("{}", encoding="utf-8")
    (trial / "outputs" / "real_outputs" / "trial_001_package_detail.csv").write_text("x\n", encoding="utf-8")
    (trial / "logs" / "train.log").write_text("ok\n", encoding="utf-8")
    (trial / "experiment_plan.yaml").write_text("trial_id: trial_001\n", encoding="utf-8")
    (trial / "final_report.md").write_text("# report\n", encoding="utf-8")

    archive_trial_files(trial)

    assert (trial / "archive" / "inputs" / "input_manifest.json").exists()
    assert (trial / "archive" / "outputs" / "real_outputs").exists()
    assert (trial / "archive" / "logs" / "train.log").exists()
    assert (trial / "archive" / "agent1" / "experiment_plan.yaml").exists()
    assert (trial / "archive" / "reports" / "final_report.md").exists()
    assert (trial / "agent1" / "experiment_plan.yaml").exists()
    assert (trial / "reports" / "final_report.md").exists()
    assert (trial / "final_report.md").exists()
    assert not (trial / "experiment_plan.yaml").exists()
    manifest = json.loads((trial / "archive" / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["groups"]["inputs"]

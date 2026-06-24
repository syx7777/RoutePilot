from __future__ import annotations

import json
from pathlib import Path

from comboscope.agents.report_agent import write_experiment_review, write_final_report


def test_report_agent_uses_chinese_reason_text(tmp_path: Path) -> None:
    comparison = {
        "decision": "keep",
        "reason": "wape improved enough and bias stayed within threshold",
        "wape_delta": -0.01,
        "bias_delta": -0.01,
    }
    run_status = {
        "train_success": True,
        "eval_success": True,
        "generated_train_path": "runs/trial_001/code/train.py",
        "real_output_dir": "runs/trial_001/outputs/real_outputs",
    }
    experiment_review = tmp_path / "experiment_review.md"
    forecast_report = tmp_path / "forecast_report.md"
    review_result = tmp_path / "review_result.json"
    final_report = tmp_path / "final_report.md"

    forecast_report.write_text("# 预测实验评测报告\n", encoding="utf-8")
    review_result.write_text(json.dumps(comparison, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "run_status.json").write_text(json.dumps(run_status, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "metric_comparison.json").write_text(
        json.dumps(
            {
                **comparison,
                "primary_metric_label": "WAPE",
                "old_primary": 0.62,
                "new_primary": 0.61,
                "primary_delta": -0.01,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"wape": 0.62, "bias": -0.1}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "new_metrics.json").write_text(json.dumps({"wape": 0.61, "bias": -0.11}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "experiment_plan.yaml").write_text(
        "changes:\n  - feature_name: combo_rolling_7d_mean\neditable_files:\n  - runs/trial_001/code/train.py\nexpected_effect: reduce WAPE\n",
        encoding="utf-8",
    )

    write_experiment_review(experiment_review, comparison, run_status)
    write_final_report(
        forecast_report_path=forecast_report,
        experiment_review_path=experiment_review,
        review_result_path=review_result,
        output_path=final_report,
    )

    text = final_report.read_text(encoding="utf-8")
    assert "ComboScope 实验验证结论报告" in text
    assert "本轮 `combo_rolling_7d_mean` 实验有效" in text
    assert "当前最佳方案切换为修改后方案" in text
    assert "wape improved enough" not in text


def test_experiment_review_explains_bad_result_reason(tmp_path: Path) -> None:
    comparison = {
        "decision": "rollback",
        "reason": "wape improvement is below threshold",
        "wape_delta": -0.001,
        "bias_delta": 0.0,
    }
    run_status = {"train_success": True, "eval_success": True}
    path = tmp_path / "experiment_review.md"

    write_experiment_review(path, comparison, run_status)

    text = path.read_text(encoding="utf-8")
    assert "效果不好原因" in text
    assert "WAPE 改善幅度未达到 keep 阈值" in text


def test_experiment_review_explains_unapplied_feature(tmp_path: Path) -> None:
    comparison = {
        "decision": "rollback",
        "reason": "feature change was not applied",
        "wape_delta": 0.0,
        "bias_delta": 0.0,
    }
    run_status = {
        "train_success": False,
        "eval_success": False,
        "feature_application_success": False,
        "feature_application_audit_path": "runs/trial_001/code/agent2_feature_application_audit.yaml",
    }
    path = tmp_path / "experiment_review.md"

    write_experiment_review(path, comparison, run_status)

    text = path.read_text(encoding="utf-8")
    assert "本轮特征改动未被真实训练代码消费" in text
    assert "特征应用审计未通过" in text
    assert "agent2_feature_application_audit.yaml" in text


def test_experiment_review_explains_agent2_codegen_failure(tmp_path: Path) -> None:
    comparison = {
        "decision": "rollback",
        "reason": "agent2 code generation failed",
        "wape_delta": 0.0,
        "bias_delta": 0.0,
    }
    run_status = {
        "train_success": False,
        "eval_success": False,
        "train_returncode": "agent2_code_generation_failed",
        "agent2_code_generation_success": False,
        "agent2_code_modification_path": "runs/trial_001/code/agent2_code_modification.yaml",
    }
    path = tmp_path / "experiment_review.md"

    write_experiment_review(path, comparison, run_status)

    text = path.read_text(encoding="utf-8")
    assert "Agent2 代码生成失败" in text
    assert "agent2_code_modification.yaml" in text


def test_final_report_explains_unapplied_feature(tmp_path: Path) -> None:
    comparison = {
        "decision": "rollback",
        "reason": "feature change was not applied",
        "wape_delta": 0.0,
        "bias_delta": 0.0,
    }
    run_status = {
        "train_success": False,
        "eval_success": False,
        "train_returncode": "feature_application_failed",
        "feature_application_success": False,
        "feature_application_audit_path": "runs/trial_001/code/agent2_feature_application_audit.yaml",
    }
    forecast_report = tmp_path / "forecast_report.md"
    experiment_review = tmp_path / "experiment_review.md"
    review_result = tmp_path / "review_result.json"
    final_report = tmp_path / "final_report.md"

    forecast_report.write_text("# 预测实验评测报告\n", encoding="utf-8")
    experiment_review.write_text("# 实验复核\n", encoding="utf-8")
    review_result.write_text(json.dumps(comparison, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "run_status.json").write_text(json.dumps(run_status, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "metric_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"wape": 0.62, "bias": -0.1}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "new_metrics.json").write_text(json.dumps({"wape": 0.62, "bias": -0.1}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "experiment_plan.yaml").write_text(
        "changes:\n  - feature_name: store_package_recent_trend_ratio\neditable_files:\n  - runs/trial_001/code/train.py\nexpected_effect: reduce WAPE\n",
        encoding="utf-8",
    )

    write_final_report(
        forecast_report_path=forecast_report,
        experiment_review_path=experiment_review,
        review_result_path=review_result,
        output_path=final_report,
    )

    text = final_report.read_text(encoding="utf-8")
    assert "特征应用审计未通过" in text
    assert "没有发现该特征被真实训练代码消费" in text
    assert "效果被证伪" in text

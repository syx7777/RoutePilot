from __future__ import annotations

from pathlib import Path

from comboscope.core.reports import build_final_report_context, write_final_report


def test_final_report_rollback_train_failed_is_decision_report(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Traceback (most recent call last):",
                "  File \"train.py\", line 1, in <module>",
                "TypeError: build_features() missing 1 required positional argument: 'target_col'",
            ]
        ),
        encoding="utf-8",
    )
    context = build_final_report_context(
        review_result={"decision": "rollback", "reason": "train or evaluation failed"},
        metric_comparison={"decision": "rollback", "primary_metric_label": "test_bias"},
        old_metrics={"wape": 0.62, "bias": -0.1205},
        new_metrics=None,
        problem_context={
            "main_problem": "systematic_bias",
            "evidence": ["整体低估", "weekend;normal_target;overestimate;not_holiday 较差"],
        },
        experiment_plan={
            "changes": [
                {
                    "feature_name": "festival_model_features",
                    "cli_args": ["--enable-festival-features", "--enable-festival-model-features"],
                }
            ],
            "editable_files": ["runs/trial_001/code/train.py"],
            "expected_effect": "改善 test_bias",
            "target_problem": "降低 test_bias",
        },
        run_status={"train_success": False, "eval_success": False, "train_returncode": 2, "train_log_path": train_log.as_posix()},
        artifacts={"trace": "trace.jsonl"},
    )
    output = tmp_path / "final_report.md"

    write_final_report(context, output)

    text = output.read_text(encoding="utf-8")
    assert "# ComboScope 实验验证结论报告" in text
    assert "本轮 `festival_model_features` 实验无效" in text
    assert "当前最佳方案仍然是 baseline" in text
    assert "训练失败" in text
    assert "TypeError: build_features() missing 1 required positional argument: 'target_col'" in text
    assert "函数签名" in text
    assert "baseline 当前仍可优化的方向" in text
    assert "## 2. 任务理解" not in text
    assert "## 3. 实验目录扫描结果" not in text
    assert "## 4. 日志解析结果" not in text
    assert "## 6. 评测数据可用性" not in text
    assert "better than baseline" not in text
    assert "worse than baseline" not in text
    assert "优于 baseline" not in text
    assert "低于 baseline" not in text


def test_final_report_keep_uses_modified_solution_template(tmp_path: Path) -> None:
    context = build_final_report_context(
        review_result={"decision": "keep", "reason": "llm objective improved"},
        metric_comparison={
            "decision": "keep",
            "primary_metric_label": "WAPE",
            "old_primary": 0.6217,
            "new_primary": 0.6082,
            "primary_delta": -0.0135,
            "wape_delta": -0.0135,
            "bias_delta": 0.001,
        },
        old_metrics={"wape": 0.6217, "bias": -0.12},
        new_metrics={"wape": 0.6082, "bias": -0.119},
        problem_context={"main_problem": "high_sales_underestimate", "evidence": ["高目标值低估"]},
        experiment_plan={
            "changes": [{"feature_name": "combo_rolling_7d_mean"}],
            "editable_files": ["runs/trial_001/code/train.py"],
            "expected_effect": "降低高销量低估",
            "target_problem": "high_sales_underestimate",
        },
        run_status={"train_success": True, "eval_success": True, "train_returncode": 0},
        artifacts={"forecast_report": "forecast_report.md"},
    )
    output = tmp_path / "final_report.md"

    write_final_report(context, output)

    text = output.read_text(encoding="utf-8")
    assert "本轮 `combo_rolling_7d_mean` 实验有效" in text
    assert "当前最佳方案切换为修改后方案" in text
    assert "## 3. 本轮修改了什么" in text
    assert "## 4. 效果改善在哪里" in text
    assert "## 6. 下一步优化建议" in text


def test_final_report_test_bias_failure_prioritizes_primary_metric(tmp_path: Path) -> None:
    context = build_final_report_context(
        review_result={"decision": "rollback", "reason": "test_bias未降低：0.621684 -> 0.700871"},
        metric_comparison={
            "decision": "rollback",
            "primary_metric_label": "test_bias",
            "old_primary": 0.621684,
            "new_primary": 0.700871,
            "primary_delta": 0.079187,
            "wape_delta": 0.079187,
            "bias_delta": 0.05,
        },
        old_metrics={"wape": 0.62, "bias": -0.12},
        new_metrics={"wape": 0.70, "bias": 0.17},
        problem_context={"main_problem": "systematic_bias", "evidence": ["test_bias 未收敛"]},
        experiment_plan={
            "changes": [{"feature_name": "bias_calibration_group"}],
            "editable_files": ["runs/trial_001/code/train.py"],
            "expected_effect": "降低 test_bias",
            "target_problem": "systematic_bias",
        },
        run_status={"train_success": True, "eval_success": True, "train_returncode": 0},
        artifacts={"forecast_report": "forecast_report.md"},
    )
    output = tmp_path / "final_report.md"

    write_final_report(context, output)

    text = output.read_text(encoding="utf-8")
    assert "test_bias" in text
    assert "0.621684" in text
    assert "0.700871" in text
    assert "test_bias 从 0.120456 变为 0.169484" not in text
    assert "修改后 test Bias" in text


def test_final_report_wape_failure_uses_comparison_wape_bias_and_sample_count(tmp_path: Path) -> None:
    context = build_final_report_context(
        review_result={"decision": "rollback", "reason": "wape improvement is below threshold"},
        metric_comparison={
            "decision": "rollback",
            "reason": "wape improvement is below threshold",
            "wape_delta": 0.040354889906664915,
            "bias_delta": 0.006584179696865283,
            "old_wape": 0.6216838718871088,
            "new_wape": 0.6620387617937737,
            "old_bias": -0.12045647275304211,
            "new_bias": 0.1270406524499074,
        },
        old_metrics={"wape": 0.6216838718871088, "bias": -0.12045647275304211, "sample_count": 56074},
        new_metrics={"wape": 0.6620387617937737, "bias": 0.1270406524499074, "sample_count": 57406},
        problem_context={"main_problem": "high_target_underestimate", "evidence": ["overall_wape=0.6216838718871088"]},
        experiment_plan={
            "changes": [{"feature_name": "extend_rolling_windows_14_30"}],
            "expected_effect": "validate whether extend_rolling_windows_14_30 improves the observed forecast error pattern",
            "target_problem": "high_target_underestimate",
        },
        run_status={"train_success": True, "eval_success": True, "train_returncode": 0, "eval_returncode": 0},
        artifacts={"forecast_report": "forecast_report.md"},
    )
    output = tmp_path / "final_report.md"

    write_final_report(context, output)

    text = output.read_text(encoding="utf-8")
    assert "n/a -> n/a" not in text
    assert "主指标 WAPE：0.621684 -> 0.662039" in text
    assert "升高 0.0403549" in text
    assert "越低越好的 WAPE 变差" in text
    assert "修改后 test Bias：0.127041" in text
    assert "Bias 方向变化：整体低估 -> 整体高估" in text
    assert "口径风险：评测样本数不一致，56074 -> 57406" in text


def test_final_report_agent2_codegen_failure_includes_categories_and_rejections(tmp_path: Path) -> None:
    context = build_final_report_context(
        review_result={"decision": "rollback", "reason": "agent2 code generation failed"},
        metric_comparison={"decision": "rollback", "primary_metric_label": "WAPE"},
        old_metrics={"wape": 0.6217, "bias": -0.12},
        new_metrics=None,
        problem_context={"main_problem": "${context.should_not_leak}"},
        experiment_plan={
            "changes": [{"feature_name": "zero_inflation_recent_zero_count"}],
            "editable_files": ["runs/trial_001/code/train.py", "runs/trial_001/code/util.py"],
            "expected_effect": "减少零销量高估",
        },
        run_status={
            "train_success": False,
            "eval_success": False,
            "train_returncode": "agent2_code_generation_failed",
            "agent2_code_generation_success": False,
            "agent2_code_generation_failure_reason": "generated code failed runtime call contract validation",
            "agent2_failure_categories": ["smoke_validation_failed", "invalid_package", "runtime_contract_error"],
            "agent2_normalized_rejections": [
                {
                    "path": "util.py",
                    "category": "smoke_validation_failed",
                    "reason": "util.py::build_features smoke failed: ModuleNotFoundError: No module named 'lightgbm'",
                },
                {"path": "util.py", "category": "invalid_package", "reason": "content is empty or not a string"},
                {
                    "path": "util.py",
                    "category": "runtime_contract_error",
                    "reason": "util.py:296 call infer_categorical_features missing required parameter exclude_cols",
                },
            ],
            "agent2_code_modification_path": "code/agent2_code_modification.yaml",
            "agent2_source_locator_path": "code/agent2_source_locator.yaml",
            "agent2_modified_files": [],
        },
        artifacts={"agent2_code_modification": "code/agent2_code_modification.yaml"},
    )
    output = tmp_path / "final_report.md"

    write_final_report(context, output)

    text = output.read_text(encoding="utf-8")
    assert "smoke_validation_failed、invalid_package、runtime_contract_error" in text
    assert "infer_categorical_features missing required parameter exclude_cols" in text
    assert "未产生有效 trial 修改" in text
    assert "${" not in text

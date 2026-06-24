from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_writer(repo_root: Path):
    path = repo_root / "skills" / "forecast-report-writer" / "scripts" / "write_report.py"
    spec = importlib.util.spec_from_file_location("forecast_report_writer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_writer_uses_business_chinese_explanations(repo_root: Path) -> None:
    writer = _load_writer(repo_root)
    context = {
        "mode": "full",
        "task": {"ask": "分析套餐预测误差"},
        "scan_result": {
            "code_files": ["src/lgb_package_to_dish_online_0319.py", "src/t2_components.py"],
            "config_files": ["README.md"],
            "log_files": [],
            "data_files": ["outputs/trial_001_package_detail.csv"],
            "candidate_prediction_files": ["outputs/trial_001_package_detail.csv"],
            "actual_files": [],
            "metrics_files": [],
            "domain_artifact_files": ["outputs/trial_001_package_detail.csv"],
            "possible_entrypoints": ["src/lgb_package_to_dish_online_0319.py"],
            "scan_strategy": "focused_dir_and_header_scan",
            "requested_dir": "/tmp/exp/src",
            "experiment_dir": "/tmp/exp",
            "fallback_used": True,
            "scanned_dirs": [".", "src", "outputs"],
            "warnings": ["log file not found"],
        },
        "artifact_summary": {
            "prediction_path": "runs/trial_001/standardized/prediction.csv",
            "actual_path": "runs/trial_001/standardized/actual.csv",
            "missing_artifacts": [],
            "ambiguous_artifacts": {},
        },
        "code_analysis": {
            "entrypoints": ["src/lgb_package_to_dish_online_0319.py"],
            "metric_functions": ["wape", "bias"],
        },
        "log_summary": {"available": False, "status": "unavailable", "notes": ["未传入训练日志"]},
        "metrics": [
            {"metric": "wape", "value": "0.54"},
            {"metric": "bias", "value": "-0.10"},
        ],
        "scene_metrics": [
            {"scene": "weekend;high_sales;underestimate", "rows": "10", "wape": "0.4", "bias": "-0.4"}
        ],
        "badcases": [
            {"badcase_type": "top_abs_error", "actual": "85", "prediction": "10", "abs_error": "75"}
        ],
        "anomaly_summary": {"anomalies": [{"type": "systematic_bias"}]},
        "experiment_plan": {
            "source_entrypoint": "src/lgb_package_to_dish_online_0319.py",
            "changes": [{"feature_name": "festival_model_features", "cli_args": ["--enable-festival-model-features"]}],
        },
        "run_status": {
            "generated_train_path": "runs/trial_001/code/train.py",
            "data_manifest_path": "runs/trial_001/data/input_manifest.json",
            "real_output_dir": "runs/trial_001/outputs/real_outputs",
        },
        "metric_comparison": {"decision": "rollback", "wape_delta": 0.01, "bias_delta": 0.05},
    }

    report = writer.build_template_report(context)

    assert "主要问题定位" in report
    assert "不是模型能力不可用" in report
    assert "扫描文件样例" in report
    assert "扫描策略" in report
    assert "业务产物候选" in report
    assert "训练/评估入口候选" in report
    assert "扫描缺失提醒" in report
    assert "WAPE（主指标" in report
    assert "Bias（整体偏差" in report
    assert "场景标签说明" in report
    assert "badcase 是误差最大的样本" in report
    assert "原实验入口" in report
    assert "trial 训练副本" in report
    assert "中文结论" in report
    assert "运行模式：" not in report.split("## 1. 一句话结论", 1)[1].split("## 2.", 1)[0]


def test_optimization_suggestions_include_specific_feature_experiments(repo_root: Path) -> None:
    path = repo_root / "skills" / "forecast-optimization-advisor" / "scripts" / "write_suggestions.py"
    spec = importlib.util.spec_from_file_location("forecast_suggestion_writer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = {
        "metrics": [
            {"metric": "wape", "value": "0.6216838718871088"},
            {"metric": "bias", "value": "-0.12045647275304211"},
        ],
        "scene_metrics": [
            {
                "scene": "weekend;normal_target;overestimate;not_holiday",
                "rows": "2986",
                "actual_sum": "5755",
                "abs_error_sum": "6852",
                "wape": "1.1905307315120148",
                "bias": "1.1905307315120148",
            },
            {
                "scene": "weekend;high_target;underestimate;not_holiday",
                "rows": "5187",
                "actual_sum": "81399",
                "abs_error_sum": "41998",
                "wape": "0.5158361191193029",
                "bias": "-0.5158361191193029",
            }
        ],
        "code_analysis": {
            "function_defs": ["build_features", "generate_rolling_features", "add_package_lifecycle_features", "fit_group_calibrator"],
            "code_identifiers": ["days_to_end"],
            "argparse_args": ["--rolling_windows", "--enable_group_calibration"],
        },
        "badcases": [
            {
                "badcase_type": "top_abs_error",
                "package_dish_name": "捞满爱意双人餐",
                "package_age_days": "2",
                "days_to_end": "11",
                "actual": "64",
                "prediction": "3.7",
            }
        ],
        "anomaly_summary": {"anomalies": [{"type": "systematic_bias"}]},
    }

    suggestions = module.render_markdown(module.build_suggestions(context))

    assert "extend_rolling_windows_14_30" in suggestions
    assert "rolling_windows -> build_features -> generate_rolling_features" in suggestions
    assert "priority_scene_by_abs_error=weekend;high_target;underestimate" in suggestions
    assert "store_package_rolling_7d_14d_30d_mean" not in suggestions
    assert "package_lifecycle_bucket" not in suggestions
    assert "days_to_end_bucket" not in suggestions
    assert "source_aligned_group_calibration" in suggestions
    assert "围绕对应场景和 badcase 做特征实验" not in suggestions
    assert "拆解高误差来源" not in suggestions
    assert "复核系统性偏差" not in suggestions
    assert "按场景复核误差" not in suggestions


def test_report_writer_embeds_specific_feature_experiments(repo_root: Path) -> None:
    writer = _load_writer(repo_root)
    context = {
        "mode": "full",
        "task": {"ask": "优化预测模型，目标是降低 test_bias"},
        "scan_result": {},
        "artifact_summary": {},
        "log_summary": {},
        "metrics": [
            {"metric": "wape", "value": "0.6216838718871088"},
            {"metric": "bias", "value": "-0.12045647275304211"},
        ],
        "scene_metrics": [
            {
                "scene": "weekend;normal_target;overestimate;not_holiday",
                "rows": "2986",
                "actual_sum": "5755",
                "abs_error_sum": "6852",
                "wape": "1.1905307315120148",
                "bias": "1.1905307315120148",
            },
            {
                "scene": "weekend;high_target;underestimate;not_holiday",
                "rows": "5187",
                "actual_sum": "81399",
                "abs_error_sum": "41998",
                "wape": "0.5158361191193029",
                "bias": "-0.5158361191193029",
            }
        ],
        "code_analysis": {
            "function_defs": ["build_features", "generate_rolling_features", "add_package_lifecycle_features"],
            "code_identifiers": ["days_to_end"],
            "argparse_args": ["--rolling_windows"],
        },
        "badcases": [
            {
                "badcase_type": "top_abs_error",
                "package_dish_name": "捞满爱意双人餐",
                "package_age_days": "2",
                "days_to_end": "11",
                "actual": "64",
                "prediction": "3.7",
            }
        ],
        "anomaly_summary": {"anomalies": [{"type": "systematic_bias"}]},
        "experiment_plan": {
            "evaluation_metric": {
                "objective_label": "test_bias",
                "decision_metric": "package_test_t2_bias_rate",
                "direction": "minimize_abs",
                "metric_formula": "sum(pred - true) / sum(true)",
                "metric_definition_source": ["src/evaluate.py:42"],
            }
        },
    }

    report = writer.build_template_report(context)

    assert "## 10. 优化建议" in report
    assert "test_bias` 已由 Agent1 根据源码解析为 `package_test_t2_bias_rate" in report
    assert "test_bias（业务口径，对应 WAPE/整体加权绝对误差，越低越好）" not in report
    assert "主要问题集中在 `weekend;high_target;underestimate;not_holiday`" in report
    assert "extend_rolling_windows_14_30" in report
    assert "rolling_windows -> build_features -> generate_rolling_features" in report
    assert "store_package_rolling_7d_14d_30d_mean" not in report
    assert "package_lifecycle_bucket" not in report
    assert "days_to_end_bucket" not in report
    assert "围绕对应场景和 badcase 做特征实验" not in report
    assert "优先围绕 `" not in report


def test_report_writer_separates_baseline_and_trial_metrics(repo_root: Path) -> None:
    writer = _load_writer(repo_root)
    context = {
        "mode": "full",
        "task": {"ask": "项目地址 /data/zhangxiaotian/package_predict/baseline，预测误差效果需要优化"},
        "scan_result": {"warnings": ["log file not found"]},
        "artifact_summary": {
            "prediction_path": "runs/baseline_trial_005/standardized_prediction.csv",
            "actual_path": "runs/baseline_trial_005/standardized_actual.csv",
            "missing_artifacts": [],
            "ambiguous_artifacts": {},
        },
        "code_analysis": {},
        "log_summary": {"available": False},
        "metrics": [
            {"metric": "wape", "value": "0.6216838718871088"},
            {"metric": "mape", "value": "0.7601131345513272"},
            {"metric": "bias", "value": "-0.12045647275304211"},
            {"metric": "mae", "value": "2.4123316557505543"},
            {"metric": "rmse", "value": "4.412003234695309"},
            {"metric": "rows", "value": "56074"},
        ],
        "new_metrics": {
            "wape": 0.6620387617937737,
            "mape": 0.7520281038063625,
            "bias": 0.1270406524499074,
            "mae": 3.800766307473965,
            "rmse": 7.20962010175636,
            "sample_count": 57406,
        },
        "scene_metrics": [
            {"scene": "weekend;high_target;underestimate;not_holiday", "rows": "5187", "abs_error_sum": "41998.34514645541", "wape": "0.5158361191193029", "bias": "-0.5158361191193029"}
        ],
        "badcases": [{"badcase_type": "top_abs_error", "actual": "85", "prediction": "13.5", "abs_error": "71.5"}],
        "anomaly_summary": {},
        "experiment_plan": {
            "source_entrypoint": "src/lgb_package_to_dish_online_0319.py",
            "changes": [{"feature_name": "extend_rolling_windows_14_30", "cli_args": ["--rolling_windows", "3", "7", "14", "30"]}],
        },
        "run_status": {
            "generated_train_path": "runs/baseline_trial_005/code/train.py",
            "train_log_path": "runs/baseline_trial_005/logs/train.log",
        },
        "metric_comparison": {
            "decision": "rollback",
            "reason": "wape improvement is below threshold",
            "wape_delta": 0.040354889906664915,
            "bias_delta": 0.006584179696865283,
            "old_wape": 0.6216838718871088,
            "new_wape": 0.6620387617937737,
            "old_bias": -0.12045647275304211,
            "new_bias": 0.1270406524499074,
        },
    }

    report = writer.build_template_report(context)

    assert "baseline 原实验 vs baseline_trial_005 修改后实验" in report
    assert "| WAPE | 0.621684 | 0.662039 | +0.0403549 | 越低越好的 WAPE 升高，效果变差 |" in report
    assert "| Bias | -0.120456 | 0.127041 | +0.00658418 | 方向：整体低估 -> 整体高估；绝对 Bias 变化 +0.00658418 |" in report
    assert "口径风险：baseline 与 trial 参与评测样本数不一致，56074 -> 57406" in report
    assert "本节场景拆解来自 baseline 原始评测产物" in report
    assert "本节 badcase 来自 baseline 原始评测产物" in report
    assert "用户指定优化目标" not in report


def test_report_writer_does_not_show_trial_metrics_when_train_failed(repo_root: Path, tmp_path: Path) -> None:
    writer = _load_writer(repo_root)
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Traceback (most recent call last):",
                "  File \"train.py\", line 1332, in run_t2_package_backtest",
                "TypeError: build_features() missing 1 required positional argument: 'target_col'",
            ]
        ),
        encoding="utf-8",
    )
    context = {
        "mode": "full",
        "task": {"ask": "排查 trial 训练失败"},
        "scan_result": {},
        "artifact_summary": {},
        "code_analysis": {},
        "log_summary": {"available": False},
        "metrics": [
            {"metric": "wape", "value": "0.6216838718871088"},
            {"metric": "bias", "value": "-0.12045647275304211"},
            {"metric": "rows", "value": "56074"},
        ],
        "new_metrics": {
            "wape": 0.6216838718871088,
            "bias": -0.12045647275304211,
            "sample_count": 56074,
        },
        "scene_metrics": [],
        "badcases": [],
        "anomaly_summary": {},
        "experiment_plan": {"changes": [{"feature_name": "days_to_end_bucket"}]},
        "run_status": {
            "train_success": False,
            "eval_success": False,
            "train_returncode": 1,
            "train_log_path": train_log.as_posix(),
        },
        "metric_comparison": {
            "decision": "rollback",
            "reason": "train or evaluation failed",
            "wape_delta": 0.0,
            "bias_delta": 0.0,
            "old_wape": 0.6216838718871088,
            "new_wape": 0.6216838718871088,
            "old_bias": -0.12045647275304211,
            "new_bias": -0.12045647275304211,
        },
    }

    report = writer.build_template_report(context)

    assert "执行状态与可用指标" in report
    assert "未产生有效 trial 指标" in report
    assert "TypeError: build_features() missing 1 required positional argument: 'target_col'" in report
    assert "carry-forward/fallback" in report
    assert "baseline 原始评测细项" in report
    assert "trial 修改后评测细项" not in report
    assert "| WAPE | 0.621684 | 0.621684 | 0 | WAPE 持平 |" not in report


def test_report_writer_surfaces_feature_application_audit_failure(repo_root: Path) -> None:
    writer = _load_writer(repo_root)
    context = {
        "mode": "full",
        "task": {"ask": "验证 store_package_recent_trend_ratio"},
        "scan_result": {},
        "artifact_summary": {},
        "code_analysis": {},
        "log_summary": {},
        "metrics": [{"metric": "wape", "value": "0.62"}, {"metric": "bias", "value": "-0.10"}],
        "scene_metrics": [{"scene": "weekend;high_target;underestimate", "rows": "10", "wape": "0.4", "bias": "-0.4"}],
        "badcases": [],
        "anomaly_summary": {},
        "experiment_plan": {
            "source_entrypoint": "src/lgb_package_to_dish_online_0319.py",
            "changes": [{"feature_name": "store_package_recent_trend_ratio", "cli_args": ["--enable-store-package-recent-trend-ratio"]}],
        },
        "run_status": {
            "generated_train_path": "runs/trial_001/code/train.py",
            "data_manifest_path": "runs/trial_001/data/input_manifest.json",
            "real_output_dir": "runs/trial_001/outputs/real_outputs",
            "feature_application_success": False,
            "feature_application_audit_path": "runs/trial_001/code/agent2_feature_application_audit.yaml",
        },
        "feature_application_audit": {
            "success": False,
            "features": [{"feature_name": "store_package_recent_trend_ratio", "applied": False}],
        },
        "metric_comparison": {"decision": "rollback", "reason": "feature change was not applied"},
    }

    report = writer.build_template_report(context)

    assert "特征应用审计未通过" in report
    assert "未落地特征" in report
    assert "Agent2 未找到本轮特征被真实训练代码消费的证据" in report
    assert "不建议保留该特征改动" not in report


def test_report_writer_surfaces_agent2_codegen_failure(repo_root: Path) -> None:
    writer = _load_writer(repo_root)
    context = {
        "mode": "full",
        "task": {"ask": "验证 store_package_rolling_7d_14d_30d_mean"},
        "scan_result": {},
        "artifact_summary": {},
        "code_analysis": {},
        "log_summary": {},
        "metrics": [{"metric": "wape", "value": "0.62"}, {"metric": "bias", "value": "-0.10"}],
        "scene_metrics": [],
        "badcases": [],
        "anomaly_summary": {},
        "experiment_plan": {
            "source_entrypoint": "src/lgb_package_to_dish_online_0319.py",
            "changes": [{"feature_name": "store_package_rolling_7d_14d_30d_mean"}],
        },
        "run_status": {
            "generated_train_path": "runs/trial_001/code/train.py",
            "train_returncode": "agent2_code_generation_failed",
            "agent2_code_generation_success": False,
            "agent2_code_modification_path": "runs/trial_001/code/agent2_code_modification.yaml",
            "feature_application_success": False,
        },
        "metric_comparison": {"decision": "rollback", "reason": "agent2 code generation failed"},
    }

    report = writer.build_template_report(context)

    assert "Agent2 代码生成失败" in report
    assert "agent2_code_modification.yaml" in report
    assert "不建议保留该特征改动" not in report

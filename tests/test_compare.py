from __future__ import annotations

from comboscope.core.compare import compare_metrics


def test_compare_metrics_keeps_when_wape_improves_and_bias_is_stable() -> None:
    result = compare_metrics(
        {"wape": 0.12, "bias": -0.03},
        {"wape": 0.11, "bias": -0.035},
        {"train_success": True, "eval_success": True},
    )

    assert result["decision"] == "keep"
    assert result["wape_delta"] < 0


def test_compare_metrics_rolls_back_when_bias_regresses() -> None:
    result = compare_metrics(
        {"wape": 0.12, "bias": -0.03},
        {"wape": 0.10, "bias": -0.08},
        {"train_success": True, "eval_success": True},
    )

    assert result["decision"] == "rollback"
    assert "bias" in result["reason"].lower()


def test_compare_metrics_rolls_back_when_feature_was_not_applied() -> None:
    result = compare_metrics(
        {"wape": 0.12, "bias": -0.03},
        {"wape": 0.12, "bias": -0.03},
        {"train_success": False, "eval_success": False, "feature_application_success": False},
    )

    assert result["decision"] == "rollback"
    assert result["reason"] == "feature change was not applied"


def test_compare_metrics_rolls_back_when_agent2_codegen_failed() -> None:
    result = compare_metrics(
        {"wape": 0.12, "bias": -0.03},
        {"wape": 0.12, "bias": -0.03},
        {
            "train_success": False,
            "eval_success": False,
            "train_returncode": "agent2_code_generation_failed",
            "agent2_code_generation_success": False,
            "feature_application_success": False,
        },
    )

    assert result["decision"] == "rollback"
    assert result["reason"] == "agent2 code generation failed"


def test_compare_metrics_rolls_back_when_sample_counts_differ() -> None:
    result = compare_metrics(
        {"wape": 0.12, "bias": -0.03, "sample_count": 56074},
        {"wape": 0.10, "bias": -0.031, "sample_count": 57406},
        {"train_success": True, "eval_success": True},
    )

    assert result["decision"] == "rollback"
    assert result["reason"] == "evaluation sample mismatch"
    assert result["sample_count_consistent"] is False

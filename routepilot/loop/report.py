"""优化闭环的运行报告：机器可读 JSON + 人读 Markdown。"""

from __future__ import annotations

import json
from pathlib import Path

from routepilot.loop.trial import OptimizationOutcome


def write_report(outcome: OptimizationOutcome, output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "run_report.json"
    markdown_path = directory / "run_report.md"
    json_path.write_text(
        json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(outcome), encoding="utf-8")
    _write_revert_files(outcome, directory)
    return json_path, markdown_path


def _write_revert_files(outcome: OptimizationOutcome, directory: Path) -> None:
    """keep 是就地生效的，把原始 editable 文件留档到 revert/ 以便回滚。"""
    if not outcome.kept_trials or not outcome.original_editable_files:
        return
    revert_root = directory / "revert"
    for relative, content in outcome.original_editable_files.items():
        target = revert_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (revert_root / "README.md").write_text(
        "# 回滚说明\n\n"
        "下列文件是本轮基线的原始内容（对应 trial 0）。\n"
        "RoutePilot 的 keep 会就地修改 editable 文件；如需回滚，把这些文件覆盖回项目根目录即可。\n",
        encoding="utf-8",
    )


def render_markdown(outcome: OptimizationOutcome) -> str:
    primary = outcome.primary_metric
    before = outcome.baseline_metrics.get(primary)
    after = outcome.final_metrics.get(primary)
    lines = [
        f"# RoutePilot 优化报告：{outcome.project}",
        "",
        f"- 目标：{outcome.goal}",
        f"- 主指标：{primary}（越小越好）",
        f"- 基线 {primary}：{_fmt(before)}",
        f"- 最终 {primary}：{_fmt(after)}",
        f"- 改善：{_fmt(_delta(before, after))}",
        f"- 采纳 trial：{outcome.kept_trials or '无'}",
        f"- 总耗时：{outcome.total_duration_sec:.1f}s",
        "",
        "## 逐轮结果",
        "",
        "| trial | 提案 | 文件 | 决策 | 原因 | 指标 | 耗时(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in outcome.trials:
        metrics = (
            " ".join(f"{key}={value:.4f}" for key, value in record.metrics.items() if key != "sample_count")
            if record.metrics
            else "-"
        )
        lines.append(
            f"| {record.trial_index} | {record.summary} | {', '.join(record.files)} | "
            f"{record.decision} | {record.reason} | {metrics} | {record.duration_sec:.1f} |"
        )
    if not outcome.trials:
        lines.append("| - | 无提案 | - | - | - | - | - |")
    lines.extend(_render_routing(outcome.routing))
    lines.extend(_render_profile(outcome.profile))
    lines.append("")
    return "\n".join(lines)


def _render_profile(profile: dict) -> list[str]:
    if not profile:
        return []
    lines = [
        "",
        "## 链路性能诊断",
        "",
        f"- 埋点总耗时：{profile.get('instrumented_seconds', 0):.2f}s",
        "",
        "| 环节 | 耗时(s) | 占比 |",
        "|---|---|---|",
    ]
    total = profile.get("instrumented_seconds") or 1.0
    for category, seconds in (profile.get("attribution") or {}).items():
        lines.append(f"| {category} | {seconds:.2f} | {seconds / total:.0%} |")
    findings = profile.get("findings") or []
    if findings:
        lines += ["", "### 瓶颈与回流", "", "| 瓶颈 | 占比 | 严重度 | 说明 |", "|---|---|---|---|"]
        for finding in findings:
            lines.append(
                f"| {finding['bottleneck']} | {finding['share']:.0%} | "
                f"{finding['severity']} | {finding['detail']} |"
            )
    else:
        lines += ["", "未发现超过阈值的单一瓶颈。"]
    return lines


def _render_routing(routing: dict) -> list[str]:
    if not routing:
        return []
    lines = [
        "",
        "## 路由与成本",
        "",
        f"- LLM 调用：{routing.get('calls', 0)}",
        f"- 总成本：${routing.get('cost_usd', 0):.6f}",
        f"- 总 tokens：{routing.get('tokens', 0)}",
        f"- 延迟 P50 / P95：{routing.get('latency_p50_sec', 0):.2f}s / "
        f"{routing.get('latency_p95_sec', 0):.2f}s",
        f"- 调用成功率：{routing.get('success_rate', 0):.0%}",
        "",
        "### 按档位",
        "",
        "| 档位 | 调用 | 成本(USD) | tokens | P95(s) | 成功率 |",
        "|---|---|---|---|---|---|",
    ]
    for name, item in (routing.get("by_tier") or {}).items():
        lines.append(
            f"| {name} | {item['calls']} | {item['cost_usd']:.6f} | {item['tokens']} | "
            f"{item['latency_p95_sec']:.2f} | {item['success_rate']:.0%} |"
        )
    lines += [
        "",
        "### 按 step",
        "",
        "| step | 调用 | 成本(USD) | tokens | P95(s) | 成功率 |",
        "|---|---|---|---|---|---|",
    ]
    for name, item in (routing.get("by_step") or {}).items():
        lines.append(
            f"| {name} | {item['calls']} | {item['cost_usd']:.6f} | {item['tokens']} | "
            f"{item['latency_p95_sec']:.2f} | {item['success_rate']:.0%} |"
        )
    return lines


def _delta(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return float("nan")
    return before - after


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"

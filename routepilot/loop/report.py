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
    lines.append("")
    return "\n".join(lines)


def _delta(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return float("nan")
    return before - after


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"

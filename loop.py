from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from comboscope.agents.llm_decision_agent import select_best_trial, write_best_trial_review
from comboscope.orchestrators.langgraph_flow import run_once
from comboscope.orchestrators.sequential_flow import run_once_sequential
from comboscope.runtime.agent_run_recorder import AgentRunRecorder
from comboscope.runtime.llm_client import create_llm_client
from comboscope.runtime.ask_paths import resolve_experiment_dir
from comboscope.runtime.cli_summary import print_loop_summary, print_trial_summary


def run_loop(
    *,
    experiment: Path | None,
    ask: str,
    max_trials: int,
    model: str | None,
    orchestrator: str,
    output: Path,
    llm_provider: str | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    repo = repo_root or Path(__file__).resolve().parent
    history: list[dict[str, Any]] = []
    runner = run_once_sequential if orchestrator == "sequential" else run_once
    candidate_limit: int | None = None
    for index in range(1, max_trials + 1):
        if candidate_limit is not None and index > candidate_limit:
            break
        trial_id = f"trial_{index:03d}"
        trial_dir = output / trial_id
        try:
            result = runner(
                {
                    "experiment_dir": resolve_experiment_dir(experiment, ask).as_posix(),
                    "ask": ask,
                    "model": model,
                    "output_dir": trial_dir.resolve().as_posix(),
                    "trial_id": trial_id,
                    "repo_root": repo.resolve().as_posix(),
                    "llm_provider": llm_provider,
                }
            )
        except Exception as exc:
            print(f"{trial_id}: failed={exc}")
            error_report = trial_dir / "error_report.md"
            if error_report.exists():
                print(f"{trial_id}: error_report={error_report.as_posix()}")
            raise
        review_path = _trial_artifact(trial_dir, "review_result.json")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        comparison = json.loads(_trial_artifact(trial_dir, "metric_comparison.json").read_text(encoding="utf-8"))
        row = {
            "trial_id": trial_id,
            "decision": review["decision"],
            "reason": review["reason"],
            "primary_metric": comparison.get("primary_metric_label"),
            "primary_delta": comparison.get("primary_delta"),
            "objective_score": comparison.get("new_primary"),
            "final_report": result.get("final_report_path", ""),
        }
        history.append(row)
        print_trial_summary(trial_id, trial_dir)
        candidate_limit = _candidate_experiment_count(trial_dir) or candidate_limit
    loop_recorder = AgentRunRecorder(output)
    with loop_recorder.step(
        "Agent2",
        "SelectBestTrial",
        artifacts={"best_trial_review": output / "best_trial_review.json"},
    ):
        best_client = create_llm_client(provider=llm_provider, model=model)
        best_review = select_best_trial(
            ask=ask,
            history=history,
            trial_summaries=_trial_summaries(output, history),
            llm_client=best_client,
        )
        loop_recorder.record_llm_call(best_client.last_call)
        (output / "best_trial_review.json").write_text(json.dumps(best_review, ensure_ascii=False, indent=2), encoding="utf-8")
        write_best_trial_review(output / "best_trial_review.md", best_review)
    with (output / "run_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trial_id", "decision", "reason", "primary_metric", "primary_delta", "objective_score", "is_best", "final_report"],
        )
        writer.writeheader()
        for row in history:
            row["is_best"] = row["trial_id"] == best_review.get("best_trial_id")
        writer.writerows(history)
    _write_loop_final_report(output, history, best_review)
    print_loop_summary(output, history)
    return history


def _candidate_experiment_count(trial_dir: Path) -> int | None:
    path = _trial_artifact(trial_dir, "candidate_experiments.yaml")
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    candidates = data.get("candidate_experiments") if isinstance(data, dict) else None
    if isinstance(candidates, list) and candidates:
        return len(candidates)
    return None


def _trial_summaries(output: Path, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in history:
        trial_dir = output / str(row["trial_id"])
        comparison_path = _trial_artifact(trial_dir, "metric_comparison.json")
        review_path = _trial_artifact(trial_dir, "review_result.json")
        summaries.append(
            {
                "trial_id": row["trial_id"],
                "comparison": json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else {},
                "review": json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {},
                "final_report": row.get("final_report"),
            }
        )
    return summaries


def _trial_artifact(trial_dir: Path, filename: str) -> Path:
    aliases = {
        "review_result.json": "agent2/review_result.json",
        "metric_comparison.json": "evaluation/metric_comparison.json",
        "candidate_experiments.yaml": "agent1/candidate_experiments.yaml",
    }
    root = trial_dir / filename
    if root.exists():
        return root
    return trial_dir / aliases.get(filename, filename)


def _write_loop_final_report(output: Path, history: list[dict[str, Any]], best_review: dict[str, Any]) -> None:
    best_report = next((Path(row["final_report"]) for row in history if row["trial_id"] == best_review.get("best_trial_id")), None)
    lines = [
        "# ComboScope Loop Final Report",
        "",
        "## 最佳方案",
        f"- 最佳 trial：{best_review.get('best_trial_id')}",
        f"- 来源：{best_review.get('source')}",
        f"- 判断依据：{best_review.get('decision_basis')}",
        f"- 建议动作：{best_review.get('recommended_action')}",
        "",
        "## 全部实验",
    ]
    for row in history:
        marker = "（模型选择）" if row.get("trial_id") == best_review.get("best_trial_id") else ""
        lines.append(
            f"- {row.get('trial_id')} {marker}: decision={row.get('decision')}, "
            f"primary={row.get('primary_metric')}, score={row.get('objective_score')}, delta={row.get('primary_delta')}"
        )
    if best_report and best_report.exists():
        lines.extend(["", "## 最佳单轮报告摘要", best_report.read_text(encoding="utf-8")])
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ComboScope multi-trial loop.")
    parser.add_argument("--experiment")
    parser.add_argument("--ask", required=True)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--llm-provider")
    parser.add_argument("--orchestrator", choices=["langgraph", "sequential"], default="langgraph")
    parser.add_argument("--output", default="runs")
    args = parser.parse_args()
    run_loop(
        experiment=Path(args.experiment) if args.experiment else None,
        ask=args.ask,
        max_trials=args.max_trials,
        model=args.model,
        llm_provider=args.llm_provider,
        orchestrator=args.orchestrator,
        output=Path(args.output),
        repo_root=Path(__file__).resolve().parent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from comboscope.orchestrators.langgraph_flow import run_once
from comboscope.orchestrators.sequential_flow import run_once_sequential
from comboscope.runtime.ask_paths import resolve_experiment_dir
from comboscope.runtime.cli_summary import print_run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ComboScope CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--experiment")
    run.add_argument("--ask", required=True)
    run.add_argument("--model")
    run.add_argument("--llm-provider")
    run.add_argument("--orchestrator", choices=["langgraph", "sequential"], default="langgraph")
    run.add_argument("--output", default="runs/trial_001")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    experiment_dir = resolve_experiment_dir(args.experiment, args.ask)
    request = {
        "experiment_dir": experiment_dir.as_posix(),
        "ask": args.ask,
        "model": args.model,
        "llm_provider": args.llm_provider,
        "output_dir": Path(args.output).resolve().as_posix(),
        "trial_id": Path(args.output).name,
        "repo_root": repo_root.as_posix(),
        "original_command": " ".join(sys.argv),
    }
    try:
        if args.orchestrator == "sequential":
            run_once_sequential(request)
        else:
            run_once(request)
    except Exception as exc:  # noqa: BLE001 - CLI should point humans at the persisted error report.
        error_report = Path(request["output_dir"]) / "error_report.md"
        print(f"ComboScope run failed: {exc}")
        if error_report.exists():
            print(f"error_report: {error_report.as_posix()}")
        return 1
    print_run_summary(request["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

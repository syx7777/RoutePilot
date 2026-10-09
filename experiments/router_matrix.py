"""路由模式对照实验矩阵。

对同一被接入项目，重复运行 strong / weak / static / dynamic 四种路由模式，
汇总成本、token、P95 延迟与任务改善的均值±标准差。

用法（在 RoutePilot 仓库根目录执行）：

    python experiments/router_matrix.py \
        --project ../routepilot-project-a \
        --repeats 3 --trials 4 \
        --out runs/experiment-matrix

注意：keep 会就地修改被接入项目的 editable 文件，因此每次运行前都会把
editable 文件恢复成实验开始时的快照，保证各次运行从同一基线出发。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ("strong", "weak", "static", "dynamic")
DEFAULT_GOAL = "在保持 Bias 稳定的前提下降低 WAPE"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_editable(project: Path, manifest: dict) -> dict[str, str]:
    """快照 editable 匹配到的现存文件，用于跨次运行恢复基线。"""
    snapshot: dict[str, str] = {}
    for relative in _editable_files(project, manifest):
        path = project / relative
        try:
            snapshot[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return snapshot


def _editable_files(project: Path, manifest: dict) -> list[str]:
    sys.path.insert(0, str(REPO_ROOT))
    from routepilot.adapter.manifest import ProjectManifest, matches_glob

    spec = ProjectManifest.model_validate(manifest)
    protected = spec.protected
    found: list[str] = []
    for path in project.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(project).as_posix()
        if any(part in {".git", ".venv", "__pycache__"} for part in relative.split("/")[:-1]):
            continue
        if any(matches_glob(relative, pattern) for pattern in protected):
            continue
        if any(matches_glob(relative, pattern) for pattern in spec.editable):
            found.append(relative)
    return sorted(found)


def _restore(project: Path, snapshot: dict[str, str]) -> None:
    for relative, content in snapshot.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _fmt(values: list[float], digits: int = 4) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f}±{statistics.stdev(values):.{digits}f}"


def run_matrix(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else project / "routepilot.yaml"
    if not manifest_path.is_file():
        print(f"[matrix] manifest not found: {manifest_path}")
        return 2
    manifest = _load_manifest_mapping(manifest_path)
    snapshot = _snapshot_editable(project, manifest)
    if not snapshot:
        print("[matrix] 未匹配到任何 editable 文件，实验无意义")
        return 2

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for mode in args.modes:
        for repeat in range(1, args.repeats + 1):
            _restore(project, snapshot)
            out = out_root / f"{mode}-r{repeat}"
            completed = subprocess.run(
                [
                    sys.executable, "-m", "routepilot.cli", "run",
                    "--manifest", str(manifest_path),
                    "--goal", args.goal,
                    "--router", mode,
                    "--max-trials", str(args.trials),
                    "--output", str(out),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            report_path = out / "run_report.json"
            if completed.returncode != 0 or not report_path.is_file():
                print(f"[FAIL] {mode}-r{repeat} rc={completed.returncode}")
                print(completed.stdout[-500:], completed.stderr[-500:])
                continue
            record = _collect(mode, repeat, _load_json(report_path))
            records.append(record)
            print(
                f"[ok] {mode}-r{repeat} base={record['baseline_wape']:.4f} "
                f"final={record['final_wape']:.4f} kept={record['kept_trials']} "
                f"cost=${record['cost_usd']:.6f} p95={record['p95_sec']:.1f}s"
            )

    _restore(project, snapshot)
    (out_root / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = _summarize(records, args.modes)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "summary.md").write_text(_render(summary, records), encoding="utf-8")
    print(f"\n[matrix] records={len(records)}  out={out_root}")
    return 0


def _load_manifest_mapping(manifest_path: Path) -> dict:
    import yaml

    return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}


def _collect(mode: str, repeat: int, report: dict) -> dict:
    routing = report.get("routing") or {}
    profile = report.get("profile") or {}
    return {
        "mode": mode,
        "repeat": repeat,
        "baseline_wape": report["baseline_metrics"]["wape"],
        "final_wape": report["final_metrics"]["wape"],
        "kept_trials": report["kept_trials"],
        "calls": routing.get("calls", 0),
        "cost_usd": routing.get("cost_usd", 0.0),
        "tokens": routing.get("tokens", 0),
        "p95_sec": routing.get("latency_p95_sec", 0.0),
        "success_rate": routing.get("success_rate", 0.0),
        "by_tier": {key: value["calls"] for key, value in (routing.get("by_tier") or {}).items()},
        "instrumented_sec": profile.get("instrumented_seconds", 0.0),
        "attribution": profile.get("attribution", {}),
        "bottlenecks": [item["bottleneck"] for item in (profile.get("findings") or [])],
    }


def _summarize(records: list[dict], modes) -> dict:
    summary: dict[str, dict] = {}
    for mode in modes:
        rows = [item for item in records if item["mode"] == mode]
        if not rows:
            continue
        finals = [item["final_wape"] for item in rows]
        deltas = [item["baseline_wape"] - item["final_wape"] for item in rows]
        costs = [item["cost_usd"] for item in rows]
        summary[mode] = {
            "repeats": len(rows),
            "final_wape_mean": statistics.mean(finals),
            "final_wape_stdev": statistics.stdev(finals) if len(finals) > 1 else 0.0,
            "improvement_mean": statistics.mean(deltas),
            "improvement_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "cost_mean": statistics.mean(costs),
            "cost_stdev": statistics.stdev(costs) if len(costs) > 1 else 0.0,
            "tokens_mean": statistics.mean([float(item["tokens"]) for item in rows]),
            "p95_mean": statistics.mean([item["p95_sec"] for item in rows]),
            "kept_total": sum(len(item["kept_trials"]) for item in rows),
            "improved_runs": sum(1 for value in deltas if value > 0),
            "run_success_rate": statistics.mean([item["success_rate"] for item in rows]),
        }
    return summary


def _render(summary: dict, records: list[dict]) -> str:
    lines = [
        "# 路由模式对照实验",
        "",
        f"- 运行次数：{len(records)}（每模式 {len(records) // max(len(summary), 1)} 次重复）",
        "- 指标含义：final wape 越小越好；improvement = baseline − final",
        "",
        "| 模式 | 重复 | final wape | improvement | 成本(USD) | tokens | P95(s) | 采纳数 | 改善次数 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, item in summary.items():
        lines.append(
            f"| {mode} | {item['repeats']} | "
            f"{item['final_wape_mean']:.4f}±{item['final_wape_stdev']:.4f} | "
            f"{item['improvement_mean']:.4f}±{item['improvement_stdev']:.4f} | "
            f"{item['cost_mean']:.6f}±{item['cost_stdev']:.6f} | "
            f"{item['tokens_mean']:.0f} | {item['p95_mean']:.2f} | "
            f"{item['kept_total']} | {item['improved_runs']}/{item['repeats']} |"
        )
    lines += [
        "",
        "> 说明：LLM 提案本身是随机的，单次运行的差异不足以支撑质量结论；",
        "> 重复次数过少时只能比较成本/延迟，不能断言质量优劣。",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoutePilot 路由模式对照实验")
    parser.add_argument("--project", required=True, help="被接入项目根目录")
    parser.add_argument("--manifest", default=None, help="默认 <project>/routepilot.yaml")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--out", default="runs/experiment-matrix")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_matrix(build_parser().parse_args()))

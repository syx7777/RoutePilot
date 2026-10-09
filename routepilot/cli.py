"""RoutePilot 接入协议命令行：init / validate。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from routepilot.adapter.command import build_adapter
from routepilot.adapter.discovery import ManifestDraft, draft_manifest
from routepilot.adapter.manifest import (
    ProjectManifest,
    load_manifest,
    resolve_project_root,
    validate_manifest,
)
from routepilot.loop import LlmProposer, ScriptedProposer, run_optimization, write_report
from routepilot.runtime.llm_client import create_llm_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routepilot",
        description="RoutePilot：即插即用的多 Agent 自动优化与推理加速框架",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="扫描项目并生成 routepilot.yaml 草稿")
    init_parser.add_argument("--root", default=".", help="待接入项目根目录")
    init_parser.add_argument("--out", default="routepilot.yaml", help="草稿输出路径")
    init_parser.add_argument("--name", default=None, help="覆盖 project.name")
    init_parser.add_argument("--report", default=None, help="将发现证据写入该 JSON 文件")
    init_parser.add_argument("--force", action="store_true", help="覆盖已存在的草稿")

    validate_parser = subparsers.add_parser(
        "validate", help="校验 routepilot.yaml 是否满足接入协议"
    )
    validate_parser.add_argument("--manifest", default="routepilot.yaml")

    run_parser = subparsers.add_parser("run", help="按 manifest 驱动一次自动优化闭环")
    run_parser.add_argument("--manifest", default="routepilot.yaml")
    run_parser.add_argument("--goal", required=True, help="本轮优化目标，例如“降低 WAPE”")
    run_parser.add_argument("--output", default=None, help="运行产物目录")
    run_parser.add_argument("--max-trials", type=int, default=None)
    run_parser.add_argument(
        "--proposals",
        default=None,
        help="离线候选提案文件（YAML）；给定时不调用 LLM",
    )
    run_parser.add_argument("--provider", default=None)
    run_parser.add_argument("--model", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        return _init(args)
    if args.command == "validate":
        return _validate(Path(args.manifest))
    if args.command == "run":
        return _run(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def _init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    try:
        draft = draft_manifest(root, project_name=args.name)
    except FileNotFoundError as exc:
        print(f"[routepilot] {exc}")
        return 2

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    if out_path.exists() and not args.force:
        print(f"[routepilot] draft already exists (use --force to overwrite): {out_path}")
        return 2

    try:
        manifest = ProjectManifest.model_validate(draft.manifest)
    except ValidationError as exc:
        print("[routepilot] 推断出的草稿未通过 schema 校验，需要人工修正：")
        print(exc)
        return 1

    text = yaml.safe_dump(
        manifest.model_dump(exclude_none=False), allow_unicode=True, sort_keys=False
    )
    out_path.write_text(_with_header(text), encoding="utf-8")

    _print_report(draft, out_path, root)
    if args.report:
        report_path = Path(args.report)
        report_path.write_text(
            json.dumps(
                {
                    "confidence": draft.confidence,
                    "questions": draft.questions,
                    "evidence": draft.evidence,
                    "manifest": manifest.model_dump(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report   : {report_path}")
    return 0


def _with_header(text: str) -> str:
    header = (
        "# 由 `routepilot init` 生成的接入草稿 —— 请人工审阅后再运行。\n"
        "# 重点确认：project.entrypoint / run.command / artifacts.columns / editable / protected\n"
    )
    return header + text


def _print_report(draft: ManifestDraft, out_path: Path, root: Path) -> None:
    manifest = draft.manifest
    print(f"project   : {manifest['project']['name']}")
    print(f"root      : {root}")
    print(f"entrypoint: {manifest['project'].get('entrypoint') or '<未识别>'}")
    print(f"command   : {' '.join(manifest['run']['command'])}")
    print(f"artifacts : prediction={manifest['artifacts']['prediction'] or '<未识别>'} "
          f"actual={manifest['artifacts']['actual'] or '<未识别>'}")
    columns = manifest["artifacts"]["columns"]
    print(f"columns   : prediction={columns['prediction'] or '?'} actual={columns['actual'] or '?'} "
          f"date={columns.get('date') or '?'} id={columns.get('id') or []}")
    print(f"editable  : {', '.join(manifest['editable']) or '<未推断出>'}")
    print(f"protected : {', '.join(manifest['protected']) or '<空>'}")
    print(f"confidence: {draft.confidence}")
    if draft.questions:
        print("待人工确认：")
        for item in draft.questions:
            print(f"  - {item}")
    print(f"draft     : {out_path}")
    print("next      : 审阅草稿后运行 `routepilot validate --manifest " + out_path.name + "`")


def _run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[routepilot] manifest not found: {manifest_path}")
        return 2
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        print(f"[routepilot] {exc}")
        return 1

    project_root = resolve_project_root(manifest, manifest_path)
    report = validate_manifest(manifest, project_root)
    for warning in report.warnings:
        print(f"warning  : {warning}")
    if not report.ok:
        for error in report.errors:
            print(f"error    : {error}")
        return 1

    adapter = build_adapter(manifest, project_root)
    if args.proposals:
        proposer = ScriptedProposer.from_file(args.proposals)
        print(f"proposer : scripted ({args.proposals})")
    else:
        client = create_llm_client(provider=args.provider, model=args.model)
        if not client.available():
            print(
                "[routepilot] 未检测到可用的 LLM 凭据。请配置 API key，"
                "或用 --proposals 指定离线候选提案。"
            )
            return 2
        proposer = LlmProposer(client)
        print(f"proposer : llm ({client.provider}/{client.model})")

    output_dir = Path(args.output) if args.output else _default_output_dir(manifest)
    print(f"output   : {output_dir}")
    try:
        outcome = run_optimization(
            adapter,
            manifest,
            goal=args.goal,
            proposer=proposer,
            max_trials=args.max_trials,
        )
    except RuntimeError as exc:
        print(f"[routepilot] run failed: {exc}")
        return 1

    json_path, markdown_path = write_report(outcome, output_dir)
    primary = outcome.primary_metric
    print("")
    print(f"baseline : {primary}={outcome.baseline_metrics.get(primary, float('nan')):.4f}")
    print(f"final    : {primary}={outcome.final_metrics.get(primary, float('nan')):.4f}")
    print(f"kept     : {outcome.kept_trials or '无'}")
    print(f"report   : {markdown_path}")
    print(f"json     : {json_path}")
    return 0


def _default_output_dir(manifest: ProjectManifest) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "runs" / f"{manifest.project.name}-{stamp}"


def _validate(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        print(f"[routepilot] manifest not found: {manifest_path}")
        return 2
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        print(f"[routepilot] {exc}")
        return 1

    project_root = resolve_project_root(manifest, manifest_path)
    report = validate_manifest(manifest, project_root)
    primary = manifest.metrics.primary

    print(f"project  : {manifest.project.name}")
    print(f"root     : {project_root}")
    print(f"command  : {' '.join(manifest.run.command)}")
    print(f"primary  : {primary.name} ({primary.direction}, min_delta={primary.min_delta})")
    print(f"editable : {', '.join(manifest.editable)}")
    if manifest.protected:
        print(f"protected: {', '.join(manifest.protected)}")
    print(f"budget   : max_trials={manifest.budget.max_trials}")

    for warning in report.warnings:
        print(f"warning  : {warning}")
    for error in report.errors:
        print(f"error    : {error}")

    if report.ok:
        print("OK: manifest satisfies the RoutePilot adapter protocol")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

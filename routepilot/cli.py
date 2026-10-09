"""RoutePilot 接入协议命令行（当前仅实现 validate）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from routepilot.adapter.manifest import (
    load_manifest,
    resolve_project_root,
    validate_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routepilot",
        description="RoutePilot：即插即用的多 Agent 自动优化与推理加速框架",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="校验 routepilot.yaml 是否满足接入协议"
    )
    validate_parser.add_argument("--manifest", default="routepilot.yaml")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(Path(args.manifest))
    parser.error(f"unknown command: {args.command}")
    return 2


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

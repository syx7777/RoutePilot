"""接入成本度量 harness。

回答设计文档 §6.2 的"通用性：接入耗时(min) / 改动 LOC / 适配成功率"三个口径，
并且**只输出量得出来的数**，不合成单一分数。

两个子命令：

    # 快照：给框架与被接入项目各取一次可复现指纹（幂等、零副作用）
    python experiments/onboarding_cost.py snapshot \
        --project ../routepilot-project-b --out runs/onboarding/b-after.json

    # 报告：在临时副本上重复跑 init / validate，并与人工终稿做行级 + 字段级 diff
    python experiments/onboarding_cost.py report \
        --project ../routepilot-project-b \
        --final-manifest ../routepilot-project-b/routepilot.yaml \
        --before runs/onboarding/b-before.json --after runs/onboarding/b-after.json \
        --repeats 3 --out runs/onboarding

四条防混淆规则（务必遵守，否则数字会误导）：
1. 三个时间口径分列、**永不相加**：init 子进程耗时（自动、可重复）/
   人工耗时（自报）/ 环境搭建耗时（不计入接入）。
2. 一次性人工动作（例如首次手动跑一遍训练看输出）不进任何计时。
3. 自动指标重复 N 次不一致时置 `auto_metric_unstable=true`，**拒绝输出单一数字**，只给区间。
4. 框架自身改动用目录指纹而非 git 工作区状态，避免受暂存/提交影响。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORE_DIRS = {".venv", "venv", "runs", "__pycache__", ".git", ".pytest_cache", ".ruff_cache"}
IGNORE_PREFIXES = ("outputs/autogluon_models",)
FRAMEWORK_SUFFIXES = {".py"}
PROJECT_INCLUDE_DIRS = ("src", "configs", "scripts")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_text_normalized(path: Path) -> str:
    """按行尾归一化后的字节做哈希。

    否则同一份源码在 CRLF 工作区与 LF（git archive / Linux checkout）下会得出
    不同指纹，把"框架没改"误判成"改了很多行"。
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return _sha256_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def _iter_files(root: Path, include, skip_ignored=True) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if skip_ignored:
            parts = relative.split("/")
            if any(part in IGNORE_DIRS for part in parts[:-1]):
                continue
            if any(relative.startswith(prefix) for prefix in IGNORE_PREFIXES):
                continue
        if include(relative):
            found.append(path)
    return found


def _tree_fingerprint(root: Path, include) -> tuple[str, int, list[str]]:
    """返回 (tree_hash, loc_total, 相对路径列表)；tree_hash 由逐文件 sha256 排序后二次哈希。"""
    files = _iter_files(root, include)
    entries: list[tuple[str, str]] = []
    loc = 0
    relatives: list[str] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        relatives.append(relative)
        entries.append((relative, _sha256_text_normalized(path)))
        try:
            loc += len(path.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            pass
    digest = _sha256_bytes(repr(sorted(entries)).encode("utf-8"))
    return digest, loc, relatives


def _framework_fingerprint(repo_root: Path) -> dict:
    include = lambda rel: Path(rel).suffix in FRAMEWORK_SUFFIXES  # noqa: E731
    tree_hash, loc, files = _tree_fingerprint(repo_root / "routepilot", include)
    return {"tree_hash": tree_hash, "loc_total": loc, "files": files}


def _project_fingerprint(project: Path) -> dict:
    def include(rel: str) -> bool:
        return rel.startswith(PROJECT_INCLUDE_DIRS) and Path(rel).suffix in {
            ".py",
            ".yaml",
            ".yml",
        }

    tree_hash, loc, files = _tree_fingerprint(project, include)
    return {"tree_hash": tree_hash, "loc_total": loc, "files": files}


def _venv_python(project: Path) -> Path | None:
    for parts in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        candidate = project.joinpath(*parts)
        if candidate.is_file():
            return candidate
    return None


def _env_fingerprint(project: Path) -> dict:
    """锁定环境的可复现指纹：uv.lock（若存在）+ 项目解释器版本。"""
    lock_path = project / "uv.lock"
    python_version = None
    interpreter = _venv_python(project)
    if interpreter is not None:
        try:
            completed = subprocess.run(
                [str(interpreter), "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            python_version = (completed.stdout or completed.stderr).strip()
        except (OSError, subprocess.SubprocessError):
            python_version = None
    return {
        "uv_lock_sha256": _sha256_file(lock_path) if lock_path.is_file() else None,
        "python": python_version,
    }


def cmd_snapshot(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    payload = {
        "project": project.name,
        "framework": _framework_fingerprint(REPO_ROOT),
        "project_tree": _project_fingerprint(project),
        "env": _env_fingerprint(project),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[snapshot] framework_loc={payload['framework']['loc_total']} "
        f"framework_hash={payload['framework']['tree_hash'][:16]} "
        f"project_hash={payload['project_tree']['tree_hash'][:16]} -> {out}"
    )
    return 0


def _copy_project(project: Path) -> Path:
    """把被接入项目复制到临时目录，用于在副本上重复跑 init（不改动真项目）。

    保留 outputs/prediction.csv 与 data/*.csv：discovery 依赖真实产物列来识别
    prediction/actual，缺了它们测出来的 confidence 不代表真实接入难度。
    """
    temp_root = Path(tempfile.mkdtemp(prefix="routepilot-onboarding-"))
    target = temp_root / project.name
    shutil.copytree(
        project,
        target,
        ignore=shutil.ignore_patterns(
            ".venv", "venv", "runs", "__pycache__", ".git", "autogluon_models"
        ),
    )
    return target


def _run_init(copy_root: Path, name: str, report_path: Path) -> dict:
    """跑一次 routepilot init 并返回耗时与草稿指标。"""
    draft_path = copy_root / "routepilot.yaml"
    if draft_path.exists():
        draft_path.unlink()
    if report_path.exists():
        report_path.unlink()
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable, "-m", "routepilot.cli", "init",
            "--root", str(copy_root),
            "--name", name,
            "--out", "routepilot.yaml",
            "--report", str(report_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    seconds = time.perf_counter() - started
    if completed.returncode != 0 or not draft_path.is_file():
        raise RuntimeError(f"init failed rc={completed.returncode}: {completed.stderr[-400:]}")
    discovery = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return {
        "seconds": seconds,
        "draft_text": draft_path.read_text(encoding="utf-8"),
        "confidence": discovery.get("confidence"),
        "questions": list(discovery.get("questions") or []),
        "evidence": discovery.get("evidence") or {},
        "draft_manifest": discovery.get("manifest") or {},
    }


def _run_validate(manifest_path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "routepilot.cli", "validate", "--manifest", str(manifest_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout
    return {
        "ok": completed.returncode == 0,
        "warnings": len(re.findall(r"^warning\s*:", output, flags=re.MULTILINE)),
        "errors": re.findall(r"^error\s*:\s*(.+)$", output, flags=re.MULTILINE),
    }


def _run_draft_loop(project_root: Path, draft_text: str, proposals: Path, output_dir: Path) -> dict:
    """在**真实项目根目录**上试跑一次闭环，验证"validate 通过 ≠ 可运行"这条边界。

    不能拿复制出来的副本跑：副本没有 `.venv`，裸 `python` 记号会回退到框架自己的
    解释器（没装 AutoGluon），失败原因就变成了"缺依赖"而不是我们要观测的 manifest 缺陷。
    因此把草稿 manifest 临时写到项目根（`project.root: "."` 要求 manifest 与项目同级），
    跑完立即删除。草稿在 baseline 阶段就会失败，不会触碰 editable 文件。
    """
    check_manifest = project_root / ".routepilot-draft-check.yaml"
    check_manifest.write_text(draft_text, encoding="utf-8")
    try:
        completed = subprocess.run(
            [
                sys.executable, "-m", "routepilot.cli", "run",
                "--manifest", str(check_manifest),
                "--goal", "降低主指标",
                "--proposals", str(proposals),
                "--max-trials", "1",
                "--output", str(output_dir),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        check_manifest.unlink(missing_ok=True)
    tail = [line for line in completed.stdout.splitlines() if "run failed" in line]
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "error": tail[-1].strip() if tail else "",
    }


def _flatten(mapping: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in (mapping or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _field_diff(draft: dict, final: dict) -> dict:
    left, right = _flatten(draft), _flatten(final)
    changed = sorted(k for k in left.keys() & right.keys() if left[k] != right[k])
    added = sorted(right.keys() - left.keys())
    removed = sorted(left.keys() - right.keys())
    return {"changed": changed, "added": added, "removed": removed}


def _line_diff(draft_text: str, final_text: str, *, drop_comments: bool = False) -> dict:
    left = _clean_lines(draft_text, drop_comments)
    right = _clean_lines(final_text, drop_comments)
    added = removed = 0
    for line in difflib.unified_diff(left, right, lineterm="", n=0):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {"added": added, "removed": removed}


def _clean_lines(text: str, drop_comments: bool) -> list[str]:
    lines = text.splitlines()
    if not drop_comments:
        return lines
    return [
        line for line in lines if line.strip() and not line.lstrip().startswith("#")
    ]


def _read_manual_minutes(args: argparse.Namespace) -> dict:
    """人工耗时只接受显式输入，并标注来源为自报。"""
    if args.manual_minutes_file:
        path = Path(args.manual_minutes_file)
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        total = 0.0
        for row in rows[1:]:
            parts = row.split(",")
            if len(parts) >= 2 and parts[1].strip():
                total += float(parts[1])
        return {"value": round(total, 3), "source": "self-reported", "steps": max(len(rows) - 1, 0)}
    if args.manual_minutes is not None:
        return {"value": float(args.manual_minutes), "source": "self-reported", "steps": None}
    return {"value": None, "source": "self-reported", "steps": None}


def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _framework_delta(before: dict | None, after: dict | None, numstat: str | None) -> dict:
    if before and after:
        b, a = before["framework"], after["framework"]
        return {
            "source": "tree-fingerprint",
            "before_hash": b["tree_hash"],
            "after_hash": a["tree_hash"],
            "loc_delta": a["loc_total"] - b["loc_total"],
            "zero_change_asserted": b["tree_hash"] == a["tree_hash"],
        }
    if numstat:
        match = re.match(r"\s*(\d+)\s+(\d+)", numstat)
        if match:
            added, removed = int(match.group(1)), int(match.group(2))
            return {
                "source": "git-numstat",
                "before_hash": None,
                "after_hash": None,
                "loc_delta": added - removed,
                "lines_added": added,
                "lines_removed": removed,
                "zero_change_asserted": added == 0 and removed == 0,
            }
    return {
        "source": None,
        "before_hash": None,
        "after_hash": None,
        "loc_delta": None,
        "zero_change_asserted": False,
    }


def cmd_report(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    final_manifest_path = Path(args.final_manifest).resolve()
    final_text = final_manifest_path.read_text(encoding="utf-8")
    import yaml

    final_manifest = yaml.safe_load(final_text) or {}

    copy_root = _copy_project(project)
    report_json = copy_root.parent / "discovery.json"
    inits = [_run_init(copy_root, project.name, report_json) for _ in range(args.repeats)]

    confidences = {item["confidence"] for item in inits}
    question_sets = {tuple(item["questions"]) for item in inits}
    draft_texts = {item["draft_text"] for item in inits}
    unstable = len(confidences) > 1 or len(question_sets) > 1 or len(draft_texts) > 1
    seconds = [item["seconds"] for item in inits]
    draft_text = inits[0]["draft_text"]
    draft_manifest = inits[0]["draft_manifest"]

    validate = _run_validate(copy_root / "routepilot.yaml")
    draft_loop = (
        _run_draft_loop(
            project, draft_text, Path(args.proposals).resolve(), Path(args.out) / "draft-run"
        )
        if args.proposals
        else None
    )

    manual = _read_manual_minutes(args)
    field_diff = _field_diff(draft_manifest, final_manifest)
    leaf_count = max(len(_flatten(final_manifest)), 1)
    changed = len(field_diff["changed"]) + len(field_diff["added"]) + len(field_diff["removed"])

    payload = {
        "project": project.name,
        "repeats": args.repeats,
        "auto": {
            "init_seconds_all": [round(value, 4) for value in seconds],
            "init_seconds_median": round(statistics.median(seconds), 4),
            "confidence": inits[0]["confidence"],
            "auto_metric_unstable": unstable,
            "questions": inits[0]["questions"],
            "evidence_resolved_fields": (inits[0]["evidence"] or {}).get("resolved_fields"),
            "draft_sha256": _sha256_bytes(draft_text.encode("utf-8")),
            "validate_first_pass": validate["ok"],
            "validate_warnings": validate["warnings"],
            "validate_errors": validate["errors"],
            "draft_manifest_leaf_count": len(_flatten(draft_manifest)),
        },
        "human": {
            "manifest_loc": _line_diff(draft_text, final_text),
            "manifest_loc_noncomment": _line_diff(draft_text, final_text, drop_comments=True),
            "field_level": field_diff,
            "field_agreement_rate": round(1.0 - changed / leaf_count, 4),
            "manual_minutes": manual,
        },
        "adaptation": {
            "validate_first_pass_on_draft": validate["ok"],
            "run_first_pass_on_draft": (draft_loop or {}).get("ok"),
            "run_first_error": (draft_loop or {}).get("error"),
        },
        "framework": _framework_delta(
            _load_json(args.before), _load_json(args.after), args.framework_numstat
        ),
        "env": _load_json(args.after)["env"] if _load_json(args.after) else None,
    }

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "onboarding_cost.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "onboarding_cost.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(_render_markdown(payload))
    print(f"[report] -> {out_root / 'onboarding_cost.json'}")
    shutil.rmtree(copy_root.parent, ignore_errors=True)
    return 0


def _render_markdown(payload: dict) -> str:
    auto, human = payload["auto"], payload["human"]
    adaptation, framework = payload["adaptation"], payload["framework"]
    manual = human["manual_minutes"]
    if manual["value"] is None:
        manual_line = "- 人工耗时：**未采集**（需专门的计时协议；见下方客观代理指标）"
    else:
        manual_line = (
            f"- 人工耗时（{manual['source']}）：**{manual['value']} min**"
        )
    lines = [
        f"# 接入成本报告：{payload['project']}",
        "",
        "> 三个时间口径分列、永不相加；人工耗时一律标注为自报。",
        "",
        "## 自动阶段（可重复）",
        "",
        f"- `init` 耗时中位数：**{auto['init_seconds_median']} s**（{payload['repeats']} 次：{auto['init_seconds_all']}）",
        f"- 自动发现置信度：**{auto['confidence']}**",
        f"- 待人工确认项：**{len(auto['questions'])}** 条",
        f"- 自动指标是否稳定：**{not auto['auto_metric_unstable']}**",
        f"- 草稿直接 `validate`：**{'通过' if auto['validate_first_pass'] else '不通过'}**"
        f"（warning={auto['validate_warnings']}, error={auto['validate_errors']}）",
        "",
        "## 人工阶段（自报 / 可复核）",
        "",
        f"- manifest 行级改动：+{human['manifest_loc']['added']} / -{human['manifest_loc']['removed']}"
        f"（去注释后 +{human['manifest_loc_noncomment']['added']} / -{human['manifest_loc_noncomment']['removed']}）",
        f"- 字段级改动：changed={human['field_level']['changed']}",
        f"  added={human['field_level']['added']} removed={human['field_level']['removed']}",
        f"- 字段一致率：**{human['field_agreement_rate']}**",
        manual_line,
        "",
        "## 适配成功率（不合成单一百分比）",
        "",
        f"- 草稿 `validate` 一次通过：**{adaptation['validate_first_pass_on_draft']}**",
        f"- 草稿直接 `run` 一次成功：**{adaptation['run_first_pass_on_draft']}**",
    ]
    if adaptation.get("run_first_error"):
        lines.append(f"  - 失败原因：`{adaptation['run_first_error']}`")
    lines += [
        "",
        "## 框架自身改动",
        "",
        f"- 来源：{framework['source']}",
        f"- 改动 LOC：**{framework['loc_delta']}**",
        f"- 断言零改动：**{framework['zero_change_asserted']}**",
        "",
        "## 环境",
        "",
        f"- `uv.lock` sha256：{(payload.get('env') or {}).get('uv_lock_sha256')}",
        f"- Python：{(payload.get('env') or {}).get('python')}",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoutePilot 接入成本度量")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="给框架与被接入项目取一次可复现指纹")
    snap.add_argument("--project", required=True)
    snap.add_argument("--out", required=True)
    snap.set_defaults(func=cmd_snapshot)

    report = sub.add_parser("report", help="量自动阶段/人工阶段/适配成功率")
    report.add_argument("--project", required=True)
    report.add_argument("--final-manifest", required=True)
    report.add_argument("--out", default="runs/onboarding")
    report.add_argument("--repeats", type=int, default=3)
    report.add_argument("--before", default=None, help="接入前的框架快照 JSON")
    report.add_argument("--after", default=None, help="接入后的框架快照 JSON")
    report.add_argument(
        "--framework-numstat",
        default=None,
        help='框架改动行数，格式"<added> <removed>"；缺省时只能用快照指纹比较',
    )
    report.add_argument("--manual-minutes", type=float, default=None)
    report.add_argument("--manual-minutes-file", default=None, help="CSV：step,minutes,note")
    report.add_argument(
        "--proposals", default=None, help="给定则在自动草稿上试跑一次闭环，用于验证可运行性"
    )
    report.set_defaults(func=cmd_report)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))

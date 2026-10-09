from __future__ import annotations

from pathlib import Path


START = "# ROUTEPILOT_POLICY_START"
END = "# ROUTEPILOT_POLICY_END"


def render_policy(feature_names: list[str]) -> str:
    lines = [START, "ENABLED_FEATURES = ["]
    for name in feature_names:
        lines.append(f'    "{name}",')
    lines.extend(["]", END])
    return "\n".join(lines)


def update_train_policy(train_path: str | Path, feature_names: list[str]) -> None:
    path = Path(train_path)
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("train.py policy markers not found")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(before + render_policy(feature_names) + after, encoding="utf-8")

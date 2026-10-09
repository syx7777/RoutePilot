from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactContractError(ValueError):
    pass


def write_artifact_contract(path: str | Path, contract: dict[str, Any]) -> dict[str, Any]:
    Path(path).write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract


def standardize_from_contract(
    contract: dict[str, Any],
    prediction_output: str | Path,
    actual_output: str | Path,
    *,
    base_dirs: list[str | Path],
) -> dict[str, Any]:
    prediction_path = _resolve_existing_path(contract.get("prediction_path"), base_dirs, "prediction_path")
    actual_path = _resolve_existing_path(contract.get("actual_path") or contract.get("prediction_path"), base_dirs, "actual_path")
    prediction_column = _required_str(contract, "prediction_column")
    actual_column = _required_str(contract, "actual_column")

    prediction_df = _read_csv(prediction_path)
    actual_df = prediction_df if prediction_path.resolve() == actual_path.resolve() else _read_csv(actual_path)
    prediction_df = _filter_split(prediction_df, contract)
    actual_df = _filter_split(actual_df, contract)

    _require_column(prediction_df, prediction_column, "prediction_column")
    _require_column(actual_df, actual_column, "actual_column")

    key_columns = _contract_columns(contract, "id_columns")
    date_column = contract.get("date_column")
    if date_column:
        key_columns.insert(0, str(date_column))
    split_column = contract.get("split_column")
    if split_column:
        key_columns.append(str(split_column))
    passthrough_columns = _contract_columns(contract, "passthrough_columns")

    prediction_standard = _standard_frame(prediction_df, prediction_column, "prediction", key_columns, passthrough_columns)
    actual_standard = _standard_frame(actual_df, actual_column, "actual", key_columns, passthrough_columns)
    if prediction_standard.empty or actual_standard.empty:
        raise ArtifactContractError("standardized prediction or actual output is empty")

    Path(prediction_output).parent.mkdir(parents=True, exist_ok=True)
    Path(actual_output).parent.mkdir(parents=True, exist_ok=True)
    prediction_standard.to_csv(prediction_output, index=False)
    actual_standard.to_csv(actual_output, index=False)
    return {
        "prediction_path": prediction_path.as_posix(),
        "actual_path": actual_path.as_posix(),
        "standardized_prediction_path": Path(prediction_output).resolve().as_posix(),
        "standardized_actual_path": Path(actual_output).resolve().as_posix(),
        "prediction_column": prediction_column,
        "actual_column": actual_column,
        "date_column": date_column,
        "id_columns": key_columns,
        "passthrough_columns": passthrough_columns,
        "prediction_rows": int(len(prediction_standard)),
        "actual_rows": int(len(actual_standard)),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 - surface readable contract error
            raise ArtifactContractError(f"failed to read CSV artifact: {path}: {exc}") from exc
    else:
        raise ArtifactContractError(f"failed to decode CSV artifact: {path}: {last_error}") from last_error
    df.columns = [str(column).lstrip("\ufeff") for column in df.columns]
    return df


def _resolve_existing_path(value: Any, base_dirs: list[str | Path], label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactContractError(f"{label} is required in artifact contract")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [Path(base) / raw for base in base_dirs]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    joined = ", ".join(path.as_posix() for path in candidates)
    raise ArtifactContractError(f"{label} does not resolve to an existing file: {joined}")


def _required_str(contract: dict[str, Any], key: str) -> str:
    value = contract.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactContractError(f"{key} is required in artifact contract")
    return value


def _filter_split(df: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    split_column = contract.get("split_column")
    split_value = contract.get("split_value")
    if not split_column or split_value in {None, ""}:
        return df
    _require_column(df, str(split_column), "split_column")
    selected = df[df[str(split_column)].astype(str).str.lower() == str(split_value).lower()].copy()
    if selected.empty:
        raise ArtifactContractError(f"split filter produced no rows: {split_column}={split_value}")
    return selected


def _contract_columns(contract: dict[str, Any], key: str) -> list[str]:
    raw = contract.get(key) or []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        raise ArtifactContractError(f"{key} must be a list of column names")
    return [str(item) for item in raw if str(item).strip()]


def _standard_frame(
    df: pd.DataFrame,
    value_column: str,
    output_column: str,
    key_columns: list[str],
    passthrough_columns: list[str],
) -> pd.DataFrame:
    columns: list[str] = []
    for column in [*key_columns, *passthrough_columns]:
        if column and column in df.columns and column not in columns and column != value_column:
            columns.append(column)
    _require_column(df, value_column, output_column)
    standard = df[[*columns, value_column]].copy()
    standard = standard.rename(columns={value_column: output_column})
    standard[output_column] = pd.to_numeric(standard[output_column], errors="coerce")
    return standard.dropna(subset=[output_column])


def _require_column(df: pd.DataFrame, column: str, label: str) -> None:
    if column not in df.columns:
        raise ArtifactContractError(f"{label} column not found: {column}; available={list(df.columns)}")

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ALLOWED_EDITABLE_FILES = {
    "benchmark/feature_config.yaml",
    "benchmark/feature_policy.py",
    "benchmark/train.py",
}


class ExperimentChange(BaseModel):
    action: str
    feature_name: str
    feature_type: str = "rolling_stat"
    group_by: str | None = None
    window: int | None = None
    cli_flag: str | None = None
    cli_args: list[str] = Field(default_factory=list)
    enabled: bool | None = None
    evidence: list[str] = Field(default_factory=list)
    field_sources: list[str] = Field(default_factory=list)
    construction: str = ""
    code_locations: list[str] = Field(default_factory=list)
    validation_metrics: list[str] = Field(default_factory=list)

    @field_validator("feature_name")
    @classmethod
    def feature_name_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("feature_name is required")
        return value

    @field_validator("cli_args")
    @classmethod
    def cli_args_are_argv_tokens(cls, value: list[str]) -> list[str]:
        tokens: list[str] = []
        seen_flag = False
        for raw in value:
            token = str(raw)
            if not token or token != token.strip() or any(ch.isspace() for ch in token):
                raise ValueError("cli_args entries must be single argv tokens without whitespace")
            if any(ch in token for ch in "()"):
                raise ValueError("cli_args entries must not include explanatory parentheses")
            if token.startswith("--"):
                seen_flag = True
            elif not seen_flag:
                raise ValueError("cli_args must start with a CLI flag token")
            tokens.append(token)
        return tokens


class FeatureHypothesis(BaseModel):
    hypothesis_id: str = "h001"
    target_problem: str
    hypothesis: str
    evidence: list[str] = Field(min_length=1)
    proposed_features: list[ExperimentChange] = Field(min_length=1)
    confidence: Literal["low", "medium", "high"] = "medium"


class PlanHypothesis(BaseModel):
    description: str
    evidence: list[str] = Field(min_length=1)


class CandidateExperiment(BaseModel):
    experiment_id: str
    title: str
    priority: int = 100
    feature_actions: list[ExperimentChange] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    expected_effect: str
    risk: str


class ExperimentPlan(BaseModel):
    trial_id: str
    target_problem: str
    hypothesis: PlanHypothesis
    editable_files: list[str] = Field(min_length=1)
    changes: list[ExperimentChange] = Field(min_length=1)
    expected_effect: str
    risk: str
    scenario: str | None = None
    model_family: str | None = None
    objective: str | None = None
    source_entrypoint: str | None = None
    generated_train_path: str | None = None
    output_contract: str | None = None
    evaluation_metric: dict[str, object] = Field(default_factory=dict)
    candidate_experiments: list[CandidateExperiment] = Field(default_factory=list)

    @field_validator("trial_id")
    @classmethod
    def trial_id_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trial_id is required")
        return value

    @model_validator(mode="after")
    def editable_files_are_allowed(self) -> "ExperimentPlan":
        invalid = [path for path in self.editable_files if not is_editable_file_allowed(path)]
        if invalid:
            raise ValueError(f"editable file not allowed: {invalid}")
        return self


class ReviewResult(BaseModel):
    decision: Literal["keep", "rollback"]
    reason: str
    wape_delta: float
    bias_delta: float


def is_editable_file_allowed(path: str) -> bool:
    if path in ALLOWED_EDITABLE_FILES:
        return True
    normalized = path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    return len(parts) == 4 and parts[0] == "runs" and parts[2] == "code" and normalized.endswith(".py")

from pydantic import BaseModel, ConfigDict, Field


class ActionArgs(BaseModel):
    model_config = ConfigDict(extra="allow")
    duration_ms: int = Field(default=200, ge=0, le=5000)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_summary: str
    controlled_entity: str
    current_subgoal: str
    action: str
    actions: list[str] = Field(default_factory=list)
    action_args: ActionArgs = Field(default_factory=ActionArgs)
    expected_result: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class PlannerContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    objective: str
    legal_actions: list[str]
    recent_outcomes: list[str] = Field(default_factory=list)
    learned_rules: str = ""
    screenshot_png: bytes = b""
    partner_recent_actions: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    episode_id: str
    step_index: int
    mode: str
    game: str
    decision: Decision
    primary_metric: str
    primary_delta: float
    improved: bool
    metric_delta: dict[str, float]
    rule_added: str | None = None
    screenshot_ref: str | None = None


class EpisodeResult(BaseModel):
    episode_id: str
    game: str
    mode: str
    steps: int
    legal_action_rate: float
    final_metrics: dict[str, float]
    rules: list[str]

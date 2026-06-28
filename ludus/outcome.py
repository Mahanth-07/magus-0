from pydantic import BaseModel


class Outcome(BaseModel):
    delta: dict[str, float]
    primary_metric: str
    primary_delta: float
    improved: bool
    summary: str


class OutcomeDetector:
    def detect(
        self,
        pre: dict[str, float],
        post: dict[str, float],
        primary_metric: str,
        higher_is_better: bool,
    ) -> Outcome:
        keys = set(pre) | set(post)
        delta = {k: post.get(k, 0.0) - pre.get(k, 0.0) for k in keys}
        pd = delta.get(primary_metric, 0.0)
        improved = pd > 0 if higher_is_better else pd < 0
        sign = "+" if pd >= 0 else ""
        summary = f"{primary_metric} {sign}{pd:.2f} ({'improved' if improved else 'no improvement'})"
        return Outcome(
            delta=delta, primary_metric=primary_metric,
            primary_delta=pd, improved=improved, summary=summary,
        )

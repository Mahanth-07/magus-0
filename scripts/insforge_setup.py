"""Create the Ludus tables in InsForge via the admin API (idempotent-ish).

InsForge creates tables through POST /api/database/tables (not raw SQL), so this is
the real mechanism behind supabase/migrations/0001_ludus.sql. Run once after setting
INSFORGE_BASE_URL + INSFORGE_API_KEY in ~/ludus/.env:

    /opt/anaconda3/bin/python scripts/insforge_setup.py
"""
import os
from pathlib import Path
import httpx


def _load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def col(name, type_, nullable=True, unique=False):
    # InsForge create-table column schema: columnName / type / isNullable / isUnique.
    return {"columnName": name, "type": type_, "isNullable": nullable, "isUnique": unique}


TABLES = [
    {
        "tableName": "ludus_episodes",
        "rlsEnabled": False,
        "columns": [
            col("episode_id", "string", nullable=False),
            col("game", "string", nullable=False),
            col("mode", "string", nullable=False),
            col("steps", "integer", nullable=False),
            col("legal_action_rate", "float", nullable=False),
            col("final_metrics", "json"),
            col("rules", "json"),
        ],
    },
    {
        "tableName": "ludus_steps",
        "rlsEnabled": False,
        "columns": [
            col("episode_id", "string", nullable=False),
            col("step_index", "integer", nullable=False),
            col("mode", "string"),
            col("game", "string"),
            col("action", "string"),
            col("expected_result", "string"),
            col("primary_metric", "string"),
            col("primary_delta", "float"),
            col("improved", "boolean"),
            col("metric_delta", "json"),
            col("rule_added", "string"),
            col("screenshot_ref", "string"),
            col("confidence", "float"),
        ],
    },
]


def main():
    _load_env()
    base = os.environ["INSFORGE_BASE_URL"].rstrip("/")
    key = os.environ["INSFORGE_API_KEY"]
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for t in TABLES:
        r = httpx.post(f"{base}/api/database/tables", headers=h, json=t, timeout=30)
        status = "created" if r.status_code < 300 else f"skip/err {r.status_code}"
        msg = "" if r.status_code < 300 else f" :: {r.text[:160]}"
        print(f"[{status}] {t['tableName']}{msg}")


if __name__ == "__main__":
    main()

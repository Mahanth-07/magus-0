#!/usr/bin/env python3
"""Poll (and optionally deploy + smoke-test) the Fireworks LoRA Tetris student.

The dataset upload + SFT job CREATE were done with firectl (the reliable path):

    firectl dataset create ludus-tetris-text     data/tetris/nebius_text_train.jsonl
    firectl dataset create ludus-tetris-text-val data/tetris/nebius_text_val.jsonl
    firectl supervised-fine-tuning-job create \
        --base-model accounts/fireworks/models/llama-v3p2-3b-instruct \
        --dataset ludus-tetris-text --evaluation-dataset ludus-tetris-text-val \
        --output-model ludus-tetris --epochs 3 --lora-rank 16

This script is the no-browser poll/deploy/smoke companion. It hits the control
plane (GET supervisedFineTuningJobs/{id}) and the OpenAI-compatible inference
endpoint directly with httpx + the FIREWORKS_API_KEY from .env, so it works
headless and is exactly reproducible.

    /opt/anaconda3/bin/python scripts/fireworks_finetune.py --job tdfqc4ng --budget-min 12
    /opt/anaconda3/bin/python scripts/fireworks_finetune.py --job tdfqc4ng --no-poll --smoke

States (control-plane enum): JOB_STATE_RUNNING / JOB_STATE_COMPLETED /
JOB_STATE_FAILED / JOB_STATE_CANCELLED (terminal = the last three).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.cli import _load_dotenv  # no-dep .env loader

ACCOUNT = "mahanthk7"
CONTROL_BASE = "https://api.fireworks.ai/v1"
INFERENCE_BASE = "https://api.fireworks.ai/inference/v1"
OUTPUT_MODEL = f"accounts/{ACCOUNT}/models/ludus-tetris"

TERMINAL = {"JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def get_job(key: str, job_id: str) -> dict:
    url = f"{CONTROL_BASE}/accounts/{ACCOUNT}/supervisedFineTuningJobs/{job_id}"
    r = httpx.get(url, headers=_headers(key), timeout=60)
    if r.status_code >= 400:
        raise SystemExit(f"job fetch failed ({r.status_code}): {r.text}")
    return r.json()


def poll(key: str, job_id: str, budget_s: float, interval_s: float = 15.0) -> dict:
    """Poll until terminal OR budget elapses. Prints status transitions. Never hangs."""
    deadline = time.monotonic() + budget_s
    last = None
    job = get_job(key, job_id)
    while True:
        state = job.get("state")
        if state != last:
            print(f"[{time.strftime('%H:%M:%S')}] state: {last or '(start)'} -> {state}")
            last = state
        if state in TERMINAL:
            return job
        if time.monotonic() >= deadline:
            print(f"[budget ~{int(budget_s)}s elapsed] still '{state}'; stopping poll (not terminal)")
            return job
        time.sleep(min(interval_s, max(1.0, deadline - time.monotonic())))
        job = get_job(key, job_id)


def smoke(key: str, model: str) -> bool:
    """POST one board-state prompt to the inference endpoint; parse via parse_decision.

    Uses the EXACT runtime prompt (build_user_text + _SYSTEM) so this is a true
    end-to-end check of the served LoRA. Returns True iff a Decision with a
    non-empty actions macro comes back.
    """
    from ludus.providers.insforge import _SYSTEM, build_user_text, parse_decision

    user_text = build_user_text(
        objective=(
            "Tetris. The active piece spawns at the TOP-CENTER column. Each turn, "
            "return an \"actions\" sequence that FIRST moves the piece toward the "
            "lowest/flattest part of the board, ROTATE as needed, and ONLY THEN end "
            "with \"drop\"."
        ),
        legal_actions=["left", "right", "rotate", "drop"],
        state_text=(
            "current=j at_cols=3-4-5 next=[o,z,s]\n"
            "Board 10x20 (col heights, 0=empty, left->right):\n"
            "0 0 0 0 0 0 0 0 0 0\n"
            "Aggregate height=0 max=0 holes=0\n"
            "score=0 level=1 lines_remaining=5"
        ),
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
    }
    print(f"\nsmoke: POST {INFERENCE_BASE}/chat/completions model={model}")
    r = httpx.post(
        f"{INFERENCE_BASE}/chat/completions",
        headers=_headers(key), json=payload, timeout=90,
    )
    if r.status_code >= 400:
        print(f"  inference FAILED ({r.status_code}): {r.text[:500]}")
        return False
    content = r.json()["choices"][0]["message"]["content"]
    print(f"  raw: {content[:300]}")
    try:
        d = parse_decision(content)
    except Exception as e:  # noqa: BLE001
        print(f"  parse_decision FAILED: {e}")
        return False
    print(f"  parsed Decision: action={d.action!r} actions={d.actions} confidence={d.confidence}")
    ok = bool(d.actions)
    print(f"  smoke {'OK' if ok else 'WARN (empty actions macro)'}")
    return ok


def main() -> None:
    _load_dotenv()
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, help="SFT job id (e.g. tdfqc4ng)")
    ap.add_argument("--model", default=OUTPUT_MODEL, help="served model id for smoke")
    ap.add_argument("--budget-min", type=float, default=12.0)
    ap.add_argument("--no-poll", action="store_true", help="skip polling; just report + optional smoke")
    ap.add_argument("--smoke", action="store_true", help="run an inference smoke test (after completion)")
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY not set (expected in .env)")

    if args.no_poll:
        job = get_job(key, args.job)
        print(f"state: {job.get('state')}")
    else:
        job = poll(key, args.job, args.budget_min * 60.0)

    state = job.get("state")
    print("\n==== FINE-TUNE STATUS ====")
    print(f"job id      : {args.job}")
    print(f"state       : {state}")
    print(f"output_model: {job.get('output_model', OUTPUT_MODEL)}")
    if state == "JOB_STATE_FAILED":
        print(f"status      : {json.dumps(job.get('status'))}")
    if state not in TERMINAL:
        print("\nJob STILL RUNNING (budget elapsed). Resume polling with:")
        print(f"  /opt/anaconda3/bin/python scripts/fireworks_finetune.py --job {args.job} --budget-min 12 --smoke")
        return

    if state == "JOB_STATE_COMPLETED" and args.smoke:
        smoke(key, args.model)


if __name__ == "__main__":
    main()

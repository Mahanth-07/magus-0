#!/usr/bin/env python3
"""Deploy the fine-tuned Tetris LoRA on Fireworks and smoke-test it.

IMPORTANT — serverless is NOT available for the base llama-v3.2-3b-instruct on
this account (a direct /chat/completions to the base returns NOT_FOUND,
"not deployed"). Serverless LoRA addons attach only to serverless base models,
so this LoRA must be loaded onto a DEDICATED on-demand deployment of the base.
The inference call shape (and the A/B) is identical either way — you just call
the addon's model id, which routes through the deployment.

Path (all via firectl, the reliable route; this script wraps it):
  1. firectl deployment create accounts/fireworks/models/llama-v3p2-3b-instruct \
         --enable-addons --accelerator-type NVIDIA_A100_80GB --wait
  2. firectl load-lora accounts/mahanthk7/models/ludus-tetris \
         --deployment <deployment-id> --wait
  3. POST a board-state prompt to .../inference/v1/chat/completions with model
     accounts/mahanthk7/models/ludus-tetris and parse via parse_decision.

Usage (anaconda python; needs FIREWORKS_API_KEY in .env, firectl on PATH):
    PATH="$PWD/.local/bin:$PATH" /opt/anaconda3/bin/python scripts/fireworks_deploy.py
    # smoke only against an already-deployed addon:
    /opt/anaconda3/bin/python scripts/fireworks_deploy.py --smoke-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.cli import _load_dotenv

ACCOUNT = "mahanthk7"
BASE_MODEL = "accounts/fireworks/models/llama-v3p2-3b-instruct"
LORA_MODEL = f"accounts/{ACCOUNT}/models/ludus-tetris"
INFERENCE_BASE = "https://api.fireworks.ai/inference/v1"


def _firectl(*args: str) -> str:
    cmd = ["firectl", *args]
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out)
    if r.returncode != 0:
        raise SystemExit(f"firectl failed (exit {r.returncode})")
    return out


def deploy(accelerator: str) -> None:
    # 1. dedicated deployment of the base, addons enabled.
    out = _firectl(
        "deployment", "create", BASE_MODEL,
        "--enable-addons", "--accelerator-type", accelerator,
        "--min-replica-count", "1", "--max-replica-count", "1",
        "--wait", "-o", "json",
    )
    # parse the deployment resource name out of the json
    import json as _json
    dep = _json.loads(out[out.index("{"): out.rindex("}") + 1])
    dep_name = dep.get("name", "")
    print(f"deployment: {dep_name} state={dep.get('state')}")
    # 2. load the LoRA addon onto it.
    _firectl("load-lora", LORA_MODEL, "--deployment", dep_name, "--wait")


def smoke(key: str, model: str, retries: int = 6) -> bool:
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
    for attempt in range(retries):
        r = httpx.post(
            f"{INFERENCE_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=90,
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            print(f"  raw: {content[:300]}")
            try:
                d = parse_decision(content)
            except Exception as e:  # noqa: BLE001
                print(f"  parse_decision FAILED: {e}")
                return False
            print(f"  parsed: action={d.action!r} actions={d.actions} conf={d.confidence}")
            ok = bool(d.actions)
            print(f"  smoke {'OK' if ok else 'WARN (empty actions)'}")
            return ok
        print(f"  attempt {attempt+1}/{retries}: HTTP {r.status_code} {r.text[:160]}")
        time.sleep(15)
    return False


def main() -> None:
    _load_dotenv()
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accelerator-type", default="NVIDIA_A100_80GB")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--model", default=LORA_MODEL)
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY not set (expected in .env)")

    if not args.smoke_only:
        deploy(args.accelerator_type)
    ok = smoke(key, args.model)
    print(f"\nDEPLOY+SMOKE {'PASS' if ok else 'FAIL'}; set FIREWORKS_MODEL={args.model}")


if __name__ == "__main__":
    main()

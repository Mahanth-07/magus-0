#!/usr/bin/env python3
"""Launch a Nebius LoRA fine-tune of the TEXT-only Tetris distillation student.

Uploads the text-SFT train/val files, creates a LoRA fine-tuning job on
meta-llama/Llama-3.2-3B-Instruct, then polls until terminal or a time budget,
printing every status transition. On success it prints the fine-tuned model id —
that string is what you set as NEBIUS_MODEL to drive the student in the loop
(with NEBIUS_MULTIMODAL=0, since this student is text-only).

API shape (Nebius Token Factory, OpenAI-compatible — verified from the docs):
  - POST /v1/files          multipart, purpose=fine-tune  -> {"id": "file-..."}
  - POST /v1/fine_tuning/jobs  {model, training_file, validation_file,
        hyperparameters:{lora:true, lora_r, lora_alpha, lora_dropout,
                         n_epochs, learning_rate, batch_size, ...}}  -> {"id": "ftjob-..."}
  - GET  /v1/fine_tuning/jobs/{id}  -> {"status": ..., "fine_tuned_model": ...|null, ...}
    status enum: validating_files, queued, running, succeeded, failed, cancelled
    terminal: succeeded / failed / cancelled

This script ONLY hits the network (no browser). Run with a python that has httpx
and the .env loaded (NEBIUS_API_KEY, NEBIUS_BASE_URL):

    /opt/anaconda3/bin/python scripts/finetune_nebius.py
    /opt/anaconda3/bin/python scripts/finetune_nebius.py --no-create --job ftjob-xxx   # resume polling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

# repo root on path so `import ludus...` resolves when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.cli import _load_dotenv  # reuse the no-dep .env loader

DATA_DIR = REPO_ROOT / "data" / "tetris"
DEFAULT_TRAIN = DATA_DIR / "nebius_text_train.jsonl"
DEFAULT_VAL = DATA_DIR / "nebius_text_val.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

TERMINAL = {"succeeded", "failed", "cancelled"}


def _client(base: str, key: str) -> httpx.Client:
    return httpx.Client(
        base_url=base.rstrip("/"),
        headers={"Authorization": f"Bearer {key}"},
        timeout=120.0,
    )


def upload_file(client: httpx.Client, path: Path) -> str:
    """POST /v1/files (multipart, purpose=fine-tune). Returns the file id."""
    with path.open("rb") as fh:
        r = client.post(
            "/files",
            files={"file": (path.name, fh, "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
    if r.status_code >= 400:
        raise SystemExit(f"file upload failed ({r.status_code}) for {path.name}: {r.text}")
    fid = r.json()["id"]
    print(f"uploaded {path.name} -> {fid}")
    return fid


def create_job(
    client: httpx.Client,
    train_id: str,
    val_id: str | None,
    *,
    model: str,
    n_epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict:
    """POST /v1/fine_tuning/jobs with LoRA hyperparameters. Returns the job object.

    LoRA is enabled by hyperparameters.lora=true (flat object, Nebius shape), NOT a
    nested method/lora block. If the API rejects the request we surface the exact
    error body and stop (per the run constraints) rather than guessing.
    """
    hyper = {
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora": True,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }
    body: dict = {"model": model, "training_file": train_id, "hyperparameters": hyper}
    if val_id:
        body["validation_file"] = val_id

    print(f"creating LoRA job: model={model} hyper={json.dumps(hyper)}")
    r = client.post("/fine_tuning/jobs", json=body)
    if r.status_code >= 400:
        # Explicit, do-not-fake-success failure path.
        raise SystemExit(
            f"\nFINE-TUNE CREATE FAILED ({r.status_code}).\n"
            f"request body: {json.dumps(body)}\n"
            f"response    : {r.text}\n"
        )
    return r.json()


def get_job(client: httpx.Client, job_id: str) -> dict:
    r = client.get(f"/fine_tuning/jobs/{job_id}")
    if r.status_code >= 400:
        raise SystemExit(f"job fetch failed ({r.status_code}): {r.text}")
    return r.json()


def _model_id_from(job: dict) -> str | None:
    """Best-effort extraction of the resulting fine-tuned model id across the field
    names Nebius/OpenAI may use (the OpenAI-canonical field is fine_tuned_model;
    Nebius checkpoints expose fine_tuned_model_checkpoint / result_files)."""
    for key in ("fine_tuned_model", "fine_tuned_model_checkpoint"):
        val = job.get(key)
        if val:
            return val
    rf = job.get("result_files")
    if isinstance(rf, list) and rf:
        return rf[0]
    return None


def poll(client: httpx.Client, job_id: str, budget_s: float, interval_s: float = 12.0) -> dict:
    """Poll GET /v1/fine_tuning/jobs/{id} until terminal OR budget elapsed.

    Prints each status transition. Returns the last fetched job object (caller
    decides whether it's terminal). Does NOT hang past the budget.
    """
    deadline = time.monotonic() + budget_s
    last_status = None
    job = get_job(client, job_id)
    while True:
        status = job.get("status")
        if status != last_status:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] status: {last_status or '(start)'} -> {status}")
            last_status = status
        if status in TERMINAL:
            return job
        if time.monotonic() >= deadline:
            print(f"[budget ~{int(budget_s)}s elapsed] still '{status}'; stopping poll (not terminal)")
            return job
        # sleep but don't overshoot the deadline
        time.sleep(min(interval_s, max(1.0, deadline - time.monotonic())))
        job = get_job(client, job_id)


def main() -> None:
    _load_dotenv()
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--n-epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--budget-min", type=float, default=8.0, help="max minutes to poll before stopping")
    ap.add_argument("--no-create", action="store_true", help="skip upload+create; just poll --job")
    ap.add_argument("--job", default=None, help="existing job id to poll (with --no-create)")
    args = ap.parse_args()

    base = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")
    key = os.environ.get("NEBIUS_API_KEY", "")
    if not key:
        raise SystemExit("NEBIUS_API_KEY not set (expected in .env)")

    budget_s = args.budget_min * 60.0
    client = _client(base, key)
    try:
        if args.no_create:
            if not args.job:
                raise SystemExit("--no-create requires --job <id>")
            job_id = args.job
            print(f"resuming poll for job {job_id}")
        else:
            if not args.train.exists():
                raise SystemExit(
                    f"train file not found: {args.train}\n"
                    f"run: python scripts/export_nebius.py --mode text"
                )
            train_id = upload_file(client, args.train)
            val_id = upload_file(client, args.val) if args.val.exists() else None
            if val_id is None:
                print(f"[warn] no val file at {args.val}; creating job without validation_file")

            job = create_job(
                client, train_id, val_id,
                model=args.model,
                n_epochs=args.n_epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            job_id = job["id"]
            print(f"\nJOB CREATED: {job_id}  (status={job.get('status')})")

        job = poll(client, job_id, budget_s)
        status = job.get("status")

        print("\n==== FINE-TUNE RESULT ====")
        print(f"job id : {job_id}")
        print(f"status : {status}")
        if status == "succeeded":
            model_id = _model_id_from(job)
            print(f"fine_tuned_model: {model_id}")
            print("\nTo drive the student:")
            print(f'  export NEBIUS_MODEL="{model_id}"')
            print("  export NEBIUS_MULTIMODAL=0   # text-only student")
        elif status == "failed":
            print(f"error  : {json.dumps(job.get('error'))}")
        elif status not in TERMINAL:
            print("Job is STILL RUNNING (budget elapsed). Resume polling with:")
            print(f"  python scripts/finetune_nebius.py --no-create --job {job_id}")
        # full terminal/last object for the record (model id, checkpoints, etc.)
        print("\n--- job object ---")
        print(json.dumps(job, indent=2, default=str))
    finally:
        client.close()


if __name__ == "__main__":
    main()

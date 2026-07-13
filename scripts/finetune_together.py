#!/usr/bin/env python3
"""Launch a Together AI VLM LoRA fine-tune of the Stage-4 student.

Uploads the vision-SFT train/val files (together_train.jsonl / together_val.jsonl),
creates a LoRA fine-tuning job on Qwen/Qwen3-VL-8B-Instruct, then polls until
terminal or a time budget, printing every status transition.

On SUCCESS the script prints the fine-tuned model id and the exact env-vars to
set so the student can be served via the Together OpenAI-compatible API. If the
succeeded job has no servable model id (the id starts with "ft-" or is null),
the script FAILS LOUDLY — do not guess or invent a model id.

API (Together OpenAI-compatible, authenticated with TOGETHER_API_KEY):
  Files  : POST /v1/files   multipart, purpose=fine-tune
  Jobs   : POST /v1/fine_tuning/jobs  {model, training_file, ...}
  Poll   : GET  /v1/fine_tuning/jobs/{id}
  Status : queued / running / pending / compiling / uploading / cancel_requested
           / succeeded / failed / cancelled / error
  Terminal: succeeded / failed / cancelled / error

Run:
    python scripts/finetune_together.py
    python scripts/finetune_together.py --no-create --job ft-abc123   # resume poll
    python scripts/finetune_together.py --budget-min 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.cli import _load_dotenv  # reuse the no-dep .env loader

DATA_DIR = REPO_ROOT / "data" / "student"
DEFAULT_TRAIN = DATA_DIR / "together_train.jsonl"
DEFAULT_VAL = DATA_DIR / "together_val.jsonl"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

TERMINAL = {"succeeded", "failed", "cancelled", "error", "completed"}


def _client(api_key: str):
    import together
    return together.Together(api_key=api_key)


def upload_file(client, path: Path) -> str:
    """Upload a JSONL file to Together's files API. Returns the file id.

    The Together SDK's files.upload() accepts a Path or str (not a tuple).
    check=True enables client-side format validation before sending.
    """
    print(f"uploading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
    resp = client.files.upload(
        file=path,
        purpose="fine-tune",
        check=True,
    )
    fid = resp.id
    print(f"  -> {fid}")
    return fid


def create_job(
    client,
    train_id: str,
    val_id: str | None,
    *,
    model: str,
    n_epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    suffix: str,
) -> object:
    """Create a LoRA fine-tuning job. Returns the job response object.

    Fails LOUDLY on any API error (per-contract: no silent fallbacks).
    train_vision=False keeps costs lower (language layers only, vision encoder frozen).
    """
    kwargs = dict(
        training_file=train_id,
        model=model,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        lora=True,
        lora_r=lora_r,
        lora_alpha=float(lora_alpha),
        lora_dropout=lora_dropout,
        train_vision=False,  # freeze vision encoder — language-only LoRA
        suffix=suffix,
    )
    if val_id:
        kwargs["validation_file"] = val_id

    print(f"creating LoRA job: model={model}")
    print(f"  hyper: n_epochs={n_epochs}, lr={learning_rate}, lora_r={lora_r}, "
          f"lora_alpha={lora_alpha}, dropout={lora_dropout}", flush=True)

    try:
        job = client.fine_tuning.create(**kwargs)
    except Exception as exc:
        raise SystemExit(
            f"\nFINE-TUNE CREATE FAILED.\n"
            f"kwargs: {json.dumps({k: v for k, v in kwargs.items() if k != 'training_file'})}\n"
            f"error : {exc}\n"
        ) from exc

    return job


def get_job(client, job_id: str) -> dict:
    """Fetch latest job state as a dict."""
    try:
        job = client.fine_tuning.retrieve(job_id)
        # Together SDK returns an object; convert to dict for uniform access
        return job.model_dump() if hasattr(job, "model_dump") else dict(job)
    except Exception as exc:
        raise SystemExit(f"job fetch failed: {exc}") from exc


def _model_id_from(job: dict) -> str | None:
    """Extract the servable fine-tuned model id from a succeeded/completed job dict.

    Together AI uses several field names across API versions:
      model_output_name, x_model_output_name, output_name, fine_tuned_model

    Never returns a raw file id (starting with "file-"). Returns None if no
    servable id is found — the caller then fails LOUDLY.
    """
    for key in ("model_output_name", "x_model_output_name", "output_name", "fine_tuned_model"):
        val = job.get(key)
        if val and isinstance(val, str) and not val.startswith("file-"):
            return val
    return None


def poll(client, job_id: str, budget_s: float, interval_s: float = 15.0) -> dict:
    """Poll until terminal or budget elapsed. Returns last fetched job dict."""
    deadline = time.monotonic() + budget_s
    last_status = None
    job = get_job(client, job_id)
    while True:
        status = job.get("status")
        if status != last_status:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] status: {last_status or '(start)'} -> {status}", flush=True)
            last_status = status
        if status in TERMINAL:
            return job
        if time.monotonic() >= deadline:
            print(f"[budget ~{int(budget_s)}s elapsed] still '{status}'; stopping poll (not terminal)")
            return job
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
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--suffix", default="magus-student-v1")
    ap.add_argument("--budget-min", type=float, default=20.0,
                    help="max minutes to poll before stopping (default 20)")
    ap.add_argument("--no-create", action="store_true",
                    help="skip upload+create; just poll --job")
    ap.add_argument("--job", default=None,
                    help="existing job id to poll (with --no-create)")
    args = ap.parse_args()

    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise SystemExit("TOGETHER_API_KEY not set (expected in .env)")

    budget_s = args.budget_min * 60.0
    client = _client(key)

    if args.no_create:
        if not args.job:
            raise SystemExit("--no-create requires --job <id>")
        job_id = args.job
        print(f"resuming poll for job {job_id}")
    else:
        if not args.train.exists():
            raise SystemExit(
                f"train file not found: {args.train}\n"
                f"run: python scripts/export_together.py"
            )

        train_id = upload_file(client, args.train)
        val_id = upload_file(client, args.val) if args.val.exists() else None
        if val_id is None:
            print(f"[warn] no val file at {args.val}; creating job without validation_file")

        job_obj = create_job(
            client, train_id, val_id,
            model=args.model,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            suffix=args.suffix,
        )
        # SDK returns an object; normalise to dict
        job_dict = job_obj.model_dump() if hasattr(job_obj, "model_dump") else dict(job_obj)
        job_id = job_dict.get("id") or job_dict.get("job_id")
        print(f"\nJOB CREATED: {job_id}  (status={job_dict.get('status')})")

    job = poll(client, job_id, budget_s)
    status = job.get("status")

    print("\n==== FINE-TUNE RESULT ====")
    print(f"job id : {job_id}")
    print(f"status : {status}")

    if status in ("succeeded", "completed"):
        model_id = _model_id_from(job)
        if model_id is None:
            print("\n--- job object ---")
            print(json.dumps(job, indent=2, default=str))
            raise SystemExit(
                "\nJOB SUCCEEDED but NO SERVABLE MODEL ID was returned.\n"
                "Do NOT use a 'file-...' id — it cannot serve completions.\n"
                "Check the Together console for the deployed model id."
            )
        print(f"fine_tuned_model: {model_id}")
        print("\nTo drive the student (Together OpenAI-compatible endpoint):")
        print(f'  export TOGETHER_STUDENT_MODEL="{model_id}"')
        print(f'  export TOGETHER_API_KEY="{key[:8]}..."')
        print("  # Base URL: https://api.together.xyz/v1 (OpenAI-compatible)")
        print("  # wire a TogetherProvider (mirror NebiusProvider) pointing to this model")
    elif status == "failed":
        print(f"error  : {json.dumps(job.get('error') or job.get('failure_reason'))}")
    elif status not in TERMINAL:
        print("Job is STILL RUNNING (budget elapsed). Resume polling with:")
        print(f"  python scripts/finetune_together.py --no-create --job {job_id}")

    print("\n--- job object ---")
    print(json.dumps(job, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export data/student/sft_v1.jsonl to Together AI VLM fine-tuning format.

Input rows (sft_v1.jsonl):
  {"image": <abs png path>, "system": str, "user": str, "assistant": str(JSON), "meta": {...}}

Output format (Together conversational VLM SFT):
  {"messages": [
    {"role": "system",    "content": [{"type": "text", "text": <system>}]},
    {"role": "user",      "content": [{"type": "text", "text": <user>},
                                      {"type": "image_url",
                                       "image_url": {"url": "data:image/jpeg;base64,..."}}]},
    {"role": "assistant", "content": [{"type": "text", "text": <assistant>}]}
  ]}

Images are resized to max 512px longest-side and JPEG-encoded (q=80) to stay
well under Together's 10 MB per-image cap. Train/val split via --val-frac /
--seed (mirrors export_nebius.py patterns).

Run:
    python scripts/export_together.py
    python scripts/export_together.py --val-frac 0.1 --seed 13
    python scripts/export_together.py --in data/student/sft_v1.jsonl --out data/student/together_sft.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_IN = REPO_ROOT / "data" / "student" / "sft_v1.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "student" / "together_sft.jsonl"

MAX_PX = 512   # longest side after downscale
JPEG_Q = 80    # JPEG quality


def _encode_image(image_bytes: bytes) -> str:
    """Downscale PNG bytes to max 512px longest-side, JPEG-encode at q=80,
    and return a data-URI string suitable for Together's image_url field.

    Raises PIL.UnidentifiedImageError for stub/fake images (e.g. synthetic
    games) — callers should catch and skip those rows.
    """
    img = Image.open(io.BytesIO(image_bytes))
    # Convert palette / RGBA modes before JPEG encoding
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_PX:
        scale = MAX_PX / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_Q, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def row_to_together(row: dict, image_bytes: bytes) -> dict:
    """Convert one sft_v1 row + its raw image bytes into a Together vision SFT row.

    Pure function (no I/O): caller supplies bytes so this is unit-testable with
    fake PNG bytes. The image is downscaled + JPEG-encoded inside _encode_image.
    System/user/assistant come directly from the pre-built fields in sft_v1.jsonl
    (already built by collect_trajectories + build_dataset at the exact runtime
    prompt distribution — no reconstruction needed).
    """
    data_uri = _encode_image(image_bytes)
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": row["system"]}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": row["user"]},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["assistant"]}],
            },
        ]
    }


def _read_rows(in_path: Path) -> list[dict]:
    rows = []
    with in_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split_paths(out_path: Path) -> tuple[Path, Path]:
    """Derive sibling train/val paths from the combined output path.

    together_sft.jsonl -> together_train.jsonl + together_val.jsonl
    """
    stem = out_path.stem   # "together_sft"
    suffix = out_path.suffix  # ".jsonl"
    base = stem[: -len("_sft")] if stem.endswith("_sft") else stem  # "together"
    return (
        out_path.with_name(f"{base}_train{suffix}"),
        out_path.with_name(f"{base}_val{suffix}"),
    )


def export(
    in_path: Path,
    out_path: Path,
    val_frac: float = 0.1,
    seed: int = 13,
) -> dict:
    """Read sft_v1.jsonl, convert each row (with downscaled image), write train+val splits.

    Returns {"total", "train", "val", "train_path", "val_path", "out"}.
    Images are loaded from the absolute paths stored in each row's "image" field.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(in_path)

    # Deterministic shuffle for split
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(rows) * val_frac)) if rows else 0
    val_idx = set(order[:n_val])

    train_path, val_path = _split_paths(out_path)
    n_train = n_val_written = 0

    skipped = 0
    with (
        out_path.open("w") as fout,
        train_path.open("w") as ftrain,
        val_path.open("w") as fval,
    ):
        for i, row in enumerate(rows):
            img_path = Path(row["image"])
            image_bytes = img_path.read_bytes()
            try:
                converted = row_to_together(row, image_bytes)
            except Exception:
                # Synthetic/stub images (e.g. b'\x89PNG synthetic-frame') are
                # not valid PNGs — skip silently, count for reporting.
                skipped += 1
                continue
            line = json.dumps(converted) + "\n"
            fout.write(line)
            if i in val_idx:
                fval.write(line)
                n_val_written += 1
            else:
                ftrain.write(line)
                n_train += 1

            if (i + 1) % 100 == 0:
                print(f"  converted {i + 1}/{len(rows)} rows...", flush=True)

    return {
        "total": len(rows),
        "written": n_train + n_val_written,
        "skipped": skipped,
        "train": n_train,
        "val": n_val_written,
        "out": out_path,
        "train_path": train_path,
        "val_path": val_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of rows in validation split (default 0.1)")
    ap.add_argument("--seed", type=int, default=13,
                    help="Random seed for train/val shuffle (default 13)")
    args = ap.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"input not found: {args.in_path}")

    print(f"Converting {args.in_path} -> {args.out_path}")
    print(f"  val_frac={args.val_frac}, seed={args.seed}, max_px={MAX_PX}, jpeg_q={JPEG_Q}")

    info = export(args.in_path, args.out_path, val_frac=args.val_frac, seed=args.seed)

    train_mb = info["train_path"].stat().st_size / 1024 / 1024
    val_mb = info["val_path"].stat().st_size / 1024 / 1024
    total_mb = info["out"].stat().st_size / 1024 / 1024

    print(f"\nWrote {info['written']}/{info['total']} rows -> {info['out']} ({total_mb:.1f} MB)")
    if info["skipped"]:
        print(f"  skipped {info['skipped']} rows with invalid/stub images")
    print(f"  train: {info['train']} rows -> {info['train_path']} ({train_mb:.1f} MB)")
    print(f"  val  : {info['val']} rows -> {info['val_path']} ({val_mb:.1f} MB)")


if __name__ == "__main__":
    main()

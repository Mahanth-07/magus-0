#!/usr/bin/env python3
"""Export the teacher self-play dataset to Nebius/OpenAI SFT format.

Two export modes over data/tetris/examples.jsonl, each one JSON object per line
with a "messages" array (system / user / assistant):

VISION (default, --mode vision) -> data/tetris/nebius_sft.jsonl
  system    : the _SYSTEM preamble from ludus.providers.insforge
  user      : [ {text: build_user_text(objective, legal_actions, state_text)},
                {image_url: data:image/png;base64,<the example image>} ]
  assistant : json.dumps(target)   (the Decision dict the student should produce)

TEXT (--mode text) -> data/tetris/nebius_text_sft.jsonl (+ a train/val split)
  system    : the _SYSTEM preamble
  user      : build_user_text(objective, legal_actions, state_text)   (a STRING)
  assistant : json.dumps(target)   (a STRING)
  Same prompt, but text-only so a non-multimodal student (e.g. Llama-3.2-3B-Instruct)
  can be LoRA-fine-tuned on it. The student sees the board via state_text at inference.

In BOTH modes the user TEXT is built with the SAME helper the runtime uses
(build_user_text), so the student trains on the exact prompt it will see at
inference — the vision mode additionally embeds the live screenshot as a data URI.

Run:
    python scripts/export_nebius.py                       # vision (default)
    python scripts/export_nebius.py --mode text           # text-only + 90/10 split
    python scripts/export_nebius.py --mode text --val-frac 0.1 --seed 7
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

# repo root on path so `import ludus...` works when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.providers.insforge import _SYSTEM, build_user_text

DEFAULT_IN = REPO_ROOT / "data" / "tetris" / "examples.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "tetris" / "nebius_sft.jsonl"
DEFAULT_TEXT_OUT = REPO_ROOT / "data" / "tetris" / "nebius_text_sft.jsonl"


def example_to_sft(example: dict, image_bytes: bytes) -> dict:
    """Convert one collected example + its image bytes into a Nebius vision SFT row.

    Pure function (no I/O) so it's unit-testable: the caller supplies the image
    bytes. The user text matches the runtime prompt (build_user_text); the
    assistant content is the JSON the student should emit (json.dumps(target)).
    """
    b64 = base64.b64encode(image_bytes).decode()
    user_text = build_user_text(
        objective=example["objective"],
        legal_actions=example["legal_actions"],
        state_text=example.get("state_text", ""),
    )
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
            {"role": "assistant", "content": json.dumps(example["target"])},
        ]
    }


def example_to_text_sft(example: dict) -> dict:
    """Convert one collected example into a TEXT-only Nebius/OpenAI SFT row.

    Pure function (no I/O), no image. All three message contents are plain STRINGS:
      system    -> _SYSTEM (the runtime preamble)
      user      -> build_user_text(...) (the EXACT runtime user prompt, sans image)
      assistant -> json.dumps(target) (the Decision JSON the student should emit)

    This is what a non-multimodal student (Llama-3.2-3B-Instruct) trains on; at
    inference it gets the board through state_text inside the user prompt.
    """
    user_text = build_user_text(
        objective=example["objective"],
        legal_actions=example["legal_actions"],
        state_text=example.get("state_text", ""),
    )
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(example["target"])},
        ]
    }


def export(in_path: Path, out_path: Path) -> int:
    """Read examples.jsonl, embed each image, write nebius_sft.jsonl. Returns count."""
    base_dir = in_path.parent
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            img_path = base_dir / example["image"]
            image_bytes = img_path.read_bytes()
            fout.write(json.dumps(example_to_sft(example, image_bytes)) + "\n")
            written += 1
    return written


def _read_examples(in_path: Path) -> list[dict]:
    examples = []
    with in_path.open() as fin:
        for line in fin:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _split_paths(out_path: Path) -> tuple[Path, Path]:
    """Derive train/val sibling paths from the combined text-SFT output path.

    nebius_text_sft.jsonl -> nebius_text_train.jsonl + nebius_text_val.jsonl
    """
    stem = out_path.name
    suffix = out_path.suffix  # ".jsonl"
    base = stem[: -len(suffix)] if suffix else stem  # "nebius_text_sft"
    base = base[: -len("_sft")] if base.endswith("_sft") else base  # "nebius_text"
    return out_path.with_name(f"{base}_train{suffix}"), out_path.with_name(f"{base}_val{suffix}")


def export_text(in_path: Path, out_path: Path, val_frac: float, seed: int) -> dict:
    """Write the combined text-SFT file plus a shuffled train/val split.

    Returns {"total", "train", "val", "out", "train_path", "val_path"}.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples = _read_examples(in_path)
    rows = [example_to_text_sft(ex) for ex in examples]

    # combined file (all rows, source order)
    with out_path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    # deterministic shuffle, then carve off the validation tail
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(rows) * val_frac)) if rows else 0
    val_idx = set(order[:n_val])

    train_path, val_path = _split_paths(out_path)
    n_train = n_written_val = 0
    with train_path.open("w") as ftrain, val_path.open("w") as fval:
        for i, row in enumerate(rows):
            line = json.dumps(row) + "\n"
            if i in val_idx:
                fval.write(line)
                n_written_val += 1
            else:
                ftrain.write(line)
                n_train += 1

    return {
        "total": len(rows),
        "train": n_train,
        "val": n_written_val,
        "out": out_path,
        "train_path": train_path,
        "val_path": val_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["vision", "text"], default="vision",
                    help="vision (image data-URI) or text-only SFT export")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", type=Path, default=None,
                    help="output jsonl (defaults: vision->nebius_sft.jsonl, text->nebius_text_sft.jsonl)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="text mode: validation fraction (default 0.1)")
    ap.add_argument("--seed", type=int, default=13, help="text mode: split shuffle seed")
    args = ap.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"input not found: {args.in_path} (run scripts/collect_dataset.py first)")

    if args.mode == "text":
        out_path = args.out_path or DEFAULT_TEXT_OUT
        info = export_text(args.in_path, out_path, val_frac=args.val_frac, seed=args.seed)
        print(f"Wrote {info['total']} text-SFT rows -> {info['out']}")
        print(f"  train: {info['train']} -> {info['train_path']}")
        print(f"  val  : {info['val']} -> {info['val_path']}")
    else:
        out_path = args.out_path or DEFAULT_OUT
        n = export(args.in_path, out_path)
        print(f"Wrote {n} vision-SFT rows -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the teacher self-play dataset to Nebius/OpenAI vision SFT format.

Pure transform: data/tetris/examples.jsonl (+ the referenced PNG per example) ->
data/tetris/nebius_sft.jsonl, one JSON object per line with a "messages" array:

  system    : the _SYSTEM preamble from ludus.providers.insforge
  user      : [ {text: build_user_text(objective, legal_actions, state_text)},
                {image_url: data:image/png;base64,<the example image>} ]
  assistant : json.dumps(target)   (the Decision dict the student should produce)

The user TEXT is built with the SAME helper the runtime uses (build_user_text), so
the student trains on the exact prompt it will see at inference — minus the live
screenshot, which is embedded here as a data URI instead.

Run:
    python scripts/export_nebius.py            # default in/out under data/tetris/
    python scripts/export_nebius.py --in data/tetris/examples.jsonl --out data/tetris/nebius_sft.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

# repo root on path so `import ludus...` works when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.providers.insforge import _SYSTEM, build_user_text

DEFAULT_IN = REPO_ROOT / "data" / "tetris" / "examples.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "tetris" / "nebius_sft.jsonl"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"input not found: {args.in_path} (run scripts/collect_dataset.py first)")

    n = export(args.in_path, args.out_path)
    print(f"Wrote {n} SFT rows -> {args.out_path}")


if __name__ == "__main__":
    main()

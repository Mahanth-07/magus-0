import sys
from pathlib import Path

# Put the repo root on sys.path so tests can import the `scripts/` package
# (collect_dataset / export_nebius). The `ludus` package is already importable
# via its editable install; `scripts/` is not packaged, so we expose it here.
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

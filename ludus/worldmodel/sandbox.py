"""Contained execution for LLM-written world models.

Two layers (spec: "the agent runs code it wrote — contain it"):
  1. check_source: AST gate — imports outside the allowlist, dangerous names
     (eval/exec/open/...), and syntax errors are rejected BEFORE execution.
  2. run_predictions: the source runs in a SUBPROCESS with a hard timeout;
     a hang or crash marks the candidate failed, never the agent. Cases are
     batched per subprocess call (one interpreter spawn per validation pass,
     not per transition).

Honest limits: this is crash/hang isolation plus a static gate, not a
security boundary against a genuinely adversarial payload."""

from __future__ import annotations

import ast
import json
import subprocess
import sys

ALLOWED_IMPORTS = {"math", "copy", "itertools", "collections"}
FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "input", "__import__",
                   "globals", "locals", "vars", "breakpoint"}

# Runs inside the subprocess: reads {"source", "cases"} JSON on stdin, execs
# the model source, calls predict per case, writes a JSON list on stdout
# (per-case result or null on per-case error).
_RUNNER = r"""
import json, sys
payload = json.load(sys.stdin)
ns = {}
exec(payload["source"], ns)
predict = ns.get("predict")
out = []
for case in payload["cases"]:
    if predict is None:
        out.append(None)
        continue
    try:
        out.append(predict(case["state"], case["action"]))
    except Exception:
        out.append(None)
print(json.dumps(out))
"""


def check_source(source: str) -> list[str]:
    """Static gate. Returns a list of violation strings ([] = clean)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    violations.append(f"forbidden import: {root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                violations.append(f"forbidden import: {root}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            violations.append(f"forbidden attribute: {node.attr}")
    return violations


def run_predictions(source: str, cases: list[dict], timeout_s: float = 30.0) -> list:
    """predict(state, action) per case, batched in one subprocess.
    Returns a list aligned with `cases`; None marks a per-case error. A
    timeout, crash, or gate violation returns all-None (candidate failed)."""
    if check_source(source):
        return [None] * len(cases)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=json.dumps({"source": source, "cases": cases}),
            capture_output=True, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            return [None] * len(cases)
        out = json.loads(proc.stdout)
        if not isinstance(out, list) or len(out) != len(cases):
            return [None] * len(cases)
        return out
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return [None] * len(cases)

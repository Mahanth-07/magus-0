"""Plain-text chat seam for code synthesis (the inducer's LLM).

The Decision-shaped providers in ludus/providers are the wrong interface for
synthesis — this is system+user string in, raw text out. Backends:
  anthropic — best-in-class code synthesis (default when key present).
  gateway   — InsForge model gateway (same endpoint the vision provider uses).
Model/backend via env: LUDUS_SYNTH_BACKEND, LUDUS_SYNTH_MODEL."""

from __future__ import annotations

import os

import httpx

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def default_backend() -> str:
    if os.environ.get("LUDUS_SYNTH_BACKEND"):
        return os.environ["LUDUS_SYNTH_BACKEND"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "gateway"


def chat(system: str, user: str, *, backend: str | None = None,
         max_tokens: int = 8192, timeout: float = 180.0) -> str:
    backend = backend or default_backend()
    if backend == "anthropic":
        model = os.environ.get("LUDUS_SYNTH_MODEL", DEFAULT_ANTHROPIC_MODEL)
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                     "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    if backend == "gateway":
        base = os.environ.get("INSFORGE_GATEWAY_URL", "")
        r = httpx.post(
            f"{base}/chat/completion",
            headers={"Authorization": f"Bearer {os.environ.get('INSFORGE_API_KEY', '')}"},
            json={"model": os.environ.get("LUDUS_SYNTH_MODEL",
                                          os.environ.get("INSFORGE_VISION_MODEL", "")),
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0},
            timeout=timeout,
        )
        r.raise_for_status()
        body = r.json()
        text = body.get("text")
        if text is None:
            text = body["choices"][0]["message"]["content"]
        return text
    raise ValueError(f"unknown synthesis backend {backend!r}")

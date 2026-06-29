# Magus-0 — Design Spec

**Date:** 2026-06-28
**Status:** Approved (brainstorming → plan)
**Context:** Live 6-hour hackathon build. Target track: GameCraft Arena + **Best Use of InsForge**.

## 1. What Magus-0 is

A **shared, game-agnostic planner** that plays multiple GameWorld browser games by reasoning over
screenshots, choosing **semantic macro-actions**, predicting outcomes, reflecting on prediction error,
and accumulating a compact **learned rulebook**. The same planner drives every game; only thin
per-game adapters differ. Episode traces, decisions, predictions/outcomes, rules, and screenshots
are persisted to **InsForge** (Postgres + S3) with a local fallback, and surfaced on a live view.

The thesis being demonstrated: **memory (learned rules) beats baseline (no rules)** on a verifiable,
state-based metric — across more than one game, using one shared brain.

## 2. Verified external dependencies (real, not assumed)

- **GameWorld** — [gameworld-project/gameworld](https://github.com/gameworld-project/gameworld),
  arXiv [2604.07429](https://arxiv.org/abs/2604.07429). 34 browser games, 170 tasks. Supports
  **generalist agents acting via a semantic action space with deterministic Semantic Action Parsing**
  (what our planner emits) and **outcome-based, state-verifiable evaluation** (reads serialized game
  state — what our metric deltas read). Python 3.12 + Playwright Chromium.
- **InsForge** — [insforge.dev](https://insforge.dev/). Agent-native BaaS: Postgres (+pgvector),
  S3-compatible storage, OpenAI-compatible **model gateway**, edge functions, realtime. Official
  **MCP server for Claude Code**.
- **Unverified, confirm at build time:** the `gameworld-dev/gameworld-games` companion repo from the
  spec's clone command. The main repo exists; confirm the games-library layout before relying on it.

## 3. Tech stack

- **Agent runtime:** Python 3.12, GameWorld + Playwright Chromium, Pydantic v2 (structured model
  output), httpx (API), Tenacity (bounded retries), PyYAML (game configs), pytest (core tests).
- **Live state:** FastAPI + WebSockets (minimal) — or InsForge realtime feeding the view.
- **Model (vision):** provider abstraction, priority order:
  1. **InsForge model gateway (primary)** — serves the "Best Use of InsForge" prize.
  2. Anthropic API (fallback — wired from the start).
  3. OpenAI-compatible endpoint.
  4. **Deterministic Mock (testing / offline loop proof).**
- **Dashboard (flex):** Next.js + TS + Tailwind + Recharts, WebSocket/realtime. Cut-candidate; may
  fall back to GameWorld's built-in monitor.

## 4. Architecture

```
GameWorld screenshot
  → Shared Magus-0 planner (visual reasoning, action selection, expected-result prediction)
  → Semantic macro-action
  → GameWorld semantic-control mapper
  → Browser action
  → New screenshot + evaluator metric delta
  → Outcome detector (expected_result vs actual delta)
  → Prediction-error reflection
  → Compact learned rulebook
  → InsForge persistence (+ local fallback) + live view
```

**Shared across all games (the core, never cut):** visual reasoning, action selection,
expected-result prediction, reflection, memory, persistence, evaluation.

**Game-specific (thin adapters only):** legal action names, control mappings, relevant evaluation
metrics, safe timing bounds. Adapters **must not** contain aiming logic, Tetris placement search,
physics prediction, or any strategy. Strategy lives only in the planner (via the model).

## 5. Components & interfaces

### `Decision` (Pydantic v2) — the model contract
```json
{
  "scene_summary": "string",
  "controlled_entity": "string",
  "current_subgoal": "string",
  "action": "string (must be a legal action for the active game)",
  "action_args": { "duration_ms": 300 },
  "expected_result": "string",
  "reason": "string",
  "confidence": 0.82
}
```
Validation failure → Tenacity retry, feeding the validation error back into the prompt. The planner
**always** returns a parseable `Decision` or fails the step explicitly (never silent).

### `ModelProvider` (protocol)
`complete(prompt, image) -> Decision`. Impls: `InsForgeGatewayProvider`, `AnthropicProvider`,
`OpenAICompatProvider`, `MockProvider`. Selection by config + availability; the gateway has the
Anthropic fallback wired so a flaky/rate-limited gateway never kills the loop.

### `MagusPlanner` (game-agnostic)
Inputs: current screenshot, objective, legal semantic actions, recent action outcomes, learned rules.
Output: `Decision`. Contains all reasoning/reflection/memory logic. Knows nothing about any specific
game's strategy.

### `GameAdapter` (protocol — thin)
`legal_actions`, `semantic_to_gameworld(action, args)` (control mapping), `relevant_metrics`,
`timing_bounds`. Implementations: `TetrisAdapter`, `Wolf3DAdapter`, `FireboyWatergirlAdapter`.

### `OutcomeDetector`
Compares the model's `expected_result` against the actual evaluator **metric delta** between
pre/post screenshots. Produces a prediction-error signal.

### `Reflector` → `Rulebook`
Prediction error → a compact natural-language rule retained across steps and attempts.
Default: **model-based reflection**; cut-fallback: **metric-based reflection** (rules derived from
metric deltas alone, no extra model call).

### Persistence (`InsForgeStore` + `LocalStore`, dual-write)
Every episode/step/decision/prediction/outcome/rule + screenshot writes to **both** InsForge
(Postgres + S3) and local (JSONL + PNG) simultaneously. Loop never blocks on InsForge.

### Modes
- **baseline** — loop with rules withheld.
- **memory** — identical loop with the learned rulebook supplied.
The experiment compares the two on the verifiable metric.

## 6. Game targets (order)

1. **Tetris** (build first — slow, discrete decisions).
   Goal: place ≥8 pieces minimizing holes; clear a line when possible.
   Minimum success: 5 consecutive legal placements; structured outputs always parse; ≥1 learned
   rule retained; both baseline and memory modes run.
2. **Wolf3D.** Goal: kill one enemy while preserving health. Macro-actions: `move_forward`,
   `move_backward`, `turn_left`, `turn_right`, `attack`, `wait`. Tactical decisions every few hundred
   ms — not real-time expert FPS play.
3. **Fireboy & Watergirl** (human-agent co-op). Human controls Fireboy via physical keyboard;
   Playwright injects Watergirl inputs; a browser event listener records recent human actions which
   become planner context. Goal: pass the first coordinated obstacle together.
   **Co-op caveat:** if live human input fights Playwright injection in the same browser, fall back to
   a scripted/recorded partner so the cooperative interaction still demos.
4. **Rocket League 2D** — stretch only; do not start until everything above + memory + persistence
   + one co-op interaction work.

## 7. Build order (6 hours, risk front-loaded)

| Time | Build |
|---|---|
| 0:00–0:30 | Create `~/ludus`; clone GameWorld; conda env + `playwright install chromium`; **inspect real GameWorld API** + confirm games repo; **prove InsForge login + one gateway vision call**; launch Tetris manually. Write `IMPLEMENTATION_PLAN.md` from real APIs. |
| 0:30–1:15 | `Decision` schema; `ModelProvider` (Mock + InsForge gateway + Anthropic fallback); local persistence. Loop runs end-to-end on **Mock** vs Tetris. |
| 1:15–2:00 | Real Tetris loop on **gateway vision** — 5 consecutive legal placements. |
| 2:00–2:45 | `OutcomeDetector`, `Reflector`, `Rulebook`, **baseline vs memory**. (Tetris "do not cut" set complete.) |
| 2:45–3:30 | `Wolf3DAdapter` + measurable run. |
| 3:30–4:15 | Fireboy & Watergirl co-op (human keys + Playwright inject + event-listener context). |
| 4:15–5:00 | InsForge deep: full Postgres traces + S3 screenshots + applied migration. |
| 5:00–5:35 | Live view via InsForge realtime (or GameWorld monitor). |
| 5:35–6:00 | Baseline-vs-memory numbers; screen recording; README; submission. |

## 8. The survival hedge (non-negotiable)

InsForge gateway is on the **critical path of the agent loop**. Therefore:
- The full loop is **proven on `MockProvider` before any network call**.
- The InsForge gateway provider has the **Anthropic fallback wired from the start**.
- Persistence is **dual-write** (InsForge + local), so an InsForge outage degrades to local without
  stopping the demo.

## 9. Cut order (when time slips)

1. Rocket League 2D
2. Custom Next.js dashboard (→ GameWorld monitor / InsForge realtime page)
3. Model-based reflection (→ metric-based reflection)
4. Extra attempts & analytics

**Never cut:** shared planner, Tetris, Wolf3D, explicit expected outcomes, learned rulebook,
baseline vs memory, persistent traces, one cooperative interaction.

## 10. Definition of done (demo)

- Shared planner drives **Tetris + Wolf3D** with parseable structured outputs every step.
- **Baseline-vs-memory** comparison produces real numbers on a state-verifiable metric, for ≥1 game.
- **One cooperative interaction** demos (Fireboy & Watergirl, or its scripted-partner fallback).
- Episode traces, decisions, predictions/outcomes, rules, and screenshots persisted to **InsForge**
  (Postgres + S3), with vision served via the **gateway** — local fallback intact.
- A live view (realtime or monitor), a screen recording, and a README.

## 11. Out of scope

- Real-time expert-level play of any game.
- Any game-specific strategy baked into adapters.
- Anything in `~/ludus` touching the Human Archive repo.

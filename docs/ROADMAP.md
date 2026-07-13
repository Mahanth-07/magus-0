# Magus Roadmap — canonical, current (2026-07-12)

Single source of truth for what's done and what's next. Detailed strategy rationale:
`docs/specs/2026-07-05-stage4-pi-style-student.md` (π0.7 analysis). Session-resume
operational facts (model ids, costs, gotchas): CLAUDE.md "SESSION STATE" sections.
Results evidence: `docs/results/` (sweep-table.md is the scoreboard).

## Done

- **M0** Distillation proof: Tetris student 1022 vs gemini-flash 466 (local 3B LoRA).
- **M1** Auto-onboarding: any GameWorld game → playable GameProfile, zero hand-written config.
- **M2** World-model induction: LLM writes game rules as validated Python (grid INDUCED;
  honest FAIL on tetris with measured wall-clock boundary).
- **M3** Planner + duel harness: beam search over induced models, INDUCED-refusal gate.
- **M4** THE MILESTONE: planner beats zero-shot frontier VLM on real 2048 (212–56).
- **Sweep** 33 games: 12 INDUCED, 3 duel wins; honest failure table.
- **Stage 4a** VLM student v1 (Qwen3-VL-8B LoRA on planner trajectories, Together):
  N=5 verified — WINS doodle-jump vs its own teacher (106.8±0.4 vs 83±19.6),
  near-deterministic everywhere; loses 2048/stack.
- **Stage 4b** RECAP-lite (advantage-conditioned SFT): built + evaluated; honest null
  vs v1 at current data scale.
- **Variance harness**: duel --repeats N, winner by means — N=1 claims retracted/upgraded.
- **Onboarding robustness round 1**: wake-up keys for menu games, empty-controls
  guardrails, honest sweep errors. Re-sweep of 14 dead games IN PROGRESS.

## Next (priority order)

1. **Finish onboarding round 1** — read re-sweep verdicts; graduate any new games
   through induce→duel automatically.
2. **Client-level boot fixes + mouse support** — DONE (2026-07-13). Added 15s
   boot wait + keyboard/mouse wake sequence in `_connect()`, mouse pseudo-keys
   in WAKE_KEYS, `_is_actionable()` accepting "ready"||"playing". 369→385 tests.
   Net from sweep3: 4 of 9 boot-hung games escaped (core-ball DUEL_WON, ovo
   DUEL_LOST, worlds-hardest-game DUEL_TIE, boxel-rebound INDUCTION_FAILED).
   Remaining blocked: 3 mouse-only games (minesweeper, monkey-mart, run-3) need
   DEFAULT_PROBE_KEYS mouse extension; vex-3 needs canvas-specific click coords;
   pacman/worlds-hardest-game-2 hit the explore "ready" bug (fixed, need re-sweep).
3. **Expert-only v2 ablation** — retrain student on expert-labeled episodes only,
   eval at N=5 (~$15). Tests whether novice rows diluted v2.
4. **Scale teacher data + student round 3** — more games, longer horizons; consider
   frontier-VLM reward-filtered trajectories for non-INDUCED games (π0.7's mixed-
   teacher recipe).
5. **Stage 5: Fireboy & Watergirl co-op** — partner-key prediction + joint planning;
   requires human partner (user) present. The demo closer.
6. **Research writeup** — the arc (distillation → induced world models → sweep →
   student-beats-teacher) with all honest negatives; docs/results has everything.

## Standing decisions

- Stack is flexible: Together for VLM fine-tune+serve (Nebius retired — can't serve
  LoRA); local ollama for text students; anaconda python for browser, .venv for tests.
- Every endpoint gets inactive_timeout and explicit DELETE after eval.
- Claims require N≥5 with std. Failures are recorded verbatim next to wins.
- Holo-3.1-4B as alternative student base: deferred until an A/B is worth the RunPod
  setup (revisit at student round 3).

## Long-term arc (the comprehensive plan — quarters, not weeks)

The end state from the original brainstorm: **an agent that catches on to any game
quickly** — general (no per-game engineering), skilled (beats strong baselines), and
eventually cooperative. The staged path there, with the measured boundaries each
stage must break:

### Horizon 1 — GameWorld mastery (current, ~weeks)
Items 1–6 above. Exit criteria: >20/33 games onboarded; >5 duel wins; one student
model that transfers to held-out GameWorld games it never trained on (the true
generalization test — never yet run); co-op demo working.

### Horizon 2 — Surpass the teacher (RL past imitation)
Everything so far caps at the best teacher available. Paths past it, in cost order:
1. **Ranking-function improvement**: tune planner scoring (or a small value net) via
   CMA-ES/self-play against INDUCED world models — no GPUs, pure simulator rollouts.
   The whole pipeline inherits improvements since the planner is the label source.
2. **Full RECAP** (not -lite): iterative rounds — deploy student, collect its own
   episodes, advantage-label, retrain. Games' verifiable rewards + free resets make
   this cheaper than robotics. Needs the variance harness (done) to measure honestly.
3. **Online RL** (PPO-class) against induced world models as simulators — only if
   1–2 plateau; requires the sim-fidelity work below.
Exit criteria: a student that beats ITS OWN teacher's mean on >half the games it
trained on (doodle-jump already shows this is possible).

### Horizon 3 — Beyond the state API (true generality)
GameWorld exposes getState(); the real world doesn't. The StateSource seam was
designed for this from day one:
1. **Vision-derived state**: a VLM (or the student itself) emits structured state
   from pixels; everything downstream (induction, planning, validation) unchanged.
2. **Pixels-only games**: itch.io/CrazyGames titles with no API — onboarding via
   vision-state + mouse/keyboard probing (mouse support lands in Horizon 1 item 2).
3. **Time-aware world models**: predict(state, action, dt) — breaks the measured
   wall-clock boundary (tetris gravity, runners) that blocks ~1/3 of induction.
Exit criteria: one game with NO state API onboarded, induced, and dueled.

### Horizon 4 — The foundation-model claim
Consolidation: one model, many games, fast adaptation to new ones.
1. **Multi-game student at scale**: train across every INDUCED game + reward-filtered
   frontier trajectories elsewhere; measure few-shot adaptation (episodes-to-competence
   on an unseen game) as the headline metric.
2. **In-context game learning**: can the student, given a few exploration transitions
   in-prompt, play a new game without weight updates? (The induction pipeline becomes
   an inference-time tool.)
3. **Writeup/artifact**: the full arc as a paper-style report — code-as-world-model
   induction over a 33-game benchmark, honest boundary map, student-beats-teacher
   distillation, RECAP-at-small-scale null — plus the repo as the reproducible artifact.

### Non-goals (explicit, to protect focus)
- Real-money game APIs, anti-cheat evasion, or multiplayer beyond the co-op demo.
- Robotics transfer (π0.7 comparisons stay analogical).
- Beating specialized game AI (Stockfish-class engines) — the claim is generality.

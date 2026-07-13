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
2. **Client-level boot fixes** (new bucket found today): games hanging in
   wait_until_actionable at connect (boxel, core-ball...) — needs canvas clicks or
   status-agnostic boot; mouse support unlocks minesweeper-class games too.
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

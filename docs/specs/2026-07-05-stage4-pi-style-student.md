# Stage 4: π-style student — strategy + roadmap (post-sweep reassessment)

**Date:** 2026-07-05. **Status:** approved direction. Supersedes nothing — extends
`2026-07-04-magus-1-induced-world-models-design.md` Stage 4 with current facts.

## The reassessment (user asked: rethink entirely?)

**Verdict: converge, don't rethink.** π0.7 (Physical Intelligence, Apr 2026 — Gemma3-4B
VLM + 860M flow-matching action expert + BAGEL-14B world model for subgoals) is
architecturally what Magus already is: world model + specialist policy generating
autonomous rollouts + distillation into a generalist + RL-from-experience. Their measured
lesson (π*0.6 RECAP): imitation alone plateaus; advantage-conditioned experience added
2.1–2.3× throughput / +25pp success. We are missing exactly their last two stages:
a generalist learned policy, and the experience loop.

Sweep buckets tell us what's limiting (33 games): 14 died at onboarding/exploration
(plumbing — no model choice fixes it), 7 induction-resistant (VLM policy helps exactly
there), 12 INDUCED (3 duel wins — the planner is a superhuman teacher on these).

## The plan (recommended order, user-approved)

1. **Stage 4 — π-style student (BEGIN NOW)**
   - Trajectory collection: planner episodes on INDUCED games (super-teacher) +
     reward-FILTERED frontier-VLM episodes elsewhere. runs/ StepRecords already carry
     decision + screenshot + metric deltas; add a converter to VLM-SFT format with
     reward/outcome metadata (for RECAP labels later).
   - Fine-tune A/B: `Hcompany/Holo-3.1-4B` (Apache 2.0, Qwen3.5-based, UI-pixel-native,
     GGUF-servable) vs `Qwen3-VL-8B` — Holo is NOT a game player out of the box
     (click/grounding paradigm), it's a candidate BASE.
   - Train+serve: **Together AI** (managed LoRA train AND serve for Qwen3-VL-8B;
     needs TOGETHER_API_KEY) or **Unsloth+RunPod** for control; local serving via
     ollama/GGUF (proven in M0). **Nebius is retired for this** (trains LoRA, cannot
     serve — the M0 wall). InsForge optional glue only.
   - Eval: re-run sweep duels with the student as a third column.
2. **RECAP-lite** — advantage-conditioned SFT using game scores as outcomes: train a
   value head/simple regressor for expected score, label trajectory steps
   positive/negative by advantage, condition training samples on the label, deploy
   conditioned on "positive". Pure SFT mechanics, no policy gradients. Games give
   dense verifiable rewards + free resets — better RECAP substrate than robotics.
3. **Onboarding/exploration robustness** (parallel, boring, biggest coverage gain):
   mouse support, menu/start-screen handling for the 14 dead games.

## Stage 4 chunk 1 (this session): data layer

- `scripts/collect_trajectories.py`: batch-run episodes (game × provider × episodes),
  unique episode ids (no runs/ collisions), records provider + final score per episode
  into a manifest.
- `ludus/student/dataset.py`: runs/ episode dirs → VLM SFT JSONL: per step
  {image path, system, user prompt (objective/state_text as at runtime), assistant =
  Decision JSON} + metadata {game, provider, step score delta, episode final score,
  episode outcome} for later reward filtering / RECAP labeling. Filtering hook:
  keep episodes with final score >= per-game threshold (planner episodes auto-keep).

Key M0 lesson that carries: train/inference prompt EXACTLY matching (reuse
build_user_text / _SYSTEM), and labels must survive blind macro execution.

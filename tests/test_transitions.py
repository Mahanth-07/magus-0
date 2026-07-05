"""TransitionStore — the induction dataset: per-press (state, action, state')."""

from ludus.worldmodel.transitions import Transition, TransitionStore


def _t(episode: str, step: int, score: int) -> Transition:
    return Transition(
        game_id="g", episode_id=episode, step=step,
        action="move_right", key="ArrowRight",
        before={"status": "playing", "metrics": {"score": score}},
        after={"status": "playing", "metrics": {"score": score + 1}},
        terminal_after=False,
    )


def test_append_and_load_roundtrip(tmp_path):
    store = TransitionStore(tmp_path / "g" / "transitions.jsonl")
    store.append(_t("ep0", 0, 0))
    store.append(_t("ep0", 1, 1))
    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].after["metrics"]["score"] == 1
    assert loaded[1].step == 1


def test_load_missing_file_is_empty(tmp_path):
    assert TransitionStore(tmp_path / "nope.jsonl").load() == []


def test_split_holds_out_whole_episodes(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    for ep in ("ep0", "ep1", "ep2", "ep3"):
        for step in range(5):
            store.append(_t(ep, step, step))
    train, holdout = store.split(holdout_frac=0.25, seed=7)
    assert len(train) == 15 and len(holdout) == 5
    train_eps = {t.episode_id for t in train}
    holdout_eps = {t.episode_id for t in holdout}
    assert not (train_eps & holdout_eps)  # split by EPISODE, never within one


def test_split_deterministic_for_seed(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    for ep in ("a", "b", "c", "d"):
        store.append(_t(ep, 0, 0))
    h1 = {t.episode_id for t in store.split(holdout_frac=0.25, seed=3)[1]}
    h2 = {t.episode_id for t in store.split(holdout_frac=0.25, seed=3)[1]}
    assert h1 == h2

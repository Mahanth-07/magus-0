# ludus/state_source.py
"""StateSource — the pluggable state-extraction seam (Magus-1 spec commitment #1).

Anything that can report the game's current structured state satisfies this.
Today: GameWorldClient (window.gameAPI.getState()), FakeGameWorld, synthetic
games. Later: a vision-based extractor for games with no state API — nothing
downstream changes.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class StateSource(Protocol):
    def raw_state(self) -> dict: ...

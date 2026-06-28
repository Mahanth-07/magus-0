"""Expert teacher for Ludus Tetris.

A pure-Python, near-optimal placement policy used to GENERATE training data
(state -> action macro) for later distillation into a learned model. It is a
data-generation tool, separate from the live VLM/agent runtime in ludus.loop.

Public surface:
    ludus.teacher.tetris_heuristic.best_move(board, shape, ...) -> dict
"""

from ludus.teacher.tetris_heuristic import best_move  # noqa: F401

__all__ = ["best_move"]

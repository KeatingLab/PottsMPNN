"""Iterative Pareto optimization loop for PottsMPNN binder design.

Closes the loop between the mutation search (stability x binding Pareto
exploration) and the AF3 + PISA + ipSAE structural scoring stage: each round
searches, folds the best candidates, promotes those that beat wildtype on the
structural metrics, and re-seeds the next round from them.

Entry point is ``run_optimization.py`` at the repository root.
"""

from __future__ import annotations

__all__ = [
    "config",
    "executors",
    "gating",
    "search_stage",
    "seeding",
    "selection",
    "state",
    "structure_stage",
]

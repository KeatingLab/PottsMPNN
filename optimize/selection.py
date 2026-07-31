"""Cutting the pooled search output down to an AF3 folding set.

AF3 is the expensive stage, so a round's ~1000 kept search candidates must be
reduced to a few dozen. Three composable layers do that, all config-driven:

1. **Constraints** -- pandas ``.query()`` strings that hard-filter candidates.
2. **Objective**   -- what to rank by: a column, a derived Pareto metric, or an
   arbitrary expression.
3. **Diversity**   -- greedy MMR re-rank so the folding set is not fifty
   variations on one position.

Together these express requests like *"rank by best binding, but the mutant
cannot lose stability"*: ``objective: binding_score`` with
``constraints: ["stability_score <= 0"]``.

The MMR routine is ported from ``mutation_search.ipynb`` cells 4/9, which
computed it but only ``display()``-ed the result. Writing it here finally makes
``ranked_mutations_depth_N.csv`` -- the file the AF3 stage has always consumed
-- a reproducible artifact rather than a manual notebook step.

Pareto helpers are imported from ``mutation_search`` rather than reimplemented;
the import is function-local because that module pulls in torch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .seeding import parse_mutation_tokens

# Objectives computed from the search columns rather than read directly.
DERIVED_OBJECTIVES = ("pareto_rank", "pareto_front", "pareto_distance", "rrf")


# ------------------------------------------------------------------ similarity


def token_similarity(a_tokens: Sequence[str], b_tokens: Sequence[str], metric: str = "jaccard") -> float:
    """Set similarity between two mutation-token collections.

    Ported verbatim from ``mutation_search.ipynb`` so behaviour matches the
    notebook exactly, including the empty-set conventions.
    """
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if metric == "overlap":
        return inter / min(len(a), len(b))
    return inter / len(a | b)


def _token_matrix(token_lists: Sequence[Sequence[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Binary membership matrix over the union of all tokens, plus set sizes."""
    vocab: Dict[str, int] = {}
    for tokens in token_lists:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    matrix = np.zeros((len(token_lists), max(1, len(vocab))), dtype=np.float64)
    for row, tokens in enumerate(token_lists):
        for token in tokens:
            matrix[row, vocab[token]] = 1.0
    return matrix, matrix.sum(axis=1)


def _similarity_to(
    matrix: np.ndarray, sizes: np.ndarray, index: int, metric: str
) -> np.ndarray:
    """Similarity of every row to row ``index``. Vectorized ``token_similarity``."""
    inter = matrix @ matrix[index]
    size_j = sizes[index]
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "overlap":
            denom = np.minimum(sizes, size_j)
        else:
            denom = sizes + size_j - inter
        sim = np.where(denom > 0, inter / np.where(denom > 0, denom, 1.0), 0.0)
    # Empty-set conventions: both empty -> 1.0, exactly one empty -> 0.0.
    both_empty = (sizes == 0) & (size_j == 0)
    one_empty = ((sizes == 0) ^ (size_j == 0))
    sim = np.where(both_empty, 1.0, sim)
    sim = np.where(one_empty, 0.0, sim)
    return sim


# ------------------------------------------------------------ diversity rerank


def diversity_rerank(
    df: pd.DataFrame,
    objective_col: str,
    diversity_weight: float = 0.0,
    metric: str = "jaccard",
    mutation_col: str = "mutations",
    max_candidates: Optional[int] = None,
) -> pd.DataFrame:
    """Greedy MMR re-rank: lower objective is better, higher uniqueness is better.

    Selects ``argmin(objective_norm - weight * uniqueness)`` where
    ``uniqueness = 1 - max(similarity to already-selected)``, exactly as the
    notebook did, but vectorized (O(n*k) rather than O(n^2) pandas ``.loc``
    lookups) and honouring ``max_candidates``.

    The notebook defined ``top_n`` but never applied it, so it always ranked the
    entire pool; stopping at ``max_candidates`` is both the fix and the reason
    this is affordable on a 1000-row pool.
    """
    if df.empty:
        out = df.copy()
        out["uniqueness_score"] = pd.Series(dtype=float)
        out["combined_rank_score"] = pd.Series(dtype=float)
        return out

    limit = len(df) if max_candidates is None else min(int(max_candidates), len(df))

    if diversity_weight <= 0:
        out = df.sort_values(objective_col, ascending=True).reset_index(drop=True)
        out["uniqueness_score"] = np.nan
        out["combined_rank_score"] = out[objective_col]
        return out.head(limit).reset_index(drop=True)

    work = df.copy().reset_index(drop=True)
    values = work[objective_col].to_numpy(dtype=float)
    # Normalize so the objective and the uniqueness term are on comparable scales.
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    span = (vmax - vmin) if vmax > vmin else 1.0
    normalized = (values - vmin) / span

    token_lists = [parse_mutation_tokens(x) for x in work[mutation_col].tolist()]
    matrix, sizes = _token_matrix(token_lists)

    n = len(work)
    remaining = np.ones(n, dtype=bool)
    # Max similarity to anything already selected; nothing selected yet -> 0,
    # which makes the first pick's uniqueness 1.0 for every candidate.
    max_sim = np.zeros(n, dtype=float)
    first_pick = True

    order: List[int] = []
    uniqueness = np.full(n, np.nan, dtype=float)
    combined = np.full(n, np.nan, dtype=float)

    while remaining.any() and len(order) < limit:
        uniq = np.ones(n, dtype=float) if first_pick else 1.0 - max_sim
        scores = np.where(remaining, normalized - diversity_weight * uniq, np.inf)
        pick = int(np.argmin(scores))

        order.append(pick)
        uniqueness[pick] = uniq[pick]
        combined[pick] = scores[pick]
        remaining[pick] = False
        first_pick = False

        if remaining.any():
            sims = _similarity_to(matrix, sizes, pick, metric)
            max_sim = np.maximum(max_sim, sims)

    work["uniqueness_score"] = uniqueness
    work["combined_rank_score"] = combined
    return work.iloc[order].reset_index(drop=True)


# ------------------------------------------------------------ derived metrics


def needs_derived_metrics(objective: str, constraints: Sequence[str] = ()) -> bool:
    """Whether any derived Pareto/RRF column is actually referenced.

    Computing them imports ``mutation_search``, which pulls in torch, so this
    is skipped when the objective and constraints only mention plain columns.
    """
    haystack = " ".join([str(objective), *(str(c) for c in constraints or ())])
    return any(name in haystack for name in DERIVED_OBJECTIVES)


def add_derived_metrics(
    df: pd.DataFrame,
    stability_col: str = "stability_score",
    binding_col: str = "binding_score",
    rrf_k: int = 60,
) -> pd.DataFrame:
    """Attach Pareto / RRF columns, reusing ``mutation_search``'s implementations.

    All derived metrics are oriented so that **lower is better**, matching the
    convention of the raw energy scores.
    """
    from mutation_search import _pareto_front, _pareto_rank, _rank_scores, _rrf_scores

    out = df.copy()
    if stability_col not in out.columns or binding_col not in out.columns:
        return out
    if out.empty:
        for name in DERIVED_OBJECTIVES:
            out[name] = pd.Series(dtype=float)
        return out

    stability = out[stability_col].to_numpy(dtype=float)
    binding = out[binding_col].to_numpy(dtype=float)

    out["pareto_rank"] = _pareto_rank(stability, binding).astype(float)
    front = _pareto_front(stability, binding)
    # Boolean front as a minimizable score: 0 on the front, 1 off it.
    out["pareto_front"] = (~front).astype(float)
    out["pareto_distance"] = _pareto_distance(stability, binding, front)
    # _rrf_scores is higher-is-better, so negate to keep "lower is better".
    out["rrf"] = -_rrf_scores(_rank_scores(stability), _rank_scores(binding), rrf_k)
    return out


def _pareto_distance(
    stability: np.ndarray, binding: np.ndarray, front_mask: np.ndarray
) -> np.ndarray:
    """Min-max-normalized Euclidean distance from each point to the Pareto front.

    Points on the front score 0. Both objectives are scaled to [0, 1] first so
    neither dominates the distance purely through its units.
    """
    if not front_mask.any():
        return np.zeros(len(stability), dtype=float)

    def _scale(values: np.ndarray) -> np.ndarray:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        span = (vmax - vmin) if vmax > vmin else 1.0
        return (values - vmin) / span

    points = np.stack([_scale(stability), _scale(binding)], axis=1)
    front_points = points[front_mask]
    # (n, n_front) pairwise distances; front members land on themselves at 0.
    deltas = points[:, None, :] - front_points[None, :, :]
    return np.sqrt((deltas**2).sum(axis=-1)).min(axis=1)


# ------------------------------------------------------- constraints/objective


def apply_constraints(df: pd.DataFrame, constraints: Sequence[str]) -> pd.DataFrame:
    """Apply pandas ``.query()`` constraints in order, ANDed together."""
    out = df
    for expression in constraints or []:
        try:
            out = out.query(expression)
        except Exception as exc:
            raise ValueError(
                f"Could not apply selection constraint {expression!r}: {exc}. "
                f"Available columns: {sorted(df.columns)}"
            ) from exc
    return out.copy()


def resolve_objective(df: pd.DataFrame, objective: str, direction: str = "min") -> pd.Series:
    """Return a minimizable Series for ``objective``.

    ``objective`` may be a column name, one of :data:`DERIVED_OBJECTIVES`, or a
    pandas-eval expression such as ``"binding_score - 0.5 * stability_score"``.
    """
    if objective in df.columns:
        values = df[objective].astype(float)
    else:
        try:
            evaluated = df.eval(objective)
        except Exception as exc:
            raise ValueError(
                f"selection.objective {objective!r} is neither a column nor a valid "
                f"expression: {exc}. Available columns: {sorted(df.columns)}"
            ) from exc
        values = pd.Series(evaluated, index=df.index).astype(float)

    if direction == "max":
        return -values
    return values


# ---------------------------------------------------------------- entry points


def select_folding_set(
    df: pd.DataFrame,
    *,
    objective: str = "binding_score",
    direction: str = "min",
    constraints: Sequence[str] = (),
    max_candidates: int = 50,
    diversity_enabled: bool = True,
    diversity_weight: float = 10.0,
    diversity_metric: str = "jaccard",
    stability_col: str = "stability_score",
    binding_col: str = "binding_score",
    mutation_col: str = "mutations",
    rrf_k: int = 60,
) -> pd.DataFrame:
    """Constrain, rank, and diversify a pooled candidate frame.

    Returns the selected rows, best first, with ``_objective``,
    ``uniqueness_score`` and ``combined_rank_score`` attached.
    """
    if df.empty:
        return df.copy()

    enriched = df
    if needs_derived_metrics(objective, constraints):
        enriched = add_derived_metrics(
            df, stability_col=stability_col, binding_col=binding_col, rrf_k=rrf_k
        )
    filtered = apply_constraints(enriched, constraints)
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    filtered["_objective"] = resolve_objective(filtered, objective, direction)
    filtered = filtered.dropna(subset=["_objective"])
    if filtered.empty:
        return filtered

    weight = diversity_weight if diversity_enabled else 0.0
    return diversity_rerank(
        filtered,
        objective_col="_objective",
        diversity_weight=weight,
        metric=diversity_metric,
        mutation_col=mutation_col,
        max_candidates=max_candidates,
    )


def _select_with_cfg(df: pd.DataFrame, cfg, max_candidates: int) -> pd.DataFrame:
    sel = cfg.selection
    return select_folding_set(
        df,
        objective=sel.objective,
        direction=sel.direction,
        constraints=list(sel.constraints),
        max_candidates=max_candidates,
        diversity_enabled=sel.diversity.enabled,
        diversity_weight=sel.diversity.weight,
        diversity_metric=sel.diversity.metric,
        stability_col=sel.stability_column,
        binding_col=sel.binding_column,
        mutation_col=sel.mutation_column,
        rrf_k=cfg.search.rrf_k,
    )


def select_from_config(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Build the AF3 folding set, ranking within each structure separately.

    PottsMPNN produces a different energy table per structure, so ``stability_score``
    and ``binding_score`` are only meaningful *within* one seed's search. Ranking a
    pooled frame would let a seed whose structure happens to yield systematically
    lower energies crowd out every other structure's candidates -- and would compute
    the Pareto front across incomparable numbers.

    So each seed is ranked against its own structure-mates only, receives a share
    of ``max_candidates``, and the per-seed picks are unioned. Comparison *between*
    structures happens later, on the AF3/PISA metrics, which are absolute.
    """
    sel = cfg.selection
    grouped = (
        sel.scope == "per_seed"
        and "seed_id" in df.columns
        and df["seed_id"].nunique() > 1
    )
    if not grouped:
        return _select_with_cfg(df, cfg, sel.max_candidates)

    groups = list(df.groupby("seed_id", sort=True))
    # Split the AF3 budget across structures, with a floor of one each.
    quota = max(1, sel.max_candidates // len(groups))
    parts = []
    for seed_id, group in groups:
        picked = _select_with_cfg(group, cfg, quota)
        if not picked.empty:
            parts.append(picked)
        print(
            f"  selection[{seed_id}]: {len(picked)} of {len(group)} candidates "
            f"(quota {quota})"
        )
    if not parts:
        return df.head(0)

    out = pd.concat(parts, ignore_index=True)
    # Different structures often converge on the same mutant; fold it once.
    if "sequence" in out.columns:
        out = out.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
    return out

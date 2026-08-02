"""Promotion and termination rules.

Two independent decisions are made from a round's structural results:

* **Promotion** -- which mutants beat the wildtype (on *all* the metrics listed
  in ``gating.beats_wt_on``) and therefore seed the next round.
* **Termination** -- whether any cutoff has been reached, ending the run early.

Every comparison derives its operator from the metric's declared ``direction``:
ipSAE is better high, PISA dG is better low.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import metric_is_better, metric_meets_cutoff
from .seeding import split_mutations
from .state import SeedRecord

# Termination reasons, recorded verbatim in the run state.
CUTOFF_MET = "cutoff_met"
MAX_ITERATIONS = "max_iterations"
NO_WINNERS = "no_winners"
NO_CANDIDATES = "no_candidates"


def annotate_wildtype_comparison(
    df: pd.DataFrame, cfg, baseline: Dict[str, float]
) -> pd.DataFrame:
    """Add per-metric ``beats_wt_<metric>`` flags and an overall ``beats_wt``.

    A mutant is a winner only if it beats the wildtype on every metric listed in
    ``gating.beats_wt_on``.
    """
    out = df.copy()
    if out.empty:
        out["beats_wt"] = pd.Series(dtype=bool)
        return out

    required = list(cfg.gating.beats_wt_on)
    overall = pd.Series(True, index=out.index)

    for metric in cfg.gating.metrics:
        if metric not in out.columns:
            continue
        direction = cfg.gating.metrics[metric].direction
        reference = baseline.get(metric)
        if reference is None:
            continue
        flags = out[metric].map(
            lambda v, r=reference, d=direction: bool(pd.notna(v)) and metric_is_better(float(v), r, d)
        )
        out[f"beats_wt_{metric}"] = flags
        if metric in required:
            overall &= flags

    out["beats_wt"] = overall.fillna(False).astype(bool)
    return out


def annotate_cutoffs(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add per-metric ``meets_cutoff_<metric>`` flags and an overall ``meets_cutoff``.

    ``gating.stop_when`` decides whether reaching *any* declared cutoff counts,
    or whether *all* of them must be reached by the same mutant.
    """
    out = df.copy()
    declared = {
        name: spec for name, spec in cfg.gating.metrics.items() if spec.cutoff is not None
    }
    if out.empty or not declared:
        out["meets_cutoff"] = pd.Series([False] * len(out), index=out.index, dtype=bool)
        return out

    per_metric = []
    for metric, spec in declared.items():
        if metric not in out.columns:
            continue
        flags = out[metric].map(
            lambda v, c=float(spec.cutoff), d=spec.direction: bool(pd.notna(v))
            and metric_meets_cutoff(float(v), c, d)
        )
        out[f"meets_cutoff_{metric}"] = flags
        per_metric.append(flags)

    if not per_metric:
        out["meets_cutoff"] = False
        return out

    stacked = pd.concat(per_metric, axis=1)
    if cfg.gating.stop_when == "all":
        out["meets_cutoff"] = stacked.all(axis=1)
    else:
        out["meets_cutoff"] = stacked.any(axis=1)
    return out


def apply_rmsd_gate(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add ``passes_rmsd`` and fold it into ``beats_wt`` and ``meets_cutoff``.

    A mutant that drifts past ``gating.rmsd_gate.max_rmsd`` (or whose RMSD could
    not be computed -- ``NaN``) is disqualified from both promotion and the
    stop-cutoff: its structural metrics describe a structure other than the one
    that was searched, so neither should count.
    """
    out = df.copy()
    gate = cfg.gating.rmsd_gate
    if not gate.enabled:
        out["passes_rmsd"] = pd.Series([True] * len(out), index=out.index, dtype=bool)
        return out

    if "rmsd" in out.columns:
        rmsd = pd.to_numeric(out["rmsd"], errors="coerce")
        passes = rmsd.notna() & (rmsd <= float(gate.max_rmsd))
    else:
        # Gate is on but no RMSD was computed: fail closed rather than promote
        # an unverified structure.
        passes = pd.Series([False] * len(out), index=out.index, dtype=bool)

    out["passes_rmsd"] = passes.astype(bool)
    if "beats_wt" in out.columns:
        out["beats_wt"] = out["beats_wt"] & out["passes_rmsd"]
    if "meets_cutoff" in out.columns:
        out["meets_cutoff"] = out["meets_cutoff"] & out["passes_rmsd"]
    return out


def evaluate_round(
    df: pd.DataFrame, cfg, baseline: Dict[str, float]
) -> pd.DataFrame:
    """Annotate a scored round with wildtype comparison, cutoffs, and RMSD gate."""
    out = annotate_wildtype_comparison(df, cfg, baseline)
    out = annotate_cutoffs(out, cfg)
    return apply_rmsd_gate(out, cfg)


def winners(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that beat the wildtype on every required metric."""
    if df.empty or "beats_wt" not in df.columns:
        return df.head(0)
    return df[df["beats_wt"]].copy()


def rank_for_promotion(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Order winners best-first for seeding the next round.

    Ranks on the **structural** metrics only: this compares mutants from
    different structures, and each structure has its own PottsMPNN energy table,
    so ``binding_score`` from seed A means nothing relative to seed B. ipSAE and
    the PISA terms are absolute measurements of a predicted complex.

    ``gating.promote_by`` picks a single metric; otherwise winners are ordered by
    their mean normalized rank across every metric in ``gating.beats_wt_on``.
    """
    if df.empty:
        return df

    if cfg.gating.promote_by:
        metric = cfg.gating.promote_by
        ascending = cfg.gating.metrics[metric].direction == "min"
        return df.sort_values(metric, ascending=ascending, kind="mergesort").reset_index(drop=True)

    ranked = df.copy()
    ranked["_promote_rank"] = _mean_structural_rank(ranked, cfg)
    return ranked.sort_values(
        "_promote_rank", ascending=True, kind="mergesort"
    ).reset_index(drop=True)


def _mean_structural_rank(df: pd.DataFrame, cfg) -> pd.Series:
    """Mean percentile rank across the gating metrics, lower being better.

    Each metric is oriented by its declared ``direction`` before ranking, so
    mixing better-high (ipSAE) and better-low (dG_binding) metrics is safe.
    """
    columns = []
    for metric in cfg.gating.beats_wt_on:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if cfg.gating.metrics[metric].direction == "max":
            values = -values
        columns.append(values.rank(pct=True, na_option="bottom"))
    if not columns:
        # Nothing to rank on; preserve the incoming order.
        return pd.Series(range(len(df)), index=df.index, dtype=float)
    return pd.concat(columns, axis=1).mean(axis=1)


def select_promotions(df: pd.DataFrame, cfg, round_index: int) -> List[SeedRecord]:
    """Build the next round's seeds from this round's winners.

    Each seed's AF3 model is resolved separately by the loop (see
    ``structure_stage.resolve_seed_model``) so that a missing prediction is
    reported against a named seed rather than silently producing a bad backbone.
    """
    ranked = rank_for_promotion(winners(df), cfg)
    if ranked.empty:
        return []

    metrics = list(cfg.gating.metrics.keys())
    seeds: List[SeedRecord] = []
    for position, (_, row) in enumerate(ranked.head(cfg.gating.promote_top_n).iterrows()):
        # Order-preserving: these mutations are rejoined into the AF3 job name
        # when the seed's model is looked up, so re-sorting them would rename
        # the job and make the existing prediction unfindable.
        mutations = split_mutations(row.get("mutations", ""))
        seeds.append(
            SeedRecord(
                seed_id=f"r{round_index + 1}s{position}",
                sequence=row["sequence"],
                mutations=mutations,
                parent_seed_id=row.get("seed_id"),
                round_index=round_index + 1,
                metrics={m: float(row[m]) for m in metrics if m in row and pd.notna(row[m])},
            )
        )
    return seeds


def should_stop(
    df: pd.DataFrame, cfg, round_index: int, n_winners: int
) -> Tuple[bool, Optional[str]]:
    """Decide whether the run ends after this round, and why.

    Checked in priority order: cutoff reached, no candidates at all, no winners
    to promote, iteration budget exhausted.
    """
    if not df.empty and "meets_cutoff" in df.columns:
        n_passing = int(df["meets_cutoff"].sum())
        if n_passing >= cfg.gating.require_n_passing:
            return True, CUTOFF_MET

    if df.empty:
        return True, NO_CANDIDATES

    if n_winners == 0 and cfg.gating.stop_on_no_winners:
        return True, NO_WINNERS

    if round_index + 1 >= cfg.run.max_iterations:
        return True, MAX_ITERATIONS

    return False, None


def summarize_round(df: pd.DataFrame, cfg, baseline: Dict[str, float]) -> Dict[str, object]:
    """Compact per-round statistics for logging and the final summary."""
    summary: Dict[str, object] = {
        "n_scored": int(len(df)),
        "n_winners": int(df["beats_wt"].sum()) if "beats_wt" in df.columns else 0,
        "n_meeting_cutoff": int(df["meets_cutoff"].sum()) if "meets_cutoff" in df.columns else 0,
    }
    if cfg.gating.rmsd_gate.enabled and "passes_rmsd" in df.columns:
        summary["n_failing_rmsd"] = int((~df["passes_rmsd"]).sum())
        if "rmsd" in df.columns and not df["rmsd"].dropna().empty:
            summary["max_rmsd"] = float(df["rmsd"].max())
    for metric in cfg.gating.metrics:
        if metric in df.columns and not df[metric].dropna().empty:
            direction = cfg.gating.metrics[metric].direction
            best = df[metric].max() if direction == "max" else df[metric].min()
            summary[f"best_{metric}"] = float(best)
            summary[f"wt_{metric}"] = baseline.get(metric)
    return summary

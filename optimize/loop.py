"""The round loop: search -> select -> fold -> gate -> re-seed.

Every stage writes its output to disk and records a completion marker in the run
state before the next begins, so a preempted multi-day run resumes at the last
completed stage instead of restarting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import cleanup, gating, search_stage, seeding, selection, structure_stage
from .config import resolve_partitions
from .executors import build_executor
from .state import RunState, SeedRecord

STAGE_SEARCH = "search"
STAGE_SELECT = "select"
STAGE_STRUCTURE = "structure"

POOLED_CSV = "pooled_candidates.csv"
FOLDING_CSV = "folding_set.csv"
SCORED_CSV = "scored_candidates.csv"
SUMMARY_CSV = "round_summary.csv"


def _round_dir(out_dir: Path, index: int) -> Path:
    return out_dir / f"round_{index}"


def _write_report(cfg, out_dir: Path, label: str) -> None:
    """Generate the run report, never letting it break a completed run.

    The results it plots are already on disk, so any failure here is reported
    and swallowed.
    """
    from . import report as report_module

    try:
        report_module.generate(out_dir, top=25)
    except Exception as exc:  # noqa: BLE001 - a report must never sink a run
        print(f"[{label}] WARNING: could not write the report ({type(exc).__name__}: {exc})")


def _load_or_none(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _initial_seed(wt_sequence: str) -> SeedRecord:
    return SeedRecord(seed_id="r0s0", sequence=wt_sequence, mutations=[], round_index=0)


def run_optimization(cfg) -> Dict[str, object]:
    """Run the full iterative optimization and return a summary dict."""
    out_dir = Path(cfg.run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.load_or_create(str(out_dir), force=cfg.run.force)

    # --- target structure -------------------------------------------------
    wt_atoms = seeding.read_backbone_atoms(str(cfg.target.pdb))
    chain_order = seeding.infer_chain_order(wt_atoms)
    chain_lengths = seeding.backbone_chain_lengths(wt_atoms, chain_order)
    wt_sequence = seeding.backbone_sequence(wt_atoms, chain_order)
    pdb_name = Path(str(cfg.target.pdb)).stem
    partitions = resolve_partitions(cfg, pdb_name)

    print(f"target      : {pdb_name} chains={chain_order} lengths={chain_lengths}")
    print(f"wildtype    : {len(wt_sequence)} residues")
    print(f"partitions  : {partitions}")

    executor = build_executor(cfg, str(out_dir / "executor"))

    # The AF3 pipeline prepends a wildtype row to every run. The baseline is
    # captured from the first round that folds anything, then reused throughout.
    baseline: Dict[str, float] = dict(state.wt_baseline)

    seeds: List[SeedRecord] = state.round(0).seeds or [_initial_seed(wt_sequence)]
    state.set_seeds(0, seeds)

    history: List[Dict[str, object]] = []
    termination = gating.MAX_ITERATIONS

    for index in range(cfg.run.max_iterations):
        round_dir = _round_dir(out_dir, index)
        round_dir.mkdir(parents=True, exist_ok=True)
        seeds = state.round(index).seeds or seeds
        print(f"\n=== round {index} | {len(seeds)} seed(s) ===")

        # --- search ------------------------------------------------------
        pooled_path = round_dir / POOLED_CSV
        pooled = _load_or_none(pooled_path) if state.is_stage_complete(index, STAGE_SEARCH) else None
        if pooled is None:
            pooled = search_stage.run_round_search(
                seeds, cfg, round_dir, wt_sequence, wt_atoms, chain_order,
                chain_lengths, partitions, index, out_dir=out_dir,
            )
            pooled.to_csv(pooled_path, index=False)
            state.set_seeds(index, seeds)  # backbone paths were filled in
            state.mark_stage_complete(index, STAGE_SEARCH)
        else:
            print(f"[round {index}] search: reusing {pooled_path.name} ({len(pooled)} rows)")

        if pooled.empty:
            termination = gating.NO_CANDIDATES
            state.finish(termination, index)
            break

        # --- select ------------------------------------------------------
        folding_path = round_dir / FOLDING_CSV
        folding = _load_or_none(folding_path) if state.is_stage_complete(index, STAGE_SELECT) else None
        if folding is None:
            folding = selection.select_from_config(pooled, cfg)
            folding.to_csv(folding_path, index=False)
            state.mark_stage_complete(index, STAGE_SELECT)
            print(f"[round {index}] selected {len(folding)} of {len(pooled)} candidates to fold")
        else:
            print(f"[round {index}] select: reusing {folding_path.name} ({len(folding)} rows)")

        if folding.empty:
            print(
                f"[round {index}] no candidate satisfied selection.constraints "
                f"({list(cfg.selection.constraints)})"
            )
            termination = gating.NO_CANDIDATES
            state.finish(termination, index)
            break

        # --- fold and score ----------------------------------------------
        scored_path = round_dir / SCORED_CSV
        scored = _load_or_none(scored_path) if state.is_stage_complete(index, STAGE_STRUCTURE) else None
        if scored is None:
            seed_backbone_map = {s.seed_id: s.backbone_pdb for s in seeds}
            scored, round_baseline = structure_stage.run_structure_stage(
                folding, cfg, out_dir, round_dir, executor, state, index,
                seed_backbone_map=seed_backbone_map, chain_order=chain_order,
            )
            scored.to_csv(scored_path, index=False)
            if round_baseline and not state.wt_baseline:
                state.set_wt(wt_sequence, round_baseline)
                baseline = dict(round_baseline)
                print(f"[wildtype] baseline from pipeline: {baseline}")
            state.mark_stage_complete(index, STAGE_STRUCTURE)
        else:
            print(f"[round {index}] structure: reusing {scored_path.name} ({len(scored)} rows)")

        if not baseline:
            baseline = dict(state.wt_baseline)
        if not baseline:
            raise RuntimeError(
                "No wildtype baseline available. The AF3 pipeline prepends a 'WT' row to its "
                "results; none was found, so mutants cannot be compared to wildtype."
            )

        # --- gate --------------------------------------------------------
        evaluated = gating.evaluate_round(scored, cfg, baseline)
        evaluated.to_csv(round_dir / SUMMARY_CSV, index=False)

        round_winners = gating.winners(evaluated)
        stats = gating.summarize_round(evaluated, cfg, baseline)
        stats["round"] = index
        history.append(stats)
        state.set_counts(
            index,
            n_pooled=len(pooled),
            n_folded=len(folding),
            n_winners=len(round_winners),
            n_meeting_cutoff=int(stats.get("n_meeting_cutoff", 0)),
        )
        print(f"[round {index}] {_format_stats(stats)}")

        # --- prune AF3 byproducts ------------------------------------------
        # After gating, so the round's winners are known and its results are
        # already on disk.
        if cfg.structure.cleanup.mode != "none":
            winner_keys = [str(k) for k in round_winners.get("mutations", [])]
            cleanup.run_cleanup_for_round(cfg, out_dir, winner_keys, index)

        if cfg.run.report_each_round:
            _write_report(cfg, out_dir, f"round {index}")

        stop, reason = gating.should_stop(evaluated, cfg, index, len(round_winners))
        if stop:
            termination = reason
            state.finish(termination, index)
            print(f"[round {index}] stopping: {reason}")
            break

        # --- promote -----------------------------------------------------
        seeds = gating.select_promotions(evaluated, cfg, index)
        if cfg.run.backbone_source == "af3":
            # Resolve each winner's AF3 model now, so a missing prediction is
            # reported against a named seed instead of failing mid-search.
            for seed in seeds:
                seed.af3_dir = structure_stage.resolve_seed_model(
                    cfg, out_dir, ",".join(seed.mutations)
                )
        state.set_seeds(index + 1, seeds)
        print(
            f"[round {index}] promoting {len(seeds)} winner(s): "
            + ", ".join(f"{s.seed_id}({','.join(s.mutations) or 'wt'})" for s in seeds)
        )
    else:
        state.finish(termination, cfg.run.max_iterations - 1)

    summary = {
        "termination_reason": state.termination_reason or termination,
        "wildtype_baseline": baseline,
        "rounds": history,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "optimization_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"\n=== finished: {summary['termination_reason']} ===")
    _print_history(history)

    # After the summary, so the report reflects every round including the last.
    if cfg.run.report:
        _write_report(cfg, out_dir, "report")
    return summary


def _format_stats(stats: Dict[str, object]) -> str:
    parts = [
        f"scored={stats.get('n_scored')}",
        f"winners={stats.get('n_winners')}",
        f"cutoff={stats.get('n_meeting_cutoff')}",
    ]
    if "n_failing_rmsd" in stats:
        parts.append(f"rmsd_fail={stats.get('n_failing_rmsd')}")
    for key, value in stats.items():
        if key.startswith("best_"):
            parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return " ".join(parts)


def _print_history(history: Sequence[Dict[str, object]]) -> None:
    if not history:
        return
    print(f"{'round':>5} {'scored':>7} {'winners':>8} {'cutoff':>7}  best metrics")
    for row in history:
        best = " ".join(
            f"{k[5:]}={v:.4g}" for k, v in row.items() if k.startswith("best_") and isinstance(v, float)
        )
        print(
            f"{row.get('round'):>5} {row.get('n_scored'):>7} {row.get('n_winners'):>8} "
            f"{row.get('n_meeting_cutoff'):>7}  {best}"
        )

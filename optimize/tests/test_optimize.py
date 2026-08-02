"""Tests for the optimization loop.

Runs without torch, a GPU, or cluster access: the mutation search and the
AF3/PISA/ipSAE pipeline are both replaced by the stubs in ``optimize.testing``.

Run directly::

    python -m optimize.tests.test_optimize

or under pytest if it is installed::

    pytest optimize/tests/test_optimize.py
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from optimize import gating, loop, seeding, selection
from optimize.config import OptimizationConfig, metric_is_better, metric_meets_cutoff, validate
from optimize.executors import Job, LocalExecutor, SlurmExecutor
from optimize.search_stage import dedupe_by_sequence, pool_seed_results
from optimize.state import RunState, SeedRecord
from optimize.testing.stub_search import stub_run_round_search

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PDB = REPO_ROOT / "double_mutant_cycle" / "2ZTA.pdb"
STUB_PIPELINE = REPO_ROOT / "optimize" / "testing" / "stub_af3_pipeline.py"

ENTRY = {
    "chain_order": ["A", "B"],
    "seq_chain_A": "ACDEFG",
    "seq_chain_B": "HIKLMNPQRS",
    "seq": "ACDEFGHIKLMNPQRS",
    "name": "t",
}


# --------------------------------------------------------------------- seeding


def test_sequence_override_keeps_chains_consistent():
    entry = dict(ENTRY)
    mutant = seeding.apply_mutation_tokens(entry["seq"], ["B:K3W", "A:C2Y"], entry)
    seeding.apply_sequence_override(entry, mutant)
    assert entry["seq"] == mutant
    assert entry["seq"] == entry["seq_chain_A"] + entry["seq_chain_B"]
    assert len(entry["seq_chain_A"]) == 6 and len(entry["seq_chain_B"]) == 10


def test_sequence_override_rejects_bad_input():
    for bad in ["ACDEF", "X" * 16]:
        try:
            seeding.apply_sequence_override(dict(ENTRY), bad)
            raise AssertionError(f"should have rejected {bad!r}")
        except ValueError:
            pass


def test_diff_to_wt_round_trips_and_handles_reversion():
    wt = ENTRY["seq"]
    mutant = seeding.apply_mutation_tokens(wt, ["B:K3W", "A:C2Y"], ENTRY)
    tokens = seeding.diff_to_wt(mutant, wt, ENTRY)
    assert tokens == ["A:C2Y", "B:K3W"]
    assert seeding.apply_mutation_tokens(wt, tokens, ENTRY) == mutant
    # Reverting a mutation makes it disappear from the diff.
    reverted = seeding.apply_mutation_tokens(mutant, ["A:Y2C"], ENTRY)
    assert seeding.diff_to_wt(reverted, wt, ENTRY) == ["B:K3W"]
    assert seeding.diff_to_wt(wt, wt, ENTRY) == []


def test_backbone_reader_excludes_solvent():
    """An HOH oxygen is named 'O' and would otherwise pass the backbone filter."""
    atoms = seeding.read_backbone_atoms(str(EXAMPLE_PDB))
    assert atoms
    assert not ({"HOH", "WAT"} & {a["resname"] for a in atoms})


def test_backbone_round_trip_preserves_structure():
    atoms = seeding.read_backbone_atoms(str(EXAMPLE_PDB))
    order = seeding.infer_chain_order(atoms)
    out = Path(tempfile.mkdtemp()) / "rt.pdb"
    seeding._write_backbone_pdb(atoms, out, order)
    again = seeding.read_backbone_atoms(str(out))
    assert len(again) == len(atoms)
    for a, b in zip(atoms, again):
        assert (a["chain"], a["resseq"], a["atom"]) == (b["chain"], b["resseq"], b["atom"])
        assert abs(a["x"] - b["x"]) < 1e-3


def test_self_describing_backbone_encodes_sequence():
    atoms = seeding.read_backbone_atoms(str(EXAMPLE_PDB))
    order = seeding.infer_chain_order(atoms)
    lengths = seeding.backbone_chain_lengths(atoms, order)
    wt = seeding.backbone_sequence(atoms, order)
    mutant = wt[:4] + ("W" if wt[4] != "W" else "Y") + wt[5:]

    out = Path(tempfile.mkdtemp()) / "seeded.pdb"
    seeding.write_backbone_with_sequence(atoms, mutant, order, str(out))
    reread = seeding.read_backbone_atoms(str(out))

    assert seeding.backbone_sequence(reread, order) == mutant
    assert seeding.backbone_chain_lengths(reread, order) == lengths
    # Coordinates must survive the residue renaming untouched.
    assert all(abs(a["x"] - b["x"]) < 1e-3 for a, b in zip(atoms, reread))


def test_chain_order_matches_parse_pdb_alphabet():
    atoms = [{"chain": c, "resseq": 1, "resname": "ALA", "atom": "CA", "x": 0, "y": 0, "z": 0}
             for c in ["B", "A", "C"]]
    assert seeding.infer_chain_order(atoms) == ["A", "B", "C"]


# ------------------------------------------------------------------- selection


def _notebook_rerank(df, binding_col, diversity_weight, metric, mut_col="mutations"):
    """Naive O(n^2) MMR, used as the oracle for the vectorized implementation."""
    if diversity_weight <= 0:
        out = df.sort_values(binding_col, ascending=True).reset_index(drop=True)
        out["uniqueness_score"] = np.nan
        out["combined_rank_score"] = out[binding_col]
        return out
    work = df.copy().reset_index(drop=True)
    bind = work[binding_col].to_numpy(dtype=float)
    bmin, bmax = bind.min(), bind.max()
    bspan = (bmax - bmin) if bmax > bmin else 1.0
    work["_binding_norm"] = (bind - bmin) / bspan
    selected, remaining = [], list(range(len(work)))
    tokens = [seeding.parse_mutation_tokens(x) for x in work[mut_col].tolist()]
    while remaining:
        best_idx = best_score = best_unique = None
        for idx in remaining:
            if not selected:
                uniq = 1.0
            else:
                uniq = 1.0 - max(
                    selection.token_similarity(tokens[idx], tokens[j], metric=metric)
                    for j in selected
                )
            score = work.loc[idx, "_binding_norm"] - diversity_weight * uniq
            if (best_score is None) or (score < best_score):
                best_idx, best_score, best_unique = idx, score, uniq
        selected.append(best_idx)
        work.loc[best_idx, "uniqueness_score"] = best_unique
        work.loc[best_idx, "combined_rank_score"] = best_score
        remaining.remove(best_idx)
    return work.loc[selected].drop(columns=["_binding_norm"]).reset_index(drop=True)


def _random_candidates(n, seed):
    import random

    rng = random.Random(seed)
    pool = [
        f"B:{rng.choice('ACDEFGHIKLMNPQRSTVWY')}{rng.randint(1, 40)}{rng.choice('ACDEFGHIKLMNPQRSTVWY')}"
        for _ in range(25)
    ]
    return pd.DataFrame([
        {
            "mutations": ",".join(sorted(rng.sample(pool, rng.randint(1, 3)))),
            "binding_score": round(rng.uniform(-5, 2), 4),
        }
        for _ in range(n)
    ])


def test_diversity_rerank_matches_notebook():
    """The vectorized MMR must reproduce the naive selection exactly."""
    for n in (40, 90):
        for weight in (0.0, 1.0, 10.0):
            for metric in ("jaccard", "overlap"):
                df = _random_candidates(n, seed=n + int(weight * 10) + len(metric))
                expected = _notebook_rerank(df, "binding_score", weight, metric)
                actual = selection.diversity_rerank(
                    df, "binding_score", weight, metric, max_candidates=None
                )
                assert list(expected["mutations"]) == list(actual["mutations"]), (n, weight, metric)
                assert np.allclose(
                    expected["combined_rank_score"].to_numpy(dtype=float),
                    actual["combined_rank_score"].to_numpy(dtype=float),
                    equal_nan=True,
                )


def test_max_candidates_is_prefix_of_full_ranking():
    """max_candidates must truncate the full ranking, not change its order."""
    df = _random_candidates(120, seed=7)
    full = selection.diversity_rerank(df, "binding_score", 10.0, "jaccard", max_candidates=None)
    cut = selection.diversity_rerank(df, "binding_score", 10.0, "jaccard", max_candidates=15)
    assert len(cut) == 15
    assert list(full["mutations"][:15]) == list(cut["mutations"])


def test_constraints_enforce_stability_floor():
    """The motivating case: best binding, subject to stability not getting worse."""
    df = pd.DataFrame({
        "mutations": ["B:A1C", "B:D2E", "B:F3G"],
        "stability_score": [-2.0, 1.5, -0.5],
        "binding_score": [-1.0, -9.0, -2.0],
    })
    out = selection.select_folding_set(
        df, objective="binding_score", direction="min",
        constraints=["stability_score <= 0"], diversity_enabled=False, max_candidates=10,
    )
    # B:D2E has the best binding but loses stability, so it must be excluded.
    assert set(out["mutations"]) == {"B:A1C", "B:F3G"}
    assert out.iloc[0]["mutations"] == "B:F3G"


def test_invalid_constraint_and_objective_raise():
    df = pd.DataFrame({"mutations": ["B:A1C"], "binding_score": [1.0]})
    for call in (
        lambda: selection.apply_constraints(df, ["no_such_column > 1"]),
        lambda: selection.resolve_objective(df, "not valid((", "min"),
    ):
        try:
            call()
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_derived_metrics_only_needed_when_referenced():
    assert not selection.needs_derived_metrics("binding_score", ["stability_score <= 0"])
    assert selection.needs_derived_metrics("pareto_rank", [])
    assert selection.needs_derived_metrics("binding_score", ["pareto_front == 0"])


# ---------------------------------------------------------------------- gating


def _gating_cfg(**over):
    user = OmegaConf.create({
        "run": {"out_dir": "x", "max_iterations": 3},
        "target": {"pdb": "x.pdb", "fasta": "x.fasta", "binding_partitions": [["A"], ["B"]]},
        "search": {"cfg_path": "x.yaml"},
        "structure": {"pipeline_script": "x.py"},
        "gating": {
            "metrics": {"ipsae": {"direction": "max", "cutoff": 0.75},
                        "dG_binding": {"direction": "min", "cutoff": -16.0}},
            "beats_wt_on": ["ipsae", "dG_binding"],
            # These fixtures carry no structures, so default off; the RMSD gate
            # has its own tests that supply an rmsd column.
            "rmsd_gate": {"enabled": False},
        },
    })
    cfg = OmegaConf.merge(OmegaConf.structured(OptimizationConfig), user, OmegaConf.create(over))
    validate(cfg)
    return cfg


# dG_binding is better LOW (interface solvation energy), unlike dG_diss.
SCORED = pd.DataFrame({
    "sequence": ["S1", "S2", "S3", "S4"],
    "mutations": ["B:A1C", "B:D2E", "B:F3G", "B:H4I"],
    "ipsae": [0.80, 0.60, 0.72, 0.40],            # wt 0.55 -> S1,S2,S3 beat it
    "dG_binding": [-20.0, -10.0, -16.0, -5.0],    # wt -13.4 -> S1,S3 beat it
})
BASELINE = {"ipsae": 0.55, "dG_binding": -13.4}


def test_beats_wt_requires_all_listed_metrics():
    out = gating.evaluate_round(SCORED, _gating_cfg(), BASELINE)
    assert list(out["beats_wt"]) == [True, False, True, False]


def test_direction_is_respected_per_metric():
    assert metric_is_better(0.8, 0.5, "max") and not metric_is_better(0.4, 0.5, "max")
    assert metric_is_better(-20, -12, "min") and not metric_is_better(-5, -12, "min")
    assert metric_meets_cutoff(0.75, 0.75, "max") and metric_meets_cutoff(-15.0, -15.0, "min")


def test_default_pisa_metric_is_dg_binding_minimised():
    """dG_binding (solvation energy) is better LOW and is what the plots use.

    dG_diss is the opposite direction; a wrong direction inverts the gate
    silently rather than erroring, so pin the defaults.
    """
    from optimize.config import OptimizationConfig as _Cfg

    defaults = OmegaConf.structured(_Cfg)
    assert defaults.gating.metrics["dG_binding"].direction == "min"
    assert defaults.gating.metrics["ipsae"].direction == "max"
    assert defaults.structure.adapter.metric_columns["ipsae"] == "ipSAE"
    assert defaults.structure.adapter.metric_columns["dG_binding"] == "dG_binding"
    # dG_diss stays available as a carried-through column.
    assert "dG_diss" in defaults.structure.adapter.extra_columns


def test_rmsd_gate_defaults():
    """The post-AF3 RMSD gate is on by default at 2 A over CA vs. the seed."""
    from optimize.config import OptimizationConfig as _Cfg

    gate = OmegaConf.structured(_Cfg).gating.rmsd_gate
    assert gate.enabled is True
    assert gate.max_rmsd == 2.0
    assert gate.atoms == "CA"
    assert gate.reference == "seed"


def test_dg_binding_gate_matches_reference_plot():
    """Reproduces the binder7 plot: WT ipSAE=0.275, dG_binding=-13.40.

    Only mutants that are BOTH higher-ipSAE and more-negative-dG_binding than WT
    (the upper-left quadrant) may be promoted.
    """
    cfg = _gating_cfg()
    frame = pd.DataFrame({
        "sequence": ["upper_left", "upper_right", "lower_left", "lower_right"],
        "mutations": ["B:A1C", "B:D2E", "B:F3G", "B:H4I"],
        "ipsae": [0.40, 0.40, 0.20, 0.20],
        "dG_binding": [-15.0, -12.0, -15.0, -12.0],
    })
    out = gating.evaluate_round(frame, cfg, {"ipsae": 0.275, "dG_binding": -13.40})
    assert list(out["beats_wt"]) == [True, False, False, False]


def test_stop_when_any_versus_all():
    any_cfg = _gating_cfg(**{"gating": {"stop_when": "any"}})
    all_cfg = _gating_cfg(**{"gating": {"stop_when": "all"}})
    n_any = int(gating.evaluate_round(SCORED, any_cfg, BASELINE)["meets_cutoff"].sum())
    n_all = int(gating.evaluate_round(SCORED, all_cfg, BASELINE)["meets_cutoff"].sum())
    # S1 meets both; S3 meets only dG_binding. "any" is therefore strictly looser.
    assert n_any == 2 and n_all == 1


def test_single_metric_gating_via_null_cutoff():
    """Leaving one cutoff null gates on the other metric alone."""
    cfg = _gating_cfg(**{"gating": {"metrics": {"dG_binding": {"cutoff": None}}}})
    out = gating.evaluate_round(SCORED, cfg, BASELINE)
    assert list(out["meets_cutoff"]) == [True, False, False, False]


def test_termination_priority():
    cfg = _gating_cfg()
    evaluated = gating.evaluate_round(SCORED, cfg, BASELINE)
    assert gating.should_stop(evaluated, cfg, 0, 2) == (True, gating.CUTOFF_MET)

    no_cut = _gating_cfg(**{"gating": {"metrics": {"ipsae": {"cutoff": None},
                                                   "dG_binding": {"cutoff": None}}}})
    ev2 = gating.evaluate_round(SCORED, no_cut, BASELINE)
    assert gating.should_stop(ev2, no_cut, 0, 0) == (True, gating.NO_WINNERS)
    assert gating.should_stop(ev2, no_cut, 0, 2) == (False, None)
    assert gating.should_stop(ev2, no_cut, 2, 2) == (True, gating.MAX_ITERATIONS)
    assert gating.should_stop(SCORED.head(0), no_cut, 0, 0)[1] == gating.NO_CANDIDATES


def test_promotions_are_nan_safe():
    """An empty mutations string returns from CSV as NaN, not ''."""
    frame = SCORED.copy()
    frame.loc[0, "mutations"] = np.nan
    evaluated = gating.evaluate_round(frame, _gating_cfg(), BASELINE)
    seeds = gating.select_promotions(evaluated, _gating_cfg(), 0)
    assert seeds and seeds[0].mutations == []
    assert seeds[0].round_index == 1


# ----------------------------------------------------------- state / executors


def test_state_resume_and_force():
    d = tempfile.mkdtemp()
    st = RunState.load_or_create(d)
    st.set_wt("ACDEF", {"ipsae": 0.4})
    st.mark_stage_complete(0, "search")
    st.cache_put_many({"ACDEG": {"ipsae": 0.6}})

    resumed = RunState.load_or_create(d)
    assert resumed.is_stage_complete(0, "search")
    assert resumed.cache_get("ACDEG") == {"ipsae": 0.6}
    assert resumed.wt_baseline == {"ipsae": 0.4}
    # force ignores every marker and the cache.
    forced = RunState.load_or_create(d, force=True)
    assert not forced.is_stage_complete(0, "search")
    assert forced.cache_get("ACDEG") is None


def test_local_executor_reports_failures():
    d = Path(tempfile.mkdtemp())
    results = LocalExecutor(max_parallel=2).run([
        Job(name="ok", argv=[sys.executable, "-c", "print(1)"], log_dir=d),
        Job(name="bad", argv=[sys.executable, "-c", "raise SystemExit(3)"], log_dir=d),
        Job(name="missing", argv=["no_such_binary_xyz"], log_dir=d),
    ])
    assert [r.ok for r in results] == [True, False, False]
    assert results[1].returncode == 3


def test_slurm_script_is_posix():
    """Generated sbatch scripts run on Linux even when rendered from Windows."""
    ex = SlurmExecutor(work_dir="/scratch/run", partition="pi_keating", conda_env="PottsMPNN")
    script = ex._render_script(Path("/scratch/run/manifest.txt"))
    assert "\\" not in script
    assert "/scratch/run/manifest.txt" in script
    assert "#SBATCH --partition=pi_keating" in script


# ----------------------------------------------------------------- search pool


def test_pooling_rewrites_mutations_against_original_wt():
    wt = "ACDEFGHIKL"
    order, lengths = ["A", "B"], {"A": 5, "B": 5}
    s0 = SeedRecord(seed_id="s0", sequence=wt, mutations=[], round_index=2)
    s1 = SeedRecord(seed_id="s1", sequence="AYDEFGHIKL", mutations=["A:C2Y"], round_index=2)
    # Each seed's search reports mutations relative to its own seed.
    r0 = {1: pd.DataFrame({"sequence": ["AWDEFGHIKL"], "mutations": ["A:C2W"], "score": [-1.0]})}
    r1 = {1: pd.DataFrame({"sequence": ["AYDEFGHIKW"], "mutations": ["B:L5W"], "score": [-3.0]})}

    pooled = pool_seed_results({"s0": r0, "s1": r1}, {"s0": s0, "s1": s1}, wt, order, lengths, 2)
    by_seq = dict(zip(pooled["sequence"], pooled["mutations"]))
    assert by_seq["AYDEFGHIKW"] == "A:C2Y,B:L5W"  # cumulative, not just this round
    assert by_seq["AWDEFGHIKL"] == "A:C2W"
    assert set(pooled["round"]) == {2}


def _two_seed_pool():
    """Two structures whose Potts energies live on incomparable scales.

    Seed A's energy table happens to yield much lower binding scores than seed
    B's. Those numbers say nothing about each other -- different structure,
    different energy table.
    """
    rows = []
    for i in range(4):
        rows.append({"seed_id": "sA", "sequence": f"A{i}", "mutations": f"B:A{i + 1}C",
                     "stability_score": -1.0, "binding_score": -20.0 - i})
    for i in range(4):
        rows.append({"seed_id": "sB", "sequence": f"B{i}", "mutations": f"B:D{i + 1}E",
                     "stability_score": -1.0, "binding_score": -2.0 - i})
    return pd.DataFrame(rows)


def test_selection_is_per_structure_not_pooled():
    """Potts energies must never be ranked across structures.

    A pooled ranking would hand every slot to seed A purely because its energy
    table is offset lower. Per-seed selection gives each structure its share.
    """
    df = _two_seed_pool()
    cfg = _gating_cfg(**{"selection": {"scope": "per_seed", "max_candidates": 4,
                                       "constraints": [], "diversity": {"enabled": False}}})
    picked = selection.select_from_config(df, cfg)
    per_seed = picked["seed_id"].value_counts().to_dict()
    assert per_seed == {"sA": 2, "sB": 2}, per_seed

    # Demonstrate the failure mode the grouping prevents.
    pooled = selection._select_with_cfg(df, cfg, 4)
    assert set(pooled["seed_id"]) == {"sA"}


def test_pooled_scope_rejected_with_multiple_seeds():
    """The invalid configuration must be refused, not silently mis-rank."""
    try:
        _gating_cfg(**{"selection": {"scope": "pooled"}, "gating": {"promote_top_n": 5}})
        raise AssertionError("should have rejected pooled scope with multiple seeds")
    except ValueError as exc:
        assert "energy table" in str(exc)


def _ensure_pareto_importable():
    """Make ``mutation_search``'s pure-numpy Pareto helpers importable.

    They live in a module that imports torch at top level. Off-cluster we stub
    the heavy imports so the real Pareto code still runs; where torch is
    installed this is a no-op.
    """
    import types

    try:
        import mutation_search  # noqa: F401
        return
    except ImportError:
        pass
    for name in ("torch", "data_utils", "potts_mpnn_utils", "run_utils"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__getattr__ = lambda attr: type(attr, (), {})
            sys.modules[name] = module
    sys.modules["torch"].load = lambda *a, **k: {}


def test_pareto_front_computed_within_structure():
    """A derived Pareto objective must also be per-structure."""
    _ensure_pareto_importable()
    df = _two_seed_pool()
    cfg = _gating_cfg(**{"selection": {"scope": "per_seed", "objective": "pareto_rank",
                                       "max_candidates": 4, "constraints": [],
                                       "diversity": {"enabled": False}}})
    picked = selection.select_from_config(df, cfg)
    # Both structures contribute front members; a pooled front would be dominated
    # entirely by seed A's lower binding scores.
    assert set(picked["seed_id"]) == {"sA", "sB"}


def test_pooling_keeps_cross_seed_duplicates_for_per_seed_ranking():
    """Pooling must not collapse across seeds before selection runs."""
    wt = "ACDEFGHIKL"
    order, lengths = ["A", "B"], {"A": 5, "B": 5}
    s0 = SeedRecord(seed_id="s0", sequence=wt, mutations=[], round_index=1)
    s1 = SeedRecord(seed_id="s1", sequence=wt, mutations=[], round_index=1)
    same = pd.DataFrame({"sequence": ["AWDEFGHIKL"], "mutations": ["A:C2W"], "score": [-1.0]})
    pooled = pool_seed_results({"s0": {1: same}, "s1": {1: same.assign(score=-9.0)}},
                               {"s0": s0, "s1": s1}, wt, order, lengths, 1)
    assert len(pooled) == 2  # one row per structure, both preserved
    assert set(pooled["seed_id"]) == {"s0", "s1"}


def test_promotion_ranks_on_structural_metrics_only():
    """The cross-structure comparison must use ipSAE/PISA, not Potts energies."""
    cfg = _gating_cfg()
    # Seed A's Potts binding_score is far better, but its structural metrics are
    # worse. Promotion must follow the structural metrics.
    df = pd.DataFrame({
        "sequence": ["fromA", "fromB"],
        "mutations": ["B:A1C", "B:D2E"],
        "seed_id": ["sA", "sB"],
        "binding_score": [-99.0, -1.0],     # incomparable across structures
        "ipsae": [0.30, 0.45],
        "dG_binding": [-14.0, -18.0],
    })
    evaluated = gating.evaluate_round(df, cfg, {"ipsae": 0.275, "dG_binding": -13.4})
    ranked = gating.rank_for_promotion(gating.winners(evaluated), cfg)
    assert list(ranked["sequence"]) == ["fromB", "fromA"]
    # promote_by pins it to a single metric instead.
    by_metric = gating.rank_for_promotion(
        gating.winners(evaluated), _gating_cfg(**{"gating": {"promote_by": "ipsae"}})
    )
    assert list(by_metric["sequence"]) == ["fromB", "fromA"]


def test_global_keep_budget_is_flat_in_seed_count():
    """Adding seeds must not multiply the number of sequences scored."""
    from optimize.search_stage import project_scored_sequences, seed_keep_budget

    cfg = _gating_cfg(**{"search": {"max_keep_per_depth": 1000, "max_mutations": 3,
                                    "keep_budget_scope": "global"}})
    assert seed_keep_budget(cfg, 1) == 1000
    assert seed_keep_budget(cfg, 5) == 200
    assert seed_keep_budget(cfg, 10_000) == 1  # never drops below 1

    # Depth 2+ ceilings are identical whether the budget is spent on 1 seed or 5.
    one = project_scored_sequences(cfg, 1, 30)
    five = project_scored_sequences(cfg, 5, 30)
    assert five[1] == 5 * one[1]          # depth 1 is one enumeration per seed
    assert five[3] == one[3]              # deeper levels stay flat


def test_per_seed_scope_multiplies_cost():
    """The opt-out reproduces the naive per-seed behaviour."""
    from optimize.search_stage import project_scored_sequences, seed_keep_budget

    cfg = _gating_cfg(**{"search": {"max_keep_per_depth": 1000, "max_mutations": 3,
                                    "keep_budget_scope": "per_seed"}})
    assert seed_keep_budget(cfg, 5) == 1000
    one = project_scored_sequences(cfg, 1, 30)
    five = project_scored_sequences(cfg, 5, 30)
    assert five[3] == 5 * one[3]


def test_projection_grows_with_depth_not_silently():
    """Depth is where cost explodes; the projection must make that visible."""
    from optimize.search_stage import project_scored_sequences

    cfg = _gating_cfg(**{"search": {"max_mutations": 4, "max_keep_per_depth": 1000,
                                    "top_percent": 10.0}})
    projection = project_scored_sequences(cfg, 1, 30)
    assert projection[1] == 30 * 19
    assert projection[4] > 100 * projection[1]


def test_dedupe_keeps_best_score():
    df = pd.DataFrame({"sequence": ["S", "S", "T"], "score": [-1.0, -5.0, 0.0]})
    out = dedupe_by_sequence(df)
    assert len(out) == 2
    assert float(out[out.sequence == "S"]["score"].iloc[0]) == -5.0


# ------------------------------------------------------------------- loop (e2e)


def _write_target_fasta(directory: Path) -> str:
    """FASTA in the pipeline's format: '>name|A:B' then 'seqA:seqB'."""
    atoms = seeding.read_backbone_atoms(str(EXAMPLE_PDB))
    order = seeding.infer_chain_order(atoms)
    lengths = seeding.backbone_chain_lengths(atoms, order)
    sequence = seeding.backbone_sequence(atoms, order)
    offset, chunks = 0, []
    for chain in order:
        chunks.append(sequence[offset:offset + lengths[chain]])
        offset += lengths[chain]
    path = directory / "target.fasta"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">2ZTA|{':'.join(order)}\n{':'.join(chunks)}\n", encoding="utf-8")
    return str(path)


def _loop_cfg(out_dir, over=None, emit_structures=False):
    extra = ["--drift", "6"] + (["--emit_structures"] if emit_structures else [])
    user = OmegaConf.create({
        "run": {"out_dir": str(out_dir), "out_name": "t", "max_iterations": 3,
                "executor": "local", "backbone_source": "wt"},
        "target": {"pdb": str(EXAMPLE_PDB), "fasta": _write_target_fasta(Path(out_dir)),
                   "binding_partitions": [["A"], ["B"]]},
        "search": {"cfg_path": "x.yaml", "binder_chain": "B"},
        "selection": {"objective": "binding_score", "direction": "min",
                      "constraints": ["stability_score <= 0"], "max_candidates": 6},
        # Adapter left at its defaults: this exercises the real pinned schema.
        "structure": {"pipeline_script": str(STUB_PIPELINE), "python_executable": sys.executable,
                      "max_parallel": 1, "extra_args": extra},
        # The stub does not fold real structures, so the RMSD gate is off here;
        # it has dedicated tests with real coordinates.
        "gating": {"promote_top_n": 2, "rmsd_gate": {"enabled": False}},
    })
    cfg = OmegaConf.merge(OmegaConf.structured(OptimizationConfig), user,
                          OmegaConf.create(over or {}))
    validate(cfg)
    return cfg


def _run_quiet(cfg):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = loop.run_optimization(cfg)
    return summary, buf.getvalue()


def test_loop_reseeds_and_accumulates_mutations():
    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    summary, _ = _run_quiet(_loop_cfg(d))
    assert summary["termination_reason"] == gating.MAX_ITERATIONS

    state = json.loads((d / "run_state.json").read_text())
    rounds = {r["index"]: r for r in state["rounds"]}
    assert rounds[1]["seeds"][0]["parent_seed_id"] == "r0s0"

    # Mutation count must grow as rounds build on their predecessors.
    depths = []
    for index in range(3):
        df = pd.read_csv(d / f"round_{index}" / "scored_candidates.csv")
        counts = df["mutations"].fillna("").map(
            lambda m: len(seeding.parse_mutation_tokens(m))
        )
        depths.append(int(counts.max()))
    assert depths == sorted(depths) and depths[-1] > depths[0]


def test_loop_resume_replays_nothing():
    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    _run_quiet(_loop_cfg(d))

    # Simulate preemption during the final round.
    shutil.rmtree(d / "round_2", ignore_errors=True)
    state = json.loads((d / "run_state.json").read_text())
    for record in state["rounds"]:
        if record["index"] == 2:
            record["stages"] = {}
    state["finished"] = False
    (d / "run_state.json").write_text(json.dumps(state, indent=2))

    _, output = _run_quiet(_loop_cfg(d))
    for index in (0, 1):
        for stage in ("search", "select", "structure"):
            assert f"[round {index}] {stage}: reusing" in output, (index, stage)


def test_loop_terminates_on_cutoff_and_empty_selection():
    loop.search_stage.run_round_search = stub_run_round_search

    cutoff, _ = _run_quiet(_loop_cfg(
        Path(tempfile.mkdtemp()), {"gating": {"metrics": {"ipsae": {"cutoff": 0.60}}}}
    ))
    assert cutoff["termination_reason"] == gating.CUTOFF_MET

    empty, _ = _run_quiet(_loop_cfg(
        Path(tempfile.mkdtemp()), {"selection": {"constraints": ["stability_score <= -99"]}}
    ))
    assert empty["termination_reason"] == gating.NO_CANDIDATES


def test_wildtype_baseline_comes_from_pipeline_wt_row():
    """The pipeline prepends a 'WT' row; no separate wildtype job is needed."""
    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    summary, output = _run_quiet(_loop_cfg(d))
    baseline = summary["wildtype_baseline"]
    assert set(baseline) == {"ipsae", "dG_binding"}
    # The stub's wildtype has zero mutations, so its ipSAE sits at the 0.50 floor.
    assert 0.50 <= baseline["ipsae"] < 0.65
    assert json.loads((d / "run_state.json").read_text())["wt_sequence"]
    # The WT row must not leak into the scored candidates.
    scored = pd.read_csv(d / "round_0" / "scored_candidates.csv")
    assert "WT" not in set(scored["mutations"].astype(str))


def test_af3_backbone_source_reseeds_from_predicted_structures():
    """The full run.backbone_source='af3' path: promote -> locate model -> convert."""
    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    cfg = _loop_cfg(d, {"run": {"backbone_source": "af3", "max_iterations": 2}},
                    emit_structures=True)
    summary, _ = _run_quiet(cfg)

    state = json.loads((d / "run_state.json").read_text())
    round1 = next(r for r in state["rounds"] if r["index"] == 1)
    assert round1["seeds"], "round 1 should have been seeded from round 0 winners"

    for seed in round1["seeds"]:
        # Each seed resolved to a real AF3 model chosen by ipSAE...
        assert seed["af3_dir"] and Path(seed["af3_dir"]).exists()
        assert "seed-1_sample-0" in seed["af3_dir"]  # stub makes sample 0 best
        # ...and that model became a backbone carrying the seed's own sequence.
        backbone = Path(seed["backbone_pdb"])
        assert backbone.exists()
        atoms = seeding.read_backbone_atoms(str(backbone))
        order = seeding.infer_chain_order(atoms)
        assert seeding.backbone_sequence(atoms, order) == seed["sequence"]


def test_rmsd_gate_disqualifies_in_full_loop():
    """With the gate on and structures that do not match the seed, no mutant is
    promoted despite strong ipSAE/PISA -- proving the gate is wired into the loop."""
    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    cfg = _loop_cfg(d, {"gating": {"rmsd_gate": {"enabled": True, "max_rmsd": 2.0}}},
                    emit_structures=True)
    summary, output = _run_quiet(cfg)
    assert summary["termination_reason"] == gating.NO_WINNERS
    r0 = pd.read_csv(d / "round_0" / "round_summary.csv")
    assert r0["rmsd"].notna().all()          # every folded mutant got an RMSD
    assert not r0["passes_rmsd"].any()        # all disqualified by the gate
    assert "rmsd_fail=" in output


def test_all_rounds_share_one_af3_directory():
    """A shared output root lets the pipeline's own skip logic avoid re-folding."""
    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    _run_quiet(_loop_cfg(d))
    shared = d / "structure"
    assert shared.is_dir()
    # One results CSV per round, all in the same directory.
    assert len(list(shared.glob("*_with_af3.csv"))) >= 2
    assert not (d / "round_0" / "structure").exists()


# ------------------------------------------------------------------ rmsd gate


def test_kabsch_rmsd_is_superposition_invariant():
    """Identical structures in any frame give 0; more noise gives more RMSD."""
    from optimize.rmsd import kabsch_rmsd

    rng = np.random.default_rng(0)
    p = rng.normal(size=(80, 3))

    # A random rigid transform must not change the RMSD from zero.
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    q = p @ rot.T + np.array([5.0, -3.0, 2.0])
    assert kabsch_rmsd(p, q) < 1e-9

    small = kabsch_rmsd(p, p + rng.normal(scale=0.3, size=p.shape))
    large = kabsch_rmsd(p, p + rng.normal(scale=2.0, size=p.shape))
    assert 0 < small < large


def test_structure_rmsd_zero_against_self_and_pairs_positionally():
    """A structure compared to a rigid transform of itself has ~0 RMSD."""
    from optimize import rmsd

    atoms = seeding.read_backbone_atoms(str(EXAMPLE_PDB))
    order = seeding.infer_chain_order(atoms)

    # Rigidly move a copy and renumber its residues; positional pairing must
    # still line them up, and Kabsch must undo the motion.
    theta = 0.4
    rot = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    moved = []
    for a in atoms:
        x, y, z = np.array([a["x"], a["y"], a["z"]]) @ rot.T + np.array([10.0, 0.0, -4.0])
        b = dict(a); b["x"], b["y"], b["z"] = float(x), float(y), float(z)
        b["resseq"] = a["resseq"] + 500  # different numbering, same order
        moved.append(b)

    d = Path(tempfile.mkdtemp())
    seeding._write_backbone_pdb(atoms, d / "ref.pdb", order)
    seeding._write_backbone_pdb(moved, d / "model.pdb", order)
    # Threshold is above PDB's 0.001 A coordinate precision, far below any
    # meaningful conformational change.
    assert rmsd.structure_rmsd(str(d / "model.pdb"), str(d / "ref.pdb"), order) < 0.01


def _perturbed_copies(atoms, order, distal_cutoff=20.0):
    """(distal-domain-moved, binder-displaced) variants of a two-chain complex.

    "Distal" is chosen by distance from the binder, not by residue index: a
    coiled-coil like 2ZTA has no residue far from its partner, so an index-based
    split would perturb the binding site itself.
    """
    chains = sorted({a["chain"] for a in atoms})
    target_chain, binder_chain = chains[0], chains[-1]

    binder_xyz = np.array([[a["x"], a["y"], a["z"]] for a in atoms if a["chain"] == binder_chain])
    dist_by_res = {}
    for a in atoms:
        if a["chain"] != target_chain:
            continue
        d = np.linalg.norm(binder_xyz - np.array([a["x"], a["y"], a["z"]]), axis=1).min()
        dist_by_res[a["resseq"]] = min(dist_by_res.get(a["resseq"], 1e9), d)
    distal = {r for r, d in dist_by_res.items() if d > distal_cutoff}
    if not distal:
        raise AssertionError(
            f"no target residue is >{distal_cutoff} A from the binder; "
            "this structure has no distal region to perturb"
        )

    theta = 0.35
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = []
    for a in atoms:
        b = dict(a)
        if a["chain"] == target_chain and a["resseq"] in distal:
            v = np.array([a["x"], a["y"], a["z"]]) @ rot.T + np.array([3.0, 0.0, 0.0])
            b["x"], b["y"], b["z"] = map(float, v)
        moved.append(b)

    shifted = []
    for a in atoms:
        b = dict(a)
        if a["chain"] == binder_chain:
            b["x"] = a["x"] + 5.0
        shifted.append(b)
    return moved, shifted, binder_chain


MULTIDOMAIN_PDB = REPO_ROOT / "energy_benchmark_datasets" / "covid_pdbs" / "6m0j.pdb"


def test_rmsd_scope_interface_ignores_distal_domain_motion():
    """A global RMSD over a multi-domain target fails candidates for domain motion
    far from the binding site -- the bug that rejected every real candidate."""
    from optimize import rmsd

    atoms = seeding.read_backbone_atoms(str(MULTIDOMAIN_PDB))
    order = seeding.infer_chain_order(atoms)
    moved, _, binder = _perturbed_copies(atoms, order)

    d = Path(tempfile.mkdtemp())
    seeding._write_backbone_pdb(atoms, d / "ref.pdb", order)
    seeding._write_backbone_pdb(moved, d / "moved.pdb", order)

    glob_rmsd = rmsd.structure_rmsd(str(d / "moved.pdb"), str(d / "ref.pdb"), order,
                                    scope="complex")
    iface_rmsd = rmsd.structure_rmsd(str(d / "moved.pdb"), str(d / "ref.pdb"), order,
                                     scope="interface", binder_chains=[binder])
    # The binder never moved, so an interface-local measure must see ~nothing...
    assert iface_rmsd < 0.01, iface_rmsd
    # ...while the global one registers the untouched-binding-site domain motion.
    assert glob_rmsd > iface_rmsd + 0.5, (glob_rmsd, iface_rmsd)


def test_rmsd_scope_interface_catches_binder_displacement():
    """The converse: a real mis-dock is diluted by a large target under 'complex'."""
    from optimize import rmsd

    atoms = seeding.read_backbone_atoms(str(MULTIDOMAIN_PDB))
    order = seeding.infer_chain_order(atoms)
    _, shifted, binder = _perturbed_copies(atoms, order)

    d = Path(tempfile.mkdtemp())
    seeding._write_backbone_pdb(atoms, d / "ref.pdb", order)
    seeding._write_backbone_pdb(shifted, d / "shift.pdb", order)

    iface = rmsd.structure_rmsd(str(d / "shift.pdb"), str(d / "ref.pdb"), order,
                                scope="interface", binder_chains=[binder])
    glob = rmsd.structure_rmsd(str(d / "shift.pdb"), str(d / "ref.pdb"), order, scope="complex")
    # The binder moved 5 A; the interface scope must report it faithfully.
    assert abs(iface - 5.0) < 0.05, iface
    # The global scope understates it, because the stationary target dominates.
    assert glob < iface


def test_rmsd_scope_binder_is_fold_only():
    """'binder' scope ignores placement entirely: a rigid shift reads as zero."""
    from optimize import rmsd

    atoms = seeding.read_backbone_atoms(str(MULTIDOMAIN_PDB))
    order = seeding.infer_chain_order(atoms)
    _, shifted, binder = _perturbed_copies(atoms, order)
    d = Path(tempfile.mkdtemp())
    seeding._write_backbone_pdb(atoms, d / "ref.pdb", order)
    seeding._write_backbone_pdb(shifted, d / "shift.pdb", order)
    v = rmsd.structure_rmsd(str(d / "shift.pdb"), str(d / "ref.pdb"), order,
                            scope="binder", binder_chains=[binder])
    assert v < 1e-6, v


def test_rmsd_scope_default_is_interface_and_needs_a_binder():
    from optimize.config import OptimizationConfig as _Cfg, resolve_binder_chains

    defaults = OmegaConf.structured(_Cfg)
    assert defaults.gating.rmsd_gate.scope == "interface"
    assert defaults.gating.rmsd_gate.interface_cutoff == 10.0

    # binder_chain wins; otherwise the second binding partition is used.
    cfg = _gating_cfg(**{"search": {"binder_chain": "B"}})
    assert resolve_binder_chains(cfg) == ["B"]
    cfg2 = _gating_cfg(**{"search": {"binder_chain": None},
                          "target": {"binding_partitions": [["A"], ["C", "D"]]}})
    assert resolve_binder_chains(cfg2) == ["C", "D"]

    # With neither, an interface-scoped gate must be refused rather than crash later.
    try:
        _gating_cfg(**{"search": {"binder_chain": None},
                       "target": {"binding_partitions": None},
                       "gating": {"rmsd_gate": {"enabled": True, "scope": "interface"}}})
        raise AssertionError("should have rejected interface scope with no binder chain")
    except ValueError as exc:
        assert "binder" in str(exc)


def test_structure_rmsd_rejects_mismatched_residue_counts():
    from optimize import rmsd

    atoms = seeding.read_backbone_atoms(str(EXAMPLE_PDB))
    order = seeding.infer_chain_order(atoms)
    d = Path(tempfile.mkdtemp())
    seeding._write_backbone_pdb(atoms, d / "ref.pdb", order)
    seeding._write_backbone_pdb(atoms[:-8], d / "short.pdb", order)  # drop a residue
    try:
        rmsd.structure_rmsd(str(d / "short.pdb"), str(d / "ref.pdb"), order)
        raise AssertionError("should have rejected mismatched residue counts")
    except ValueError as exc:
        assert "residue counts" in str(exc)


def test_rmsd_gate_disqualifies_high_rmsd_and_nan():
    """A drifted or uncomputable mutant is out of both winners and the cutoff."""
    cfg = _gating_cfg(**{"gating": {"rmsd_gate": {"enabled": True, "max_rmsd": 2.0}}})
    df = SCORED.copy()
    df["rmsd"] = [0.5, 1.0, 3.0, float("nan")]   # S3 too far, S4 uncomputable
    out = gating.evaluate_round(df, cfg, BASELINE)

    assert list(out["passes_rmsd"]) == [True, True, False, False]
    # Metrics alone would give beats_wt = [T, F, T, F]; S3 is now disqualified.
    assert list(out["beats_wt"]) == [True, False, False, False]
    # S1 met the cutoffs but if it had drifted it could not end the run.
    drifted = SCORED.copy()
    drifted["rmsd"] = [9.0, 1.0, 1.0, 1.0]
    out2 = gating.evaluate_round(drifted, cfg, BASELINE)
    assert bool(out2.loc[0, "meets_cutoff"]) is False


def test_rmsd_gate_disabled_is_noop():
    cfg = _gating_cfg()  # helper disables the gate
    df = SCORED.copy()
    df["rmsd"] = [9.0, 9.0, 9.0, 9.0]
    out = gating.evaluate_round(df, cfg, BASELINE)
    assert list(out["passes_rmsd"]) == [True, True, True, True]
    assert list(out["beats_wt"]) == [True, False, True, False]  # unaffected by rmsd


def test_rmsd_gate_fails_closed_when_enabled_but_uncomputed():
    """Enabling the gate without an rmsd column must not promote unverified rows."""
    cfg = _gating_cfg(**{"gating": {"rmsd_gate": {"enabled": True}}})
    out = gating.evaluate_round(SCORED.copy(), cfg, BASELINE)  # no rmsd column
    assert not out["beats_wt"].any()


def _write_model_cif(path: Path, atoms) -> None:
    header = (
        "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "_atom_site.pdbx_PDB_model_num\n"
    )
    lines = [
        f"ATOM {i} {a['atom']} . {a['resname']} {a['chain']} {a['resseq']} "
        f"{a['x']:.3f} {a['y']:.3f} {a['z']:.3f} 1"
        for i, a in enumerate(atoms, 1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(lines) + "\n#\n", encoding="utf-8")


def _emit_af3_model(structure_dir, base_name, mutations, atoms, ipsae=0.9, pae=10, dist=15):
    from optimize.af3_layout import af3_job_name

    job = Path(structure_dir) / af3_job_name(base_name, mutations).lower()
    sample = job / "seed-1_sample-0"
    model = sample / "m_model.cif"
    _write_model_cif(model, atoms)
    (sample / f"m_model_{pae:02d}_{dist:02d}.txt").write_text(
        f"Chn1 Chn2 Type ipSAE\nA B asym {ipsae:.4f}\n", encoding="utf-8"
    )
    return str(model)


def test_structure_stage_computes_rmsd_against_seed_backbone():
    """End to end through structure_stage: a drifted AF3 model gets NaN-free rmsd
    and is disqualified, while a near-identical one passes."""
    from omegaconf import OmegaConf
    from optimize.structure_stage import compute_rmsd_column, shared_structure_dir

    d = Path(tempfile.mkdtemp())
    fasta = _write_target_fasta(d)
    atoms = seeding.read_backbone_atoms(str(EXAMPLE_PDB))
    order = seeding.infer_chain_order(atoms)
    base_name = "2ZTA"

    structure_dir = shared_structure_dir(d)

    # Near-identical model: a rigid transform (RMSD ~0 after superposition).
    near = [dict(a) for a in atoms]
    for a in near:
        a["x"] += 7.0  # pure translation, removed by Kabsch
    _emit_af3_model(structure_dir, base_name, "B:R2W", near)

    # Drifted model: large random displacement (RMSD well over 2 A).
    rng = np.random.default_rng(1)
    far = []
    for a in atoms:
        b = dict(a)
        b["x"] += float(rng.normal(scale=6.0))
        b["y"] += float(rng.normal(scale=6.0))
        b["z"] += float(rng.normal(scale=6.0))
        far.append(b)
    _emit_af3_model(structure_dir, base_name, "B:K4D", far)

    cfg = OmegaConf.merge(
        OmegaConf.structured(OptimizationConfig),
        OmegaConf.create({
            "run": {"out_dir": str(d)},
            "target": {"pdb": str(EXAMPLE_PDB), "fasta": fasta, "binding_partitions": [["A"], ["B"]]},
            "search": {"cfg_path": "x"}, "structure": {"pipeline_script": "x"},
            "gating": {"rmsd_gate": {"enabled": True, "max_rmsd": 2.0, "reference": "seed"}},
        }),
    )
    validate(cfg)

    scored = pd.DataFrame({
        "sequence": ["s_near", "s_far"],
        "mutations": ["B:R2W", "B:K4D"],
        "seed_id": ["r0s0", "r0s0"],
        "ipsae": [0.9, 0.9],
        "dG_binding": [-20.0, -20.0],
    })
    out = compute_rmsd_column(scored, cfg, d, {"r0s0": str(EXAMPLE_PDB)}, order)
    assert out.loc[0, "rmsd"] < 0.5      # near model
    assert out.loc[1, "rmsd"] > 2.0      # drifted model

    gated = gating.evaluate_round(out, cfg, {"ipsae": 0.5, "dG_binding": -13.4})
    assert list(gated["beats_wt"]) == [True, False]


# --------------------------------------------------------------------- report


def _fake_run_dir(tmp: Path, rounds=3) -> Path:
    """A minimal finished run: round summaries plus a state file."""
    (tmp / "run_state.json").write_text(json.dumps({
        "version": 1, "wt_sequence": "AAA",
        "wt_baseline": {"ipsae": 0.50, "dG_binding": -12.0},
        "termination_reason": "max_iterations", "rounds": [], "result_cache": {},
        "finished": True,
    }), encoding="utf-8")
    for r in range(rounds):
        d = tmp / f"round_{r}"
        d.mkdir(parents=True, exist_ok=True)
        n = 5
        pd.DataFrame({
            "sequence": [f"s{r}{i}" for i in range(n)],
            "mutations": [",".join(f"B:A{j+1}C" for j in range(i % 3 + 1)) for i in range(n)],
            "seed_id": [f"r{r}s0"] * n,
            "round": [r] * n,
            # improves with round; ipsae up, dG down
            "ipsae": [0.50 + 0.03 * r + 0.01 * i for i in range(n)],
            "dG_binding": [-12.0 - 0.5 * r - 0.2 * i for i in range(n)],
            "rmsd": [0.4] * n,
            "beats_wt_ipsae": [True] * n,
            "beats_wt_dG_binding": [True] * n,
            "beats_wt": [True] * n,
        }).to_csv(d / "round_summary.csv", index=False)
    return tmp


def test_report_infers_metric_directions_from_the_data():
    """Directions come from who beat wildtype, so a run gated on other metrics works."""
    from optimize.report import load_run

    run = load_run(_fake_run_dir(Path(tempfile.mkdtemp())))
    assert set(run.metrics) == {"ipsae", "dG_binding"}
    assert run.directions["ipsae"] == "max"
    assert run.directions["dG_binding"] == "min"
    assert run.baseline["ipsae"] == 0.50
    assert run.rounds == [0, 1, 2]


def test_report_ranking_puts_pareto_first_and_dedupes():
    from optimize.report import load_run, rank_candidates

    run = load_run(_fake_run_dir(Path(tempfile.mkdtemp())))
    ranked = rank_candidates(run)
    # duplicate mutation strings across rounds collapse to their first appearance
    assert ranked["mutations"].is_unique
    assert ranked["pareto"].iloc[0]           # a front member leads
    # the front is non-dominated: nothing beats a member on both metrics
    front = ranked[ranked["pareto"]]
    for _, row in front.iterrows():
        better = ranked[(ranked["ipsae"] >= row["ipsae"]) &
                        (ranked["dG_binding"] <= row["dG_binding"]) &
                        ((ranked["ipsae"] > row["ipsae"]) |
                         (ranked["dG_binding"] < row["dG_binding"]))]
        assert better.empty, row["mutations"]


def test_report_writes_self_contained_html():
    from optimize.report import generate

    d = _fake_run_dir(Path(tempfile.mkdtemp()))
    path = generate(d, top=5)
    text = path.read_text(encoding="utf-8")
    assert path.exists() and (d / "report" / "best_mutants.csv").exists()
    # plots inlined, nothing fetched from the network or neighbouring files
    assert text.count("data:image/png;base64,") >= 2
    assert 'src="http' not in text and "<link" not in text
    assert "Wildtype baseline" in text


def test_report_lineage_walks_back_through_ancestors():
    """A final mutant must trace to its seed, its seed's parent, and so on."""
    from optimize.report import lineage, load_run, rank_candidates

    tmp = Path(tempfile.mkdtemp())
    _fake_run_dir(tmp, rounds=3)
    # Give the state file a real promotion chain: r2s0 <- r1s0 <- r0s0.
    state = json.loads((tmp / "run_state.json").read_text(encoding="utf-8"))
    state["rounds"] = [
        {"index": 0, "seeds": [{"seed_id": "r0s0", "sequence": "A", "mutations": [],
                                "parent_seed_id": None, "round_index": 0, "metrics": {}}]},
        {"index": 1, "seeds": [{"seed_id": "r1s0", "sequence": "B", "mutations": ["B:A1C"],
                                "parent_seed_id": "r0s0", "round_index": 1,
                                "metrics": {"ipsae": 0.55, "dG_binding": -12.5}}]},
        {"index": 2, "seeds": [{"seed_id": "r2s0", "sequence": "C",
                                "mutations": ["B:A1C", "B:A2C"],
                                "parent_seed_id": "r1s0", "round_index": 2,
                                "metrics": {"ipsae": 0.60, "dG_binding": -13.5}}]},
    ]
    (tmp / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    run = load_run(tmp)
    assert set(run.seeds) == {"r0s0", "r1s0", "r2s0"}

    row = rank_candidates(run).iloc[0].copy()
    row["round"] = 2
    row["seed_id"] = "r2s0"
    chain = lineage(run, row)

    # Oldest first, and a seed promoted into round k is plotted at round k-1,
    # because that is where it was scored.
    assert [p["round"] for p in chain] == [0, 1, 2]
    assert chain[0]["metrics"]["ipsae"] == 0.55     # r1s0, scored in round 0
    assert chain[1]["metrics"]["ipsae"] == 0.60     # r2s0, scored in round 1
    assert chain[-1]["is_final"] and not chain[0]["is_final"]


def test_report_lineage_survives_a_broken_chain():
    """A missing or self-referential parent must not hang or crash the report."""
    from optimize.report import lineage, load_run

    tmp = Path(tempfile.mkdtemp())
    _fake_run_dir(tmp, rounds=2)
    state = json.loads((tmp / "run_state.json").read_text(encoding="utf-8"))
    state["rounds"] = [
        {"index": 1, "seeds": [{"seed_id": "r1s0", "sequence": "B", "mutations": ["B:A1C"],
                                "parent_seed_id": "r1s0",   # cycle
                                "round_index": 1, "metrics": {"ipsae": 0.55}}]},
    ]
    (tmp / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    run = load_run(tmp)
    row = run.frame.iloc[0].copy()
    row["seed_id"] = "r1s0"
    chain = lineage(run, row)               # must terminate
    assert len(chain) == 2

    row2 = run.frame.iloc[0].copy()
    row2["seed_id"] = "does_not_exist"
    assert len(lineage(run, row2)) == 1     # just the candidate itself


def test_report_runs_automatically_at_the_end_of_a_run():
    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    cfg = _loop_cfg(d, {"run": {"max_iterations": 2, "report": True},
                        "gating": {"promote_top_n": 2, "rmsd_gate": {"enabled": False}}},
                    emit_structures=True)
    _run_quiet(cfg)
    assert (d / "report" / "report.html").exists()
    assert (d / "report" / "best_mutants.csv").exists()


def test_report_failure_never_sinks_a_finished_run():
    """Results are already on disk; a plotting error must not discard them."""
    from optimize import report as report_module

    loop.search_stage.run_round_search = stub_run_round_search
    d = Path(tempfile.mkdtemp())
    cfg = _loop_cfg(d, {"run": {"max_iterations": 1, "report": True},
                        "gating": {"rmsd_gate": {"enabled": False}}},
                    emit_structures=True)
    original = report_module.generate
    report_module.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        summary, output = _run_quiet(cfg)
    finally:
        report_module.generate = original
    assert summary["termination_reason"]                       # the run still finished
    assert (d / "round_0" / "round_summary.csv").exists()      # results intact
    assert "could not write the report" in output              # and it said so


def test_report_writes_every_artifact_to_one_place():
    """Plots as PNGs, the raw combined CSV, the ranked CSV, and a FASTA."""
    from optimize.report import generate, load_run

    d = _fake_run_dir(Path(tempfile.mkdtemp()), rounds=3)
    generate(d, top=5)
    report_dir = d / "report"

    for name in ("report.html", "all_candidates.csv", "best_mutants.csv",
                 "top_sequences.fasta"):
        assert (report_dir / name).exists(), name

    plots = {p.name for p in (report_dir / "plots").glob("*.png")}
    # one per round, plus the summary plots
    for rnd in load_run(d).rounds:
        assert f"round_{rnd:02d}_tradeoff.png" in plots, rnd
    assert {"progression.png", "tradeoff_all_rounds.png"} <= plots
    assert all(p.stat().st_size > 1000 for p in (report_dir / "plots").glob("*.png"))


def test_all_candidates_csv_keeps_every_row():
    """The raw record must not deduplicate -- that is best_mutants.csv's job."""
    from optimize.report import generate, load_run

    d = _fake_run_dir(Path(tempfile.mkdtemp()), rounds=3)
    generate(d, top=5)
    raw = pd.read_csv(d / "report" / "all_candidates.csv")
    ranked = pd.read_csv(d / "report" / "best_mutants.csv")

    assert len(raw) == len(load_run(d).frame)      # every scored row, all rounds
    assert set(raw["round"]) == {0, 1, 2}
    assert len(ranked) <= len(raw)                 # ranked is the deduped view
    assert ranked["mutations"].is_unique
    for col in ("sequence", "mutations", "ipsae", "dG_binding", "round"):
        assert col in raw.columns, col


def test_top_sequences_fasta_is_usable():
    from optimize.report import generate

    d = _fake_run_dir(Path(tempfile.mkdtemp()), rounds=2)
    generate(d, top=3)
    lines = (d / "report" / "top_sequences.fasta").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6                          # 3 records, header + sequence
    assert lines[0].startswith(">rank001 round=")
    assert "ipsae=" in lines[0] and "mutations=" in lines[0]
    assert lines[1] and not lines[1].startswith(">")


def test_round_scatter_counts_better_than_wt():
    """The title's better-than-WT count must respect each metric's direction."""
    from optimize.report import load_run, plot_round_scatter

    d = _fake_run_dir(Path(tempfile.mkdtemp()), rounds=1)
    run = load_run(d)
    out = Path(tempfile.mkdtemp()) / "r0.png"
    assert plot_round_scatter(run, 0, out) is not None
    assert out.exists() and out.stat().st_size > 1000

    sub = run.frame[run.frame["round"] == 0]
    better = ((sub["ipsae"] > run.baseline["ipsae"]) &
              (sub["dG_binding"] < run.baseline["dG_binding"])).sum()
    # The fixture's first row sits exactly on the wildtype values, and beating
    # wildtype is a strict inequality -- so it must be excluded.
    assert better == len(sub) - 1, (better, len(sub))


def test_report_errors_clearly_on_a_non_run_directory():
    from optimize.report import load_run

    try:
        load_run(Path(tempfile.mkdtemp()))
        raise AssertionError("should have raised")
    except FileNotFoundError as exc:
        assert "round_summary.csv" in str(exc)


# -------------------------------------------------------------------- cleanup


def _fake_job(root: Path, job: str, n_samples: int = 3) -> Path:
    """An AF3 job dir with the file mix the real pipeline produces."""
    job_dir = root / job.lower()
    for s in range(n_samples):
        d = job_dir / f"seed-1_sample-{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "m_model.cif").write_text("data_x\n" * 50, encoding="utf-8")
        (d / "m_model_10_15.txt").write_text("Chn1 Chn2 Type ipSAE\nA B asym 0.5\n",
                                             encoding="utf-8")
        # The O(N^2) bulk.
        (d / "m_confidences.json").write_text('{"pae": [' + "0.0," * 5000 + "0.0]}",
                                              encoding="utf-8")
        (d / "m_summary_confidences.json").write_text('{"iptm": 0.5}', encoding="utf-8")
        (d / "m_pisa.xml").write_text("<x/>" * 500, encoding="utf-8")
        (d / "m_pisa_assemblies.xml").write_text("<y/>" * 500, encoding="utf-8")
    return job_dir


def test_cleanup_never_removes_structures_or_ipsae_reports():
    """The RMSD gate and re-seeding depend on these; pruning must not touch them."""
    from optimize.cleanup import prune_jobs

    root = Path(tempfile.mkdtemp())
    job = _fake_job(root, "T__BA1C")
    for mode in ("compress", "delete"):
        stats = prune_jobs(root, mode=mode)
        assert stats.files > 0
        for s in range(3):
            d = job / f"seed-1_sample-{s}"
            assert (d / "m_model.cif").exists(), f"{mode} removed a structure"
            assert (d / "m_model_10_15.txt").exists(), f"{mode} removed an ipSAE report"
        # rebuild for the second mode
        job = _fake_job(root, "T__BA1C")


def test_cleanup_compresses_pae_and_reclaims_space():
    from optimize.cleanup import prune_jobs

    root = Path(tempfile.mkdtemp())
    job = _fake_job(root, "T__BA1C")
    before = sum(p.stat().st_size for p in job.rglob("*") if p.is_file())
    stats = prune_jobs(root, mode="compress")
    after = sum(p.stat().st_size for p in job.rglob("*") if p.is_file())

    assert stats.bytes_saved > 0 and after < before
    d = job / "seed-1_sample-0"
    assert not (d / "m_confidences.json").exists()
    assert (d / "m_confidences.json.gz").exists()
    # gzip must be lossless
    import gzip as gz
    with gz.open(d / "m_confidences.json.gz", "rt", encoding="utf-8") as fh:
        assert fh.read().startswith('{"pae":')


def test_cleanup_protects_winners():
    from optimize.cleanup import job_names_to_protect, prune_jobs

    root = Path(tempfile.mkdtemp())
    winner = _fake_job(root, "T__BA1C")
    loser = _fake_job(root, "T__BD2E")
    protected = job_names_to_protect(["B:A1C"], "T")
    stats = prune_jobs(root, mode="delete", protected_jobs=protected)

    assert stats.jobs_protected == 1 and stats.jobs_pruned == 1
    assert (winner / "seed-1_sample-0" / "m_confidences.json").exists()
    assert not (loser / "seed-1_sample-0" / "m_confidences.json").exists()


def test_cleanup_prunes_summary_with_pae():
    """Keeping the completeness marker while dropping the PAE would make a forced
    rerun skip inference and then silently produce NaN metrics."""
    from optimize.cleanup import prune_jobs

    root = Path(tempfile.mkdtemp())
    job = _fake_job(root, "T__BA1C")
    prune_jobs(root, mode="delete", targets=["pae", "summary"])
    d = job / "seed-1_sample-0"
    assert not (d / "m_confidences.json").exists()
    assert not (d / "m_summary_confidences.json").exists()

    # Requesting only "pae" must leave the summary alone despite the glob overlap.
    job2 = _fake_job(root, "T__BD2E")
    prune_jobs(root, mode="delete", targets=["pae"])
    d2 = job2 / "seed-1_sample-0"
    assert not (d2 / "m_confidences.json").exists()
    assert (d2 / "m_summary_confidences.json").exists()


def test_cleanup_leaves_find_best_model_working():
    """After pruning, re-seeding must still locate each mutant's best model."""
    from optimize.af3_layout import find_best_model
    from optimize.cleanup import prune_jobs

    root = Path(tempfile.mkdtemp())
    _fake_job(root, "T__BA1C")
    before = find_best_model(str(root), "T__BA1C", 10, 15)
    prune_jobs(root, mode="delete")
    after = find_best_model(str(root), "T__BA1C", 10, 15)
    assert before is not None and after == before


def test_cleanup_requires_the_sequence_cache():
    """Pruning the completeness marker is only safe if our cache prevents re-folds."""
    try:
        _gating_cfg(**{"structure": {"cleanup": {"mode": "compress"},
                                     "cache_by_sequence": False}})
        raise AssertionError("should have rejected cleanup without the sequence cache")
    except ValueError as exc:
        assert "cache_by_sequence" in str(exc)


def test_cleanup_mode_none_is_inert():
    from optimize.cleanup import prune_jobs

    root = Path(tempfile.mkdtemp())
    job = _fake_job(root, "T__BA1C")
    n_before = len(list(job.rglob("*")))
    stats = prune_jobs(root, mode="none")
    assert stats.files == 0 and len(list(job.rglob("*"))) == n_before


# ----------------------------------------------------------------- af3 layout


def test_af3_job_name_matches_pipeline():
    """Reproduces run_mutation_af3_pipeline.py's job naming."""
    from optimize import af3_layout

    assert af3_layout.mutation_af3_tokens("B:W102E,B:I110S") == ["BW102E", "BI110S"]
    assert af3_layout.af3_job_name("350d", "B:W102E,B:I110S") == "350d__BW102E_BI110S"
    # Wildtype and empty/NaN all collapse to the pipeline's "WT" token.
    for wt in ("WT", "", None, float("nan")):
        assert af3_layout.af3_job_name("350d", wt) == "350d__WT"


def test_mutation_order_is_preserved_for_job_names():
    """The AF3 job name joins tokens in order, so re-sorting renames the job.

    diff_to_wt emits positional order (B:L5D,B:E6D); alphabetical sorting would
    produce B:E6D,B:L5D and point at a directory that does not exist.
    """
    from optimize.af3_layout import af3_job_name, candidate_job_names

    assert seeding.split_mutations("B:L5D,B:E6D") == ["B:L5D", "B:E6D"]   # order kept
    assert seeding.parse_mutation_tokens("B:L5D,B:E6D") == ("B:E6D", "B:L5D")  # set, sorted
    assert af3_job_name("T", "B:L5D,B:E6D") != af3_job_name("T", "B:E6D,B:L5D")
    # Lookup tries both orderings, so either spelling finds the same job.
    names = candidate_job_names("T", "B:L5D,B:E6D")
    assert "T__BL5D_BE6D" in names and "T__BE6D_BL5D" in names


def test_promotion_preserves_mutation_order():
    """Promoted seeds must keep the order their model was named with."""
    cfg = _gating_cfg()
    df = pd.DataFrame({
        "sequence": ["s1"],
        "mutations": ["B:L5D,B:E6D"],     # positional order, as diff_to_wt emits
        "ipsae": [0.80],
        "dG_binding": [-20.0],
    })
    evaluated = gating.evaluate_round(df, cfg, BASELINE)
    seeds = gating.select_promotions(evaluated, cfg, 0)
    assert seeds[0].mutations == ["B:L5D", "B:E6D"], seeds[0].mutations


def test_find_model_tolerates_either_token_order():
    """A model stored under one ordering is found when asked for by the other."""
    from optimize.af3_layout import find_model_for_mutant

    root = Path(tempfile.mkdtemp())
    _fake_job(root, "T__BL5D_BE6D")           # stored positionally
    found = find_model_for_mutant(str(root), "T", "B:E6D,B:L5D")   # asked alphabetically
    assert "bl5d_be6d" in found.replace("\\", "/")


def test_af3_best_model_selection_reads_ipsae_reports():
    """The best model is the one with the highest min-asym ipSAE, as the pipeline chose."""
    from optimize import af3_layout

    root = Path(tempfile.mkdtemp())
    job = root / "350d__bw102e"
    scores = {0: 0.61, 1: 0.88, 2: 0.42}   # sample 1 should win
    for sample, score in scores.items():
        sample_dir = job / f"seed-1_sample-{sample}"
        sample_dir.mkdir(parents=True)
        model = sample_dir / "m_model.cif"
        model.write_text("data_x\n", encoding="utf-8")
        (sample_dir / "m_model_10_15.txt").write_text(
            f"Chn1 Chn2 Type ipSAE\nA B asym {score:.4f}\nB A asym {score + 0.2:.4f}\n",
            encoding="utf-8",
        )
    best = af3_layout.find_best_model(str(root), "350d__BW102E", 10, 15)
    assert best is not None and "sample-1" in best


def test_af3_missing_model_raises_with_diagnosis():
    from optimize import af3_layout

    try:
        af3_layout.find_model_for_mutant(tempfile.mkdtemp(), "350d", "B:W102E")
        raise AssertionError("should have raised")
    except FileNotFoundError as exc:
        assert "350d__BW102E" in str(exc)


def test_fasta_parsing_matches_pipeline_format():
    from optimize import af3_layout

    path = Path(tempfile.mkdtemp()) / "t.fasta"
    path.write_text(">350d_binder12|A:B\nACDEF:GHIKL\n", encoding="utf-8")
    assert af3_layout.read_fasta_base_name(str(path)) == "350d_binder12"
    name, chains = af3_layout.read_fasta_chain_seqs(str(path))
    assert name == "350d_binder12" and chains == {"A": "ACDEF", "B": "GHIKL"}


def test_af3_model_converts_to_valid_backbone():
    """A stub AF3 mmCIF must convert into a backbone matching the expected layout."""
    from optimize.testing.stub_af3_pipeline import _write_fake_model

    chains = {"A": "ACDEF", "B": "GHIKL"}
    d = Path(tempfile.mkdtemp())
    model = d / "m_model.cif"
    _write_fake_model(model, chains)

    out = d / "backbone.pdb"
    seeding.prepare_backbone_from_af3(
        str(model), str(out),
        expected_chain_order=["A", "B"],
        expected_lengths={"A": 5, "B": 5},
        expected_sequence="ACDEFGHIKL",
        validate=False,   # parse_PDB validation needs torch
    )
    atoms = seeding.read_backbone_atoms(str(out))
    assert seeding.backbone_sequence(atoms, ["A", "B"]) == "ACDEFGHIKL"


def test_af3_backbone_rejects_wrong_sequence():
    """AF3 folding something other than what we asked for must be caught."""
    from optimize.testing.stub_af3_pipeline import _write_fake_model

    d = Path(tempfile.mkdtemp())
    model = d / "m_model.cif"
    _write_fake_model(model, {"A": "ACDEF", "B": "GHIKL"})
    try:
        seeding.prepare_backbone_from_af3(
            str(model), str(d / "b.pdb"),
            expected_chain_order=["A", "B"],
            expected_lengths={"A": 5, "B": 5},
            expected_sequence="AAAAAGGGGG",
            validate=False,
        )
        raise AssertionError("should have rejected the mismatched sequence")
    except ValueError as exc:
        assert "different sequence" in str(exc)


def _write_results_csv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_no_baseline_error_names_the_cause_and_recovery():
    """A wildtype whose AF3 run failed must produce an actionable error.

    Reproduces the real failure: the pipeline writes its CSV, the WT row is
    present, but every metric is NaN because AF3 OOMed.
    """
    from optimize.af3_layout import af3_job_name
    from optimize.executors import Job
    from optimize.structure_stage import _raise_no_baseline

    d = Path(tempfile.mkdtemp())
    fasta = _write_target_fasta(d)
    cfg = _gating_cfg(**{"target": {"fasta": fasta}})
    structure_dir = d / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)

    # WT row present, all metrics NaN, and most mutants failed too.
    results = pd.DataFrame({
        "mutations": ["WT", "B:A1C", "B:D2E", "B:F3G"],
        "ipsae": [float("nan")] * 3 + [0.8],
        "dG_binding": [float("nan")] * 3 + [-20.0],
    })
    # The pipeline's permanent-skip marker for the wildtype.
    (structure_dir / f"{af3_job_name('2ZTA', 'WT')}.failed").write_text("", encoding="utf-8")

    job = Job(name="af3_round_0", argv=["x"], log_dir=structure_dir / "logs")
    try:
        _raise_no_baseline(results, cfg, structure_dir, 0, job)
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        message = str(exc)
        assert "every metric is NaN" in message          # names the cause
        assert "3 of 4" in message                       # quantifies the damage
        assert "af3_round_0.err" in message              # points at the log
        assert "retry_failed" in message                 # names the recovery
        assert "max_parallel" in message                 # flags the likely cause


def test_no_baseline_error_distinguishes_missing_wt_row():
    from optimize.executors import Job
    from optimize.structure_stage import _raise_no_baseline

    d = Path(tempfile.mkdtemp())
    cfg = _gating_cfg(**{"target": {"fasta": _write_target_fasta(d)}})
    results = pd.DataFrame({"mutations": ["B:A1C"], "ipsae": [0.8], "dG_binding": [-20.0]})
    job = Job(name="j", argv=["x"], log_dir=d)
    try:
        _raise_no_baseline(results, cfg, d, 0, job)
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        assert "no 'WT' row at all" in str(exc)


def test_failed_markers_are_reported_and_optionally_cleared():
    """Markers block a rerun silently; the loop must surface or clear them."""
    from optimize.af3_layout import af3_job_name
    from optimize.structure_stage import handle_failed_markers

    d = Path(tempfile.mkdtemp())
    cfg = _gating_cfg(**{"target": {"fasta": _write_target_fasta(d)}})
    structure_dir = d / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)
    for key in ("WT", "B:A1C"):
        (structure_dir / f"{af3_job_name('2ZTA', key)}.failed").write_text("", encoding="utf-8")

    # Default: report, do not delete.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = handle_failed_markers(["B:A1C", "B:D2E"], cfg, structure_dir, 0)
    assert n == 2
    assert "WILDTYPE" in buf.getvalue()
    assert len(list(structure_dir.glob("*.failed"))) == 2   # untouched

    # retry_failed: clear them so the mutants are re-folded.
    retry_cfg = _gating_cfg(**{
        "target": {"fasta": _write_target_fasta(d)},
        "structure": {"retry_failed": True},
    })
    with contextlib.redirect_stdout(io.StringIO()):
        n = handle_failed_markers(["B:A1C", "B:D2E"], retry_cfg, structure_dir, 0)
    assert n == 2
    assert list(structure_dir.glob("*.failed")) == []


def test_rmsd_skips_candidates_without_metrics():
    """Unscored candidates must not each emit an RMSD failure message."""
    from optimize.structure_stage import compute_rmsd_column

    d = Path(tempfile.mkdtemp())
    cfg = _gating_cfg(**{
        "target": {"fasta": _write_target_fasta(d), "pdb": str(EXAMPLE_PDB)},
        "gating": {"rmsd_gate": {"enabled": True}},
    })
    scored = pd.DataFrame({
        "sequence": ["a", "b"],
        "mutations": ["B:A1C", "B:D2E"],
        "seed_id": ["r0s0", "r0s0"],
        "ipsae": [float("nan"), float("nan")],       # both unscored
        "dG_binding": [float("nan"), float("nan")],
    })
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = compute_rmsd_column(scored, cfg, d, {"r0s0": str(EXAMPLE_PDB)},
                                  seeding.infer_chain_order(
                                      seeding.read_backbone_atoms(str(EXAMPLE_PDB))))
    assert out["rmsd"].isna().all()
    output = buf.getvalue()
    assert "skipped 2 candidate(s)" in output
    assert "could not compute RMSD" not in output   # no per-candidate noise


def test_results_glob_isolates_one_round():
    """All rounds share one AF3 dir, so a round must read only its own results.

    The glob keeps a "{stem}" placeholder for exactly this reason; without it,
    every round's CSV would match.
    """
    from optimize.structure_stage import parse_results, results_path

    cfg = _gating_cfg()
    assert "{stem}" in cfg.structure.adapter.results_glob

    shared = Path(tempfile.mkdtemp())
    for round_index, (mutation, ipsae) in enumerate([("B:A1C", 0.10), ("B:D2E", 0.90)]):
        pd.DataFrame([
            {"mutations": "WT", "ipSAE": 0.5, "dG_binding": -13.4},
            {"mutations": mutation, "ipSAE": ipsae, "dG_binding": -15.0},
        ]).to_csv(shared / f"round_{round_index}_folding_set_with_af3.csv", index=False)

    got = parse_results(shared, cfg, input_csv="/x/round_1_folding_set.csv")
    assert set(got["mutations"]) == {"WT", "B:D2E"}, "round 0's results leaked in"
    assert results_path(shared, "/x/round_1_folding_set.csv", cfg).endswith(
        "round_1_folding_set_with_af3.csv"
    )


def test_pipeline_invocation_matches_submit_scripts():
    """The generated command must carry the flags the cluster scripts use."""
    from optimize.structure_stage import build_pipeline_job

    cfg = _gating_cfg(**{
        "structure": {"pipeline_script": "/path/run_mutation_af3_pipeline.py",
                      "pisa_exe": "/p/pisa", "pisa_cfg": "/p/cfg.txt",
                      "ipsae_script": "/p/ipsae.py", "max_parallel": 16},
        "target": {"msa_json": "/p/msa.json"},
    })
    job = build_pipeline_job("/tmp/in.csv", Path(tempfile.mkdtemp()), cfg, "j")
    argv = job.argv
    for flag in ("--mutations_csv", "--fasta", "--out_dir", "--msa_json", "--pisa_exe",
                 "--pisa_cfg", "--pisa_name", "--ipsae_script", "--pae_cutoff",
                 "--dist_cutoff", "--max_parallel"):
        assert flag in argv, flag
    assert argv[argv.index("--pae_cutoff") + 1] == "10"
    assert argv[argv.index("--dist_cutoff") + 1] == "15"


# ---------------------------------------------------------------------- config


def test_validation_rejects_inconsistent_configs():
    cases = {
        "search.rank_by=pareto with energy_mode=stability":
            {"search": {"rank_by": "pareto", "energy_mode": "stability"}},
        "max_iterations below 1": {"run": {"max_iterations": 0}},
        "top_percent out of range": {"search": {"top_percent": 0}},
        "use_depths beyond max_mutations":
            {"search": {"max_mutations": 3, "use_depths": [9]}},
        "unknown gating metric": {"gating": {"beats_wt_on": ["ipsae", "nope"]}},
        "slurm without partition": {"run": {"executor": "slurm"}},
        # A declared metric with no column in the results adapter: the loop
        # could never read it out of the pipeline's output.
        "gating metric with no adapter column":
            {"gating": {"metrics": {"unmapped": {"direction": "max", "cutoff": 1.0}}}},
    }
    for label, patch in cases.items():
        raised = False
        try:
            _gating_cfg(**patch)
        except ValueError:
            raised = True
        assert raised, f"should have rejected: {label}"


def test_example_config_is_valid():
    from optimize.config import load_config

    cfg = load_config(str(REPO_ROOT / "inputs" / "example_config_optimization.yaml"))
    assert cfg.run.max_iterations >= 1
    assert set(cfg.gating.beats_wt_on) <= set(cfg.gating.metrics.keys())


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner reports everything
            failures.append((test.__name__, exc))
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

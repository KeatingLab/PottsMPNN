"""Running the mutation search once per seed and pooling the results.

``recursive_mutation_search`` is called as a library function, not through
``mutation_search.py``'s CLI: that CLI unconditionally overwrites the parsed
``--disallowed_chains`` with ``['A']`` (mutation_search.py:955), so shelling out
would silently ignore the configured chains. ``analysis_pipeline_integration.py``
already imports ``mutation_search`` as a library, so this follows existing practice.

Each seed searches against its own backbone, so a round fans out into one search
per seed and the results are pooled afterwards. Every pooled candidate has its
``mutations`` recomputed against the *original* wildtype -- see
:func:`optimize.seeding.diff_to_wt` for why lineage is derived rather than tracked.

A caveat worth repeating in any downstream analysis: scores are ddG relative to
each seed's own backbone and sequence, so they are **not comparable across
rounds**. Cross-round progress is measured by the structural metrics, which
share one fixed wildtype baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import seeding
from .state import SeedRecord

# Columns added to every pooled candidate.
PROVENANCE_COLUMNS = ["round", "seed_id", "depth", "round_mutations", "parent_mutations"]


def _seed_backbone_name(round_index: int, seed_id: str) -> str:
    """Filename stem for a seed's backbone; becomes ``pdb_data[0]['name']``."""
    return f"r{round_index}_{seed_id}"


def prepare_seed_backbone(
    seed: SeedRecord,
    cfg,
    round_dir: Path,
    wt_atoms: Sequence[dict],
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    out_dir: Optional[Path] = None,
) -> str:
    """Materialize the backbone this seed will search against.

    Round 0 uses the target PDB untouched. Later rounds either convert the
    seed's best AF3 model (``backbone_source: af3``) or rewrite the original
    backbone with the seed's sequence (``backbone_source: wt``).
    """
    backbone_dir = round_dir / "backbones"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = backbone_dir / f"{_seed_backbone_name(seed.round_index, seed.seed_id)}.pdb"

    if seed.round_index == 0:
        # The wildtype round searches the target structure as provided.
        return str(cfg.target.pdb)

    if cfg.run.backbone_source == "af3":
        from .structure_stage import resolve_seed_model

        if out_dir is None:
            raise ValueError("out_dir is required to locate AF3 models for re-seeding.")
        model_path = seed.af3_dir or resolve_seed_model(cfg, out_dir, ",".join(seed.mutations))
        return seeding.prepare_backbone_from_af3(
            model_path,
            str(out_pdb),
            expected_chain_order=chain_order,
            expected_lengths=chain_lengths,
            expected_sequence=seed.sequence,
            chain_map=dict(cfg.structure.af3_chain_map) or None,
        )

    return seeding.write_backbone_with_sequence(wt_atoms, seed.sequence, chain_order, str(out_pdb))


def seed_keep_budget(cfg, n_seeds: int) -> int:
    """The per-seed ``max_keep_per_depth`` given the round's seed count.

    Under ``keep_budget_scope: global`` the configured cap is a whole-round
    budget, split evenly across seeds, so adding seeds does not multiply the
    number of sequences scored. Under ``per_seed`` each seed gets the full cap.
    """
    cap = int(cfg.search.max_keep_per_depth)
    if cfg.search.keep_budget_scope == "per_seed" or n_seeds <= 1:
        return cap
    return max(1, cap // n_seeds)


def project_scored_sequences(cfg, n_seeds: int, n_mutable_positions: int) -> Dict[int, int]:
    """Upper bound on sequences scored per depth, summed across seeds.

    Depth 1 enumerates every single mutant of each seed; deeper levels enumerate
    every single mutant of everything kept at the previous depth. This ignores
    the de-duplication of sequences reachable by different mutation orders, so it
    over-estimates -- it is meant as a ceiling to sanity-check a config against.
    """
    from math import ceil

    per_sequence = max(0, n_mutable_positions) * 19
    cap = seed_keep_budget(cfg, n_seeds)
    projection: Dict[int, int] = {}
    kept = 1  # each seed starts from a single sequence
    for depth in range(1, int(cfg.search.max_mutations) + 1):
        generated = kept * per_sequence
        projection[depth] = generated * n_seeds
        effective_pct = cfg.search.top_percent / (
            cfg.search.top_percent_decay_base ** (depth - 1)
        )
        kept = min(max(1, ceil(generated * effective_pct / 100.0)), cap)
    return projection


def run_search_for_seed(
    seed: SeedRecord,
    backbone_pdb: str,
    cfg,
    seed_dir: Path,
    partitions: Sequence[Sequence[str]],
    max_keep_per_depth: Optional[int] = None,
) -> Dict[int, pd.DataFrame]:
    """Run ``recursive_mutation_search`` for one seed against one backbone."""
    from mutation_search import recursive_mutation_search  # local import: pulls torch

    seed_dir.mkdir(parents=True, exist_ok=True)
    pdb_name = Path(backbone_pdb).stem
    partitions_json = seeding.write_partitions_json(
        str(seed_dir / "binding_partitions.json"), pdb_name, partitions
    )

    search = cfg.search
    return recursive_mutation_search(
        pdb_paths=[backbone_pdb],
        cfg_path=search.cfg_path,
        max_mutations=search.max_mutations,
        top_percent=search.top_percent,
        allowed_mutations=None,
        disallowed_chains=list(search.disallowed_chains) or None,
        binding_energy_json=partitions_json,
        binding_energy_cutoff=search.binding_energy_cutoff,
        energy_mode=search.energy_mode,
        rrf_k=search.rrf_k,
        rank_by=search.rank_by,
        show_pareto_front=search.show_pareto_front,
        plot_dir=str(seed_dir),
        top_percent_decay_base=search.top_percent_decay_base,
        max_keep_per_depth=(
            search.max_keep_per_depth if max_keep_per_depth is None else max_keep_per_depth
        ),
        per_position_quota=search.per_position_quota,
        allowed_from_aas=list(search.allowed_from_aas) if search.allowed_from_aas else None,
        allowed_to_aas=list(search.allowed_to_aas) if search.allowed_to_aas else None,
        binder_chain=search.binder_chain,
    )


def pool_seed_results(
    results_by_seed: Dict[str, Dict[int, pd.DataFrame]],
    seeds_by_id: Dict[str, SeedRecord],
    wt_sequence: str,
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    round_index: int,
    use_depths: Sequence[int] = (),
) -> pd.DataFrame:
    """Concatenate per-seed, per-depth results into one provenance-tagged frame.

    ``mutations`` is recomputed against the original wildtype so the column means
    the same thing in every round; the search's own per-round labels are kept as
    ``round_mutations`` for debugging.
    """
    frames: List[pd.DataFrame] = []
    wanted = set(use_depths or ())

    for seed_id, by_depth in results_by_seed.items():
        seed = seeds_by_id[seed_id]
        for depth, df in sorted(by_depth.items()):
            if wanted and depth not in wanted:
                continue
            if df is None or df.empty:
                continue
            tagged = df.copy()
            tagged["round"] = round_index
            tagged["seed_id"] = seed_id
            tagged["depth"] = depth
            tagged["round_mutations"] = tagged.get("mutations", "")
            tagged["parent_mutations"] = ",".join(seed.mutations)
            tagged["mutations"] = [
                ",".join(
                    seeding.diff_to_wt(
                        sequence, wt_sequence, lengths=dict(chain_lengths), chain_order=chain_order
                    )
                )
                for sequence in tagged["sequence"]
            ]
            frames.append(tagged)

    if not frames:
        return pd.DataFrame(
            columns=["sequence", "mutations", "score", *PROVENANCE_COLUMNS]
        )

    # Deliberately NOT deduplicated here. Selection runs per seed, so each seed
    # must see its own complete candidate set; collapsing across seeds first
    # would silently remove a sequence from one structure's ranking. Duplicates
    # are collapsed after selection, in selection.select_from_config.
    return pd.concat(frames, ignore_index=True)


def dedupe_by_sequence(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Collapse identical sequences, keeping the first occurrence.

    Used within a single structure's results, where ``score`` is comparable. It
    is NOT used to merge across seeds: PottsMPNN energies come from a different
    energy table per structure, so "the better score" is not defined between
    them. Cross-seed duplicates are collapsed positionally after selection.
    """
    if df.empty or "sequence" not in df.columns:
        return df
    if score_col in df.columns and df.get("seed_id", pd.Series(dtype=object)).nunique() <= 1:
        df = df.sort_values(score_col, ascending=True, kind="mergesort")
    return df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)


def run_round_search(
    seeds: Sequence[SeedRecord],
    cfg,
    round_dir: Path,
    wt_sequence: str,
    wt_atoms: Sequence[dict],
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    partitions: Sequence[Sequence[str]],
    round_index: int,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Search from every seed in a round and return the pooled candidate frame."""
    results_by_seed: Dict[str, Dict[int, pd.DataFrame]] = {}
    seeds_by_id = {s.seed_id: s for s in seeds}

    per_seed_cap = seed_keep_budget(cfg, len(seeds))
    if len(seeds) > 1:
        print(
            f"[round {round_index}] keep budget: {per_seed_cap}/depth/seed "
            f"({cfg.search.keep_budget_scope} scope, {len(seeds)} seeds, "
            f"cap {cfg.search.max_keep_per_depth})"
        )
        # An upper bound on the round's scoring cost, using every residue of the
        # mutable chains as a ceiling on mutable positions. The interface cutoff
        # will bring the real number well below this.
        mutable = sum(
            n for c, n in chain_lengths.items() if c not in set(cfg.search.disallowed_chains)
        )
        projection = project_scored_sequences(cfg, len(seeds), mutable)
        total = sum(projection.values())
        print(
            f"[round {round_index}] projected scoring ceiling: "
            + ", ".join(f"depth {d}<={n:,}" for d, n in projection.items())
            + f" (total <={total:,})"
        )

    for seed in seeds:
        seed_dir = round_dir / f"seed_{seed.seed_id}"
        backbone = prepare_seed_backbone(
            seed, cfg, round_dir, wt_atoms, chain_order, chain_lengths, out_dir=out_dir,
        )
        seed.backbone_pdb = backbone
        print(f"[round {round_index}] searching seed {seed.seed_id} on {Path(backbone).name}")
        results_by_seed[seed.seed_id] = run_search_for_seed(
            seed, backbone, cfg, seed_dir, partitions, max_keep_per_depth=per_seed_cap
        )

    pooled = pool_seed_results(
        results_by_seed,
        seeds_by_id,
        wt_sequence,
        chain_order,
        chain_lengths,
        round_index,
        use_depths=list(cfg.search.use_depths),
    )
    print(f"[round {round_index}] pooled {len(pooled)} unique candidates from {len(seeds)} seed(s)")
    return pooled

"""A torch-free stand-in for the mutation search.

``recursive_mutation_search`` needs torch, a GPU and model weights, none of
which are needed to exercise the *loop*. This generates plausible candidates
from a seed sequence with deterministic scores, so selection, gating,
promotion, re-seeding and resume can all be tested end to end.

Install it by monkeypatching ``optimize.search_stage.run_round_search``.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import pandas as pd

from .. import seeding

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _unit_hash(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def make_candidates(
    seed_sequence: str,
    wt_sequence: str,
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    mutable_chain: str = "B",
    n_positions: int = 6,
    n_substitutions: int = 3,
) -> pd.DataFrame:
    """Enumerate single substitutions on ``seed_sequence`` with fake energies."""
    offsets: Dict[str, int] = {}
    offset = 0
    for chain in chain_order:
        offsets[chain] = offset
        offset += chain_lengths[chain]

    start = offsets[mutable_chain]
    length = chain_lengths[mutable_chain]
    rows = []
    for local in range(min(n_positions, length)):
        index = start + local
        wt_residue = seed_sequence[index]
        for residue in AMINO_ACIDS[:n_substitutions]:
            if residue == wt_residue:
                continue
            candidate = seed_sequence[:index] + residue + seed_sequence[index + 1 :]
            mutations = seeding.diff_to_wt(
                candidate, wt_sequence, lengths=dict(chain_lengths), chain_order=chain_order
            )
            # More mutations trend toward better (more negative) energies, so
            # successive rounds have something to promote.
            depth = len(mutations)
            rows.append(
                {
                    "sequence": candidate,
                    "mutations": ",".join(mutations),
                    "score": round(-depth + _unit_hash(candidate, "score") * 2 - 1, 4),
                    "stability_score": round(_unit_hash(candidate, "stab") * 2 - 1.2, 4),
                    "binding_score": round(-depth + _unit_hash(candidate, "bind") * 2 - 1, 4),
                }
            )
    return pd.DataFrame(rows)


def stub_run_round_search(
    seeds,
    cfg,
    round_dir,
    wt_sequence,
    wt_atoms,
    chain_order,
    chain_lengths,
    partitions,
    round_index,
    out_dir=None,
):
    """Drop-in replacement for ``search_stage.run_round_search``.

    Only the PottsMPNN scoring is faked. Backbone preparation still runs for
    real, so the AF3-model -> next-round-backbone path is genuinely exercised.
    """
    from ..search_stage import dedupe_by_sequence, prepare_seed_backbone

    frames: List[pd.DataFrame] = []
    for seed in seeds:
        # Real backbone preparation: converts the seed's AF3 model, or rewrites
        # the wildtype backbone with the seed's sequence.
        seed.backbone_pdb = prepare_seed_backbone(
            seed, cfg, round_dir, wt_atoms, chain_order, chain_lengths, out_dir=out_dir,
        )
        candidates = make_candidates(
            seed.sequence,
            wt_sequence,
            chain_order,
            chain_lengths,
            mutable_chain=cfg.search.binder_chain or chain_order[-1],
        )
        if candidates.empty:
            continue
        candidates["round"] = round_index
        candidates["seed_id"] = seed.seed_id
        candidates["depth"] = 1
        candidates["round_mutations"] = candidates["mutations"]
        candidates["parent_mutations"] = ",".join(seed.mutations)
        frames.append(candidates)

    if not frames:
        return pd.DataFrame(
            columns=["sequence", "mutations", "score", "round", "seed_id", "depth"]
        )
    return dedupe_by_sequence(pd.concat(frames, ignore_index=True))

"""Mutation search utilities for PottsMPNN energy ranking.

This module provides a notebook-friendly wrapper that explores combinatorial
mutations by iteratively scoring all single-site variants, keeping the top
percentage at each depth, and recursing to the next mutation count.

Code in this file was originally written by or adapted from:

    1) Birnbaum and Keating "Beyond native sequence recovery: Improved modeling of the
    sequence-energy landscape of protein structures"
    bioRxiv, 2026. doi:10.64898/2026.01.14.699067
       - Code: https://github.com/KeatingLab/PottsMPNN

"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
import argparse

from potts_mpnn_utils import PottsMPNN, parse_PDB
from run_utils import chain_to_partition_map, inter_partition_contact_mask, score_seqs

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
MAX_SEQS_PER_BATCH = 1_000_000


@dataclass
class Candidate:
    sequence: str
    mutations: Tuple[str, ...]
    positions: Tuple[int, ...]
    score: Optional[float] = None


def load_model_from_config(cfg_path: str) -> Tuple[PottsMPNN, OmegaConf]:
    """Load a PottsMPNN model and config for inference."""
    cfg = OmegaConf.load(cfg_path)
    cfg.model.vocab = 22 if "msa" in cfg.model.check_path else 21

    checkpoint = torch.load(cfg.model.check_path, map_location="cpu", weights_only=False)
    model = PottsMPNN(
        ca_only=False,
        num_letters=cfg.model.vocab,
        vocab=cfg.model.vocab,
        node_features=cfg.model.hidden_dim,
        edge_features=cfg.model.hidden_dim,
        hidden_dim=cfg.model.hidden_dim,
        potts_dim=cfg.model.potts_dim,
        num_encoder_layers=cfg.model.num_layers,
        num_decoder_layers=cfg.model.num_layers,
        k_neighbors=cfg.model.num_edges,
        augment_eps=cfg.inference.noise,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model = model.to(cfg.dev)
    for param in model.parameters():
        param.requires_grad = False
    return model, cfg


def _parse_binding_partitions(binding_energy_json: Optional[str], pdb_name: str) -> List[List[str]]:
    if not binding_energy_json:
        return []
    with open(binding_energy_json, "r", encoding="utf-8") as handle:
        binding_data = json.load(handle)
    return binding_data.get(pdb_name, [])


def _chain_lengths(pdb_data: dict) -> Dict[str, int]:
    return {chain: len(pdb_data[f"seq_chain_{chain}"]) for chain in pdb_data["chain_order"]}


def _partition_sequence(
    sequence: str,
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    partition: Sequence[str],
) -> str:
    offsets = {}
    offset = 0
    for chain in chain_order:
        offsets[chain] = offset
        offset += chain_lengths[chain]
    return "".join(
        sequence[offsets[chain] : offsets[chain] + chain_lengths[chain]]
        for chain in partition
        if chain in offsets
    )


def _partition_sequences(
    sequences: Sequence[str],
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    partition: Sequence[str],
) -> List[str]:
    return [
        _partition_sequence(sequence, chain_order, chain_lengths, partition)
        for sequence in sequences
    ]


def _concat_ca_positions(pdb_entry: dict) -> torch.Tensor:
    coords = []
    for chain in pdb_entry["chain_order"]:
        chain_coords = pdb_entry[f"coords_chain_{chain}"][f"CA_chain_{chain}"]
        coords.append(np.array(chain_coords, dtype=np.float32))
    ca_pos = np.concatenate(coords, axis=0)
    return torch.from_numpy(ca_pos).unsqueeze(0)


def _chain_encoding(chain_lengths: Dict[str, int], chain_order: Sequence[str]) -> torch.Tensor:
    encoding = []
    for idx, chain in enumerate(chain_order, start=1):
        encoding.extend([idx] * chain_lengths[chain])
    return torch.tensor([encoding], dtype=torch.long)


def _interface_mask(
    pdb_entry: dict,
    chain_lengths: Dict[str, int],
    binding_partitions: List[List[str]],
    binding_energy_cutoff: float,
) -> np.ndarray:
    ca_pos = _concat_ca_positions(pdb_entry)
    chain_order = pdb_entry["chain_order"]
    chain_encoding_all = _chain_encoding(chain_lengths, chain_order).to(device=ca_pos.device)
    partition_index = chain_to_partition_map(chain_encoding_all, chain_order, binding_partitions)
    inter_mask = inter_partition_contact_mask(ca_pos, partition_index, binding_energy_cutoff)
    return inter_mask.squeeze(0).cpu().numpy().astype(bool)


def _mask_sequence_to_interface(
    sequence: str,
    wt_sequence: str,
    interface_mask: np.ndarray,
) -> str:
    return "".join(
        seq_res if interface_mask[idx] else wt_sequence[idx]
        for idx, seq_res in enumerate(sequence)
    )


def _global_position_map(chain_lengths: Dict[str, int]) -> Dict[Tuple[str, int], int]:
    mapping = {}
    offset = 0
    for chain, length in chain_lengths.items():
        for pos in range(1, length + 1):
            mapping[(chain, pos)] = offset + pos - 1
        offset += length
    return mapping


def _allowed_mutations_by_position(
    chain_lengths: Dict[str, int],
    allowed_mutations: Optional[Dict[str, Dict[int, Optional[Iterable[str]]]]] = None,
    disallowed_chains: Optional[Iterable[str]] = None,
) -> Dict[int, List[str]]:
    global_positions = _global_position_map(chain_lengths)
    disallowed_chains_set = set(disallowed_chains or [])
    if not allowed_mutations:
        allowed_by_pos: Dict[int, List[str]] = {}
        offset = 0
        for chain, length in chain_lengths.items():
            if chain in disallowed_chains_set:
                offset += length
                continue
            for pos in range(length):
                allowed_by_pos[offset + pos] = AMINO_ACIDS[:]
            offset += length
        return allowed_by_pos

    allowed_by_pos: Dict[int, List[str]] = {}
    for chain, position_map in allowed_mutations.items():
        if chain not in chain_lengths:
            raise ValueError(f"Chain '{chain}' is not present in the structure.")
        if chain in disallowed_chains_set:
            continue
        if isinstance(position_map, dict):
            for pos, residues in position_map.items():
                idx = global_positions[(chain, pos)]
                if residues is None:
                    allowed_by_pos[idx] = AMINO_ACIDS[:]
                else:
                    allowed_by_pos[idx] = [res for res in residues if res in AMINO_ACIDS]
        else:
            for pos in position_map:
                idx = global_positions[(chain, pos)]
                allowed_by_pos[idx] = AMINO_ACIDS[:]
    return allowed_by_pos


def _plot_mutation_distributions(
    results: Dict[int, pd.DataFrame],
    chain_lengths: Dict[str, int],
    output_dir: Optional[str],
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    chain_order = list(chain_lengths.keys())

    for depth, df in results.items():
        if df.empty:
            continue
        chain_counts = {chain: np.zeros(chain_lengths[chain], dtype=int) for chain in chain_order}
        for mut_list in df["mutations"].fillna(""):
            if not mut_list:
                continue
            for mut in mut_list.split(","):
                try:
                    chain, rest = mut.split(":")
                    wt = rest[0]
                    pos = int(rest[1:-1])
                    mut_res = rest[-1]
                except ValueError:
                    continue
                _ = wt, mut_res
                if chain not in chain_counts:
                    continue
                chain_counts[chain][pos - 1] += 1

        n_chains = len(chain_order)
        fig, axes = plt.subplots(
            nrows=n_chains,
            ncols=1,
            figsize=(12, max(2.5, 2.0 * n_chains)),
            sharey=True,
        )
        if n_chains == 1:
            axes = [axes]
        for ax, chain in zip(axes, chain_order):
            counts = chain_counts[chain]
            ax.bar(np.arange(len(counts)) + 1, counts, color="#4c78a8")
            ax.set_title(f"Chain {chain}")
            ax.set_xlabel("Position")
            ax.set_xlim(0.5, len(counts) + 0.5)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle(f"Mutation distribution (depth {depth})")
        axes[0].set_ylabel("Mutation count")
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.savefig(os.path.join(output_dir, f"mutation_distribution_depth_{depth}.png"))
        plt.close(fig)




def _plot_pareto_fronts(
    results: Dict[int, pd.DataFrame],
    output_dir: Optional[str],
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)

    for depth, df in results.items():
        if df.empty or "pareto_front" not in df.columns:
            continue
        if not {"stability_score", "binding_score"}.issubset(df.columns):
            continue

        pareto_mask = df["pareto_front"].fillna(False).to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(
            df["stability_score"],
            df["binding_score"],
            color="#4c78a8",
            alpha=0.6,
            label="Candidates",
        )
        if pareto_mask.any():
            ax.scatter(
                df.loc[pareto_mask, "stability_score"],
                df.loc[pareto_mask, "binding_score"],
                color="#f58518",
                edgecolor="black",
                linewidth=0.6,
                label="Pareto front",
            )
        ax.set_title(f"Pareto front (depth {depth})")
        ax.set_xlabel("Stability score")
        ax.set_ylabel("Binding score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"pareto_front_depth_{depth}.png"))
        plt.close(fig)


def _format_mutation(chain: str, pos: int, wt: str, mut: str) -> str:
    return f"{chain}:{wt}{pos}{mut}"


def _sequence_mutations(
    sequence: str,
    chain_lengths: Dict[str, int],
    allowed_by_pos: Dict[int, List[str]],
    allowed_from: Optional[Iterable[str]] = None,
    allowed_to: Optional[Iterable[str]] = None,
    disallow_positions: Optional[Iterable[int]] = None,
) -> List[Tuple[str, Tuple[str, ...], int]]:
    allowed_from_set = set(allowed_from or AMINO_ACIDS)
    allowed_to_set = set(allowed_to or AMINO_ACIDS)
    disallow_positions_set = set(disallow_positions or [])
    chain_order = list(chain_lengths.keys())
    chain_offsets = {}
    offset = 0
    for chain in chain_order:
        chain_offsets[chain] = offset
        offset += chain_lengths[chain]

    mutants = []
    for chain in chain_order:
        start = chain_offsets[chain]
        for local_pos in range(1, chain_lengths[chain] + 1):
            global_pos = start + local_pos - 1
            if global_pos not in allowed_by_pos:
                continue
            if global_pos in disallow_positions_set:
                continue
            wt = sequence[global_pos]
            if wt not in allowed_from_set:
                continue
            allowed_targets = [aa for aa in allowed_by_pos[global_pos] if aa in allowed_to_set]
            for mut in allowed_targets:
                if mut == wt:
                    continue
                new_seq = sequence[:global_pos] + mut + sequence[global_pos + 1 :]
                mutation = _format_mutation(chain, local_pos, wt, mut)
                mutants.append((new_seq, (mutation,), global_pos))
    return mutants


def _score_seqs_batched(
    model: PottsMPNN,
    cfg: OmegaConf,
    pdb_data: Sequence[dict],
    sequences: Sequence[str],
    *,
    partition: Optional[Sequence[str]] = None,
    track_progress: bool = False,
    max_batch_size: int = MAX_SEQS_PER_BATCH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences:
        empty = torch.empty((1, 0), device=cfg.dev)
        return empty, empty, empty

    scores_list = []
    seqs_list = []
    refs_list = []
    for idx in range(0, len(sequences), max_batch_size):
        batch = sequences[idx : idx + max_batch_size]
        batch_scores, batch_seqs, batch_refs = score_seqs(
            model,
            cfg,
            pdb_data,
            [0.0] * len(batch),
            list(batch),
            partition=partition,
            track_progress=track_progress,
        )
        scores_list.append(batch_scores)
        seqs_list.append(batch_seqs)
        refs_list.append(batch_refs)

    scores = torch.cat(scores_list, dim=1)
    scored_seqs = torch.cat(seqs_list, dim=1)
    reference_scores = torch.cat(refs_list, dim=1)
    return scores, scored_seqs, reference_scores


def _score_sequences(
    model: PottsMPNN,
    cfg: OmegaConf,
    pdb_data_list: Sequence[dict],
    sequences: Sequence[str],
    binding_partitions_list: Sequence[List[List[str]]],
    energy_mode: str,
    binding_energy_cutoff: Optional[float] = None,
    rrf_k: int = 60,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    cfg.inference.ddG = True
    cfg.inference.mean_norm = False
    cfg.inference.filter = False
    if len(pdb_data_list) != len(binding_partitions_list):
        raise ValueError("pdb_data_list and binding_partitions_list must be the same length.")

    all_scores = []
    all_stability = []
    all_binding = []
    for pdb_data, binding_partitions in zip(pdb_data_list, binding_partitions_list):
        pdb_entry = pdb_data[0]
        chain_order = pdb_entry["chain_order"]
        chain_lengths = _chain_lengths(pdb_entry)
        wt_sequence = pdb_entry["seq"]
        scores, _, _ = _score_seqs_batched(
            model, cfg, pdb_data, sequences, track_progress=True
        )
        scores = scores.squeeze(0)

        stability_scores = scores.cpu().numpy()
        if energy_mode == "stability":
            all_scores.append(stability_scores)
            all_stability.append(stability_scores)
            continue

        if not binding_partitions:
            raise ValueError("Binding energy scoring requires binding_energy_json partitions.")

        interface_mask = None
        if binding_energy_cutoff is not None:
            interface_mask = _interface_mask(
                pdb_entry, chain_lengths, binding_partitions, binding_energy_cutoff
            )
            binding_sequences = [
                _mask_sequence_to_interface(seq, wt_sequence, interface_mask)
                for seq in sequences
            ]
        else:
            binding_sequences = list(sequences)

        bound_scores = torch.zeros_like(scores)
        bound_indices = [idx for idx, seq in enumerate(binding_sequences) if seq != wt_sequence]
        if bound_indices:
            bound_subset = [binding_sequences[idx] for idx in bound_indices]
            bound_subset_scores, _, _ = _score_seqs_batched(
                model,
                cfg,
                pdb_data,
                bound_subset,
                track_progress=True,
            )
            bound_scores[bound_indices] = bound_subset_scores.squeeze(0)

        unbound_scores = torch.zeros_like(scores)
        for partition in binding_partitions:
            wt_partition_seq = _partition_sequence(
                wt_sequence, chain_order, chain_lengths, partition
            )
            partition_sequences = _partition_sequences(
                binding_sequences, chain_order, chain_lengths, partition
            )
            partition_indices = [
                idx
                for idx, seq in enumerate(partition_sequences)
                if seq != wt_partition_seq
            ]
            if not partition_indices:
                continue
            partition_subset = [partition_sequences[idx] for idx in partition_indices]
            partition_scores, _, _ = _score_seqs_batched(
                model,
                cfg,
                pdb_data,
                partition_subset,
                partition=partition,
                track_progress=True,
            )
            unbound_scores[partition_indices] = (
                unbound_scores[partition_indices] + partition_scores.squeeze(0)
            )

        binding_scores = bound_scores - unbound_scores
        binding_scores_np = binding_scores.cpu().numpy()
        if energy_mode == "binding":
            all_scores.append(binding_scores_np)
            all_binding.append(binding_scores_np)
        elif energy_mode == "both":
            stability_ranks = _rank_scores(stability_scores)
            binding_ranks = _rank_scores(binding_scores_np)
            rrf_scores = _rrf_scores(stability_ranks, binding_ranks, rrf_k)
            all_scores.append(-rrf_scores)
            all_stability.append(stability_scores)
            all_binding.append(binding_scores_np)
        else:
            raise ValueError("energy_mode must be one of: 'stability', 'binding', 'both'.")

    return (
        np.mean(np.stack(all_scores, axis=0), axis=0),
        np.mean(np.stack(all_stability, axis=0), axis=0) if all_stability else None,
        np.mean(np.stack(all_binding, axis=0), axis=0) if all_binding else None,
    )


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _rrf_scores(stability_ranks: np.ndarray, binding_ranks: np.ndarray, rrf_k: int) -> np.ndarray:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")
    return (1.0 / (rrf_k + stability_ranks)) + (1.0 / (rrf_k + binding_ranks))


def _pareto_front(stability_scores: np.ndarray, binding_scores: np.ndarray) -> np.ndarray:
    n = len(stability_scores)
    front = np.ones(n, dtype=bool)
    for i in range(n):
        if not front[i]:
            continue
        for j in range(n):
            if i == j or not front[i]:
                continue
            if (
                stability_scores[j] <= stability_scores[i]
                and binding_scores[j] <= binding_scores[i]
                and (
                    stability_scores[j] < stability_scores[i]
                    or binding_scores[j] < binding_scores[i]
                )
            ):
                front[i] = False
                break
    return front


def _normalize_amino_acids(amino_acids: Optional[Iterable[str]]) -> Optional[List[str]]:
    if amino_acids is None:
        return None
    normalized = [aa for aa in amino_acids if aa in AMINO_ACIDS]
    if not normalized:
        raise ValueError("Provided amino acid filter did not match any canonical residues.")
    return normalized


def recursive_mutation_search(
    pdb_paths: Union[str, Sequence[str]],
    cfg_path: str,
    max_mutations: int,
    top_percent: float,
    *,
    allowed_mutations: Optional[Dict[str, Dict[int, Optional[Iterable[str]]]]] = None,
    disallowed_chains: Optional[Iterable[str]] = None,
    binding_energy_json: Optional[str] = None,
    binding_energy_cutoff: Optional[float] = None,
    energy_mode: str = "stability",
    rrf_k: int = 60,
    show_pareto_front: bool = False,
    plot_dir: Optional[str] = None,
    top_percent_decay_base: float = 10.0,
    max_keep_per_depth: int = 1_000_000,
    per_position_quota: Optional[int] = None,
    allowed_from_aas: Optional[Iterable[str]] = None,
    allowed_to_aas: Optional[Iterable[str]] = None,
) -> Dict[int, pd.DataFrame]:
    """Search mutations iteratively and return the top percent at each depth.

    Parameters
    ----------
    pdb_paths : str or sequence of str
        Path(s) to input PDB file(s). Multiple files must share the same length.
    cfg_path : str
        Path to PottsMPNN energy prediction config (YAML).
    max_mutations : int
        Maximum number of mutations to explore.
    top_percent : float
        Percentage (0-100) of candidates to keep at each depth.
    top_percent_decay_base : float
        Base for exponential decay applied to the top_percent as depth increases.
        A value of 10.0 keeps depth-1 at top_percent, depth-2 at top_percent/10,
        depth-3 at top_percent/100, etc. Set to 1.0 to disable decay.
    max_keep_per_depth : int
        Hard cap on the number of candidates to keep per depth (after decay).
    per_position_quota : int, optional
        Maximum number of kept candidates that may include any individual position.
    allowed_mutations : dict, optional
        Mapping of chain -> {position (1-indexed): [allowed residues] or None}.
        If None, all positions and canonical residues are allowed.
    disallowed_chains : iterable, optional
        Chains to disallow from mutation entirely (e.g., ["B", "C"]).
    binding_energy_json : str, optional
        Path to JSON describing binding partitions for energy calculation.
    binding_energy_cutoff : float, optional
        Cα distance cutoff (Angstroms) for interface residues used in binding energy.
    energy_mode : str
        "stability", "binding", or "both". "both" is stability + binding.
    rrf_k : int
        Reciprocal rank fusion constant used when energy_mode is "both".
    show_pareto_front : bool
        If True and energy_mode is "both", include a Pareto front indicator column.
    plot_dir : str, optional
        If provided, save mutation distribution plots to this directory.
    allowed_from_aas : iterable, optional
        Amino acids that are allowed to be mutated from (wildtype filter).
    allowed_to_aas : iterable, optional
        Amino acids that are allowed to be mutated to (mutant filter).

    Returns
    -------
    dict
        Mapping of mutation count to a DataFrame with columns:
        sequence, mutations, mutation_order, score.
        Mutations at each depth are enforced to occur at distinct positions.
    """
    if max_mutations < 1:
        raise ValueError("max_mutations must be >= 1.")
    if not (0.0 < top_percent <= 100.0):
        raise ValueError("top_percent must be within (0, 100].")
    if binding_energy_cutoff is not None and binding_energy_cutoff <= 0:
        raise ValueError("binding_energy_cutoff must be a positive distance in Angstroms.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")
    if show_pareto_front and energy_mode != "both":
        raise ValueError("show_pareto_front requires energy_mode='both'.")
    if top_percent_decay_base <= 0:
        raise ValueError("top_percent_decay_base must be a positive number.")
    if max_keep_per_depth < 1:
        raise ValueError("max_keep_per_depth must be >= 1.")
    if per_position_quota is not None and per_position_quota < 1:
        raise ValueError("per_position_quota must be >= 1 when provided.")

    model, cfg = load_model_from_config(cfg_path)
    pdb_path_list = [pdb_paths] if isinstance(pdb_paths, str) else list(pdb_paths)
    if not pdb_path_list:
        raise ValueError("pdb_paths must contain at least one PDB path.")
    pdb_data_list = [parse_PDB(path, skip_gaps=cfg.inference.skip_gaps) for path in pdb_path_list]
    pdb_names = [pdb_data[0]["name"] for pdb_data in pdb_data_list]

    chain_lengths = _chain_lengths(pdb_data_list[0][0])
    total_length = sum(chain_lengths.values())
    chain_order = pdb_data_list[0][0]["chain_order"]
    for pdb_data in pdb_data_list[1:]:
        if pdb_data[0]["chain_order"] != chain_order:
            raise ValueError("All PDBs must have the same chain order.")
        if sum(_chain_lengths(pdb_data[0]).values()) != total_length:
            raise ValueError("All PDBs must have the same total sequence length.")
    allowed_by_pos = _allowed_mutations_by_position(
        chain_lengths,
        allowed_mutations,
        disallowed_chains=disallowed_chains,
    )
    binding_partitions_list = [
        _parse_binding_partitions(binding_energy_json, pdb_name) for pdb_name in pdb_names
    ]
    if binding_energy_cutoff is not None and energy_mode != "stability":
        if not binding_partitions_list or not binding_partitions_list[0]:
            raise ValueError("Binding energy cutoff requires binding_energy_json partitions.")
        interface_mask = _interface_mask(
            pdb_data_list[0][0],
            chain_lengths,
            binding_partitions_list[0],
            binding_energy_cutoff,
        )
        allowed_by_pos = {
            pos: residues for pos, residues in allowed_by_pos.items() if interface_mask[pos]
        }
    normalized_from = _normalize_amino_acids(allowed_from_aas)
    normalized_to = _normalize_amino_acids(allowed_to_aas)

    current = [Candidate(sequence=pdb_data_list[0][0]["seq"], mutations=tuple(), positions=tuple())]
    results: Dict[int, pd.DataFrame] = {}

    for depth in range(1, max_mutations + 1):
        print(f"Scoring mutations at depth {depth}")
        generated: Dict[str, Candidate] = {}
        for candidate in current:
            for new_seq, new_mut, global_pos in _sequence_mutations(
                candidate.sequence,
                chain_lengths,
                allowed_by_pos,
                allowed_from=normalized_from,
                allowed_to=normalized_to,
                disallow_positions=candidate.positions,
            ):
                mutations = candidate.mutations + new_mut
                if new_seq in generated:
                    continue
                generated[new_seq] = Candidate(
                    sequence=new_seq,
                    mutations=mutations,
                    positions=candidate.positions + (global_pos,),
                )

        if not generated:
            results[depth] = pd.DataFrame(
                columns=["sequence", "mutations", "mutation_order", "score"]
            )
            current = []
            continue

        sequences = list(generated.keys())
        print(f"Scoring {len(sequences)} mutations.")
        scores, stability_scores, binding_scores = _score_sequences(
            model,
            cfg,
            pdb_data_list,
            sequences,
            binding_partitions_list,
            energy_mode,
            binding_energy_cutoff=binding_energy_cutoff,
            rrf_k=rrf_k,
        )
        for seq, score in zip(sequences, scores):
            generated[seq].score = float(score)

        ranked = sorted(generated.values(), key=lambda c: c.score)
        effective_top_percent = top_percent / (top_percent_decay_base ** (depth - 1))
        keep_n = max(1, ceil(len(ranked) * (effective_top_percent / 100.0)))
        keep_n = min(keep_n, max_keep_per_depth)
        if per_position_quota is None:
            kept = ranked[:keep_n]
        else:
            kept = []
            position_counts: Dict[int, int] = {}
            for candidate in ranked:
                if len(kept) >= keep_n:
                    break
                if any(
                    position_counts.get(pos, 0) >= per_position_quota
                    for pos in candidate.positions
                ):
                    continue
                for pos in candidate.positions:
                    position_counts[pos] = position_counts.get(pos, 0) + 1
                kept.append(candidate)

        print(f"Kept {len(kept)} sequences.")

        data = {
            "sequence": [c.sequence for c in kept],
            "mutations": [",".join(c.mutations) for c in kept],
            "mutation_order": [list(c.mutations) for c in kept],
            "score": [c.score for c in kept],
        }
        if energy_mode == "both":
            if stability_scores is None or binding_scores is None:
                raise ValueError("Joint optimization requires stability and binding scores.")
            sequence_indices = {seq: idx for idx, seq in enumerate(sequences)}
            kept_indices = [sequence_indices[c.sequence] for c in kept]
            data["stability_score"] = [float(stability_scores[idx]) for idx in kept_indices]
            data["binding_score"] = [float(binding_scores[idx]) for idx in kept_indices]
            if show_pareto_front:
                pareto_flags = _pareto_front(stability_scores, binding_scores)
                data["pareto_front"] = [bool(pareto_flags[idx]) for idx in kept_indices]

        results[depth] = pd.DataFrame(data)
        current = kept

    _plot_mutation_distributions(results, chain_lengths, plot_dir)
    _plot_pareto_fronts(results, plot_dir)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run recursive mutation search with PottsMPNN")
    
    # Required arguments
    parser.add_argument("--pdb_paths", nargs="+", required=True, help="List of PDB file paths")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--plot_dir", type=str, required=True, help="Output directory for plots and CSVs")

    # Search parameters
    parser.add_argument("--max_mutations", type=int, default=4, help="Max mutation depth")
    parser.add_argument("--top_percent", type=float, default=10.0, help="Top percentage to keep")
    parser.add_argument("--top_percent_decay_base", type=float, default=1.0, help="Decay base for top_percent")
    parser.add_argument("--max_keep_per_depth", type=int, default=2000, help="Hard cap on kept candidates")
    parser.add_argument("--per_position_quota", type=int, default=None, help="Max candidates per position")
    
    # Constraints
    parser.add_argument("--disallowed_chains", nargs="+", default=[], help="Chains to exclude from mutation")
    parser.add_argument("--allowed_from_aas", type=str, default=None, help="String of allowed source AAs")
    parser.add_argument("--allowed_to_aas", type=str, default=None, help="String of allowed target AAs")

    # Energy/Scoring parameters
    parser.add_argument("--binding_energy_json", type=str, default=None, help="Path to binding energy JSON")
    parser.add_argument("--binding_energy_cutoff", type=float, default=None, help="Contact cutoff in Angstroms")
    parser.add_argument("--energy_mode", type=str, default="both", choices=["stability", "binding", "both"], help="Scoring mode")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF constant for 'both' mode")
    parser.add_argument("--no_pareto_front", action="store_true", help="Disable Pareto front calculation")

    args = parser.parse_args()

    # Convert amino acid strings to lists if provided
    allowed_from = list(args.allowed_from_aas) if args.allowed_from_aas else None
    allowed_to = list(args.allowed_to_aas) if args.allowed_to_aas else None

    # Run the search
    print(f"Starting search for {len(args.pdb_paths)} PDBs...")
    results = recursive_mutation_search(
        pdb_paths=args.pdb_paths,
        cfg_path=args.cfg_path,
        max_mutations=args.max_mutations,
        top_percent=args.top_percent,
        allowed_mutations=None, # Complex dict not supported via CLI, modify script if needed
        disallowed_chains=args.disallowed_chains,
        binding_energy_json=args.binding_energy_json,
        binding_energy_cutoff=args.binding_energy_cutoff,
        energy_mode=args.energy_mode,
        rrf_k=args.rrf_k,
        show_pareto_front=not args.no_pareto_front,
        plot_dir=args.plot_dir,
        top_percent_decay_base=args.top_percent_decay_base,
        max_keep_per_depth=args.max_keep_per_depth,
        per_position_quota=args.per_position_quota,
        allowed_from_aas=allowed_from,
        allowed_to_aas=allowed_to,
    )

    # Save results to CSV
    os.makedirs(args.plot_dir, exist_ok=True)
    for depth, result_df in results.items():
        csv_path = os.path.join(args.plot_dir, f"top_mutations_depth_{int(depth)}.csv")
        result_df.to_csv(csv_path, index=None)
        print(f"Saved depth {depth} results to {csv_path}")

if __name__ == "__main__":
    main()
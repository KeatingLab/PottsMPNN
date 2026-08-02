"""Backbone RMSD between an AF3 prediction and a reference structure.

Used by the optional post-AF3 gate: a mutant whose predicted structure moves too
far from the structure its search reasoned about is disqualified, because its
ipSAE/PISA numbers then describe a different structure than the one that was
scored.

RMSD is computed after Kabsch superposition -- AF3 output and the reference sit
in arbitrary coordinate frames. Residues are paired **positionally** within each
chain, which is robust to the two files numbering residues differently as long
as the per-chain counts match.

Only numpy is needed; structures are read with the existing torch-free readers.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .seeding import read_backbone_atoms

CA_ATOMS: Tuple[str, ...] = ("CA",)
BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")


def kabsch_transform(mobile: np.ndarray, target: np.ndarray):
    """Optimal rigid transform taking ``mobile`` onto ``target``.

    Returns ``(rotation, mobile_centroid, target_centroid)`` so the transform can
    be applied to a *different* atom set than the one it was fitted on -- which
    is what a localized RMSD requires.
    """
    if mobile.shape != target.shape:
        raise ValueError(f"point sets differ in shape: {mobile.shape} vs {target.shape}")
    if mobile.shape[0] < 3:
        raise ValueError("need at least 3 points to define a superposition")
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    p, q = mobile - mc, target - tc
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, mc, tc


def apply_transform(coords: np.ndarray, rotation, mobile_centroid, target_centroid) -> np.ndarray:
    return (coords - mobile_centroid) @ rotation.T + target_centroid


def superposed_rmsd(
    model_fit: np.ndarray,
    ref_fit: np.ndarray,
    model_eval: np.ndarray,
    ref_eval: np.ndarray,
) -> float:
    """RMSD over ``*_eval`` after superposing on ``*_fit``.

    Separating the two sets lets the binding site define the frame while the
    binder is scored, so the result is not dominated by domain motion far from
    the interface.
    """
    rotation, mc, tc = kabsch_transform(model_fit, ref_fit)
    aligned = apply_transform(model_eval, rotation, mc, tc)
    if aligned.shape != ref_eval.shape:
        raise ValueError(f"eval sets differ in shape: {aligned.shape} vs {ref_eval.shape}")
    return float(np.sqrt(((aligned - ref_eval) ** 2).sum() / len(ref_eval)))


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD between two (N, 3) point sets after optimal rigid superposition.

    Translation and rotation are removed via the Kabsch algorithm, so identical
    structures in different frames return 0.
    """
    if mobile.shape != target.shape:
        raise ValueError(f"point sets differ in shape: {mobile.shape} vs {target.shape}")
    if mobile.shape[0] == 0:
        raise ValueError("cannot compute RMSD over zero points")

    p = mobile - mobile.mean(axis=0)
    q = target - target.mean(axis=0)

    # Optimal rotation aligning p onto q.
    covariance = p.T @ q
    u, _s, vt = np.linalg.svd(covariance)
    # Correct for a reflection so the result is a proper rotation.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    aligned = p @ rotation.T
    diff = aligned - q
    return float(np.sqrt((diff**2).sum() / len(p)))


def _coords_by_residue(atoms, chain_map: Optional[Dict[str, str]]):
    """Nest atom records as ``chain -> resseq -> atom_name -> (x, y, z)``."""
    table: Dict[str, Dict[int, Dict[str, Tuple[float, float, float]]]] = {}
    for atom in atoms:
        chain = chain_map.get(atom["chain"], atom["chain"]) if chain_map else atom["chain"]
        table.setdefault(chain, {}).setdefault(atom["resseq"], {})[atom["atom"]] = (
            atom["x"],
            atom["y"],
            atom["z"],
        )
    return table


def extract_coords(
    atoms,
    chain_order: Sequence[str],
    atom_names: Sequence[str] = CA_ATOMS,
    chain_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Ordered coordinates for the requested atoms, plus per-chain residue counts.

    Residues are visited chain-by-chain in ``chain_order``, then by ascending
    residue number within each chain, so two structures extracted the same way
    line up positionally.
    """
    table = _coords_by_residue(atoms, chain_map)
    coords = []
    counts: Dict[str, int] = {}
    for chain in chain_order:
        residues = table.get(chain, {})
        resseqs = sorted(residues)
        counts[chain] = len(resseqs)
        for resseq in resseqs:
            residue = residues[resseq]
            for name in atom_names:
                if name not in residue:
                    raise ValueError(f"residue {chain}:{resseq} is missing atom {name}")
                coords.append(residue[name])
    return np.asarray(coords, dtype=float), counts


def atom_chains(
    atoms,
    chain_order: Sequence[str],
    atom_names: Sequence[str] = CA_ATOMS,
    chain_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-atom ``(chain, resseq)`` labels in the same order as :func:`extract_coords`."""
    table = _coords_by_residue(atoms, chain_map)
    chains, resseqs = [], []
    for chain in chain_order:
        for resseq in sorted(table.get(chain, {})):
            residue = table[chain][resseq]
            for name in atom_names:
                if name in residue:
                    chains.append(chain)
                    resseqs.append(resseq)
    return np.asarray(chains, dtype=object), np.asarray(resseqs, dtype=int)


def interface_fit_mask(
    ref_coords: np.ndarray,
    chains: np.ndarray,
    binder_chains: Sequence[str],
    cutoff: float,
) -> np.ndarray:
    """Target-side atoms within ``cutoff`` of the binder, in the reference.

    These define the local frame the binder's placement is judged against.
    """
    is_binder = np.isin(chains, list(binder_chains))
    target_xyz = ref_coords[~is_binder]
    binder_xyz = ref_coords[is_binder]
    if len(target_xyz) == 0 or len(binder_xyz) == 0:
        return ~is_binder
    near = np.zeros(len(target_xyz), dtype=bool)
    for i in range(0, len(target_xyz), 4000):
        d = np.linalg.norm(target_xyz[i:i + 4000, None, :] - binder_xyz[None, :, :], axis=-1)
        near[i:i + 4000] = (d <= cutoff).any(axis=1)
    mask = np.zeros(len(ref_coords), dtype=bool)
    mask[np.flatnonzero(~is_binder)[near]] = True
    return mask


def structure_rmsd(
    model_path: str,
    reference_path: str,
    chain_order: Sequence[str],
    atom_names: Sequence[str] = CA_ATOMS,
    model_chain_map: Optional[Dict[str, str]] = None,
    scope: str = "complex",
    binder_chains: Sequence[str] = (),
    interface_cutoff: float = 10.0,
) -> float:
    """Superposed RMSD between a model file and a reference file.

    ``scope`` chooses what is measured:

    * ``complex``   -- superpose and score over everything. Meaningful only for a
      small, single-domain complex: over a large multi-domain target, rigid-body
      domain motion far from the binding site swamps any real signal (a point
      mutant can easily read >10 A).
    * ``interface`` -- superpose on the target atoms within ``interface_cutoff``
      of the binder, then score the binder: CAPRI ligand-RMSD localized to the
      binding site.
    * ``binder``    -- superpose and score over the binder alone: a fold check
      that ignores placement entirely.

    ``model_chain_map`` renames the model's chains onto ``chain_order`` (the
    reference's chain ids); the reference is assumed already normalized.
    """
    model_atoms = read_backbone_atoms(str(model_path))
    reference_atoms = read_backbone_atoms(str(reference_path))

    model_coords, model_counts = extract_coords(
        model_atoms, chain_order, atom_names, model_chain_map
    )
    ref_coords, ref_counts = extract_coords(reference_atoms, chain_order, atom_names)

    if model_counts != ref_counts:
        raise ValueError(
            f"model and reference disagree on residue counts per chain "
            f"({model_counts} vs {ref_counts}); cannot pair residues for RMSD."
        )

    if scope == "complex":
        return kabsch_rmsd(model_coords, ref_coords)

    if not binder_chains:
        raise ValueError(f"scope={scope!r} needs binder_chains to be specified.")
    chains, _ = atom_chains(reference_atoms, chain_order, atom_names)
    is_binder = np.isin(chains, list(binder_chains))
    if not is_binder.any():
        raise ValueError(
            f"no atoms found for binder chain(s) {list(binder_chains)}; "
            f"chains present: {sorted(set(chains))}"
        )

    if scope == "binder":
        fit = eval_mask = is_binder
    elif scope == "interface":
        fit = interface_fit_mask(ref_coords, chains, binder_chains, interface_cutoff)
        if fit.sum() < 3:
            raise ValueError(
                f"only {fit.sum()} target atom(s) within {interface_cutoff} A of the binder; "
                "cannot define a binding-site frame (raise interface_cutoff)."
            )
        eval_mask = is_binder
    else:
        raise ValueError(f"unknown rmsd scope {scope!r}")

    return superposed_rmsd(
        model_coords[fit], ref_coords[fit], model_coords[eval_mask], ref_coords[eval_mask]
    )


def atom_names_for(mode: str) -> Tuple[str, ...]:
    """Atom set named by ``gating.rmsd_gate.atoms``."""
    if mode == "backbone":
        return BACKBONE_ATOMS
    return CA_ATOMS

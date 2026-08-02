"""Locating AF3 outputs written by ``run_mutation_af3_pipeline.py``.

That script derives its AF3 job name from the mutation string in a specific way,
and picks the best of three diffusion samples by ipSAE. Re-seeding the next round
on "the AF3 structure of this mutant" reproduces both steps exactly.

Job naming::

    tokens   = [f"{chain}{wt}{resnum}{mut}" for each mutation]   # "B:W102E" -> "BW102E"
    job_name = safe_token(f"{base_name}__{'_'.join(tokens)}")
    job_dir  = out_dir / job_name.lower()[_YYYYMMDD_HHMMSS]

Model selection: for each ``seed-*_sample-*`` directory the pipeline runs ipSAE
and keeps the structure with the highest minimum asymmetric ipSAE. It leaves the
ipSAE report next to each model, so the same choice can be recovered without
re-running anything.
"""

from __future__ import annotations

import math
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .seeding import parse_mutation_token, parse_mutation_tokens, split_mutations

WT_MUTATION_STRING = "WT"


def read_fasta_base_name(fasta_path: str) -> str:
    """The job-name prefix: the first ``|``-separated field of the FASTA header.

    Mirrors ``run_mutation_af3_pipeline.parse_fasta``.
    """
    with open(fasta_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise ValueError(f"Bad FASTA: {fasta_path}")
    return lines[0][1:].split("|")[0]


def read_fasta_chain_seqs(fasta_path: str) -> Tuple[str, dict]:
    """``(base_name, {chain_id: sequence})`` from the pipeline's FASTA format."""
    with open(fasta_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise ValueError(f"Bad FASTA: {fasta_path}")
    header = lines[0][1:]
    parts = header.split("|")
    chain_ids = parts[1].split(":") if len(parts) > 1 else ["A"]
    seqs = lines[1].split(":")
    if len(seqs) != len(chain_ids):
        raise ValueError(
            f"Chain id count ({len(chain_ids)}) != seq count ({len(seqs)}) in {fasta_path}"
        )
    return parts[0], dict(zip(chain_ids, seqs))


def _safe_token(text: str) -> str:
    """``run_mutation_af3_pipeline.safe_token``."""
    return text.replace(":", "_").replace(",", "_").replace(" ", "")


def mutation_af3_tokens(mutations) -> List[str]:
    """Convert ``"B:W102E,B:I110S"`` into ``["BW102E", "BI110S"]``.

    Matches the token format built in ``apply_mutations``, which drops the
    colon. Order follows the mutation string, as the pipeline does.
    """
    text = "" if mutations is None else str(mutations).strip()
    if text == "" or text.lower() == "nan" or text == WT_MUTATION_STRING:
        return [WT_MUTATION_STRING]
    tokens = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        chain, wt, position, mut = parse_mutation_token(token)
        tokens.append(f"{chain}{wt}{position}{mut}")
    return tokens or [WT_MUTATION_STRING]


def af3_job_name(base_name: str, mutations) -> str:
    """The AF3 job name the pipeline would use for this mutant."""
    return _safe_token(f"{base_name}__{'_'.join(mutation_af3_tokens(mutations))}")


def find_job_dirs(out_dir: str, job_name: str) -> List[str]:
    """All output directories for a job, including AF3's timestamped variants.

    Mirrors ``run_mutation_af3_pipeline.find_job_dirs``.
    """
    job_lower = job_name.lower()
    matches = glob(os.path.join(str(out_dir), job_lower)) + glob(
        os.path.join(str(out_dir), f"{job_lower}_[0-9]*")
    )
    return sorted(matches)


def _min_asym_ipsae(ipsae_txt: Path) -> float:
    """Lowest asymmetric ipSAE in a report, as ``run_mutation_af3_pipeline`` reads it."""
    import pandas as pd

    try:
        frame = pd.read_csv(ipsae_txt, sep=r"\s+")
    except (OSError, ValueError):
        return float("nan")
    if "Type" not in frame.columns or "ipSAE" not in frame.columns:
        return float("nan")
    asym = frame[frame["Type"] == "asym"]
    if asym.empty:
        return float("nan")
    return float(asym["ipSAE"].min())


def find_best_model(
    out_dir: str,
    job_name: str,
    pae_cutoff: int = 10,
    dist_cutoff: int = 15,
) -> Optional[str]:
    """The model the pipeline chose: highest min-asym ipSAE across samples.

    Reads the ipSAE reports the pipeline leaves beside each model
    (``<model>_<pae>_<dist>.txt``), so no recomputation is needed. Falls back to
    the first available model if no report can be read.
    """
    suffix = f"_{pae_cutoff:02d}_{dist_cutoff:02d}.txt"
    best_score, best_model, fallback = -math.inf, None, None

    for job_dir in sorted(find_job_dirs(out_dir, job_name), key=os.path.getmtime, reverse=True):
        for sample_dir in sorted(glob(os.path.join(job_dir, "seed-*_sample-*"))):
            models = glob(os.path.join(sample_dir, "*_model.cif"))
            if not models:
                continue
            model = models[0]
            if fallback is None:
                fallback = model
            report = Path(model.replace(".cif", "") + suffix)
            if not report.exists():
                continue
            score = _min_asym_ipsae(report)
            if not math.isnan(score) and score > best_score:
                best_score, best_model = score, model
        if best_model is not None:
            return best_model
    return best_model or fallback


def candidate_job_names(base_name: str, mutations) -> List[str]:
    """Job names for every plausible ordering of a mutant's tokens.

    The job name joins the tokens in the order they appear, so the same mutant
    yields different names depending on how the string was assembled
    (``B:L5D,B:E6D`` vs ``B:E6D,B:L5D``). The loop writes them in positional
    order, but a hand-made CSV may not, and a mismatch shows up only as a
    confusing "no model found".
    """
    tokens = split_mutations(mutations)
    if len(tokens) < 2:
        return [af3_job_name(base_name, mutations)]

    def positional(token: str):
        chain, _wt, position, _mut = parse_mutation_token(token)
        return (chain, position)

    orderings = [tokens, sorted(tokens, key=positional), sorted(tokens)]
    names: List[str] = []
    for ordering in orderings:
        name = af3_job_name(base_name, ",".join(ordering))
        if name not in names:
            names.append(name)
    return names


def find_model_for_mutant(
    out_dir: str,
    base_name: str,
    mutations,
    pae_cutoff: int = 10,
    dist_cutoff: int = 15,
) -> str:
    """Locate a mutant's best AF3 model, or raise with a diagnostic."""
    names = candidate_job_names(base_name, mutations)
    for job_name in names:
        model = find_best_model(out_dir, job_name, pae_cutoff, dist_cutoff)
        if model is not None:
            return model

    existing = [d for n in names for d in find_job_dirs(out_dir, n)]
    tried = ", ".join(repr(n) for n in names)
    raise FileNotFoundError(
        f"No AF3 model found for mutant {mutations!r} under {out_dir}. Tried job name(s): {tried}. "
        + (
            f"Job directories exist ({existing}) but contain no seed-*_sample-*/*_model.cif."
            if existing
            else "No matching job directory exists; AF3 may have failed for this mutant."
        )
    )


def failure_marker(out_dir: str, base_name: str, mutations) -> Path:
    """Path of the ``.failed`` marker the pipeline writes on AF3 failure."""
    return Path(out_dir) / f"{af3_job_name(base_name, mutations)}.failed"

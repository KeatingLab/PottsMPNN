"""The AF3 + PISA + ipSAE stage.

Wraps ``run_mutation_af3_pipeline.py``, whose contract is:

* input  -- a CSV with a ``mutations`` column holding ``"B:W102E,B:I110S"``
  strings, validated against the base FASTA. Mutations must therefore be
  expressed against the *original* wildtype, which is exactly what
  :func:`optimize.seeding.diff_to_wt` produces.
* output -- ``<out_dir>/<input_csv_stem>_with_af3.csv``: the input CSV plus
  ``ipSAE``, ``int_area``, ``dG_binding`` and ``dG_diss``, with a wildtype row
  (``mutations == "WT"``) prepended.

All rounds share **one** AF3 output directory. The pipeline skips any job whose
outputs are already complete (line 471), so a mutant seen in an earlier round --
including the wildtype -- is never re-folded, and every model lives under a
single root for re-seeding.
"""

from __future__ import annotations

import glob as globlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .executors import Job

# The pipeline joins on the mutation string; it prepends a row labelled "WT".
KEY_COLUMN = "mutations"
WT_KEY = "WT"


def shared_structure_dir(out_dir: Path) -> Path:
    """One AF3 output root for the whole run, so nothing is folded twice."""
    return Path(out_dir) / "structure"


def write_folding_csv(df: pd.DataFrame, path: Path) -> str:
    """Write the folding set in the shape the AF3 pipeline consumes.

    This is the ``ranked_mutations_*.csv`` artifact: the file the AF3 stage has
    always read but which nothing in the repository ever produced -- it came
    from an unsaved notebook cell.

    ``to_csv`` quotes any field containing a comma, which the pipeline checks
    for explicitly (line 579-588): an unquoted multi-mutation string would be
    split across columns and rejected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if KEY_COLUMN not in out.columns:
        raise ValueError(
            f"Folding set needs a {KEY_COLUMN!r} column; got {sorted(out.columns)}"
        )
    # The pipeline rejects the file if stability_score is present but non-numeric.
    if "stability_score" in out.columns:
        out["stability_score"] = pd.to_numeric(out["stability_score"], errors="coerce")
    out.to_csv(path, index=False)
    return str(path)


def build_pipeline_job(
    mutations_csv: str,
    out_dir: Path,
    cfg,
    job_name: str,
) -> Job:
    """Build the ``run_mutation_af3_pipeline.py`` invocation.

    Flag set matches ``submit_mutation_af3_pipeline_array.sh`` and
    ``submit_mutation_af3_pipeline_engaging.sh``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    structure = cfg.structure

    argv: List[str] = [
        structure.python_executable,
        str(structure.pipeline_script),
        "--mutations_csv", str(mutations_csv),
        "--fasta", str(cfg.target.fasta),
        "--out_dir", str(out_dir),
    ]
    optional: Sequence[Tuple[str, object]] = [
        ("--msa_json", cfg.target.msa_json),
        ("--pisa_exe", structure.pisa_exe),
        ("--pisa_cfg", structure.pisa_cfg),
        ("--pisa_name", cfg.run.out_name),
        ("--ipsae_script", structure.ipsae_script),
        ("--pae_cutoff", structure.pae_cutoff),
        ("--dist_cutoff", structure.dist_cutoff),
        ("--max_parallel", structure.max_parallel),
    ]
    for flag, value in optional:
        if value is not None:
            argv.extend([flag, str(value)])
    argv.extend(str(a) for a in structure.extra_args)

    return Job(name=job_name, argv=argv, log_dir=out_dir / "logs")


def results_path(out_dir: Path, input_csv: str, cfg) -> str:
    """Where the pipeline will write this input CSV's results."""
    stem = Path(input_csv).stem
    return str(Path(out_dir) / cfg.structure.adapter.results_glob.format(stem=stem))


def parse_results(out_dir: Path, cfg, input_csv: Optional[str] = None) -> pd.DataFrame:
    """Read the pipeline's output into ``[mutations, <metric>..., <extras>]``.

    Restricted to the CSV produced from ``input_csv`` when given, so one round
    never picks up another's results out of the shared output directory.
    """
    adapter = cfg.structure.adapter
    pattern = (
        results_path(out_dir, input_csv, cfg)
        if input_csv
        else str(Path(out_dir) / adapter.results_glob.format(stem="*"))
    )
    matches = sorted(globlib.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No pipeline results matched {pattern}. The AF3 pipeline may have exited "
            "before writing its CSV; check the stage logs."
        )

    frames = [pd.read_csv(m) for m in matches]
    raw = pd.concat(frames, ignore_index=True)

    if adapter.mutation_key_column not in raw.columns:
        raise ValueError(
            f"Pipeline results lack the join column {adapter.mutation_key_column!r}. "
            f"Available: {sorted(raw.columns)}"
        )

    rename = {adapter.mutation_key_column: KEY_COLUMN}
    missing = []
    for metric, column in adapter.metric_columns.items():
        if column in raw.columns:
            rename[column] = metric
        else:
            missing.append(f"{metric} <- {column}")
    if missing:
        raise ValueError(
            f"Pipeline results are missing metric column(s): {', '.join(missing)}. "
            f"Available columns: {sorted(raw.columns)}"
        )

    keep = list(rename) + [c for c in adapter.extra_columns if c in raw.columns]
    out = raw[keep].rename(columns=rename)
    out[KEY_COLUMN] = out[KEY_COLUMN].astype(str)
    return out.drop_duplicates(subset=KEY_COLUMN, keep="last").reset_index(drop=True)


def extract_wildtype_baseline(results: pd.DataFrame, cfg) -> Optional[Dict[str, float]]:
    """Pull the wildtype metrics out of a results frame.

    The pipeline prepends a ``mutations == "WT"`` row to every run (line 596),
    so the reference is produced alongside the mutants rather than needing its
    own job.
    """
    wt_rows = results[results[KEY_COLUMN] == WT_KEY]
    if wt_rows.empty:
        return None
    row = wt_rows.iloc[0]
    metrics = {}
    for metric in cfg.gating.metrics:
        if metric in row and pd.notna(row[metric]):
            metrics[metric] = float(row[metric])
    return metrics or None


def _raise_no_baseline(
    results: pd.DataFrame, cfg, structure_dir: Path, round_index: int, job
) -> None:
    """Explain *why* the wildtype reference is unusable, and how to recover.

    Distinguishes an absent ``WT`` row from one whose metrics are all NaN -- the
    latter means AF3 ran but failed for the wildtype, which is not obvious from
    the row simply being there.
    """
    from .af3_layout import failure_marker, read_fasta_base_name

    metrics = list(cfg.gating.metrics.keys())
    wt_rows = results[results[KEY_COLUMN] == WT_KEY]
    total = len(results)
    unscored = (
        int(results[metrics].isna().any(axis=1).sum())
        if total and all(m in results.columns for m in metrics)
        else total
    )

    if wt_rows.empty:
        cause = (
            "the pipeline's results contain no 'WT' row at all, which usually means it "
            "exited before writing its CSV"
        )
    else:
        cause = (
            "the 'WT' row is present but every metric is NaN, i.e. AlphaFold3 or PISA "
            "failed for the wildtype itself"
        )

    base_name = read_fasta_base_name(str(cfg.target.fasta))
    marker = failure_marker(str(structure_dir), base_name, WT_KEY)
    hints = [
        f"Round {round_index}: no usable wildtype baseline -- {cause}.",
        f"  {unscored} of {total} folded candidate(s) have no metrics.",
        f"  AF3/PISA log: {job.stderr_path}",
    ]
    if marker.exists():
        hints.append(
            f"  A failure marker exists for the wildtype ({marker.name}), so a plain rerun "
            "will SKIP it and fail identically."
        )
        hints.append(
            "  Set structure.retry_failed=true (or delete "
            f"{structure_dir}/*.failed) before retrying."
        )
    if unscored and total and unscored >= total * 0.5:
        hints.append(
            "  Most candidates failed too. A common cause is GPU out-of-memory from too "
            "many concurrent AF3 workers: look for 'RESOURCE_EXHAUSTED' in the log and "
            "lower structure.max_parallel (a ~2000-token complex typically allows only 1-2)."
        )
    raise RuntimeError("\n".join(hints))


def _split_cached(df: pd.DataFrame, cfg, state) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Partition the folding set into rows needing AF3 and rows already scored."""
    if not cfg.structure.cache_by_sequence:
        return df, {}
    cached: Dict[str, Dict[str, float]] = {}
    pending_rows = []
    for _, row in df.iterrows():
        hit = state.cache_get(row["sequence"]) if "sequence" in row else None
        if hit is None:
            pending_rows.append(row)
        else:
            cached[str(row[KEY_COLUMN])] = hit
    pending = pd.DataFrame(pending_rows).reset_index(drop=True) if pending_rows else df.head(0)
    return pending, cached


def handle_failed_markers(
    mutation_keys: Sequence[str], cfg, structure_dir: Path, round_index: int
) -> int:
    """Warn about -- or clear -- ``.failed`` markers for this round's candidates.

    ``run_mutation_af3_pipeline.py`` writes ``<job>.failed`` when AF3 dies and
    then skips that mutant **permanently** on every future run (line 467). That
    is right for a genuinely impossible input, but wrong after a transient or
    misconfigured failure such as GPU OOM -- and if the wildtype is among them,
    a rerun can never recover a baseline. Neither our sequence cache nor a resume
    clears these, so surface them explicitly.
    """
    from .af3_layout import failure_marker, read_fasta_base_name

    base_name = read_fasta_base_name(str(cfg.target.fasta))
    # The wildtype is folded alongside the mutants, so check it too.
    keys = list(dict.fromkeys([*mutation_keys, WT_KEY]))
    markers = [
        marker
        for key in keys
        if (marker := failure_marker(str(structure_dir), base_name, key)).exists()
    ]
    if not markers:
        return 0

    if cfg.structure.retry_failed:
        for marker in markers:
            marker.unlink()
        print(
            f"[round {round_index}] cleared {len(markers)} .failed marker(s) "
            "(structure.retry_failed=true); those mutants will be re-folded"
        )
        return len(markers)

    wt_blocked = failure_marker(str(structure_dir), base_name, WT_KEY).exists()
    print(
        f"[round {round_index}] WARNING: {len(markers)} of {len(keys)} candidate(s) have a "
        f".failed marker from an earlier run and will be SKIPPED permanently"
        + (" -- including the WILDTYPE, so no baseline can be produced" if wt_blocked else "")
    )
    print(
        f"[round {round_index}]   to retry them, set structure.retry_failed=true "
        f"or delete {structure_dir}/*.failed"
    )
    return len(markers)


def compute_rmsd_column(
    df: pd.DataFrame,
    cfg,
    out_dir: Path,
    seed_backbone_map: Dict[str, str],
    chain_order: Sequence[str],
) -> pd.DataFrame:
    """Add an ``rmsd`` column: predicted structure vs. the mutant's reference.

    RMSD-to-seed is not a function of the sequence alone (it depends on which
    seed the mutant came from), so it is computed fresh here every round rather
    than cached like the sequence-intrinsic ipSAE/PISA metrics. Recomputing is
    cheap: the AF3 model already exists on disk, so this only re-reads
    coordinates. Anything that cannot be located or paired yields ``NaN``, which
    the gate treats as a failure.
    """
    from .af3_layout import find_model_for_mutant, read_fasta_base_name
    from .config import resolve_binder_chains
    from .rmsd import atom_names_for, structure_rmsd

    out = df.copy()
    gate = cfg.gating.rmsd_gate
    structure_dir = shared_structure_dir(out_dir)
    base_name = read_fasta_base_name(str(cfg.target.fasta))
    atom_names = atom_names_for(gate.atoms)
    chain_map = dict(cfg.structure.af3_chain_map) or None
    binder_chains = resolve_binder_chains(cfg)

    metrics = list(cfg.gating.metrics.keys())
    values = []
    skipped = 0
    for _, row in out.iterrows():
        mutations = str(row[KEY_COLUMN])
        # A candidate with no structural metrics has no usable model either, and
        # is already excluded downstream. Reporting each one here would bury the
        # real failure under one message per candidate.
        if metrics and any(pd.isna(row.get(m)) for m in metrics):
            values.append(float("nan"))
            skipped += 1
            continue
        try:
            model = find_model_for_mutant(
                str(structure_dir), base_name, mutations,
                pae_cutoff=int(cfg.structure.pae_cutoff),
                dist_cutoff=int(cfg.structure.dist_cutoff),
            )
            if gate.reference == "target":
                reference = str(cfg.target.pdb)
            else:
                reference = seed_backbone_map.get(row.get("seed_id"))
                if not reference:
                    raise FileNotFoundError(
                        f"no seed backbone recorded for seed {row.get('seed_id')!r}"
                    )
            values.append(
                structure_rmsd(
                    model, reference, chain_order, atom_names, chain_map,
                    scope=gate.scope,
                    binder_chains=binder_chains,
                    interface_cutoff=gate.interface_cutoff,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[rmsd] {mutations}: could not compute RMSD ({exc}); treating as gate failure")
            values.append(float("nan"))

    if skipped:
        print(f"[rmsd] skipped {skipped} candidate(s) that have no structural metrics")
    out["rmsd"] = values
    return out


def run_structure_stage(
    df: pd.DataFrame,
    cfg,
    out_dir: Path,
    round_dir: Path,
    executor,
    state,
    round_index: int,
    seed_backbone_map: Optional[Dict[str, str]] = None,
    chain_order: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """Fold and score a folding set.

    Returns ``(scored_frame, wildtype_baseline)``; the baseline is ``None`` when
    the results carried no wildtype row (e.g. everything came from cache).
    """
    if df.empty:
        return df.copy(), None

    work = df.copy().reset_index(drop=True)
    work[KEY_COLUMN] = work[KEY_COLUMN].astype(str)

    structure_dir = shared_structure_dir(out_dir)
    pending, cached = _split_cached(work, cfg, state)
    if cached:
        print(f"[round {round_index}] {len(cached)} candidate(s) served from cache")

    metrics = list(cfg.gating.metrics.keys())
    results = pd.DataFrame(columns=[KEY_COLUMN, *metrics])
    baseline = None

    if not pending.empty:
        handle_failed_markers(
            [str(k) for k in pending[KEY_COLUMN]], cfg, structure_dir, round_index
        )
        # Input CSV lives outside the scanned output dir so a results glob can
        # never read it back as if it were output.
        csv_path = write_folding_csv(
            pending, round_dir / "inputs" / f"round_{round_index}_folding_set.csv"
        )
        job = build_pipeline_job(csv_path, structure_dir, cfg, f"af3_round_{round_index}")
        print(f"[round {round_index}] folding {len(pending)} candidate(s) -> {structure_dir}")
        outcomes = executor.run([job])
        failed = [o for o in outcomes if not o.ok]
        if failed:
            detail = "; ".join(
                f"{o.job.name} rc={o.returncode} {o.error or ''}".strip() for o in failed
            )
            raise RuntimeError(
                f"AF3 pipeline failed in round {round_index}: {detail}. See {job.stderr_path}"
            )
        results = parse_results(structure_dir, cfg, input_csv=csv_path)
        baseline = extract_wildtype_baseline(results, cfg)
        if baseline is None and not state.wt_baseline:
            _raise_no_baseline(results, cfg, structure_dir, round_index, job)

    merged = work.merge(
        results[results[KEY_COLUMN] != WT_KEY], on=KEY_COLUMN, how="left", suffixes=("", "_af3")
    )

    for metric in metrics:
        if metric not in merged.columns:
            merged[metric] = pd.NA
        if cached:
            filler = merged[KEY_COLUMN].map(lambda k: cached.get(k, {}).get(metric))
            merged[metric] = merged[metric].fillna(filler)

    if not pending.empty and cfg.structure.cache_by_sequence and "sequence" in merged.columns:
        fresh = {}
        for _, row in merged.iterrows():
            values = {m: row[m] for m in metrics}
            if all(pd.notna(v) for v in values.values()):
                fresh[row["sequence"]] = {m: float(v) for m, v in values.items()}
        if fresh:
            state.cache_put_many(fresh)

    unscored = int(merged[metrics].isna().any(axis=1).sum())
    if unscored:
        print(
            f"[round {round_index}] WARNING: {unscored} candidate(s) have no structural "
            "metrics (AF3 or PISA likely failed) and are ignored downstream"
        )

    if cfg.gating.rmsd_gate.enabled:
        if chain_order is None:
            raise ValueError("chain_order is required to compute the RMSD gate.")
        merged = compute_rmsd_column(
            merged, cfg, out_dir, seed_backbone_map or {}, chain_order
        )
        gate = cfg.gating.rmsd_gate
        rmsd = pd.to_numeric(merged["rmsd"], errors="coerce")
        n_over = int((rmsd > float(gate.max_rmsd)).sum())
        n_nan = int(rmsd.isna().sum())
        print(
            f"[round {round_index}] RMSD gate ({gate.scope} scope, {gate.atoms} vs "
            f"{gate.reference}, max {gate.max_rmsd} A): {n_over} over threshold, "
            f"{n_nan} uncomputable"
        )
        if n_over and n_over == int(rmsd.notna().sum()) and gate.scope == "complex":
            print(
                f"[round {round_index}]   NOTE: every candidate failed under 'complex' scope. "
                "Over a large multi-domain target this usually reflects domain motion far from "
                "the binding site, not the mutations. Try gating.rmsd_gate.scope=interface."
            )

    return merged, baseline


def resolve_seed_model(cfg, out_dir: Path, mutations) -> str:
    """Locate the AF3 model for a promoted mutant, for use as the next backbone."""
    from .af3_layout import find_model_for_mutant, read_fasta_base_name

    base_name = read_fasta_base_name(str(cfg.target.fasta))
    return find_model_for_mutant(
        str(shared_structure_dir(out_dir)),
        base_name,
        mutations,
        pae_cutoff=int(cfg.structure.pae_cutoff),
        dist_cutoff=int(cfg.structure.dist_cutoff),
    )

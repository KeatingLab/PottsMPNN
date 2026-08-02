"""Pruning AF3 byproducts between rounds.

AF3 writes a PAE/confidence matrix per diffusion sample, which is **O(tokens^2)**:
for a 1867-token complex that is tens of MB per sample, three samples per mutant,
fifty mutants per round. Over five rounds it dwarfs everything else on disk.

The loop reads back only two things after a fold:

* ``*_model.cif``            -- the RMSD gate, and re-seeding the next round
* ``*_model_<pae>_<dist>.txt`` -- the ipSAE report, to pick each mutant's best sample

Everything else is consumed once, at fold time, by ipSAE and PISA, so the big
files can be compressed or removed without touching what the loop depends on.

Two rules make this safe:

1. **Structures and ipSAE reports are never touched.** A pruned mutant can still
   be RMSD-checked and still be promoted.
2. **``*_summary_confidences.json`` is pruned along with the PAE.** The pipeline
   uses it as its "outputs are complete" marker; leaving it while removing the
   PAE would make the pipeline skip inference and then fail to find the PAE,
   silently producing NaN metrics. Removing both keeps the job marked incomplete
   so a forced rerun regenerates it. Normal resumes never reach that path
   because the loop's own sequence cache serves the metrics.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

# File classes that may be pruned, mapped to their glob patterns.
TARGET_PATTERNS: Dict[str, Sequence[str]] = {
    # The PAE / contact-probability matrices: the O(N^2) bulk.
    "pae": ("seed-*_sample-*/*confidences.json",),
    # PISA's XML dumps, already parsed into the results CSV.
    "pisa": ("seed-*_sample-*/*_pisa.xml", "seed-*_sample-*/*_pisa_assemblies.xml"),
    # The pipeline's completeness marker; see the module docstring.
    "summary": ("seed-*_sample-*/*_summary_confidences.json",),
}

# Never pruned, whatever the configuration says.
PROTECTED_SUFFIXES = (".cif", ".txt")


@dataclass
class CleanupStats:
    mode: str = "none"
    files: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    jobs_pruned: int = 0
    jobs_protected: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def bytes_saved(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)

    def describe(self) -> str:
        if self.mode == "none" or self.files == 0:
            return "nothing pruned"
        verb = "compressed" if self.mode == "compress" else "deleted"
        saved = self.bytes_saved / 1e6
        return (
            f"{verb} {self.files} file(s) across {self.jobs_pruned} job(s), "
            f"reclaiming {saved:.1f} MB"
            + (f" ({self.jobs_protected} job(s) left intact)" if self.jobs_protected else "")
        )


def _matches(job_dir: Path, targets: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for target in targets:
        for pattern in TARGET_PATTERNS.get(target, ()):
            found.extend(job_dir.glob(pattern))
    # "*confidences.json" also matches the summary file; only include it when
    # "summary" was explicitly requested.
    if "summary" not in set(targets):
        found = [p for p in found if not p.name.endswith("_summary_confidences.json")]
    return [p for p in dict.fromkeys(found) if p.suffix not in PROTECTED_SUFFIXES]


def compress_file(path: Path) -> int:
    """Gzip in place; returns the resulting size. Idempotent."""
    target = path.with_suffix(path.suffix + ".gz")
    with open(path, "rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()
    return target.stat().st_size


def prune_jobs(
    structure_dir: Path,
    mode: str = "compress",
    targets: Sequence[str] = ("pae", "pisa", "summary"),
    protected_jobs: Optional[Set[str]] = None,
    dry_run: bool = False,
) -> CleanupStats:
    """Prune AF3 byproducts under ``structure_dir``.

    ``protected_jobs`` are lowercase AF3 job-directory names left untouched,
    typically the round's winners.
    """
    stats = CleanupStats(mode=mode)
    if mode == "none":
        return stats
    if mode not in {"compress", "delete"}:
        raise ValueError(f"unknown cleanup mode {mode!r}")

    structure_dir = Path(structure_dir)
    if not structure_dir.is_dir():
        return stats
    protected = {j.lower() for j in (protected_jobs or set())}

    for job_dir in sorted(p for p in structure_dir.iterdir() if p.is_dir()):
        if job_dir.name == "logs":
            continue
        # Timestamped variants share the base job name.
        base = job_dir.name.lower()
        if any(base == p or base.startswith(p + "_") for p in protected):
            stats.jobs_protected += 1
            continue

        victims = _matches(job_dir, targets)
        if not victims:
            continue
        touched = False
        for path in victims:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if dry_run:
                stats.files += 1
                stats.bytes_before += size
                stats.bytes_after += 0 if mode == "delete" else size // 4  # rough estimate
                touched = True
                continue
            try:
                if mode == "delete":
                    path.unlink()
                    after = 0
                else:
                    after = compress_file(path)
            except OSError as exc:
                stats.errors.append(f"{path}: {exc}")
                continue
            stats.files += 1
            stats.bytes_before += size
            stats.bytes_after += after
            touched = True
        if touched:
            stats.jobs_pruned += 1

    return stats


def job_names_to_protect(mutation_keys: Iterable[str], base_name: str) -> Set[str]:
    """AF3 job-directory names (lowercased) for the given mutants."""
    from .af3_layout import af3_job_name

    return {af3_job_name(base_name, key).lower() for key in mutation_keys}


def run_cleanup_for_round(cfg, out_dir: Path, protected_keys: Iterable[str], round_index: int) -> CleanupStats:
    """Prune after a round, leaving the round's winners intact."""
    from .af3_layout import WT_MUTATION_STRING, read_fasta_base_name
    from .structure_stage import shared_structure_dir

    settings = cfg.structure.cleanup
    if settings.mode == "none":
        return CleanupStats(mode="none")

    base_name = read_fasta_base_name(str(cfg.target.fasta))
    protected: Set[str] = set()
    if settings.keep_winners:
        protected = job_names_to_protect(protected_keys, base_name)
    # The wildtype is the run's fixed baseline; always keep it.
    protected |= job_names_to_protect([WT_MUTATION_STRING], base_name)

    stats = prune_jobs(
        shared_structure_dir(out_dir),
        mode=settings.mode,
        targets=list(settings.targets),
        protected_jobs=protected,
    )
    print(f"[round {round_index}] cleanup: {stats.describe()}")
    for err in stats.errors[:3]:
        print(f"[round {round_index}]   WARNING: {err}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prune AF3 byproducts (PAE matrices, PISA XML) from a run's "
                    "structure directory. Structures and ipSAE reports are never touched.")
    ap.add_argument("structure_dir", type=Path,
                    help="The run's <out_dir>/structure directory.")
    ap.add_argument("--mode", choices=["compress", "delete"], default="compress")
    ap.add_argument("--targets", nargs="+", default=["pae", "pisa", "summary"],
                    choices=sorted(TARGET_PATTERNS))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be pruned without changing anything.")
    args = ap.parse_args()

    stats = prune_jobs(args.structure_dir, mode=args.mode, targets=args.targets,
                       dry_run=args.dry_run)
    prefix = "[dry run] would have " if args.dry_run else ""
    print(f"{prefix}{stats.describe()}")
    for err in stats.errors:
        print(f"  WARNING: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

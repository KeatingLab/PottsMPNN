"""Durable run state: resume markers, seed lineage, and the AF3 result cache.

An AF3 round is many GPU-hours, so every stage records a completion marker here
and a preempted run resumes at the last completed stage rather than from zero.

The state file is written atomically (temp file + replace) so a job killed
mid-write cannot leave a truncated ledger behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_FILENAME = "run_state.json"
STATE_VERSION = 1


def sequence_hash(sequence: str) -> str:
    """Stable short hash identifying a mutant sequence across rounds."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]


@dataclass
class SeedRecord:
    """One starting point for a round's mutation search."""

    seed_id: str
    sequence: str
    # Mutations relative to the ORIGINAL wildtype, not to the parent seed.
    mutations: List[str] = field(default_factory=list)
    backbone_pdb: Optional[str] = None
    parent_seed_id: Optional[str] = None
    round_index: int = 0
    # Structural metrics that earned this seed its promotion (empty for round 0).
    metrics: Dict[str, float] = field(default_factory=dict)
    # Where this seed's AF3 prediction lives, used to build its next backbone
    # when run.backbone_source is "af3".
    af3_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedRecord":
        return cls(
            seed_id=data["seed_id"],
            sequence=data["sequence"],
            mutations=list(data.get("mutations", [])),
            backbone_pdb=data.get("backbone_pdb"),
            parent_seed_id=data.get("parent_seed_id"),
            round_index=int(data.get("round_index", 0)),
            metrics=dict(data.get("metrics", {})),
            af3_dir=data.get("af3_dir"),
        )


@dataclass
class RoundRecord:
    """Per-round bookkeeping: inputs, stage completion, and outcome counts."""

    index: int
    seeds: List[SeedRecord] = field(default_factory=list)
    stages: Dict[str, bool] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    termination: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoundRecord":
        return cls(
            index=int(data["index"]),
            seeds=[SeedRecord.from_dict(s) for s in data.get("seeds", [])],
            stages=dict(data.get("stages", {})),
            counts=dict(data.get("counts", {})),
            termination=data.get("termination"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "seeds": [asdict(s) for s in self.seeds],
            "stages": self.stages,
            "counts": self.counts,
            "termination": self.termination,
        }


class RunState:
    """The run ledger, persisted to ``<out_dir>/run_state.json``."""

    def __init__(self, path: Path, force: bool = False):
        self.path = Path(path)
        self.force = force
        self.version: int = STATE_VERSION
        self.wt_sequence: Optional[str] = None
        self.wt_baseline: Dict[str, float] = {}
        self.rounds: Dict[int, RoundRecord] = {}
        # sequence hash -> structural metrics, so a mutant recurring across
        # rounds is never re-folded.
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        self.finished: bool = False
        self.termination_reason: Optional[str] = None

    # ---------------------------------------------------------------- load/save

    @classmethod
    def load_or_create(cls, out_dir: str, force: bool = False) -> "RunState":
        path = Path(out_dir) / STATE_FILENAME
        state = cls(path, force=force)
        if path.exists() and not force:
            state._load()
        return state

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        version = int(data.get("version", 0))
        if version != STATE_VERSION:
            raise ValueError(
                f"{self.path} was written by state version {version}, but this code expects "
                f"{STATE_VERSION}. Move it aside or rerun with run.force=true."
            )
        self.version = version
        self.wt_sequence = data.get("wt_sequence")
        self.wt_baseline = dict(data.get("wt_baseline", {}))
        self.result_cache = dict(data.get("result_cache", {}))
        self.finished = bool(data.get("finished", False))
        self.termination_reason = data.get("termination_reason")
        self.rounds = {
            int(r["index"]): RoundRecord.from_dict(r) for r in data.get("rounds", [])
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "wt_sequence": self.wt_sequence,
            "wt_baseline": self.wt_baseline,
            "result_cache": self.result_cache,
            "finished": self.finished,
            "termination_reason": self.termination_reason,
            "rounds": [self.rounds[i].to_dict() for i in sorted(self.rounds)],
        }
        # Atomic replace: a kill between write and rename leaves the old file intact.
        fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp_path, self.path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------- rounds

    def round(self, index: int) -> RoundRecord:
        """Get (creating if absent) the record for a round."""
        if index not in self.rounds:
            self.rounds[index] = RoundRecord(index=index)
        return self.rounds[index]

    def is_stage_complete(self, round_index: int, stage: str) -> bool:
        if self.force:
            return False
        return bool(self.round(round_index).stages.get(stage, False))

    def mark_stage_complete(self, round_index: int, stage: str) -> None:
        self.round(round_index).stages[stage] = True
        self.save()

    def set_counts(self, round_index: int, **counts: int) -> None:
        self.round(round_index).counts.update(counts)
        self.save()

    def set_seeds(self, round_index: int, seeds: List[SeedRecord]) -> None:
        self.round(round_index).seeds = list(seeds)
        self.save()

    def finish(self, reason: str, round_index: Optional[int] = None) -> None:
        self.finished = True
        self.termination_reason = reason
        if round_index is not None:
            self.round(round_index).termination = reason
        self.save()

    # -------------------------------------------------------------------- cache

    def cache_get(self, sequence: str) -> Optional[Dict[str, Any]]:
        if self.force:
            return None
        return self.result_cache.get(sequence_hash(sequence))

    def cache_put(self, sequence: str, metrics: Dict[str, Any]) -> None:
        self.result_cache[sequence_hash(sequence)] = dict(metrics)

    def cache_put_many(self, entries: Dict[str, Dict[str, Any]]) -> None:
        """Bulk insert keyed by sequence (not hash); saves once."""
        for sequence, metrics in entries.items():
            self.result_cache[sequence_hash(sequence)] = dict(metrics)
        self.save()

    # ----------------------------------------------------------------- baseline

    def set_wt(self, sequence: str, baseline: Dict[str, float]) -> None:
        self.wt_sequence = sequence
        self.wt_baseline = dict(baseline)
        self.save()

    def has_wt_baseline(self) -> bool:
        return bool(self.wt_baseline) and not self.force

    # ------------------------------------------------------------------ summary

    def summary(self) -> Dict[str, Any]:
        return {
            "finished": self.finished,
            "termination_reason": self.termination_reason,
            "wt_baseline": self.wt_baseline,
            "rounds": [
                {
                    "index": rec.index,
                    "n_seeds": len(rec.seeds),
                    "counts": rec.counts,
                    "termination": rec.termination,
                }
                for rec in (self.rounds[i] for i in sorted(self.rounds))
            ],
        }

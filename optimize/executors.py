"""Pluggable job execution backends.

``LocalExecutor`` runs jobs as bounded-parallel subprocesses; ``SlurmExecutor``
submits them as a single sbatch array and polls to completion. Both present the
same interface so the round loop is unaware of where work actually runs.

The SLURM backend uses the same idiom as the repository's submit scripts: a
manifest file with one job per line, indexed by ``SLURM_ARRAY_TASK_ID``.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence


def _posix(path) -> str:
    """Render a path with forward slashes.

    Generated sbatch scripts always execute on Linux, so any path embedded in
    one must be POSIX even when the script is rendered from Windows.
    """
    return Path(path).as_posix()


@dataclass
class Job:
    """A single command to run."""

    name: str
    argv: List[str]
    log_dir: Path
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(str(a)) for a in self.argv)

    @property
    def stdout_path(self) -> Path:
        return Path(self.log_dir) / f"{self.name}.out"

    @property
    def stderr_path(self) -> Path:
        return Path(self.log_dir) / f"{self.name}.err"


@dataclass
class JobResult:
    job: Job
    returncode: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class Handle:
    """Opaque reference to submitted work."""

    jobs: List[Job]
    backend: str
    payload: Any = None


class Executor(Protocol):
    def submit(self, jobs: Sequence[Job]) -> Handle: ...

    def wait(self, handle: Handle) -> List[JobResult]: ...


class _BaseExecutor:
    def run(self, jobs: Sequence[Job]) -> List[JobResult]:
        """Submit and wait. The common case for the round loop."""
        if not jobs:
            return []
        return self.wait(self.submit(jobs))


class LocalExecutor(_BaseExecutor):
    """Runs jobs as subprocesses on this machine, ``max_parallel`` at a time."""

    def __init__(self, max_parallel: int = 1, timeout_seconds: Optional[int] = None):
        self.max_parallel = max(1, int(max_parallel or 1))
        self.timeout_seconds = timeout_seconds

    def submit(self, jobs: Sequence[Job]) -> Handle:
        # Local execution is synchronous; the work happens in wait().
        return Handle(jobs=list(jobs), backend="local")

    def wait(self, handle: Handle) -> List[JobResult]:
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            return list(pool.map(self._run_one, handle.jobs))

    def _run_one(self, job: Job) -> JobResult:
        job.log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if job.env:
            env.update(job.env)
        try:
            with open(job.stdout_path, "w", encoding="utf-8") as out, open(
                job.stderr_path, "w", encoding="utf-8"
            ) as err:
                proc = subprocess.run(
                    job.argv,
                    stdout=out,
                    stderr=err,
                    cwd=job.cwd,
                    env=env,
                    timeout=self.timeout_seconds,
                )
            return JobResult(job=job, returncode=proc.returncode)
        except subprocess.TimeoutExpired:
            return JobResult(
                job=job,
                returncode=124,
                error=f"timed out after {self.timeout_seconds}s",
            )
        except OSError as exc:
            return JobResult(job=job, returncode=127, error=str(exc))


class SlurmExecutor(_BaseExecutor):
    """Submits jobs as one sbatch array and polls until the array completes."""

    def __init__(
        self,
        work_dir: str,
        partition: Optional[str] = None,
        gres: Optional[str] = None,
        mem: Optional[str] = None,
        time_limit: Optional[str] = None,
        cpus_per_task: Optional[int] = None,
        conda_env: Optional[str] = None,
        conda_root: Optional[str] = None,
        modules: Optional[Sequence[str]] = None,
        account: Optional[str] = None,
        extra_directives: Optional[Sequence[str]] = None,
        poll_interval_seconds: int = 60,
        max_parallel: Optional[int] = None,
    ):
        self.work_dir = Path(work_dir)
        self.partition = partition
        self.gres = gres
        self.mem = mem
        self.time_limit = time_limit
        self.cpus_per_task = cpus_per_task
        self.conda_env = conda_env
        self.conda_root = conda_root
        self.modules = list(modules or [])
        self.account = account
        self.extra_directives = list(extra_directives or [])
        self.poll_interval_seconds = max(5, int(poll_interval_seconds))
        self.max_parallel = max_parallel

    def submit(self, jobs: Sequence[Job]) -> Handle:
        jobs = list(jobs)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            job.log_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        manifest = self.work_dir / f"manifest_{stamp}.txt"
        script = self.work_dir / f"array_{stamp}.sh"

        # One job per line; the array task indexes into it.
        with open(manifest, "w", encoding="utf-8", newline="\n") as handle:
            for job in jobs:
                handle.write(
                    f"{job.command} > {shlex.quote(_posix(job.stdout_path))} "
                    f"2> {shlex.quote(_posix(job.stderr_path))}\n"
                )

        script.write_text(self._render_script(manifest), encoding="utf-8", newline="\n")

        array_spec = f"0-{len(jobs) - 1}"
        if self.max_parallel:
            array_spec += f"%{self.max_parallel}"
        proc = subprocess.run(
            ["sbatch", "--parsable", f"--array={array_spec}", str(script)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
        job_id = proc.stdout.strip().split(";")[0]
        if not re.fullmatch(r"\d+", job_id):
            raise RuntimeError(f"Could not parse sbatch job id from: {proc.stdout!r}")
        return Handle(jobs=jobs, backend="slurm", payload=job_id)

    def _render_script(self, manifest: Path) -> str:
        lines = ["#!/bin/bash"]
        directives = [
            ("--partition", self.partition),
            ("--gres", self.gres),
            ("--mem", self.mem),
            ("--time", self.time_limit),
            ("--cpus-per-task", self.cpus_per_task),
            ("--account", self.account),
        ]
        for flag, value in directives:
            if value:
                lines.append(f"#SBATCH {flag}={value}")
        lines.append(f"#SBATCH -o {_posix(self.work_dir / 'slurm_%A_%a.out')}")
        lines.append(f"#SBATCH -e {_posix(self.work_dir / 'slurm_%A_%a.err')}")
        for directive in self.extra_directives:
            lines.append(f"#SBATCH {directive}")

        lines.append("")
        lines.append("set -euo pipefail")
        if self.conda_env:
            root = self.conda_root or "$(conda info --base)"
            lines.append(f'source "{root}/etc/profile.d/conda.sh"')
            lines.append(f"conda activate {self.conda_env}")
        for module in self.modules:
            lines.append(f"module load {module}")
        lines.append("")
        lines.append(f'MANIFEST="{_posix(manifest)}"')
        lines.append('CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")')
        lines.append('if [[ -z "$CMD" ]]; then')
        lines.append('    echo "ERROR: empty manifest entry for task $SLURM_ARRAY_TASK_ID" >&2')
        lines.append("    exit 1")
        lines.append("fi")
        lines.append('eval "$CMD"')
        lines.append("")
        return "\n".join(lines)

    def wait(self, handle: Handle) -> List[JobResult]:
        job_id = handle.payload
        while self._array_is_active(job_id):
            time.sleep(self.poll_interval_seconds)
        return self._collect_results(handle, job_id)

    def _array_is_active(self, job_id: str) -> bool:
        proc = subprocess.run(
            ["squeue", "--job", str(job_id), "--noheader", "--format=%i"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Once the job leaves the queue entirely squeue can error; treat
            # that as "no longer active" and let sacct report the outcome.
            return False
        return bool(proc.stdout.strip())

    def _collect_results(self, handle: Handle, job_id: str) -> List[JobResult]:
        codes = self._exit_codes(job_id)
        results = []
        for index, job in enumerate(handle.jobs):
            code = codes.get(index)
            if code is None:
                results.append(
                    JobResult(job=job, returncode=1, error=f"no sacct record for task {index}")
                )
            else:
                results.append(JobResult(job=job, returncode=code))
        return results

    def _exit_codes(self, job_id: str) -> Dict[int, int]:
        proc = subprocess.run(
            ["sacct", "-j", str(job_id), "--noheader", "--parsable2", "--format=JobID,ExitCode,State"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return {}
        codes: Dict[int, int] = {}
        for line in proc.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            raw_id, exit_code, state = parts[0], parts[1], parts[2]
            # Array task ids look like "12345_7"; skip ".batch"/".extern" steps.
            match = re.fullmatch(r"\d+_(\d+)", raw_id)
            if not match:
                continue
            task_index = int(match.group(1))
            code = int(exit_code.split(":")[0]) if ":" in exit_code else 0
            if code == 0 and not state.startswith("COMPLETED"):
                code = 1
            codes[task_index] = code
        return codes


def build_executor(cfg, work_dir: str) -> _BaseExecutor:
    """Construct the executor named by ``cfg.run.executor``."""
    if cfg.run.executor == "local":
        return LocalExecutor(
            max_parallel=cfg.structure.max_parallel or 1,
            timeout_seconds=cfg.structure.timeout_seconds,
        )
    if cfg.run.executor == "slurm":
        return SlurmExecutor(
            work_dir=work_dir,
            partition=cfg.slurm.partition,
            gres=cfg.slurm.gres,
            mem=cfg.slurm.mem,
            time_limit=cfg.slurm.time,
            cpus_per_task=cfg.slurm.cpus_per_task,
            conda_env=cfg.slurm.conda_env,
            conda_root=cfg.slurm.conda_root,
            modules=cfg.slurm.modules,
            account=cfg.slurm.account,
            extra_directives=cfg.slurm.extra_directives,
            poll_interval_seconds=cfg.slurm.poll_interval_seconds,
            max_parallel=cfg.structure.max_parallel,
        )
    raise ValueError(f"Unknown executor: {cfg.run.executor!r}")

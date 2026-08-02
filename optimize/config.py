"""Configuration schema for the iterative optimization loop.

Declared with dataclasses so OmegaConf can type-check the YAML;
:func:`validate` adds the cross-field checks a type system cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

# Objectives that are computed from the search columns rather than read directly.
DERIVED_OBJECTIVES = {"pareto_rank", "pareto_front", "pareto_distance", "rrf"}

VALID_ENERGY_MODES = {"stability", "binding", "both"}
VALID_RANK_BY = {"joint", "binding", "pareto"}
VALID_EXECUTORS = {"local", "slurm"}
VALID_SIMILARITY_METRICS = {"jaccard", "overlap"}
VALID_DIRECTIONS = {"min", "max"}
VALID_STOP_WHEN = {"any", "all"}
VALID_BACKBONE_SOURCES = {"wt", "af3"}
VALID_KEEP_SCOPES = {"global", "per_seed"}
VALID_SELECTION_SCOPES = {"per_seed", "pooled"}


@dataclass
class RunConfig:
    """Top-level run settings."""

    out_dir: str = MISSING
    out_name: str = "optimization"
    max_iterations: int = 3
    executor: str = "local"
    # Structure the next round searches against. "af3" re-seeds on the
    # AF3-predicted complex of each promoted mutant; "wt" keeps the original
    # backbone and only overrides the sequence.
    backbone_source: str = "af3"
    seed: Optional[int] = None
    # Overwrite completed-stage markers and recompute everything.
    force: bool = False
    # Write <out_dir>/report/ (self-contained HTML + ranked CSV) when the run
    # ends, including on an early stop.
    report: bool = True
    # Also refresh the report after every round, so a long run can be inspected
    # while it is still going.
    report_each_round: bool = False


@dataclass
class TargetConfig:
    """The design target: starting structure and its auxiliary inputs."""

    pdb: str = MISSING
    fasta: str = MISSING
    msa_json: Optional[str] = None
    binding_energy_json: Optional[str] = None
    # Partitions for the target, e.g. [["A"], ["B"]]. If null, they are read
    # from ``binding_energy_json`` using the PDB's own name as the key.
    binding_partitions: Optional[List[List[str]]] = None


@dataclass
class SearchConfig:
    """Arguments forwarded to ``mutation_search.recursive_mutation_search``."""

    cfg_path: str = MISSING
    max_mutations: int = 3
    top_percent: float = 10.0
    top_percent_decay_base: float = 1.0
    max_keep_per_depth: int = 1000
    # Whether max_keep_per_depth is a budget for the whole round or for each
    # seed. A per-seed cap multiplies the round's scoring cost by the number of
    # seeds; "global" divides the cap across seeds so the cost stays flat as
    # promote_top_n grows.
    keep_budget_scope: str = "global"  # global | per_seed
    per_position_quota: Optional[int] = 100
    disallowed_chains: List[str] = field(default_factory=list)
    binder_chain: Optional[str] = None
    energy_mode: str = "both"
    rank_by: str = "pareto"
    rrf_k: int = 60
    binding_energy_cutoff: Optional[float] = 8.0
    allowed_from_aas: Optional[str] = None
    allowed_to_aas: Optional[str] = None
    show_pareto_front: bool = True
    # Which depths feed the folding set. Empty means every depth.
    use_depths: List[int] = field(default_factory=list)


@dataclass
class DiversityConfig:
    enabled: bool = True
    weight: float = 10.0
    metric: str = "jaccard"


@dataclass
class SelectionConfig:
    """How search candidates are cut down to the AF3 folding set."""

    # PottsMPNN builds a different energy table per structure, so its energies
    # are NOT comparable between seeds. "per_seed" ranks each seed's candidates
    # against its own structure-mates only and gives each seed a share of
    # max_candidates. "pooled" ranks everything together and is only valid when
    # a round has a single seed.
    scope: str = "per_seed"  # per_seed | pooled
    # A column name, one of DERIVED_OBJECTIVES, or a pandas-eval expression.
    objective: str = "binding_score"
    direction: str = "min"
    # pandas .query() strings, ANDed together. e.g. ["stability_score <= 0"]
    constraints: List[str] = field(default_factory=list)
    max_candidates: int = 50
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    # Column names in the search output.
    stability_column: str = "stability_score"
    binding_column: str = "binding_score"
    mutation_column: str = "mutations"


@dataclass
class ResultsAdapter:
    """How to find and read the AF3/PISA/ipSAE pipeline's output.

    Config-driven so a change in that script's output format is a config edit
    rather than a code change.
    """

    # Glob for the pipeline's result CSV, relative to the shared AF3 output dir,
    # which also holds inputs/, pisa_cfgs/ and the AF3 job directories -- so this
    # must not be a bare "*.csv". The "{stem}" placeholder is substituted with the
    # round's input CSV stem, so a round reads only its own results.
    results_glob: str = "{stem}_with_af3.csv"
    # The pipeline passes the input CSV through and requires a "mutations" column.
    mutation_key_column: str = "mutations"
    # Maps each gating metric name to its column in the pipeline's output.
    # Available columns and their directions:
    #   ipSAE       interface confidence                        better high
    #   dG_binding  interface solvation energy (int_solv_en)    better LOW
    #   dG_diss     dissociation free energy of the assembly    better HIGH
    #   int_area    buried interface area                       (not a gate)
    metric_columns: Dict[str, str] = field(
        default_factory=lambda: {"ipsae": "ipSAE", "dG_binding": "dG_binding"}
    )
    # Extra columns to carry through into the round summary, if present.
    extra_columns: List[str] = field(
        default_factory=lambda: ["int_area", "dG_diss"]
    )


@dataclass
class CleanupConfig:
    """Pruning AF3 byproducts after each round.

    AF3's PAE/confidence matrices are O(tokens^2) per diffusion sample, are
    consumed once by ipSAE, and dominate disk use over a multi-round run.
    Structures (``*_model.cif``) and ipSAE reports are never pruned -- the RMSD
    gate and re-seeding depend on them.
    """

    mode: str = "compress"  # none | compress | delete
    # Leave the round's winners untouched, so anything still in play stays
    # immediately usable.
    keep_winners: bool = True
    # File classes to prune. "summary" is the pipeline's completeness marker;
    # keeping it while removing the PAE would make a forced rerun skip inference
    # and then silently yield NaN.
    targets: List[str] = field(default_factory=lambda: ["pae", "pisa", "summary"])


@dataclass
class StructureConfig:
    """The AF3 + PISA + ipSAE stage."""

    pipeline_script: str = MISSING
    pisa_exe: Optional[str] = None
    pisa_cfg: Optional[str] = None
    ipsae_script: Optional[str] = None
    # Concurrent AF3 workers sharing one GPU. Scale to complex size: AF3 pads to
    # a token bucket and a ~2000-token complex asks ~14 GiB per allocation, so an
    # 80 GB card fits only 1-2. Too high shows up as RESOURCE_EXHAUSTED.
    max_parallel: Optional[int] = None
    # Delete this round's candidates' .failed markers before folding, so mutants
    # that died from a transient or misconfigured run (e.g. GPU OOM) are retried.
    # The pipeline otherwise skips a failed mutant permanently -- including the
    # wildtype, which leaves the run with no baseline.
    retry_failed: bool = False
    # Must match the pipeline's own defaults: the ipSAE report filename embeds
    # both cutoffs, and those reports are read back to pick each mutant's best
    # model when re-seeding.
    pae_cutoff: int = 10
    dist_cutoff: int = 15
    python_executable: str = "python"
    # Extra flags appended verbatim to the pipeline invocation.
    extra_args: List[str] = field(default_factory=list)
    adapter: ResultsAdapter = field(default_factory=ResultsAdapter)
    # Skip folding a sequence already scored in an earlier round.
    cache_by_sequence: bool = True
    timeout_seconds: Optional[int] = None
    # Renames AF3 chain IDs back onto the target's, e.g. {"A": "A", "B": "B"}.
    # AF3 uses the chain ids given in the FASTA header, so this is usually empty.
    af3_chain_map: Dict[str, str] = field(default_factory=dict)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)


@dataclass
class MetricSpec:
    """A structural metric used for WT comparison and termination."""

    direction: str = "max"
    cutoff: Optional[float] = None


@dataclass
class RmsdGate:
    """Reject a mutant whose AF3 prediction drifts too far from its reference.

    When AF3 predicts a substantially different fold or binding pose, the
    ipSAE/PISA scores describe a structure other than the one the search
    reasoned about. A mutant failing this gate is disqualified from both
    promotion and the stop-cutoff.
    """

    enabled: bool = True
    max_rmsd: float = 2.0  # Angstroms
    atoms: str = "CA"  # CA | backbone
    # "seed": compare to the structure the mutant was searched from (the target
    # PDB in round 0, the seed's own AF3 prediction later). "target": always
    # compare to the original target PDB.
    reference: str = "seed"  # seed | target
    # What the RMSD is measured over:
    #   interface -- superpose on target atoms near the binder, score the binder
    #                (CAPRI ligand-RMSD localized to the binding site)
    #   binder    -- superpose and score on the binder alone (fold check only)
    #   complex   -- superpose and score over everything
    # "complex" is only meaningful for a small single-domain complex: over a
    # large multi-domain target, distal domain motion fails good candidates while
    # a real binder displacement is diluted below the threshold.
    scope: str = "interface"  # interface | binder | complex
    # Target atoms within this distance of the binder define the frame that
    # "interface" scope superposes on.
    interface_cutoff: float = 10.0


@dataclass
class GatingConfig:
    """Promotion and termination rules."""

    # ipSAE: interface confidence, better high.
    # dG_binding: PISA interface solvation energy, better LOW (more negative is
    # more favourable). dG_diss is a dissociation free energy, so its direction
    # is "max"; mixing the two up silently inverts the gate.
    metrics: Dict[str, MetricSpec] = field(
        default_factory=lambda: {
            "ipsae": MetricSpec(direction="max", cutoff=None),
            "dG_binding": MetricSpec(direction="min", cutoff=None),
        }
    )
    # A mutant is a "winner" only if it beats WT on ALL of these metrics.
    beats_wt_on: List[str] = field(default_factory=lambda: ["ipsae", "dG_binding"])
    # Whether ANY cutoff or ALL cutoffs must be met to end the run early.
    stop_when: str = "any"
    # How many mutants must satisfy the cutoff condition to trigger the stop.
    require_n_passing: int = 1
    # How many winners seed the next round.
    promote_top_n: int = 5
    # End the run if a round produces no mutant that beats WT.
    stop_on_no_winners: bool = True
    # Rank winners by the selection objective (default) or a named metric.
    promote_by: Optional[str] = None
    # Optional post-AF3 structural sanity check (default on).
    rmsd_gate: RmsdGate = field(default_factory=RmsdGate)


@dataclass
class SlurmConfig:
    partition: Optional[str] = None
    gres: Optional[str] = None
    mem: Optional[str] = None
    time: Optional[str] = None
    cpus_per_task: Optional[int] = None
    conda_env: Optional[str] = None
    conda_root: Optional[str] = None
    modules: List[str] = field(default_factory=list)
    account: Optional[str] = None
    extra_directives: List[str] = field(default_factory=list)
    poll_interval_seconds: int = 60


@dataclass
class OptimizationConfig:
    run: RunConfig = field(default_factory=RunConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def load_config(path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load a YAML config, merge it onto the schema, and validate it.

    ``overrides`` are dotlist strings such as ``run.max_iterations=5``.
    """
    schema = OmegaConf.structured(OptimizationConfig)
    user_cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(schema, user_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    validate(cfg)
    return cfg


def _check_choice(value: Any, valid: set, label: str) -> None:
    if value not in valid:
        allowed = ", ".join(sorted(str(v) for v in valid))
        raise ValueError(f"{label} must be one of: {allowed} (got {value!r})")


def validate(cfg: DictConfig) -> None:
    """Raise ``ValueError`` on any invalid or internally inconsistent setting."""
    _check_choice(cfg.run.executor, VALID_EXECUTORS, "run.executor")
    _check_choice(cfg.run.backbone_source, VALID_BACKBONE_SOURCES, "run.backbone_source")
    if cfg.run.max_iterations < 1:
        raise ValueError("run.max_iterations must be >= 1.")

    _check_choice(cfg.search.energy_mode, VALID_ENERGY_MODES, "search.energy_mode")
    _check_choice(cfg.search.rank_by, VALID_RANK_BY, "search.rank_by")
    if cfg.search.rank_by in {"binding", "pareto"} and cfg.search.energy_mode != "both":
        raise ValueError(
            f"search.rank_by='{cfg.search.rank_by}' requires search.energy_mode='both' "
            "so that stability is still computed and tracked."
        )
    _check_choice(cfg.search.keep_budget_scope, VALID_KEEP_SCOPES, "search.keep_budget_scope")
    if cfg.search.max_mutations < 1:
        raise ValueError("search.max_mutations must be >= 1.")
    if not (0.0 < cfg.search.top_percent <= 100.0):
        raise ValueError("search.top_percent must be within (0, 100].")
    if cfg.search.use_depths:
        bad = [d for d in cfg.search.use_depths if not (1 <= d <= cfg.search.max_mutations)]
        if bad:
            raise ValueError(
                f"search.use_depths entries must be within 1..{cfg.search.max_mutations}; got {bad}"
            )

    _check_choice(cfg.selection.scope, VALID_SELECTION_SCOPES, "selection.scope")
    if cfg.selection.scope == "pooled" and cfg.gating.promote_top_n > 1:
        raise ValueError(
            "selection.scope='pooled' ranks PottsMPNN energies across different "
            "structures, which is invalid because each structure has its own energy "
            "table. Use scope='per_seed', or set gating.promote_top_n=1 so every "
            "round has a single seed."
        )
    _check_choice(cfg.selection.direction, VALID_DIRECTIONS, "selection.direction")
    _check_choice(
        cfg.selection.diversity.metric,
        VALID_SIMILARITY_METRICS,
        "selection.diversity.metric",
    )
    if cfg.selection.max_candidates < 1:
        raise ValueError("selection.max_candidates must be >= 1.")
    if cfg.selection.diversity.enabled and cfg.selection.diversity.weight < 0:
        raise ValueError("selection.diversity.weight must be >= 0.")

    # Pareto-derived objectives need both objectives present in the search output.
    if cfg.selection.objective in DERIVED_OBJECTIVES and cfg.search.energy_mode != "both":
        raise ValueError(
            f"selection.objective='{cfg.selection.objective}' needs both stability and "
            "binding scores; set search.energy_mode='both'."
        )

    _check_choice(cfg.gating.stop_when, VALID_STOP_WHEN, "gating.stop_when")
    if cfg.gating.require_n_passing < 1:
        raise ValueError("gating.require_n_passing must be >= 1.")
    if cfg.gating.promote_top_n < 1:
        raise ValueError("gating.promote_top_n must be >= 1.")
    for name, spec in cfg.gating.metrics.items():
        _check_choice(spec.direction, VALID_DIRECTIONS, f"gating.metrics.{name}.direction")
    unknown = [m for m in cfg.gating.beats_wt_on if m not in cfg.gating.metrics]
    if unknown:
        known = ", ".join(sorted(cfg.gating.metrics.keys()))
        raise ValueError(
            f"gating.beats_wt_on references unknown metric(s): {unknown}. Declared metrics: {known}"
        )
    if not cfg.gating.beats_wt_on:
        raise ValueError("gating.beats_wt_on must list at least one metric.")
    if all(spec.cutoff is None for spec in cfg.gating.metrics.values()):
        # Legal: the run can then only end via max_iterations / no winners.
        pass
    if cfg.gating.promote_by is not None and cfg.gating.promote_by not in cfg.gating.metrics:
        known = ", ".join(sorted(cfg.gating.metrics.keys()))
        raise ValueError(
            f"gating.promote_by={cfg.gating.promote_by!r} is not a declared metric. Declared: {known}"
        )

    _check_choice(cfg.gating.rmsd_gate.atoms, {"CA", "backbone"}, "gating.rmsd_gate.atoms")
    _check_choice(cfg.gating.rmsd_gate.reference, {"seed", "target"}, "gating.rmsd_gate.reference")
    _check_choice(
        cfg.gating.rmsd_gate.scope, {"interface", "binder", "complex"}, "gating.rmsd_gate.scope"
    )
    if cfg.gating.rmsd_gate.enabled and cfg.gating.rmsd_gate.max_rmsd <= 0:
        raise ValueError("gating.rmsd_gate.max_rmsd must be a positive distance in Angstroms.")
    if cfg.gating.rmsd_gate.interface_cutoff <= 0:
        raise ValueError("gating.rmsd_gate.interface_cutoff must be positive.")
    if (
        cfg.gating.rmsd_gate.enabled
        and cfg.gating.rmsd_gate.scope in {"interface", "binder"}
        and not resolve_binder_chains(cfg)
    ):
        raise ValueError(
            f"gating.rmsd_gate.scope='{cfg.gating.rmsd_gate.scope}' needs to know which chain is "
            "the binder. Set search.binder_chain, or give target.binding_partitions so the "
            "second partition can be used."
        )

    _check_choice(
        cfg.structure.cleanup.mode, {"none", "compress", "delete"}, "structure.cleanup.mode"
    )
    unknown_targets = [
        t for t in cfg.structure.cleanup.targets if t not in {"pae", "pisa", "summary"}
    ]
    if unknown_targets:
        raise ValueError(
            f"structure.cleanup.targets has unknown entries {unknown_targets}; "
            "valid: pae, pisa, summary"
        )
    if cfg.structure.cleanup.mode != "none" and not cfg.structure.cache_by_sequence:
        raise ValueError(
            "structure.cleanup prunes the AF3 completeness marker, so the loop's own "
            "sequence cache must stay on to avoid re-folding pruned mutants. Either set "
            "structure.cache_by_sequence=true or structure.cleanup.mode=none."
        )

    # Every gating metric must be readable out of the pipeline's output.
    unmapped = [m for m in cfg.gating.metrics if m not in cfg.structure.adapter.metric_columns]
    if unmapped:
        mapped = ", ".join(sorted(cfg.structure.adapter.metric_columns.keys()))
        raise ValueError(
            f"gating.metrics declares {unmapped} but structure.adapter.metric_columns has no "
            f"column for them. Mapped metrics: {mapped}"
        )

    if cfg.run.executor == "slurm" and not cfg.slurm.partition:
        raise ValueError("run.executor='slurm' requires slurm.partition to be set.")


def resolve_binder_chains(cfg) -> List[str]:
    """Which chain(s) are the binder, for interface-localized RMSD.

    Prefers the explicit ``search.binder_chain``; otherwise falls back to the
    second binding partition, which is the binder by the convention used
    throughout (``[["A"], ["B"]]`` = target, binder).
    """
    if cfg.search.binder_chain:
        return [str(cfg.search.binder_chain)]
    partitions = cfg.target.binding_partitions
    if partitions and len(partitions) >= 2 and partitions[1]:
        return [str(c) for c in partitions[1]]
    return []


def metric_is_better(value: float, reference: float, direction: str) -> bool:
    """True if ``value`` is strictly better than ``reference`` for ``direction``."""
    if direction == "max":
        return value > reference
    return value < reference


def metric_meets_cutoff(value: float, cutoff: float, direction: str) -> bool:
    """True if ``value`` reaches ``cutoff`` for ``direction`` (inclusive)."""
    if direction == "max":
        return value >= cutoff
    return value <= cutoff


def resolve_partitions(cfg: DictConfig, pdb_name: str) -> List[List[str]]:
    """Resolve binding partitions from the inline config or the partitions JSON."""
    if cfg.target.binding_partitions:
        return [list(p) for p in cfg.target.binding_partitions]
    if not cfg.target.binding_energy_json:
        raise ValueError(
            "Set either target.binding_partitions or target.binding_energy_json to define "
            "the binding partitions."
        )
    import json

    source = cfg.target.binding_energy_json
    # Mirrors run_utils.process_data: the value may be an inline mapping or a path.
    if isinstance(source, (dict, DictConfig)):
        table = OmegaConf.to_container(source, resolve=True) if isinstance(source, DictConfig) else source
    else:
        with open(source, "r", encoding="utf-8") as handle:
            table = json.load(handle)

    if pdb_name in table:
        return [list(p) for p in table[pdb_name]]
    for key in table:
        if str(key).split("|")[0] == pdb_name:
            return [list(p) for p in table[key]]
    raise KeyError(
        f"No binding partition entry for PDB {pdb_name!r} in {source}. "
        f"Available keys: {sorted(table)[:10]}"
    )

# Iterative Pareto optimization loop

Closes the loop between the mutation search and the AF3 + PISA + ipSAE
structural scoring stage, which previously had to be joined by hand.

```
round k:  mutation search (stability x binding Pareto), one search per seed
      ->  select folding set (objective + constraints + diversity)
      ->  AF3 + PISA + ipSAE
      ->  promote mutants beating wildtype on the structural metrics
      ->  re-seed round k+1 from them
```

The run stops when a configured ipSAE and/or PISA cutoff is reached, when no
mutant beats wildtype, or when the iteration budget runs out.

## Running

```bash
python run_optimization.py --config inputs/example_config_optimization.yaml
```

Any field can be overridden on the command line:

```bash
python run_optimization.py --config <cfg> run.max_iterations=5 gating.promote_top_n=3
```

Inspect the merged config without running anything:

```bash
python run_optimization.py --config <cfg> --print-config
```

## Configuration

See [`inputs/example_config_optimization.yaml`](../inputs/example_config_optimization.yaml)
for a fully commented config. The blocks that matter most:

### Choosing what to fold (`selection`)

`objective` accepts a column, a derived metric, or an expression;
`constraints` are pandas `.query()` strings that hard-filter first. Together
they cover requests like *"best binding, but the mutant cannot lose stability"*:

```yaml
selection:
  objective: binding_score
  direction: min
  constraints: ["stability_score <= 0"]
```

Derived objectives (all oriented so **lower is better**):

| objective | meaning |
|---|---|
| `pareto_rank` | non-dominated sorting front index; 0 is the front |
| `pareto_front` | 0 on the front, 1 off it |
| `pareto_distance` | normalized Euclidean distance to the front |
| `rrf` | reciprocal rank fusion of stability and binding |

Arbitrary expressions work too, e.g. `objective: "binding_score - 0.5 * stability_score"`.

`diversity` applies a greedy MMR re-rank on top, so the folding set is not fifty
variations on one position.

### Stopping (`gating`)

```yaml
gating:
  metrics:
    ipsae:   {direction: max, cutoff: 0.75}
    dG_diss: {direction: max, cutoff: 15.0}
  beats_wt_on: [ipsae, dG_diss]  # must beat WT on ALL of these to be promoted
  stop_when: any                 # "any" cutoff ends the run; "all" requires both
  require_n_passing: 1
  promote_top_n: 5
```

To gate on **one** metric only, set the other's `cutoff: null`.

An optional **RMSD gate** (`gating.rmsd_gate`, on by default) runs after AF3:
each mutant's predicted model is Kabsch-superposed onto its reference (the
structure it was searched from, or the target PDB) and the Cα RMSD is measured.
A mutant past `max_rmsd` (default 2 Å) — or whose RMSD can't be computed — is
disqualified from both promotion and the stop-cutoff, since its ipSAE/PISA then
describe a structure other than the one that was scored.

`rmsd_gate.scope` decides what is superposed and what is scored. The default
`interface` fits on target atoms within `interface_cutoff` of the binder and
scores the binder — CAPRI ligand-RMSD localized to the pocket. `complex` (global
fit and score) is only valid for a small single-domain complex: on a large
multi-domain target it both fails good candidates on distal domain motion and
dilutes real binder displacement below the threshold.

Metrics available from `run_mutation_af3_pipeline.py`, with their directions:

| column | meaning | better | default gate |
|---|---|---|---|
| `ipSAE` | interface confidence, min-asym over the best of 3 diffusion samples | high | ✅ |
| `dG_binding` | PISA interface solvation energy (`int_solv_en`) | **low** | ✅ |
| `dG_diss` | PISA dissociation free energy of the largest assembly | **high** | — |
| `int_area` | interface buried area | — | — |

> `dG_binding` and `dG_diss` point in **opposite directions**. `dG_binding` is a
> solvation energy, so more negative is more favourable — this is the metric the
> existing analysis plots use, and it is the default. `dG_diss` is the energy
> needed to pull the complex apart, so higher is tighter. Setting the wrong
> `direction` silently inverts the entire gate rather than erroring, which is why
> both defaults are pinned and covered by tests.

### Controlling how many sequences get scored

A round fans out into **one independent search per promoted seed**, and within
each search the number of sequences *scored* at depth `d` is
`(kept at depth d-1) × (mutable positions × 19)`. Costs therefore grow with
depth, not with breadth:

| lever | effect |
|---|---|
| `search.max_mutations` | **the dominant one.** Depth 3-4 is where cost explodes. Rounds compound, so keep this at 1-2 and let the loop supply depth. |
| `search.keep_budget_scope` | `global` (default) treats `max_keep_per_depth` as a whole-round budget split across seeds, so scoring cost stays flat as `promote_top_n` grows. `per_seed` gives each seed the full cap and multiplies cost by the seed count. |
| `search.max_keep_per_depth` | the budget itself; caps everything from depth 2 on. |
| `search.binding_energy_cutoff` | restricts mutable positions to the interface — the biggest reduction in the `× 19` term. |
| `search.allowed_from_aas` / `allowed_to_aas` | shrink the substitution alphabet. |
| `search.top_percent_decay_base` | `>1` shrinks the kept fraction at each successive depth. |
| `gating.promote_top_n` | seed count. Under `global` scope this costs wall-clock (searches run sequentially) but not extra scoring. |

AF3 cost is **independent of all this** — it is bounded by
`selection.max_candidates` per round.

Each round logs its per-seed budget and an upper-bound projection before
searching, so a runaway config is visible before the GPU time is spent:

```
[round 1] keep budget: 200/depth/seed (global scope, 5 seeds, cap 1000)
[round 1] projected scoring ceiling: depth 1<=5,890, depth 2<=222,300 (total <=228,190)
```

### Where work runs (`run.executor`)

`local` runs subprocesses bounded by `structure.max_parallel`; `slurm` submits an
sbatch array and polls to completion, reusing the manifest idiom from the
existing submit scripts.

## Where structures may and may not be compared

PottsMPNN builds a **different energy table for every structure**, so
`stability_score` and `binding_score` are only meaningful *within* one seed's
search. The loop enforces that boundary:

```
per structure (Potts energies only)          across structures (AF3/PISA only)
─────────────────────────────────────        ────────────────────────────────
search  ── rank ── constraints ── diversity ─┐
   (one search per seed, own backbone)       ├─► AF3 + PISA ── beat WT? ── rank ── top N
search  ── rank ── constraints ── diversity ─┘
```

* **`selection.scope: per_seed`** (default) ranks each seed's candidates only
  against its own structure-mates and gives each seed `max_candidates //
  n_seeds` slots. A pooled ranking would hand every slot to whichever structure
  happens to sit lowest on the energy scale, and would compute the Pareto front
  over incomparable numbers. Setting `scope: pooled` with `promote_top_n > 1` is
  rejected by config validation.
* **Promotion ranks on structural metrics only** — `gating.promote_by`, or the
  mean normalized rank across `gating.beats_wt_on`. It never falls back to a
  Potts objective.
* **Cross-seed duplicates are collapsed after selection**, not before, so each
  structure's ranking sees its complete candidate set. Within a single structure
  duplicates are resolved by `score`; across structures, positionally — "the
  better score" is undefined between two energy tables.

## Two things to know

**Search scores are not comparable across rounds.** With
`run.backbone_source: af3`, each round searches the AF3-predicted structure of
its seed, so ddG values are relative to *that* backbone and sequence. Cross-round
progress is measured by ipSAE/PISA, which share one fixed wildtype baseline
(folded once, in round 0, and cached).

**Mutations are always reported against the original wildtype.** The search
labels mutations relative to whatever it was seeded with, so every pooled
candidate is re-diffed against the round-0 sequence. Reversions therefore
disappear from the list rather than accumulating as noise.

**The wildtype reference is free.** `run_mutation_af3_pipeline.py` prepends a
`mutations == "WT"` row to every run, so the baseline is produced alongside the
mutants. It is captured from the first round and cached; because all rounds share
one AF3 output directory, the pipeline's own skip logic means it is folded once.

## Resuming

Every stage writes its output and records a marker in `run_state.json`, so a
preempted run resumes where it stopped:

```bash
python run_optimization.py --config <cfg>          # resumes automatically
python run_optimization.py --config <cfg> run.force=true   # recompute everything
```

Two layers of caching avoid repeated AF3 work: this loop skips any sequence
already scored in an earlier round, and the pipeline itself skips any job whose
outputs are complete (it also writes a `.failed` marker so a mutant that crashed
AF3 is not retried forever).

## Output layout

```
<out_dir>/
  run_state.json                  resume markers, seed lineage, result cache
  optimization_summary.json       termination reason + per-round statistics
  structure/                      ONE shared AF3 root for the whole run
    <name>__<TOKENS>/             AF3 job dir, e.g. 350d__BW102E_BI110S
      seed-1_sample-N/            model .cif + ipSAE report per sample
    round_<k>_folding_set_with_af3.csv    pipeline results for round k
  round_<k>/
    backbones/                    per-seed structure the search ran against
    seed_<id>/                    raw mutation_search output (CSVs + plots)
    inputs/                       pipeline input CSV, kept out of the results glob
    pooled_candidates.csv         all seeds pooled, deduplicated
    folding_set.csv               what was sent to AF3  (= ranked_mutations)
    scored_candidates.csv         folding set with ipSAE / PISA attached
    round_summary.csv             + beats_wt / meets_cutoff / rmsd / passes_rmsd
```

## Tests

Run without torch, a GPU, or cluster access — PottsMPNN scoring and the AF3
pipeline are stubbed, but backbone preparation, selection, gating and the AF3
output-layout logic all run for real:

```bash
python -m optimize.tests.test_optimize
```

## Coupling to `run_mutation_af3_pipeline.py`

The loop reproduces three things from that script; if it changes, these follow:

1. **Output contract** — `<out_dir>/<input_stem>_with_af3.csv`, joined on
   `mutations`, with a prepended `WT` row. Configured under `structure.adapter`.
2. **Job naming** — `safe_token(f"{base_name}__{'_'.join(tokens)}")` lowercased,
   where a token drops the colon (`B:W102E` → `BW102E`) and `base_name` is the
   first `|` field of the FASTA header. Implemented in `af3_layout.py`.
3. **Model selection** — the sample with the highest min-asym ipSAE. Recovered by
   reading the `*_<pae>_<dist>.txt` reports the pipeline leaves beside each model,
   so `structure.pae_cutoff` / `dist_cutoff` must match what the pipeline ran with.

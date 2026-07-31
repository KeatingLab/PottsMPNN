# Iterative Binder Optimization (PottsMPNN → AF3 → PISA)

> Heading levels are set to slot into `README.md` as section 9. Read standalone as-is.

## 9. Iterative Binder Optimization

`run_optimization.py` closes the loop between PottsMPNN's mutation search and
structural validation with AlphaFold3, ipSAE and PISA. Each round explores a
Pareto front of stability × binding, folds the most promising mutants, keeps
those that beat wildtype on the structural metrics, and starts the next round
from *their predicted structures*.

```
round k
   ┌─────────────────────────────────────────────────────────────┐
   │  for each seed structure:                                    │
   │      PottsMPNN mutation search (stability × binding Pareto)  │
   │      rank + constrain + diversify   → this seed's quota      │
   └─────────────────────────────────────────────────────────────┘
                              │  folding set (all seeds' picks)
                              ▼
            AlphaFold3  →  ipSAE  →  PISA
                              │  ipSAE, dG_binding, dG_diss, int_area
                              ▼
            keep mutants beating wildtype on ALL gated metrics
                              │
              ┌───────────────┴───────────────┐
       cutoff reached?                   otherwise
              │                               │
            STOP            top N winners become round k+1's seeds
```

Terminates when a configured ipSAE and/or PISA cutoff is met, when no mutant
beats wildtype, or when the iteration budget is exhausted.

---

### 9.1 External dependencies

PottsMPNN itself needs only the conda environment from section 2. The structural
stage shells out to three tools that **must be installed separately**:

| Tool | Purpose here | Where to get it |
|---|---|---|
| **AlphaFold3** | Folds each mutant complex using precomputed MSAs | [github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) · [installation docs](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) |
| **ipSAE** | Interface confidence score; replaces ipTM for multi-domain / disordered complexes | [github.com/DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE) · [paper](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1) |
| **PISA** | Interface area, solvation energy, and assembly dissociation energy | [CCP4 PISA docs](https://www.ccp4.ac.uk/html/pisa.html) · [CCP4 suite](https://www.ccp4.ac.uk/) · [manual (PDF)](https://ftp.ccp4.ac.uk/ekr/pisa/PISA_1.11.pdf) |

#### AlphaFold3

Source code is Apache-2.0, but **model parameters are licensed separately** and
must be requested from Google — they are not redistributable, and the terms
restrict output to theoretical modelling only (no clinical use). Follow the
installation docs to obtain them.

`run_mutation_af3_pipeline.py` invokes AF3 through **Singularity/Apptainer**, not
Docker, so you need an `alphafold3.sif` image built from the official container.
It auto-discovers the installation from these locations. Edit `find_af3_paths()` with your paths.

It runs with `--num_diffusion_samples=3` and one model seed, and disables JAX
memory preallocation so several AF3 processes can share one GPU.

#### ipSAE

A single script, `ipsae.py`. Clone the repo and point `structure.ipsae_script`
at it. The pipeline calls it as
`python ipsae.py <pae_json> <model_cif> <pae_cutoff> <dist_cutoff>` and reads the
minimum asymmetric ipSAE from the report it writes beside each model.

#### PISA

You need the **command-line** PISA binary (shipped with CCP4 6.1+, or built from
source), invoked as `pisa <session> -analyse <file> --as-is <cfg>`. Point
`structure.pisa_exe` at the binary and `structure.pisa_cfg` at a config
template — the pipeline copies the template per mutant and rewrites
`SESSION_PREFIX` so concurrent workers do not collide.

---

### 9.2 Input files

Four inputs beyond the model checkpoint:

**1. Target structure** (`target.pdb`) — the starting complex

**2. Target FASTA** (`target.fasta`) — chain ids in the header, sequences joined
by `:`. The base name (first `|` field) also determines AF3 job directory names:

```
>example_binder|A:B
LPMMPRQVYCA:EEKEFRLDQ
```

**3. MSA paths JSON** (`target.msa_json`) — per-chain precomputed MSAs, so AF3
skips its own search. Any key may be omitted:

```json
{
  "A": {
    "unpaired": "/abs/path/chainA_unpaired.a3m",
    "paired":   "/abs/path/chainA_paired.a3m",
    "templates": [
      {"mmcifPath": "/abs/path/template.cif",
       "queryIndices": [0, 1, 2], "templateIndices": [0, 1, 2]}
    ]
  },
  "B": {"unpaired": "/abs/path/chainB_unpaired.a3m", "paired": ""}
}
```

**4. Binding partitions** — how chains split for the binding-energy calculation.
Either inline in the config as `target.binding_partitions: [["A"], ["B"]]`, or a
JSON keyed by PDB name (`target.binding_energy_json`), matching the format used
elsewhere in PottsMPNN:

```json
{"example_binder": [["A"], ["B"]]}
```

---

### 9.3 Quick start

```bash
cp inputs/example_config_optimization.yaml inputs/my_opt.yaml
# edit target.*, structure.* paths, and the search/gating knobs
```

Check the merged config without running anything:

```bash
python run_optimization.py --config inputs/my_opt.yaml --print-config
```

Run it:

```bash
python run_optimization.py --config inputs/my_opt.yaml
```

Override any field inline:

```bash
python run_optimization.py --config inputs/my_opt.yaml run.max_iterations=5 gating.promote_top_n=3
```

---

### 9.4 PottsMPNN energy comparisons across structures

**PottsMPNN builds a different energy table for every structure.** Its energies
are therefore only meaningful *within* one structure — a `binding_score` of
−3.0 on structure A says nothing about −1.5 on structure B.

The pipeline enforces that boundary:

| Stage | Compares | Using |
|---|---|---|
| Search, ranking, constraints, diversity | mutants of **one** structure | PottsMPNN energies |
| Beat-wildtype test, promotion ranking | mutants **across** structures | ipSAE / PISA only |

So each seed gets `selection.max_candidates / n_seeds` folding slots and is ranked
only against its own structure-mates (`selection.scope: per_seed`). Promotion —
the single point where structures meet — ranks on structural metrics alone.
Setting `selection.scope: pooled` with more than one seed is rejected by config
validation.

As a consequence, `binding_score` in `round_summary.csv` should not be used to sort a merged summary.

On a related note, search scores are also **not comparable across rounds**, because
`run.backbone_source: af3` gives each round a different backbone. Track progress
with ipSAE/PISA, which share one fixed wildtype baseline.

---

### 9.5 Configuration reference

#### `run` — top level

| Key | Default | Meaning |
|---|---|---|
| `out_dir` | *required* | Root for all output. |
| `out_name` | `optimization` | Run label; also the PISA session prefix. |
| `max_iterations` | `3` | Maximum rounds. |
| `executor` | `local` | `local` (subprocesses) or `slurm` (sbatch array + polling). |
| `backbone_source` | `af3` | `af3` searches each seed's predicted structure; `wt` keeps the original backbone and only substitutes the sequence. |
| `seed` | `null` | RNG seed. |
| `force` | `false` | Discard resume markers and recompute everything. |
| `report` | `true` | Write `<out_dir>/report/` when the run ends (§7.11). |
| `report_each_round` | `false` | Also refresh the report after every round. |

#### `target` — the design target

| Key | Default | Meaning |
|---|---|---|
| `pdb` | *required* | Starting complex. |
| `fasta` | *required* | Sequences + chain ids; base name drives AF3 job names. |
| `msa_json` | `null` | Per-chain MSA paths (§7.2). |
| `binding_energy_json` | `null` | Partitions keyed by PDB name. |
| `binding_partitions` | `null` | Inline partitions, e.g. `[["A"], ["B"]]`. Takes precedence. |

#### `search` — forwarded to `recursive_mutation_search`

| Key | Default | Meaning |
|---|---|---|
| `cfg_path` | *required* | PottsMPNN model YAML (see `inputs/example_mut_search_config.yaml`). |
| `max_mutations` | `3` | Mutation depth **per round**. Rounds compound — see §7.6. |
| `top_percent` | `10.0` | Percentage kept at each depth. |
| `top_percent_decay_base` | `1.0` | `>1` shrinks the kept fraction as depth grows. |
| `max_keep_per_depth` | `1000` | Hard cap on kept candidates per depth. |
| `keep_budget_scope` | `global` | `global` splits that cap across seeds (flat cost); `per_seed` gives each seed the full cap. |
| `per_position_quota` | `100` | Max kept candidates touching any one position. |
| `disallowed_chains` | `[]` | Chains held fixed, e.g. `[A]` for the target. |
| `binder_chain` | `null` | Chain whose partition energy is reported as stability. |
| `energy_mode` | `both` | `stability`, `binding`, or `both`. |
| `rank_by` | `pareto` | `joint` (RRF), `binding`, or `pareto` (non-dominated sorting). |
| `rrf_k` | `60` | RRF constant when `rank_by: joint`. |
| `binding_energy_cutoff` | `8.0` | Å cutoff restricting mutations to interface residues. |
| `allowed_from_aas` | `null` | Only mutate *from* these, e.g. `"AVILMFW"`. |
| `allowed_to_aas` | `null` | Only mutate *to* these, e.g. `"RNDEQHKSTY"`. |
| `show_pareto_front` | `true` | Emit a `pareto_front` flag column. |
| `use_depths` | `[]` | Which depths feed the folding set; empty = all. |

#### `selection` — search output → AF3 folding set

| Key | Default | Meaning |
|---|---|---|
| `scope` | `per_seed` | Rank within each structure (see §7.4). `pooled` only valid with one seed. |
| `objective` | `binding_score` | A column, a derived metric, or a pandas expression. |
| `direction` | `min` | `min` or `max`. |
| `constraints` | `[]` | pandas `.query()` strings, ANDed. |
| `max_candidates` | `50` | AF3 budget per round, split across seeds. |
| `diversity.enabled` | `true` | Greedy MMR re-rank. |
| `diversity.weight` | `10.0` | Higher favours dissimilar mutation sets. |
| `diversity.metric` | `jaccard` | `jaccard` or `overlap` over mutation tokens. |
| `stability_column` / `binding_column` / `mutation_column` | — | Column names in the search output. |

Derived objectives, all oriented **lower = better**:

| Objective | Meaning |
|---|---|
| `pareto_rank` | Non-dominated sorting front index; `0` is the front. |
| `pareto_front` | `0` on the front, `1` off it. |
| `pareto_distance` | Normalized Euclidean distance to the front. |
| `rrf` | Reciprocal rank fusion of stability and binding. |

Expressions work too: `objective: "binding_score - 0.5 * stability_score"`.

**Example — best binding, but stability must not get worse:**

```yaml
selection:
  objective: binding_score
  direction: min
  constraints: ["stability_score <= 0"]
```

**Example — closest to the Pareto front, hydrophilic substitutions only:**

```yaml
selection:
  objective: pareto_distance
  constraints: ["stability_score <= 0", "binding_score < -1.0"]
```

#### `structure` — the AF3 + PISA + ipSAE stage

| Key | Default | Meaning |
|---|---|---|
| `pipeline_script` | *required* | Path to `run_mutation_af3_pipeline.py`. |
| `pisa_exe` | `null` | PISA binary. |
| `pisa_cfg` | `null` | PISA config template. |
| `ipsae_script` | `null` | Path to `ipsae.py`. |
| `max_parallel` | `null` | Concurrent AF3 workers sharing one GPU. **Scale to complex size** — see below. |
| `retry_failed` | `false` | Delete this round's `.failed` markers before folding, so previously-crashed mutants are retried. |
| `pae_cutoff` | `10` | ipSAE PAE cutoff. **Must match** what the pipeline ran with — the report filename embeds it. |
| `dist_cutoff` | `15` | ipSAE distance cutoff. Same constraint. |
| `python_executable` | `python` | Interpreter for the pipeline. |
| `extra_args` | `[]` | Extra flags appended verbatim. |
| `cache_by_sequence` | `true` | Skip re-folding a sequence scored in an earlier round. |
| `timeout_seconds` | `null` | Per-job timeout (local executor). |
| `af3_chain_map` | `{}` | Rename AF3 chain ids back onto the target's. Normally empty. |
| `cleanup.mode` | `compress` | `none`, `compress` (gzip), or `delete` — prune AF3 byproducts after each round. |
| `cleanup.keep_winners` | `true` | Leave the round's winners completely untouched. |
| `cleanup.targets` | `[pae, pisa, summary]` | Which file classes to prune. |

**Disk growth.** AF3 writes a PAE/confidence matrix per diffusion sample, sized
**O(tokens²)** — three samples per mutant, fifty mutants per round:

| Complex | per sample | per round | over 5 rounds |
|---|---|---|---|
| 1867 tokens (full length) | ~42 MB | ~6.3 GB | **~31 GB** |
| 282 tokens (truncated) | ~1 MB | ~143 MB | ~0.7 GB |

Those matrices are read once, by ipSAE, and never again — so they are pruned
after each round. **Structures (`*_model.cif`) and ipSAE reports are never
pruned**: the RMSD gate and re-seeding depend on them, so a pruned mutant remains
fully usable as a seed. `keep_winners` additionally leaves the round's winners
entirely alone.

`summary` is included in `targets` deliberately. It is the pipeline's
"outputs complete" marker; keeping it while removing the PAE would make a forced
rerun skip inference and then fail to find the PAE, silently producing NaN
metrics. Pruning both keeps the job honestly marked incomplete. This is why
`cleanup` requires `structure.cache_by_sequence: true` — the loop's own cache is
what prevents pruned mutants from being re-folded.

To prune an existing run's output directory retroactively:

```bash
python -m optimize.cleanup <out_dir>/structure --mode compress --dry-run
```
| `adapter.results_glob` | `{stem}_with_af3.csv` | How to find the pipeline's output. |
| `adapter.mutation_key_column` | `mutations` | Join column. |
| `adapter.metric_columns` | `{ipsae: ipSAE, dG_binding: dG_binding}` | Gating metric → output column. |
| `adapter.extra_columns` | `[int_area, dG_diss]` | Carried through to the summary. |

#### `gating` — promotion and termination

| Key | Default | Meaning |
|---|---|---|
| `metrics` | ipSAE (max), dG_binding (min) | Per-metric `direction` and optional `cutoff`. |
| `beats_wt_on` | `[ipsae, dG_binding]` | Must beat wildtype on **all** of these to be promoted. |
| `stop_when` | `any` | `any` cutoff ends the run; `all` requires one mutant to meet every cutoff. |
| `require_n_passing` | `1` | How many mutants must satisfy the cutoff condition. |
| `promote_top_n` | `5` | How many winners seed the next round. |
| `stop_on_no_winners` | `true` | End the run if a round promotes nothing. |
| `promote_by` | `null` | Rank winners by one metric; `null` = mean normalized rank across `beats_wt_on`. |
| `rmsd_gate.enabled` | `true` | Post-AF3 structural sanity check (below). |
| `rmsd_gate.max_rmsd` | `2.0` | Max superposed RMSD (Å) between prediction and reference. |
| `rmsd_gate.atoms` | `CA` | `CA` or `backbone` (N, Cα, C, O). |
| `rmsd_gate.reference` | `seed` | `seed` = the structure the mutant was searched from; `target` = the original target PDB. |
| `rmsd_gate.scope` | `interface` | What the RMSD measures — see below. |
| `rmsd_gate.interface_cutoff` | `10.0` | Target atoms this close to the binder define the superposition frame. |

**`scope` matters more than `max_rmsd`.** The gate asks "did the binder move or
misfold?", and the answer depends entirely on what you superpose:

| scope | superpose on | score over | measures |
|---|---|---|---|
| `interface` *(default)* | target atoms near the binder | the binder | binding-pose change (CAPRI ligand-RMSD) |
| `binder` | the binder | the binder | binder fold only, ignores placement |
| `complex` | everything | everything | global conformation |

> **Do not use `complex` with a multi-domain target.** It is wrong in *both*
> directions. On a real complex: rotating a domain far from the
> binding site — leaving the binder untouched — reads **4.37 Å** and fails the
> gate, while genuinely displacing the binder by 5 Å is diluted to **1.21 Å** and
> passes it. The `interface` scope reports 0.00 Å and 5.00 Å respectively.

**Sizing `max_parallel`.** This is the most common cause of a failed run. AF3
pads the input to a token bucket and allocates accordingly: a 2000-residue
complex rounds to 2048 tokens and requests **~14 GiB in a single allocation**. Set it by complex size, not by CPU
count:

| Complex size | `max_parallel` on an 80 GB card |
|---|---|
| ≲600 tokens | ~8 |
| ~1000 tokens | ~4 |
| ~2000 tokens | **1** |

Too high fails as `RESOURCE_EXHAUSTED: Failed to allocate request for N GiB` in
`<out_dir>/structure/logs/af3_round_<k>.err`. Note the pipeline then writes a
`.failed` marker per crashed mutant and **skips it permanently on every later
run** — so after such a failure a plain rerun reproduces it exactly. Recover with
`structure.retry_failed: true` for one run, or by deleting
`<out_dir>/structure/*.failed`. If the wildtype was among the casualties the run
cannot produce a baseline at all, and says so explicitly.

**The RMSD gate.** AF3 sometimes predicts a substantially different fold or
binding pose for a mutant; when it does, the ipSAE/PISA numbers describe a
structure other than the one the search reasoned about. After AF3, each mutant's
predicted model is superposed (Kabsch) onto its reference and the Cα RMSD is
measured. A mutant exceeding `max_rmsd` — or whose RMSD cannot be computed
(missing model, atom mismatch) — is **disqualified from both promotion and the
stop-cutoff**, so a drifted hit can neither seed the next round nor falsely end
the run. With `reference: seed` (default), the comparison is against the exact
structure the mutant was mutated from: the target PDB in round 0, and the seed's
own AF3 prediction in later rounds. The `rmsd` and `passes_rmsd` columns are
written to `round_summary.csv`. Disable with `rmsd_gate.enabled: false`.

**Metric directions — these are opposite, and a wrong one inverts the gate
silently rather than erroring:**

| Column | Meaning | Better |
|---|---|---|
| `ipSAE` | Interface confidence (min-asym, best of 3 samples) | **high** |
| `dG_binding` | Interface solvation energy (PISA `int_solv_en`) | **low** |
| `dG_diss` | Assembly dissociation free energy | **high** |
| `int_area` | Buried interface area | — |

`dG_binding` is the default because it is what the existing analysis plots use.
To gate on `dG_diss` instead:

```yaml
structure:
  adapter:
    metric_columns: {ipsae: ipSAE, dG_diss: dG_diss}
gating:
  metrics:
    ipsae:   {direction: max, cutoff: 0.40}
    dG_diss: {direction: max, cutoff: 15.0}
  beats_wt_on: [ipsae, dG_diss]
```

To gate on **one** metric only, set the other's `cutoff: null` — it still gets
computed and still counts toward `beats_wt_on`, it just cannot end the run.

#### `slurm` — only when `run.executor: slurm`

| Key | Meaning |
|---|---|
| `partition`, `gres`, `mem`, `time`, `cpus_per_task`, `account` | sbatch directives. |
| `conda_env`, `conda_root` | Environment activated in the job script. |
| `modules` | `module load` lines, e.g. `[apptainer/1.4.2]`. |
| `extra_directives` | Raw extra `#SBATCH` lines. |
| `poll_interval_seconds` | How often to check `squeue` (default 60). |

---

### 9.6 Controlling how many sequences get scored

Within one search, the number of sequences **scored** at depth *d* is
`(kept at depth d−1) × (mutable positions × 19)`. Cost grows with *depth*, not
breadth. Measured ceilings for 30 mutable positions and 5 seeds:

| Config | sequences scored |
|---|---|
| depth 4, `keep_budget_scope: per_seed` | 5,865,300 |
| depth 4, `global` | 1,305,300 |
| **depth 2, `global`** (default) | **165,300** |
| depth 1, `global` | 2,850 |

Levers, most effective first:

1. **`search.max_mutations`** — the dominant one. Rounds compound: depth 2 over
   5 rounds already reaches 10 cumulative mutations. Keep it at 1–2 and let the
   loop supply depth; it is cheaper *and* a finer-grained hill climb.
2. **`search.keep_budget_scope: global`** — keeps cost flat as `promote_top_n` grows.
3. **`search.binding_energy_cutoff`** — restricting to the interface is the
   biggest cut to the `× 19` term.
4. **`search.allowed_from_aas` / `allowed_to_aas`** — shrink the alphabet.
5. **`search.top_percent_decay_base` > 1** — taper the kept fraction with depth.

AF3 cost is **independent of all this** — it is bounded by
`selection.max_candidates` per round.

Each round logs its budget and a ceiling before spending GPU time:

```
[round 1] keep budget: 200/depth/seed (global scope, 5 seeds, cap 1000)
[round 1] projected scoring ceiling: depth 1<=5,890, depth 2<=222,300 (total <=228,190)
```

---

### 9.7 Output layout

```
<out_dir>/
  run_state.json                 resume markers, seed lineage, result cache
  optimization_summary.json      termination reason + per-round statistics
  structure/                     ONE shared AF3 root for the whole run
    <name>__<TOKENS>/            AF3 job dir, e.g. example_binder__BR1E_V2A
      seed-1_sample-N/           model .cif + ipSAE report per sample
    round_<k>_folding_set_with_af3.csv
  round_<k>/
    backbones/                   structure each seed searched against
    seed_<id>/                   raw mutation_search output (CSVs + plots)
    inputs/                      pipeline input CSV
    pooled_candidates.csv        all seeds, provenance-tagged
    folding_set.csv              what was sent to AF3 (= ranked_mutations)
    scored_candidates.csv        folding set + ipSAE / PISA
    round_summary.csv            + beats_wt / meets_cutoff / rmsd / passes_rmsd
```

Mutations are reported as `CHAIN:WT<pos><MUT>` (e.g. `B:W3E`), 1-indexed within
the chain, and always **relative to the original wildtype** — the AF3 pipeline
validates each mutation against the base FASTA, so cumulative notation is
required. Reverting a mutation makes it disappear from the list.

---

### 9.8 Resuming

Every stage records a completion marker, so a preempted run continues where it
stopped:

```bash
python run_optimization.py --config inputs/my_opt.yaml              # resumes
python run_optimization.py --config inputs/my_opt.yaml run.force=true  # from scratch
```

Three layers avoid duplicated AF3 work: the loop skips sequences scored in an
earlier round; the pipeline skips jobs whose outputs are complete; and a mutant
that crashed AF3 gets a `.failed` marker so it is not retried forever (it shows
up as an unscored candidate with a warning, not a crash).

---

### 9.9 Troubleshooting

| Symptom | Cause |
|---|---|
| `No pipeline results matched …_with_af3.csv` | AF3 pipeline exited before writing its CSV. Check `<out_dir>/structure/logs/`. |
| `no usable wildtype baseline` | AF3/PISA failed for the wildtype. The error names whether the `WT` row was absent or all-NaN, and points at the log; check for `RESOURCE_EXHAUSTED` and lower `max_parallel`. |
| `RESOURCE_EXHAUSTED: Failed to allocate request for N GiB` | Too many concurrent AF3 workers for the complex size. Lower `structure.max_parallel` (see §7.5), then clear `.failed` markers before retrying. |
| Rerun reproduces a failure exactly, skipping mutants | `.failed` markers from the earlier run. Set `structure.retry_failed: true` or delete `<out_dir>/structure/*.failed`. |
| `Pipeline results are missing metric column(s)` | `structure.adapter.metric_columns` does not match the pipeline's output. |
| `No AF3 model found for mutant … (job …)` | AF3 failed for that mutant; look for a `.failed` marker in `<out_dir>/structure/`. |
| Everything fails the RMSD gate (`rmsd_fail` = folded count) | Most often `rmsd_gate.scope: complex` on a multi-domain target: domain motion away from the interface dominates and a point mutant reads >10 Å. Use `scope: interface`. Otherwise check `af3_chain_map`, or loosen `max_rmsd` to diagnose. |
| `AF3 model … encodes a different sequence than requested` | Chain relabeling — set `structure.af3_chain_map`. |
| `Backbone … chain lengths do not match expected` | AF3 output layout differs from the target; mutation positions would misalign. |
| `selection.scope='pooled' ranks PottsMPNN energies across different structures` | Use `per_seed`, or `promote_top_n: 1`. |
| `CSV appears mis-parsed: … non-numeric stability_score` | An unquoted multi-mutation field. The loop quotes automatically; only hand-edited CSVs hit this. |
| AF3 crashes with `stream != nullptr` | Too many concurrent workers for GPU memory. Lower `structure.max_parallel`. |

---

### 9.10 Reading the results

With `run.report: true` (the default) the loop writes `<out_dir>/report/` when it
finishes — however it finishes, including an early stop:

| File | Contents |
|---|---|
| `report.html` | Self-contained: plots inlined as base64, no external files. `scp` it and open. |
| `best_mutants.csv` | Every unique mutant, ranked, Pareto-front members flagged. |

Four plots:

1. **Progression by round** — per-round distribution of each metric with the
   round median and the cumulative best overlaid, and the wildtype baseline
   marked. Shows whether the loop is still gaining: the cumulative best is
   monotone by construction and can be carried by one lucky mutant, so the
   median is what tells you the population as a whole is improving.
2. **Metric trade-off** — every candidate on ipSAE vs PISA, coloured by round,
   with the Pareto front traced and wildtype starred.
3. **Lineage of the best mutants** — each top mutant traced back through its
   ancestors via `parent_seed_id`, plotted against the round each ancestor was
   scored in. A monotone climb means the loop was genuinely optimizing; a flat
   or wandering trace means later rounds added mutations without buying
   anything.
4. **Mutation convergence** — which substitutions recur among winners, round by
   round.

Ranking is the mean normalized rank across the gated metrics, with Pareto-front
members listed first. Metric names *and directions* are discovered from the data
(the `beats_wt_<metric>` columns plus the wildtype baseline), so a run gated on
`dG_diss` reports correctly with no extra configuration.

Regenerate at any time, including for a run that predates this feature:

```bash
python -m optimize.report outputs/example_output --top 30
```

Set `report_each_round: true` to refresh it after every round and watch a long
run in flight. Report generation never sinks a run: results are already on disk,
so a plotting failure is warned about and swallowed.

### 9.11 Tests

The loop's logic is testable without torch, a GPU, or cluster access — PottsMPNN
scoring and the AF3 pipeline are stubbed, while backbone preparation, selection,
gating, and AF3 output-layout logic all run for real:

```bash
python -m optimize.tests.test_optimize
```

See [`optimize/README.md`](optimize/README.md) for developer-facing detail on how
the package is put together and which parts are coupled to
`run_mutation_af3_pipeline.py`.

---

### Sources

- AlphaFold3 — [github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3), [installation docs](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md)
- ipSAE — [github.com/DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE), Dunbrack (2025), [*Rēs ipSAE loquunt*](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1)
- PISA — [CCP4 program docs](https://www.ccp4.ac.uk/html/pisa.html), [CCP4 suite](https://www.ccp4.ac.uk/), [user manual](https://ftp.ccp4.ac.uk/ekr/pisa/PISA_1.11.pdf); Krissinel & Henrick (2007)

"""Summarize a finished optimization run: progression plots and a ranked table.

Reads a run's ``round_*/round_summary.csv`` plus ``run_state.json`` and writes a
self-contained HTML report (plots inlined, no external files) alongside a ranked
CSV of every candidate ever folded.

Metric names and directions are discovered from the data: the gating stage
writes a ``beats_wt_<metric>`` column per metric, and the wildtype baseline in
``run_state.json`` fixes which way is better, so a run gated on ``dG_diss``
reports correctly without reconfiguration.

Usage::

    python -m optimize.report outputs/350d_binder12_trunc_optimization
    python -m optimize.report <out_dir> --top 30
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Compute nodes have no display; select the non-interactive backend before
# pyplot is imported anywhere.
import matplotlib

matplotlib.use("Agg")

WT_LABEL = "WT"


# ------------------------------------------------------------------- loading


@dataclass
class RunData:
    frame: pd.DataFrame
    baseline: Dict[str, float]
    directions: Dict[str, str]
    metrics: List[str]
    out_dir: Path
    termination: Optional[str] = None
    # seed_id -> seed record from run_state.json, for lineage reconstruction
    seeds: Dict[str, dict] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = {}

    @property
    def rounds(self) -> List[int]:
        return sorted(self.frame["round"].unique().tolist())


def _infer_directions(frame: pd.DataFrame, metrics, baseline) -> Dict[str, str]:
    """Which way is better, deduced from who was flagged as beating wildtype."""
    directions: Dict[str, str] = {}
    for metric in metrics:
        flag = f"beats_wt_{metric}"
        ref = baseline.get(metric)
        if flag not in frame.columns or ref is None:
            directions[metric] = "max"
            continue
        winners = pd.to_numeric(frame.loc[frame[flag] == True, metric], errors="coerce").dropna()
        if winners.empty:
            directions[metric] = "max"
        else:
            directions[metric] = "max" if (winners > ref).mean() >= 0.5 else "min"
    return directions


def load_run(out_dir: Path) -> RunData:
    out_dir = Path(out_dir)
    frames = []
    for path in sorted(out_dir.glob("round_*/round_summary.csv"),
                       key=lambda p: int(p.parent.name.split("_")[1])):
        df = pd.read_csv(path)
        if "round" not in df.columns:
            df["round"] = int(path.parent.name.split("_")[1])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No round_*/round_summary.csv under {out_dir}. Is this an optimization out_dir?"
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["round"] = frame["round"].astype(int)

    baseline: Dict[str, float] = {}
    termination = None
    seeds: Dict[str, dict] = {}
    state_path = out_dir / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        baseline = dict(state.get("wt_baseline") or {})
        termination = state.get("termination_reason")
        for record in state.get("rounds", []):
            for seed in record.get("seeds", []):
                seeds[seed["seed_id"]] = seed

    metrics = [c[len("beats_wt_"):] for c in frame.columns if c.startswith("beats_wt_")]
    metrics = [m for m in metrics if m in frame.columns]
    if not metrics:
        metrics = [m for m in ("ipsae", "dG_binding", "dG_diss") if m in frame.columns]

    return RunData(frame=frame, baseline=baseline,
                   directions=_infer_directions(frame, metrics, baseline),
                   metrics=metrics, out_dir=out_dir, termination=termination, seeds=seeds)


def lineage(run: RunData, row: pd.Series) -> List[dict]:
    """Ancestry of one candidate, oldest first.

    Each promoted seed records its ``parent_seed_id``, so a final mutant walks
    back to the wildtype. A seed with ``round_index == k`` was scored as a
    candidate in round ``k - 1``, so its metrics belong to that round.
    """
    chain: List[dict] = [{
        "round": int(row["round"]),
        "mutations": str(row.get("mutations", "")),
        "metrics": {m: row[m] for m in run.metrics if m in row and pd.notna(row[m])},
        "is_final": True,
    }]
    seed_id = row.get("seed_id")
    seen = set()
    while seed_id and seed_id in run.seeds and seed_id not in seen:
        seen.add(seed_id)
        seed = run.seeds[seed_id]
        metrics = {k: v for k, v in (seed.get("metrics") or {}).items() if v is not None}
        if metrics:
            chain.append({
                "round": int(seed.get("round_index", 0)) - 1,
                "mutations": ",".join(seed.get("mutations") or []),
                "metrics": metrics,
                "is_final": False,
            })
        seed_id = seed.get("parent_seed_id")
    return sorted(chain, key=lambda p: p["round"])


# ------------------------------------------------------------------ ranking


def _oriented(values: pd.Series, direction: str) -> pd.Series:
    """Return values flipped so that LOWER is always better."""
    numeric = pd.to_numeric(values, errors="coerce")
    return -numeric if direction == "max" else numeric


def pareto_front(frame: pd.DataFrame, metrics, directions) -> np.ndarray:
    """Non-dominated mask over the metrics, each oriented to 'lower is better'."""
    cols = [_oriented(frame[m], directions[m]).to_numpy(dtype=float) for m in metrics]
    if not cols:
        return np.zeros(len(frame), dtype=bool)
    points = np.column_stack(cols)
    ok = ~np.isnan(points).any(axis=1)
    front = np.zeros(len(frame), dtype=bool)
    idx = np.flatnonzero(ok)
    for i in idx:
        others = points[ok]
        dominated = ((others <= points[i]).all(axis=1) & (others < points[i]).any(axis=1)).any()
        front[i] = not dominated
    return front


def rank_candidates(run: RunData) -> pd.DataFrame:
    """One row per unique mutant, best first, with Pareto and rank annotations."""
    frame = run.frame.copy()
    # The same mutant can appear in several rounds (cache hits); keep the
    # earliest, which is where it was produced.
    if "mutations" in frame.columns:
        frame = frame.sort_values("round").drop_duplicates(subset="mutations", keep="first")

    ranks = []
    for metric in run.metrics:
        ranks.append(_oriented(frame[metric], run.directions[metric]).rank(pct=True,
                                                                          na_option="bottom"))
    frame["_score"] = pd.concat(ranks, axis=1).mean(axis=1) if ranks else 0.0
    frame["pareto"] = pareto_front(frame, run.metrics, run.directions)
    frame["n_mutations"] = frame.get("mutations", pd.Series("", index=frame.index)).fillna("").map(
        lambda m: 0 if str(m).strip() in ("", "nan") else len(str(m).split(","))
    )
    # Pareto members first, then by mean normalized rank.
    return frame.sort_values(["pareto", "_score"], ascending=[False, True]).reset_index(drop=True)


# -------------------------------------------------------------------- plots


def _png(fig, save_to: Optional[Path] = None) -> str:
    """Encode a figure for inlining, and optionally also write it to disk."""
    import matplotlib.pyplot as plt

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, format="png", dpi=130, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_round_scatter(run: RunData, rnd: Optional[int] = None,
                       save_to: Optional[Path] = None) -> Optional[str]:
    """Trade-off scatter for one round, coloured by cumulative mutation count.

    ``rnd=None`` plots every round together. Points are shaded by how many
    mutations they carry relative to the original wildtype, and the quadrant
    that beats wildtype on both metrics is highlighted.
    """
    if len(run.metrics) < 2:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    y_metric, x_metric = run.metrics[0], run.metrics[1]
    frame = run.frame if rnd is None else run.frame[run.frame["round"] == rnd]
    x = pd.to_numeric(frame[x_metric], errors="coerce")
    y = pd.to_numeric(frame[y_metric], errors="coerce")
    ok = x.notna() & y.notna()
    if not ok.any():
        return None
    x, y, frame = x[ok], y[ok], frame[ok]

    counts = frame.get("mutations", pd.Series("", index=frame.index)).fillna("").map(
        lambda m: 0 if str(m).strip() in ("", "nan") else len(str(m).split(","))
    )

    wx, wy = run.baseline.get(x_metric), run.baseline.get(y_metric)
    n_better = 0
    if wx is not None and wy is not None:
        bx = x < wx if run.directions[x_metric] == "min" else x > wx
        by = y < wy if run.directions[y_metric] == "min" else y > wy
        n_better = int((bx & by).sum())

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    palette = ["#4c78a8", "#f58518", "#54a24b", "#b82e2e", "#9467bd", "#8c564b", "#17becf"]
    for i, k in enumerate(sorted(counts.unique())):
        sel = counts == k
        ax.scatter(x[sel], y[sel], s=52, alpha=0.85, color=palette[i % len(palette)],
                   edgecolor="black", linewidth=0.5,
                   label=f"{int(k)} mutation" + ("" if k == 1 else "s"))

    if wx is not None and wy is not None:
        # Shade the quadrant that beats wildtype on both metrics.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        rx = (min(x0, wx), wx) if run.directions[x_metric] == "min" else (wx, max(x1, wx))
        ry = (min(y0, wy), wy) if run.directions[y_metric] == "min" else (wy, max(y1, wy))
        ax.add_patch(plt.Rectangle((rx[0], ry[0]), rx[1] - rx[0], ry[1] - ry[0],
                                   facecolor="#2ca02c", alpha=0.07, zorder=0))
        ax.axvline(wx, ls="--", lw=1, color="#555", zorder=1)
        ax.axhline(wy, ls="--", lw=1, color="#555", zorder=1)
        ax.scatter([wx], [wy], marker="*", s=340, facecolor="white", edgecolor="black",
                   linewidth=1.3, zorder=6)
        ax.annotate(f"WT\n({y_metric}={wy:.3f}, {x_metric}={wx:.2f})",
                    xy=(wx, wy), xytext=(14, 26), textcoords="offset points",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    arrow_x = "← better" if run.directions[x_metric] == "min" else "better →"
    arrow_y = "better →" if run.directions[y_metric] == "max" else "← better"
    ax.set_xlabel(f"{x_metric}   ({arrow_x})")
    ax.set_ylabel(f"{y_metric}   ({arrow_y})")
    label = "all rounds" if rnd is None else f"round {rnd}"
    ax.set_title(f"{run.out_dir.name}  —  {label}   "
                 f"(n={len(frame)}, better-than-WT={n_better})")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", color="none", markerfacecolor="white",
                          markeredgecolor="black", markersize=15))
    labels.append("WT (reference)")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=min(5, len(labels)), fontsize=8, frameon=True)
    ax.grid(alpha=0.25)
    return _png(fig, save_to)


def plot_progression(run: RunData, save_to: Optional[Path] = None) -> str:
    import matplotlib.pyplot as plt

    metrics = run.metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes[0], metrics):
        better_high = run.directions[metric] == "max"
        best_so_far, medians, xs = [], [], []
        for rnd in run.rounds:
            vals = pd.to_numeric(run.frame.loc[run.frame["round"] == rnd, metric],
                                 errors="coerce").dropna()
            if vals.empty:
                continue
            ax.scatter(rnd + rng.uniform(-0.13, 0.13, len(vals)), vals,
                       s=16, alpha=0.45, color="#4c78a8", edgecolor="none")
            best = vals.max() if better_high else vals.min()
            best_so_far.append(best if not best_so_far else
                               (max(best_so_far[-1], best) if better_high
                                else min(best_so_far[-1], best)))
            # Per-round median, not a running one: it tracks whether the whole
            # population is improving, which the monotone cumulative best cannot
            # show.
            medians.append(vals.median())
            xs.append(rnd)
        if xs:
            ax.plot(xs, medians, "-s", color="#54a24b", lw=2, ms=5,
                    label="round median", zorder=4)
            ax.plot(xs, best_so_far, "-o", color="#f58518", lw=2, ms=6,
                    label="cumulative best", zorder=5)
        ref = run.baseline.get(metric)
        if ref is not None:
            ax.axhline(ref, ls="--", color="#333", lw=1.2, label=f"wildtype ({ref:.3f})")
        ax.set_xlabel("round")
        ax.set_ylabel(f"{metric}  ({'higher' if better_high else 'lower'} is better)")
        ax.set_xticks(run.rounds)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Per-round distribution, median and cumulative best", y=1.02)
    return _png(fig, save_to)


def plot_pareto(run: RunData, ranked: pd.DataFrame,
                save_to: Optional[Path] = None) -> Optional[str]:
    if len(run.metrics) < 2:
        return None
    import matplotlib.pyplot as plt

    x, y = run.metrics[1], run.metrics[0]   # dG on x, ipSAE on y by convention
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    cmap = plt.get_cmap("viridis")
    rounds = run.rounds
    for rnd in rounds:
        sub = ranked[ranked["round"] == rnd]
        ax.scatter(pd.to_numeric(sub[x], errors="coerce"),
                   pd.to_numeric(sub[y], errors="coerce"),
                   s=42, alpha=0.75, edgecolor="white", linewidth=0.5,
                   color=cmap(rnd / max(1, max(rounds))), label=f"round {rnd}")
    front = ranked[ranked["pareto"]]
    if not front.empty:
        order = front.sort_values(x)
        ax.plot(pd.to_numeric(order[x], errors="coerce"),
                pd.to_numeric(order[y], errors="coerce"),
                "-", color="#d62728", lw=1.4, alpha=0.8, zorder=4)
        ax.scatter(pd.to_numeric(front[x], errors="coerce"),
                   pd.to_numeric(front[y], errors="coerce"),
                   s=110, facecolor="none", edgecolor="#d62728", linewidth=1.8,
                   label="Pareto front", zorder=5)
    bx, by = run.baseline.get(x), run.baseline.get(y)
    if bx is not None and by is not None:
        ax.scatter([bx], [by], marker="*", s=420, color="white", edgecolor="black",
                   linewidth=1.3, zorder=6, label="wildtype")
        ax.axvline(bx, ls="--", lw=1, color="#888")
        ax.axhline(by, ls="--", lw=1, color="#888")
    ax.set_xlabel(f"{x}  ({'higher' if run.directions[x]=='max' else 'lower'} is better)")
    ax.set_ylabel(f"{y}  ({'higher' if run.directions[y]=='max' else 'lower'} is better)")
    ax.set_title("All folded candidates")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    return _png(fig, save_to)


def plot_lineage(run: RunData, ranked: pd.DataFrame, top: int = 6,
                 save_to: Optional[Path] = None) -> Optional[str]:
    """How each of the best final mutants got there, ancestor by ancestor.

    A flat or wandering trace means later rounds added mutations without buying
    anything; a monotone climb means the loop was optimizing.
    """
    if not run.seeds:
        return None
    import matplotlib.pyplot as plt

    best = ranked.head(top)
    chains = [(row, lineage(run, row)) for _, row in best.iterrows()]
    chains = [(r, c) for r, c in chains if len(c) >= 2]
    if not chains:
        return None

    metrics = run.metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), 4.8), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for ax, metric in zip(axes[0], metrics):
        for i, (row, chain) in enumerate(chains):
            xs = [p["round"] for p in chain if metric in p["metrics"]]
            ys = [float(p["metrics"][metric]) for p in chain if metric in p["metrics"]]
            if len(xs) < 2:
                continue
            label = str(row.get("mutations", ""))
            if len(label) > 34:
                label = label[:31] + "..."
            ax.plot(xs, ys, "-o", color=cmap(i % 10), lw=1.8, ms=5, alpha=0.9, label=label)
            ax.scatter([xs[-1]], [ys[-1]], s=90, facecolor="none",
                       edgecolor=cmap(i % 10), linewidth=1.8, zorder=5)
        ref = run.baseline.get(metric)
        if ref is not None:
            ax.axhline(ref, ls="--", color="#333", lw=1.2, label="wildtype")
        better_high = run.directions[metric] == "max"
        ax.set_xlabel("round the ancestor was scored in")
        ax.set_ylabel(f"{metric}  ({'higher' if better_high else 'lower'} is better)")
        ax.set_xticks(run.rounds)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle(f"Lineage of the top {len(chains)} mutants", y=1.02)
    return _png(fig, save_to)


def plot_mutation_frequency(run: RunData, top: int = 20,
                            save_to: Optional[Path] = None) -> Optional[str]:
    """Which substitutions the optimizer converged on, round by round."""
    if "mutations" not in run.frame.columns:
        return None
    import matplotlib.pyplot as plt
    from collections import Counter

    winners = run.frame[run.frame.get("beats_wt", False) == True] \
        if "beats_wt" in run.frame.columns else run.frame
    if winners.empty:
        winners = run.frame
    per_round: Dict[int, Counter] = {}
    total = Counter()
    for rnd in run.rounds:
        c = Counter()
        for muts in winners.loc[winners["round"] == rnd, "mutations"].fillna(""):
            for tok in str(muts).split(","):
                tok = tok.strip()
                if tok and tok.lower() != "nan":
                    c[tok] += 1
        per_round[rnd] = c
        total.update(c)
    if not total:
        return None

    labels = [m for m, _ in total.most_common(top)]
    matrix = np.array([[per_round[r][m] for r in run.rounds] for m in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(1.15 * len(run.rounds) + 4.2, 0.34 * len(labels) + 1.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(run.rounds)), [str(r) for r in run.rounds])
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("round")
    ax.set_title(f"Substitutions among winners (top {len(labels)})")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="count", shrink=0.8)
    return _png(fig, save_to)


# --------------------------------------------------------------------- html


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    columns = [c for c in columns if c in frame.columns]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                v = "" if pd.isna(v) else f"{v:.4g}"
            elif isinstance(v, (bool, np.bool_)):
                v = "yes" if v else ""
            cells.append(f"<td>{html.escape(str(v))}</td>")
        cls = ' class="pareto"' if row.get("pareto") else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_html(run: RunData, ranked: pd.DataFrame, images: Dict[str, Optional[str]],
               top: int) -> str:
    cols = ["round", "mutations", "n_mutations", *run.metrics, "rmsd", "beats_wt", "pareto",
            "seed_id"]
    best = ranked.head(top)

    summary_rows = []
    for rnd in run.rounds:
        sub = run.frame[run.frame["round"] == rnd]
        cells = [str(rnd), str(len(sub)),
                 str(int(sub["beats_wt"].sum())) if "beats_wt" in sub else "-"]
        for m in run.metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            if vals.empty:
                cells.append("-")
            else:
                cells.append(f"{(vals.max() if run.directions[m]=='max' else vals.min()):.4g}")
        summary_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    summary_head = "".join(f"<th>{h}</th>" for h in
                           ["round", "scored", "winners"] + [f"best {m}" for m in run.metrics])

    wt = ", ".join(f"{m} = {run.baseline[m]:.4g}" for m in run.metrics if m in run.baseline)
    figs = "".join(
        f'<h2>{html.escape(title)}</h2><img src="data:image/png;base64,{data}" alt="{html.escape(title)}">'
        for title, data in images.items() if data
    )
    return f"""<meta charset="utf-8"><title>Optimization report - {html.escape(run.out_dir.name)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto; max-width: 1100px;
        line-height: 1.5; color: #1a1a1a; }}
 h1 {{ margin-bottom: .2rem; }} .sub {{ color: #666; margin-bottom: 1.5rem; }}
 img {{ max-width: 100%; height: auto; border: 1px solid #e5e5e5; border-radius: 6px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: .6rem 0 1.6rem; }}
 th, td {{ border-bottom: 1px solid #e8e8e8; padding: 5px 8px; text-align: left; white-space: nowrap; }}
 th {{ background: #f6f6f6; position: sticky; top: 0; }}
 tr.pareto {{ background: #fff6f4; }}
 tr.pareto td:first-child::after {{ content: " *"; color: #d62728; font-weight: bold; }}
 .wrap {{ overflow-x: auto; }} code {{ background:#f2f2f2; padding:1px 4px; border-radius:3px; }}
</style>
<h1>Optimization report</h1>
<div class="sub">{html.escape(str(run.out_dir))} &middot; {len(run.rounds)} rounds &middot;
 {len(run.frame)} folded candidates &middot; {len(ranked)} unique mutants &middot;
 terminated: <code>{html.escape(str(run.termination))}</code></div>
<p><strong>Wildtype baseline:</strong> {html.escape(wt) or "not recorded"}</p>
<h2>Per-round summary</h2>
<div class="wrap"><table><thead><tr>{summary_head}</tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
{figs}
<h2>Top {len(best)} mutants</h2>
<p>Ranked by mean normalized rank across {', '.join(run.metrics)}; Pareto-front members
 (marked <span style="color:#d62728">*</span>) are listed first. Full table in
 <code>best_mutants.csv</code>.</p>
<div class="wrap">{_table(best, cols)}</div>
"""


# ---------------------------------------------------------------------- main


def write_fasta(ranked: pd.DataFrame, path: Path, top: int, run: RunData) -> Optional[Path]:
    """Top mutants as FASTA."""
    if "sequence" not in ranked.columns:
        return None
    lines = []
    for i, (_, row) in enumerate(ranked.head(top).iterrows(), start=1):
        muts = str(row.get("mutations", "")) or "none"
        stats = " ".join(f"{m}={row[m]:.4g}" for m in run.metrics
                         if m in row and pd.notna(row[m]))
        flag = " pareto" if row.get("pareto") else ""
        lines.append(f">rank{i:03d} round={int(row['round'])} {stats}{flag} mutations={muts}")
        lines.append(str(row["sequence"]))
    if not lines:
        return None
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate(out_dir: Path, top: int = 25) -> Path:
    run = load_run(out_dir)
    ranked = rank_candidates(run)

    report_dir = Path(out_dir) / "report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Every scored row, all rounds, no dedup.
    run.frame.to_csv(report_dir / "all_candidates.csv", index=False)
    # Unique mutants, ranked, Pareto flagged.
    ranked.drop(columns=[c for c in ("_score",) if c in ranked.columns]) \
          .to_csv(report_dir / "best_mutants.csv", index=False)
    fasta = write_fasta(ranked, report_dir / "top_sequences.fasta", top, run)

    images: Dict[str, Optional[str]] = {
        "Progression by round": plot_progression(run, plots_dir / "progression.png"),
        "Metric trade-off (all rounds)":
            plot_round_scatter(run, None, plots_dir / "tradeoff_all_rounds.png"),
        "Pareto front": plot_pareto(run, ranked, plots_dir / "pareto.png"),
        "Lineage of the best mutants": plot_lineage(run, ranked, save_to=plots_dir / "lineage.png"),
        "Mutation convergence":
            plot_mutation_frequency(run, save_to=plots_dir / "mutation_convergence.png"),
    }
    for rnd in run.rounds:
        images[f"Round {rnd}"] = plot_round_scatter(
            run, rnd, plots_dir / f"round_{rnd:02d}_tradeoff.png"
        )

    html_path = report_dir / "report.html"
    html_path.write_text(build_html(run, ranked, images, top), encoding="utf-8")

    n_plots = sum(1 for v in images.values() if v)
    print(f"run      : {out_dir}")
    print(f"  rounds : {run.rounds}   terminated: {run.termination}")
    print(f"  metrics: {', '.join(f'{m} ({run.directions[m]})' for m in run.metrics)}")
    print(f"  folded : {len(run.frame)} rows -> {len(ranked)} unique mutants")
    print(f"  pareto : {int(ranked['pareto'].sum())} non-dominated")
    print(f"\nwrote {report_dir}/")
    print(f"  report.html            self-contained, all {n_plots} plots inlined")
    print(f"  all_candidates.csv     {len(run.frame)} rows, every scored candidate in every round")
    print(f"  best_mutants.csv       {len(ranked)} unique mutants, ranked")
    if fasta:
        print(f"  top_sequences.fasta    top {min(top, len(ranked))} sequences")
    print(f"  plots/                 {n_plots} PNGs, one per round plus the summaries")

    best = ranked.head(min(top, 10))
    show = [c for c in ["round", "mutations", *run.metrics, "rmsd", "pareto"]
            if c in best.columns]
    print(f"\ntop {len(best)} mutants:")
    print(best[show].to_string(index=False))
    return html_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="The run's out_dir.")
    ap.add_argument("--top", type=int, default=25, help="Rows in the HTML table (default 25).")
    args = ap.parse_args()
    try:
        generate(args.out_dir, args.top)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

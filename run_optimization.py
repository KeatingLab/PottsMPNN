"""Master script for iterative Pareto optimization of binder designs.

Runs, for a configurable number of iterations:

    mutation search (stability x binding Pareto)
        -> select folding set (objective + constraints + diversity)
        -> AF3 + PISA + ipSAE
        -> promote mutants that beat wildtype on the structural metrics
        -> re-seed the next round from them

and stops early when the configured ipSAE and/or PISA cutoffs are reached.

Usage::

    python run_optimization.py --config inputs/example_config_optimization.yaml
    python run_optimization.py --config <cfg> run.max_iterations=5 run.force=true

Follows the repository's config convention (see ``sample_seqs.py`` and
``energy_prediction.py``): a single ``--config`` YAML, loaded with OmegaConf.
Trailing ``key=value`` arguments override any config field.
"""

from __future__ import annotations

import argparse
import sys

from optimize.config import load_config
from optimize.loop import run_optimization


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iterative mutation search + AF3/PISA/ipSAE optimization loop."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the optimization YAML.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotlist overrides, e.g. run.max_iterations=5 gating.promote_top_n=3",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully merged config and exit without running.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg))
        return 0

    run_optimization(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
#SBATCH --job-name=potts_opt
#SBATCH --mem=180G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=40
#SBATCH -p pi_keating
#SBATCH -o optimization_%x_%j.out
#SBATCH -e optimization_%x_%j.err

# Outer launcher for the iterative optimization loop (run_optimization.py).
#
# This submits the ORCHESTRATOR itself as a single GPU job. run_optimization.py
# does not self-submit: it runs the PottsMPNN mutation search in-process (needs a
# GPU) and drives the AF3 + PISA + ipSAE stage. So the whole loop must live
# inside one allocation on a GPU node -- launching it on a login node would run
# the GPU search on the login node.
#
# Default topology: run.executor=local, so the AF3/PISA pipeline runs as
# subprocesses on THIS node, sharing THIS GPU (like
# submit_mutation_af3_pipeline_engaging.sh). Tune structure.max_parallel in the
# config to what the GPU's memory can hold.
#
# Alternative topology: set run.executor=slurm in the config to have the AF3
# stage submit its own sbatch array to other nodes while this job polls. That
# frees other nodes for AF3 but leaves this GPU idle between search phases, and
# this job must still survive the entire run -- keep --time generous either way.
# The sbatch directives for that inner array come from the config's slurm: block,
# NOT from this script.
#
# Usage:
#   sbatch submit_optimization.sh [CONFIG] [extra run_optimization overrides...]
#
# Examples:
#   sbatch submit_optimization.sh inputs/my_opt.yaml
#   sbatch submit_optimization.sh inputs/my_opt.yaml run.max_iterations=5 gating.promote_top_n=3
#
# The log files land in the directory you submit from (optimization_<name>_<jobid>.out/.err).

set -euo pipefail

# --- environment (match the other submit scripts) --------------------------
CONDA_ROOT="${CONDA_ROOT:-$(conda info --base)}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate PottsMPNN
module load apptainer/1.4.2   # AF3 runs via Singularity/Apptainer under executor=local

CODE_DIR="/orcd/pool/005/keating_shared/fosterb/PottsMPNN"

# Relative paths in the configs (run.out_dir, search.cfg_path, and the model
# checkpoint inside it) resolve against the working directory, so pin it rather
# than inheriting wherever sbatch happened to be invoked from.
cd "${CODE_DIR}"

# --- config + pass-through overrides ---------------------------------------
CONFIG="${1:-inputs/example_config_optimization.yaml}"
shift || true   # remaining args (if any) are forwarded as dotlist overrides

if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${CODE_DIR}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
fi

echo "================================================================"
echo "Iterative optimization"
echo "  node     : $(hostname)"
echo "  gpu      : ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "  config   : ${CONFIG}"
echo "  overrides: $*"
echo "================================================================"

# Fail fast if the config is invalid, before any GPU time is spent.
python "${CODE_DIR}/run_optimization.py" --config "${CONFIG}" "$@" --print-config >/dev/null

python "${CODE_DIR}/run_optimization.py" --config "${CONFIG}" "$@"

echo "Optimization finished."

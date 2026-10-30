#!/usr/bin/env bash
# Submit from the repository root; create logs/ before calling sbatch.
# Supply the allocation with: sbatch --account=YOUR_ACCOUNT scripts/hpc_train.sbatch
#SBATCH --job-name=cifar10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${CIFAR_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit with sbatch}}"

if [[ -n "${CIFAR_IMAGE:-}" ]]; then
  # Use an existing site-provided CUDA image and your prepared Python overlay.
  runtime="${CIFAR_CONTAINER_RUNTIME:-apptainer}"
  if ! command -v "$runtime" >/dev/null 2>&1; then
    runtime=singularity
  fi
  options=(exec --nv --bind "$PWD:$PWD")
  if [[ -n "${CIFAR_OVERLAY:-}" ]]; then
    options+=(--overlay "${CIFAR_OVERLAY}:ro")
  fi
  # source a fixed positional argument; do not interpolate shell command text.
  srun "$runtime" "${options[@]}" "$CIFAR_IMAGE" /bin/bash -c \
    'set -e; source "$1"; shift; exec bash "$@"' bash \
    "${CIFAR_ENV_SCRIPT:-/ext3/env.sh}" "$PWD/scripts/hpc_run.sh" "$@"
else
  # CIFAR_PYTHON should point to an existing, site-compatible Linux environment.
  srun bash scripts/hpc_run.sh "$@"
fi

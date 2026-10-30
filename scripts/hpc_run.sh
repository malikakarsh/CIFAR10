#!/usr/bin/env bash
# Run inside a Slurm GPU allocation, optionally inside an NYU container.
set -euo pipefail
cd "${CIFAR_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
python_bin="${CIFAR_PYTHON:-python}"
plan="${1:-configs/generalization_plan.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
data_root="${CIFAR_DATA_ROOT:-$PWD/dataset}"
run_root="${CIFAR_RUN_ROOT:-$PWD/runs_hpc}"
mkdir -p "$run_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$python_bin" - <<'PY'
import torch, torchvision
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: use a GPU allocation and a CUDA-enabled PyTorch environment.')
x = torch.ones(1, device='cuda')
print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0), 'CUDA build:', torch.version.cuda)
print('CUDA allocation check:', x.item())
PY
exec "$python_bin" -m cifar10 plan --config "$plan" \
  --readme "$run_root/report.md" \
  --set training.device=cuda data.num_workers=4 \
  "data.root=$data_root" "experiment.output_dir=$run_root" "$@"

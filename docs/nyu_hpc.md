# Training on NYU HPC

Use a Slurm GPU job with CUDA. The instructions below assume **Torch**, based on NYU's current public documentation. Greene and course allocations may have different login hosts, accounts, and environment setup; confirm those before submitting. No cluster connection or job submission has been performed from this workspace.

**Burst Open OnDemand:** if accessing `ood.burst.hpc.nyu.edu` through Code Server, use the account and GPU partition assigned to that Burst allocation. Inside an allocated GPU session, run the Python commands in the [staged experiment plan](experiment_plan.md) directly. The Torch hostnames and account instructions below do not establish your Burst settings.

## 1. Connect and identify your allocation

Connect to the NYU network or VPN, then use your assigned login hostname. For Torch:

```bash
ssh NETID@login.torch.hpc.nyu.edu
my_slurm_accounts
```

Replace `NETID`, `YOUR_ACCOUNT`, and environment paths below with your own values. On Torch, every submitted job needs an allocation account; do not add a partition unless your allocation specifically requires it.

## 2. Transfer the project and permitted dataset

On the cluster:

```bash
mkdir -p /scratch/$USER/cifar10
cd /scratch/$USER/cifar10
git clone https://github.com/malikakarsh/CIFAR10.git .
mkdir -p logs
```

Push the new HPC scripts to your repository first, or copy the code directly. Dataset files are Git-ignored and must be transferred separately. From the Mac, in the local repository:

```bash
tar -czf /private/tmp/cifar10-dataset.tar.gz dataset
scp /private/tmp/cifar10-dataset.tar.gz NETID@dtn.torch.hpc.nyu.edu:/scratch/NETID/cifar10/
```

Back on the cluster:

```bash
cd /scratch/$USER/cifar10
tar -xzf cifar10-dataset.tar.gz
```

NYU recommends its data-transfer nodes (DTNs) for transfers. Keep data and checkpoints on scratch, and copy important results to backed-up storage: scratch is not backed up and is subject to cleanup.

## 3. Prepare a Linux CUDA environment

Do not copy the Mac `.venv`. The Mac setup checker and pinned requirements target that environment. Use a cluster-compatible Python 3.12 environment and a matched CUDA PyTorch/torchvision build supported by the allocated GPU and driver.

NYU recommends Apptainer/Singularity with a Conda overlay; follow its [environment setup guide](https://services.rt.nyu.edu/docs/hpc/containers/apptainer_with_conda/) linked from the [single-GPU tutorial](https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/pytorch_intro/). Use the actual image and overlay paths supplied by NYU or your course, rather than assuming an old CUDA image path exists. On Torch, the documented setup uses an interactive allocation and a writable overlay with `--fakeroot`; install the project inside that same environment.

Inside that prepared environment, from the repository root:

```bash
python -m pip install -e .
python -m pip check
python -m cifar10 check-data --config configs/base.yaml
python -m unittest discover -s tests -v
```

The project dependency ranges are in `pyproject.toml`; a current compatible preinstalled torch/torchvision pair satisfies them without downloading replacements. If torch/torchvision are absent, install a matched CUDA build using the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the GPU/driver environment first. Do installations in the site's supported setup session, before the batch job; do not assume compute-node internet access.

For an Apptainer/overlay environment, export these on the login node after setup:

```bash
export CIFAR_IMAGE=/absolute/path/to/nyu-cuda-image.sif
export CIFAR_OVERLAY=/absolute/path/to/your-python-overlay.ext3
export CIFAR_ENV_SCRIPT=/ext3/env.sh
# If your cluster calls the runtime singularity:
export CIFAR_CONTAINER_RUNTIME=singularity
```

The batch script mounts the overlay read-only and sources its environment. Install packages beforehand with the site's documented writable-overlay setup, then exit it before submitting. If dataset/output paths are outside standard site container binds, add their bind mounts to the script.

Alternatively, if your course provides a directly usable Linux environment, select its interpreter without container variables:

```bash
export CIFAR_PYTHON=/absolute/path/to/linux-environment/bin/python
```

## 4. Submit the augmentation experiments

From the cluster repository root, after selecting the environment:

```bash
mkdir -p logs
sbatch --account=YOUR_ACCOUNT scripts/hpc_train.sbatch
```

This requests **one GPU, six CPU cores, 16 GB RAM, and a 12-hour time limit**. These are starting resource requests, not measured runtime predictions. No particular GPU model is assumed. The default job runs `configs/generalization_plan.yaml`: RandAugment alone followed by RandAugment + mixup, 200 epochs each on the 2.44M wider SGD architecture.

The script overrides `training.device=cuda`, uses four data-loader workers, and retains the configured batch size. CUDA mixed precision is enabled by the inherited `training.amp=true`; numerical behavior differs from the float32 Mac runs. To compare in float32, pass `training.amp=false` using the command below and keep that choice unchanged when resubmitting.

Select another plan, or pass config overrides, after the script path:

```bash
# Optional: the original eight-experiment plan starts as new CUDA experiments.
sbatch --account=YOUR_ACCOUNT scripts/hpc_train.sbatch configs/all_experiments.yaml

# Short smoke run in a separate output directory; does not change full-run settings.
export CIFAR_RUN_ROOT="$PWD/runs_hpc/smoke"
sbatch --account=YOUR_ACCOUNT --time=00:20:00 scripts/hpc_train.sbatch \
  configs/generalization_plan.yaml training.epochs=2 scheduler.warmup_epochs=0
unset CIFAR_RUN_ROOT
```

To use the same command with float32, append `training.amp=false`. Default output is `runs_hpc/`, with a report at `runs_hpc/report.md`. `CIFAR_DATA_ROOT` and `CIFAR_RUN_ROOT` can override these paths. Keep all paths and config settings consistent when resubmitting so checkpoint matching works.

## 5. Monitor, stop, and resume

```bash
squeue -u "$USER"
tail -f logs/cifar10-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
scancel JOBID
```

`sbatch` jobs continue after you disconnect. Stopping `tail -f` with Ctrl+C does not cancel a submitted job; use `scancel`. A timeout or cancellation may lose the current unfinished epoch. Submit the same batch command again to resume from the last saved checkpoint and skip completed experiments. Do not submit two jobs to the same output directory concurrently.

Existing Mac runs are preserved separately. Exact config matching includes device, paths, and worker count, so copying MPS checkpoints into `runs_hpc/` does not automatically resume them as CUDA experiments. Existing model weights can still be used for CUDA inference; cross-device training migration requires an explicit separate procedure.

## 6. Retrieve results

From your Mac:

```bash
rsync -av NETID@dtn.torch.hpc.nyu.edu:/scratch/NETID/cifar10/runs_hpc/ ./runs_hpc/
```

Predictions can be generated inside a GPU allocation with the existing `predict` command and `--device cuda`. No multi-GPU support is implemented; allocating more than one GPU will not accelerate this training loop automatically.

## Official references

- [Connecting to Torch](https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/)
- [Slurm accounts and GPU jobs](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)
- [PyTorch and container environments](https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/pytorch_intro/)
- [Data transfers](https://services.rt.nyu.edu/docs/hpc/storage/data_transfers/)
- [Storage policy](https://services.rt.nyu.edu/docs/hpc/storage/intro_and_data_management/)

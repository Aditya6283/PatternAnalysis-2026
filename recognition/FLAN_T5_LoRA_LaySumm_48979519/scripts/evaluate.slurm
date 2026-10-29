#!/bin/bash
# ---------------------------------------------------------------------------
# evaluate.slurm - Rangpur batch job: predict.py demo + evaluate.py investigation -> predictions/, results/, figures/
# Project: FLAN-T5 lay summaries of radiology reports (COMP3710). Author: Aditya Raj Singh (48979519)
# Submit from the project folder:  sbatch scripts/evaluate.slurm   (logs go to logs/, which must exist)
# Extra arguments are passed on, e.g. sbatch scripts/evaluate.slurm --val-limit 3000
# Data and model weights must be downloaded first on the login node: python download.py
# Partition: 'a100' schedules reliably; switch to 'comp3710' if your course partition works.
# No --mem line on purpose: the comp3710 nodes advertise RealMemory=1 MB, so any explicit
# memory request stays pending (PartitionConfig).
# ---------------------------------------------------------------------------
#SBATCH --job-name=evaluate
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Name the env here: with no argument, `source` would hand this job's own arguments
# (e.g. --lora-r 32) to conda's activate script, which then fails.
source "$HOME/miniconda3/bin/activate" torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u predict.py --checkpoint runs/lora/best.pt --n 8 --out predictions
python -u evaluate.py --lora runs/lora/best.pt --full runs/full/best.pt \
    --out results --figures figures "$@"

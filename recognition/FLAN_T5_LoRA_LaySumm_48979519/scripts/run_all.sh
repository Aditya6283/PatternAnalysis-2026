#!/bin/bash
# ---------------------------------------------------------------------------
# run_all.sh - Submit the whole pipeline to Rangpur as a chain of dependent jobs.
# Project: FLAN-T5 lay summaries of radiology reports (COMP3710). Author: Aditya Raj Singh (48979519)
#
#   feasibility (audit + prompt check + smoke tests)
#        |-> train_lora -|
#        |-> train_full -|-> evaluate (predict.py + evaluate.py)
#
# Usage (from the project folder, on the login node, AFTER `python download.py`):
#     bash scripts/run_all.sh
# Watch with:  squeue --me      and      tail -f logs/<job>_<id>.out
# ---------------------------------------------------------------------------
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

F=$(sbatch --parsable scripts/feasibility.slurm)
L=$(sbatch --parsable --dependency=afterok:$F scripts/train_lora.slurm)
U=$(sbatch --parsable --dependency=afterok:$F scripts/train_full.slurm)
E=$(sbatch --parsable --dependency=afterok:$L:$U scripts/evaluate.slurm)

echo "feasibility $F | lora $L | full $U | evaluate $E"
echo "If a job fails, its dependants are cancelled: read logs/<job>_<id>.err, fix, resubmit from that step."

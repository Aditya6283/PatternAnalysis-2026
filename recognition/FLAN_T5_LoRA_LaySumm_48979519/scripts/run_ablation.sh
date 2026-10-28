#!/bin/bash
# ---------------------------------------------------------------------------
# run_ablation.sh - Optional experiments for the LoRA vs. full trade-off (run after run_all.sh).
# Project: FLAN-T5 lay summaries of radiology reports (COMP3710). Author: Aditya Raj Singh (48979519)
#
#   1. LoRA rank: r = 4 and r = 64 (main run r = 16)        -> how much capacity does LoRA need?
#   2. LoRA on every linear layer (q k v o wi_0 wi_1 wo)     -> where should the adapters go?
#   3. FLAN-T5-small (77M): LoRA and full                    -> does model size matter more than method?
#   4. One evaluation of all of them (no zero-shot / retrieval re-runs) -> results_ablation/
# Needs flan-t5-small downloaded: python download.py --skip-data --models google/flan-t5-small
# ---------------------------------------------------------------------------
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

J1=$(sbatch --parsable --job-name=lora_r4 scripts/train_lora.slurm --lora-r 4 --lora-alpha 8 --out runs/lora_r4)
J2=$(sbatch --parsable --job-name=lora_r64 scripts/train_lora.slurm --lora-r 64 --lora-alpha 128 --out runs/lora_r64)
J3=$(sbatch --parsable --job-name=lora_all scripts/train_lora.slurm --lora-targets q k v o wi_0 wi_1 wo --out runs/lora_all)
J4=$(sbatch --parsable --job-name=small_lora scripts/train_lora.slurm --model google/flan-t5-small --out runs/small_lora)
J5=$(sbatch --parsable --job-name=small_full scripts/train_full.slurm --model google/flan-t5-small --out runs/small_full)
sbatch --job-name=eval_ablation --dependency=afterok:$J1:$J2:$J3:$J4:$J5 --partition=a100 --gres=gpu:1 \
  --cpus-per-task=4 --time=03:00:00 --output=logs/eval_ablation_%j.out \
  --wrap="source \$HOME/miniconda3/bin/activate && conda activate torch && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
   && python -u evaluate.py --no-zeroshot --no-retrieval --val-limit 3000 --test-limit 500 \
   --lora runs/lora/best.pt --full runs/full/best.pt \
   --extra LoRA_r4=runs/lora_r4/best.pt LoRA_r64=runs/lora_r64/best.pt \
   LoRA_all_layers=runs/lora_all/best.pt Small_LoRA=runs/small_lora/best.pt Small_full=runs/small_full/best.pt \
   --out results_ablation --figures results_ablation/figures"
echo "Ablations submitted. Compare runs/*/summary.json and results_ablation/summary.md."

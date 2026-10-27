"""
download.py - Fetch the BioLaySumm data and FLAN-T5 weights once (run on the LOGIN node).

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).
Author  : Aditya Raj Singh (48979519)
Note    : Implementation assisted by Claude (Anthropic). See the README section
          "Artificial Intelligence Usage Disclosure".

Rangpur compute nodes may have no internet access, so everything is downloaded here first:
  * the three official splits (pinned commit) -> <data-dir>/{train,validation,test}.parquet
    (about 29 MB in total);
  * the model files (pinned commit) into the Hugging Face cache (~/.cache/huggingface),
    about 1 GB for flan-t5-base. The Slurm scripts then run with HF_HUB_OFFLINE=1.
It deliberately does not import PyTorch and uses a single download thread, because the
login node allows only a few processes/threads per user.

Usage (login node, inside the conda env):
    OMP_NUM_THREADS=1 python download.py                       # data + flan-t5-base
    OMP_NUM_THREADS=1 python download.py --models google/flan-t5-base google/flan-t5-small
"""

import argparse
import os
import shutil

from config import (DATASET_ID, DATASET_REVISION, DEFAULT_DATA_DIR, MODEL_FILES, MODEL_REVISIONS,
                    SPLIT_FILES)


def download_data(data_dir: str) -> None:
    from huggingface_hub import hf_hub_download
    os.makedirs(data_dir, exist_ok=True)
    for split, remote in SPLIT_FILES.items():
        local = hf_hub_download(DATASET_ID, remote, repo_type="dataset", revision=DATASET_REVISION)
        target = os.path.join(data_dir, f"{split}.parquet")
        shutil.copyfile(local, target)
        print(f"[download] {split:10s} -> {target} ({os.path.getsize(target) / 2 ** 20:.1f} MB)", flush=True)
    try:
        import pyarrow.parquet as pq
        for split in SPLIT_FILES:
            n = pq.ParquetFile(os.path.join(data_dir, f"{split}.parquet")).metadata.num_rows
            print(f"[download] {split:10s} rows: {n:,}", flush=True)
    except ImportError:
        print("[download] pyarrow not installed yet: `pip install pyarrow pandas` before training", flush=True)


def download_model(model_id: str) -> None:
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, revision=MODEL_REVISIONS.get(model_id), allow_patterns=MODEL_FILES,
                             max_workers=1)
    size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)
    print(f"[download] {model_id} -> {path} ({size / 2 ** 20:.0f} MB)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Download BioLaySumm 2025 LaymanRRG and FLAN-T5 weights")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--models", nargs="*", default=["google/flan-t5-base"])
    p.add_argument("--skip-data", action="store_true")
    args = p.parse_args()
    if not args.skip_data:
        download_data(args.data_dir)
    for model_id in args.models:
        download_model(model_id)
    print("[download] done. Training jobs can now run offline (HF_HUB_OFFLINE=1).", flush=True)


if __name__ == "__main__":
    main()

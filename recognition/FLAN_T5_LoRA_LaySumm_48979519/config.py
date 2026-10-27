"""
config.py - Constants shared by every script (no heavy imports, so download.py stays light).

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).
Author  : Aditya Raj Singh (48979519)
Note    : Implementation assisted by Claude (Anthropic). See the README section
          "Artificial Intelligence Usage Disclosure".
"""

import os

# BioLaySumm 2025, Subtask 2.1 LaymanRRG, open-source track (pinned commit for reproducibility)
DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
DATASET_REVISION = "6783ab38f9de28c4372799346f51d3927ae7eb9a"
SPLIT_FILES = {s: f"data/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")}
DEFAULT_DATA_DIR = os.environ.get("LAYSUMM_DATA",
                                  os.path.join(os.path.expanduser("~"), "data", "biolaysumm"))

# FLAN-T5 checkpoints (pinned Hugging Face commits)
DEFAULT_MODEL = "google/flan-t5-base"
MODEL_REVISIONS = {
    "google/flan-t5-base": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    "google/flan-t5-small": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
}
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "spiece.model",
               "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

# One fixed instruction for every system (zero-shot and fine-tuned), so comparisons are fair.
# The best of four candidate wordings for zero-shot FLAN-T5 on training reports
# (feasibility.py re-checks this on the dev split; see README Section 2).
PROMPT_TEMPLATE = "Translate this radiology report into simple language a patient can understand: {report}"
CANDIDATE_PROMPTS = {
    "translate": PROMPT_TEMPLATE,
    "rewrite": "Rewrite this radiology report in plain language for a patient: {report}",
    "explain": "Explain this radiology report to a patient in simple words: {report}",
    "question": "Radiology report: {report}\n\nWhat does this report mean in simple words?",
}

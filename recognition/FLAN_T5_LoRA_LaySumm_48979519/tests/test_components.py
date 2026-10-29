"""
tests/test_components.py - Unit tests used to verify the implementation (run offline).

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).
Author  : Aditya Raj Singh (48979519)
Note    : Written with assistance from Claude (Anthropic) as part of auditing the
          AI-assisted code (see README, "Artificial Intelligence Usage Disclosure").

Run from the project folder:   python -m pytest -q tests
No downloads are needed: model tests use a tiny randomly initialised T5.
Each test checks a property the report relies on (LoRA starts exactly at the pre-trained
model and merges exactly; padding never counts in the loss; no report text is shared between
train and dev; the confidence score equals the generation probabilities; the safety rules
handle negation, hedging and lay wording; ROUGE / readability / AUROC behave as expected).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (LengthGroupedBatchSampler, Seq2SeqCollator, group_dev_split,  # noqa: E402
                     token_budget_batches,  # noqa: E402
                     normalise_report, study_key)
from modules import (IGNORE_INDEX, LaySummariser, LoRALinear, count_parameters,  # noqa: E402
                     format_prompt, inject_lora, lora_state_dict, merge_lora)
import safety  # noqa: E402
from utils import auroc, bootstrap_ci, rouge_per_example  # noqa: E402


def tiny_t5(seed: int = 0):
    from transformers import T5Config, T5ForConditionalGeneration
    torch.manual_seed(seed)
    cfg = T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2,
                   num_heads=4, feed_forward_proj="gated-gelu", dropout_rate=0.0, pad_token_id=0,
                   eos_token_id=1, decoder_start_token_id=0, tie_word_embeddings=False)
    return T5ForConditionalGeneration(cfg).eval()


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
def test_lora_starts_as_the_base_layer():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x))                      # B = 0 at initialisation
    assert not base.weight.requires_grad and lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_lora_merge_is_exact():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(16, 8), r=4, alpha=8, dropout=0.0)
    nn.init.normal_(lora.lora_B)                                  # pretend it has been trained
    x = torch.randn(5, 16)
    assert torch.allclose(lora.merged_linear()(x), lora(x), atol=1e-5)


def test_inject_lora_trains_only_adapters():
    model = tiny_t5()
    replaced = inject_lora(model, ("q", "v"), r=4, alpha=8, dropout=0.0)
    # 2 encoder layers x (self-attn) + 2 decoder layers x (self-attn + cross-attn) = 6 attention blocks
    assert len(replaced) == 12
    total, trainable = count_parameters(model)
    assert trainable == 12 * 4 * (32 + 32)                         # r * (d_in + d_out) per adapted layer
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)


def test_lora_checkpoint_roundtrip_and_merge():
    model = tiny_t5()
    inject_lora(model, ("q", "v"), r=4, alpha=8, dropout=0.0)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(p, std=0.5)
    ids = torch.randint(2, 64, (2, 7))
    labels = torch.randint(2, 64, (2, 5))
    with torch.no_grad():
        ref = model(input_ids=ids, labels=labels).logits
    fresh = tiny_t5()
    inject_lora(fresh, ("q", "v"), r=4, alpha=8, dropout=0.0)
    fresh.load_state_dict(lora_state_dict(model), strict=False)   # adapters only
    with torch.no_grad():
        assert torch.allclose(fresh(input_ids=ids, labels=labels).logits, ref, atol=1e-5)
        assert merge_lora(fresh) == 12
        assert torch.allclose(fresh(input_ids=ids, labels=labels).logits, ref, atol=1e-4)
    assert not any(isinstance(m, LoRALinear) for m in fresh.modules())


def test_confidence_equals_generation_probabilities():
    """Teacher-forced re-scoring must reproduce the greedy decoder's own log-probabilities."""
    t5 = tiny_t5(seed=3)
    model = LaySummariser("tiny", mode="zeroshot", backbone=t5).eval()
    ids = torch.randint(2, 64, (3, 6))
    mask = torch.ones_like(ids)
    # A random model could emit the pad id (0) as content, which a trained model never does;
    # suppress it, and compare with the RAW logits (before the suppression is applied).
    out = t5.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                      suppress_tokens=[0], output_logits=True, return_dict_in_generate=True)
    seqs = out.sequences.clone()
    # Summaries in a batch end at different lengths (padding after </s>). Pretend rows 0 and 1
    # stopped early, so that a chunk of rows {0, 1} is cut to its own longest summary.
    seqs[0, 4:] = 0
    seqs[1, 5:] = 0
    steps = torch.stack(out.logits, dim=1).float().log_softmax(-1)   # (B, T, V)
    chosen = steps.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    valid = seqs[:, 1:] != 0                                         # padding after </s>
    assert valid.any(1).all()
    expected = (chosen * valid).sum(1) / valid.sum(1)
    for chunk in (8, 2, 1):                                          # memory-saving chunks agree
        got = model.sequence_logprobs(ids, mask, seqs, chunk=chunk)
        assert torch.allclose(got["mean_logprob"], expected, atol=1e-4)
        assert got["n_tokens"].tolist() == valid.sum(1).tolist()


def test_lora_training_step_reduces_loss():
    torch.manual_seed(0)
    model = LaySummariser("tiny", mode="lora", lora_r=4, lora_alpha=8, lora_dropout=0.0,
                          backbone=tiny_t5()).train()
    ids, labels = torch.randint(2, 64, (4, 8)), torch.randint(2, 64, (4, 6))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-2)
    first = None
    for _ in range(30):
        loss = model(ids, torch.ones_like(ids), labels).loss
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.7 * first
    assert set(model.trainable_state_dict()) == {k for k in model.state_dict() if "lora_" in k}


def test_prompt_format():
    assert format_prompt("  No   pneumothorax.\n") == \
        "Translate this radiology report into simple language a patient can understand: No pneumothorax."


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def test_collator_masks_padding_in_labels():
    batch = Seq2SeqCollator(pad_token_id=0)([{"input_ids": [5, 6, 1], "labels": [7, 1]},
                                             {"input_ids": [5, 1], "labels": [7, 8, 9, 1]}])
    assert batch["input_ids"].tolist() == [[5, 6, 1], [5, 1, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[7, 1, IGNORE_INDEX, IGNORE_INDEX], [7, 8, 9, 1]]


def test_length_grouped_sampler_covers_everything_once_and_shards():
    lengths = list(np.random.default_rng(0).integers(5, 100, size=203))
    s = LengthGroupedBatchSampler(lengths, batch_size=8, seed=1, mega_mult=4)
    batches = list(s)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(203)) and len(batches) == len(s)
    s.set_epoch(1)
    assert list(s) != batches                                        # reshuffled each epoch
    r0 = LengthGroupedBatchSampler(lengths, 8, seed=1, mega_mult=4, rank=0, world_size=2)
    r1 = LengthGroupedBatchSampler(lengths, 8, seed=1, mega_mult=4, rank=1, world_size=2)
    a, b = {i for x in r0 for i in x}, {i for x in r1 for i in x}
    assert not (a & b) and len(list(r0)) == len(list(r1)) == len(r0)


def test_token_budget_bounds_padded_tokens():
    lengths = [700, 650, 300, 120, 100, 90, 60, 50, 40, 30, 20, 10]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches = token_budget_batches(order, lengths, batch_size=8, max_tokens=800)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:   # a single over-long example may stand alone, otherwise the budget holds
        assert len(b) <= 8 and (len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 800)
    assert batches[0] == [0] and len(batches[-1]) > 1                 # long reports alone, short ones grouped


def test_dev_split_shares_no_report_text_with_train():
    reports = ["No significant findings."] * 60 + [f"Nodule number {i}." for i in range(40)] + \
              [f"nodule number {i}" for i in range(40)]              # same text, different format
    df = pd.DataFrame({"uid": range(len(reports)), "radiology_report": reports,
                       "layman_report": ["x"] * len(reports)})
    train, dev = group_dev_split(df, dev_size=20, seed=0, max_group=50)
    tk = set(train["radiology_report"].map(normalise_report))
    dk = set(dev["radiology_report"].map(normalise_report))
    assert not (tk & dk) and len(dev) >= 20
    assert "no significant findings" in tk                          # boiler-plate stays in training
    assert normalise_report("No  significant findings.") == "no significant findings"


def test_study_key_groups_the_images_of_one_examination():
    df = pd.DataFrame({"source": ["PadChest", "PadChest", "PadChest", "BIMCV-COVID19", "OpenI"],
                       "images_path": ["2168401_00-141-160.png", "2168401_00-141-123.png", "9999_00-1-1.png",
                                       "ses-E1", "['/front/CXR1_IM-1.png']"]})
    keys = study_key(df).tolist()
    assert keys[0] == keys[1]                    # two images of the same PadChest examination
    assert keys[0] != keys[2]                    # a different examination
    assert len(set(keys)) == 4                   # BIMCV sessions and Open-i studies stay separate


# ---------------------------------------------------------------------------
# Safety rules and readability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("report,summary,expected", [
    ("No pneumothorax.", "There is no collapsed lung.", set()),
    ("Pneumothorax is not visualized.", "There is no collapsed lung.", set()),
    ("Cannot exclude pneumonia in the right base.", "There may be a lung infection on the right.", set()),
    ("Cannot rule out a small pneumothorax.", "Everything looks normal.", {"omission", "false_reassurance"}),
    ("Cardiomegaly.", "The heart is enlarged.", set()),
    ("No pleural effusion.", "There is fluid around the lungs.", {"polarity_flip"}),
    ("Small left pleural effusion.", "Everything looks normal.", {"omission", "false_reassurance"}),
    ("Normal study.", "There is a 2 cm mass in the left lung.", {"hallucination"}),
    ("Possible pneumonia in the right base.", "There is pneumonia in the right lung.", {"certainty_upgrade"}),
    ("Right lower lobe consolidation.", "There might be a lung infection on the right.", {"certainty_downgrade"}),
    ("No changes in the left pleural effusion.", "The fluid around the left lung has not changed.", set()),
    ("The cardiac silhouette is not enlarged.", "The heart is normal in size.", set()),
    ("Left rib fracture.", "There is a broken rib on the left.", set()),
    ("Small left pleural effusion.", "There is a little fluid around the left lung.", set()),
    ("Small left pleural effusion.", "There is a small amount of fluid around the right lung.", {"laterality_swap"}),
    ("Right pleural effusion and left basal consolidation.",
     "There is fluid around the right lung and a solid area at the bottom of the left lung.", set()),
    ("No infiltrates, consolidations or pleural effusion are observed.",
     "We didn't see any lung infection or fluid around the lungs.", set()),
])
def test_safety_flags(report, summary, expected):
    flags = safety.compare(report, summary)
    raised = {k for k in safety.FLAG_TYPES if flags[k]}
    assert raised == expected, (raised, flags)
    assert flags["any"] == bool(expected)


def test_negation_and_hedging_cues():
    st = safety.concept_status("Pneumothorax has resolved. Effusion cannot be excluded.")
    assert st["pneumothorax"].negated and st["pleural_effusion"].affirmed
    assert st["pleural_effusion"].certainty == "hedged"
    assert not safety.is_global_normal("The heart looks normal.")    # organ-level, not global
    assert safety.is_global_normal("Everything looks normal.")
    assert safety.disagrees("There is a small effusion.", "No effusion.")
    assert not safety.disagrees("No effusion.", "The lungs are clear.")
    # 'ruling out X, more likely Y': the negation stops at the new alternative, which stays hedged
    report = "Ruling out pulmonary neoplasia, more likely suggesting an infectious pneumonic process."
    st = safety.concept_status(report)
    assert st["lung_opacity_infection"].affirmed and st["lung_opacity_infection"].certainty == "hedged"
    assert not safety.compare(report, "This is probably a lung infection, not a tumour.")["any"]
    assert safety.compare(report, "There is a lung infection.")["certainty_upgrade"]


def test_readability_and_sentences():
    assert safety.count_syllables("cat") == 1 and safety.count_syllables("water") == 2
    assert safety.count_syllables("pneumonia") >= 3
    # 6 words, 1 sentence, 6 syllables: 0.39*6 + 11.8*1 - 15.59
    assert abs(safety.fkgl("The cat sat on the mat.") - (0.39 * 6 + 11.8 - 15.59)) < 1e-9
    assert safety.fkgl("Cardiomegaly with bilateral interstitial infiltrates.") > \
        safety.fkgl("The heart is big. The lungs are cloudy.")
    assert safety.split_sentences("No effusion. Mild cardiomegaly.\nNormal bones.") == \
        ["No effusion.", "Mild cardiomegaly.", "Normal bones."]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_rouge_identity_and_empty():
    s = rouge_per_example(["The heart is enlarged.", ""], ["The heart is enlarged.", "The lungs are clear."])
    assert all(abs(s[k][0] - 1.0) < 1e-9 for k in s) and all(s[k][1] == 0 for k in s)


def test_auroc_and_bootstrap():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert auroc(np.ones(4), np.array([0, 1, 0, 1])) == 0.5          # ties
    lo, hi = bootstrap_ci(np.r_[np.zeros(50), np.ones(50)], n_boot=500)
    assert lo < 0.5 < hi

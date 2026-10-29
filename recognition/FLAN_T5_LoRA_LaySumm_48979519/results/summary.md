# Evaluation summary

Held-out official validation split: 10,000 reports (45.0% also occur verbatim in the training split; 5,505 unseen). Reference-free safety audit on 2,000 official test reports (0.2% seen in training).

## 1. ROUGE on the held-out validation split

F1 scores in [0, 1] with Porter stemming; ROUGE-Lsum 95% bootstrap CI. *Seen* = the report text also occurs in the training split.

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | ROUGE-Lsum 95% CI | ROUGE-Lsum seen | ROUGE-Lsum unseen |
|---|---|---|---|---|---|---|---|
| Copy source | 0.364 | 0.139 | 0.329 | 0.342 | [0.340, 0.345] | 0.317 | 0.363 |
| Retrieval (nearest train report) | 0.584 | 0.360 | 0.511 | 0.541 | [0.536, 0.545] | 0.582 | 0.507 |
| Zero-shot FLAN-T5 | 0.309 | 0.117 | 0.278 | 0.289 | [0.286, 0.293] | 0.269 | 0.306 |
| LoRA fine-tuned | 0.732 | 0.547 | 0.682 | 0.702 | [0.699, 0.706] | 0.741 | 0.671 |
| Full fine-tuned | 0.749 | 0.574 | 0.701 | 0.720 | [0.717, 0.723] | 0.754 | 0.692 |

ROUGE-Lsum by data source:

| Source | Reports | Copy source | Retrieval (nearest train report) | Zero-shot FLAN-T5 | LoRA fine-tuned | Full fine-tuned |
|---|---|---|---|---|---|---|
| BIMCV-COVID19 | 2,042 | 0.377 | 0.435 | 0.319 | 0.654 | 0.676 |
| OpenI | 134 | 0.341 | 0.620 | 0.255 | 0.725 | 0.749 |
| PadChest | 7,824 | 0.333 | 0.567 | 0.282 | 0.714 | 0.731 |

Paired bootstrap on ROUGE-Lsum (same reports; 1,000 resamples). The last column is the share of resamples where the first system is not better (a one-sided p-value).

| Comparison | Reports | Mean difference | 95% CI | P(diff <= 0) |
|---|---|---|---|---|
| LoRA fine-tuned - Full fine-tuned | all | -0.0179 | [-0.0197, -0.0159] | 1.000 |
| LoRA fine-tuned - Full fine-tuned | unseen | -0.0215 | [-0.0237, -0.0192] | 1.000 |
| LoRA fine-tuned - Zero-shot FLAN-T5 | all | +0.4127 | [+0.4085, +0.4169] | 0.000 |
| LoRA fine-tuned - Zero-shot FLAN-T5 | unseen | +0.3647 | [+0.3599, +0.3696] | 0.000 |
| Full fine-tuned - Zero-shot FLAN-T5 | all | +0.4306 | [+0.4261, +0.4350] | 0.000 |
| Full fine-tuned - Zero-shot FLAN-T5 | unseen | +0.3862 | [+0.3812, +0.3913] | 0.000 |
| LoRA fine-tuned - Retrieval (nearest train report) | all | +0.1614 | [+0.1567, +0.1664] | 0.000 |
| LoRA fine-tuned - Retrieval (nearest train report) | unseen | +0.1635 | [+0.1584, +0.1685] | 0.000 |

## 2. Readability (validation)

Flesch-Kincaid grade (patient material targets about grade 6-8); jargon = share of words typical of expert reports but avoided by the human lay writers (learned from the training split); near-copy = the output is (almost) the input report.

| Text | Grade (mean) | Grade (median) | Grade <= 8 | Jargon words | Words | Empty | Near-copy of report |
|---|---|---|---|---|---|---|---|
| Copy source | 13.7 | 12.7 | 8.1% | 13.1% | 23.1 | 0.0% | 100.0% |
| Retrieval (nearest train report) | 7.5 | 7.0 | 63.2% | 0.2% | 36.3 | 0.0% | 0.2% |
| Zero-shot FLAN-T5 | 13.7 | 12.7 | 11.4% | 11.2% | 19.2 | 5.2% | 64.7% |
| LoRA fine-tuned | 6.7 | 6.3 | 70.3% | 0.1% | 35.9 | 0.0% | 0.3% |
| Full fine-tuned | 6.7 | 6.4 | 70.0% | 0.0% | 36.8 | 0.0% | 0.2% |
| Human reference | 7.0 | 6.7 | 67.5% | 0.1% | 38.7 | 0.0% | 0.2% |
| Expert report (input) | 13.7 | 12.7 | 8.1% | 13.1% | 23.1 | 0.0% | 100.0% |

## 3. Clinical-safety audit against the source report

Share of summaries with at least one flag on a critical finding (pneumothorax, pleural_effusion, lung_opacity_infection, cardiomegaly, nodule_mass, fracture, pulmonary_edema), on the 5,505 UNSEEN validation reports; last column: same on the official test reports (all new). The *Human reference* row is the checker's false-alarm floor (human lay writers paraphrase in ways the rules miss, and some references are genuinely incomplete); judge models relative to it.

| Text | Any flag | Omission | Hallucination | Polarity flip | Certainty upgrade | Certainty downgrade | Left/right swap | False reassurance | Any flag (test) |
|---|---|---|---|---|---|---|---|---|---|
| Copy source | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Retrieval (nearest train report) | 38.8% | 26.6% | 8.8% | 5.4% | 1.2% | 1.4% | 0.6% | 1.2% | 38.8% |
| Zero-shot FLAN-T5 | 15.9% | 14.4% | 0.8% | 1.5% | 0.2% | 0.0% | 0.0% | 0.3% | 15.5% |
| LoRA fine-tuned | 19.1% | 11.4% | 4.4% | 2.5% | 1.5% | 1.0% | 0.1% | 0.1% | 19.5% |
| Full fine-tuned | 20.1% | 11.7% | 4.5% | 3.0% | 1.3% | 0.9% | 0.2% | 0.2% | 20.1% |
| Human reference | 22.2% | 12.9% | 5.8% | 3.2% | 1.3% | 1.1% | 0.1% | 0.2% | n/a |

## 4. Confidence and guardrails (validation)

Error label (offline audit): the summary disagrees with the human reference on whether a critical finding is present. *Auto-approved* = released to the patient without review; *errors caught* = share of all errors routed to a human; *residual error* = error rate among auto-approved summaries.

| System | Policy | Auto-approved | Errors caught | Residual error |
|---|---|---|---|---|
| Zero-shot FLAN-T5 | No guardrail | 100.0% | 0.0% | 21.2% |
| Zero-shot FLAN-T5 | Confidence filter (90% coverage) | 90.0% | 19.9% | 18.9% |
| Zero-shot FLAN-T5 | Confidence filter (80% coverage) | 80.1% | 26.9% | 19.4% |
| Zero-shot FLAN-T5 | Entity check (no safety flag) | 89.0% | 46.2% | 12.8% |
| Zero-shot FLAN-T5 | Entity check + confidence (top 90%) | 83.2% | 48.2% | 13.2% |
| Zero-shot FLAN-T5 | AUROC of low confidence for errors | 0.506 |  | (for checker flags: 0.715) |
| LoRA fine-tuned | No guardrail | 100.0% | 0.0% | 13.2% |
| LoRA fine-tuned | Confidence filter (90% coverage) | 91.2% | 9.3% | 13.1% |
| LoRA fine-tuned | Confidence filter (80% coverage) | 80.0% | 26.2% | 12.2% |
| LoRA fine-tuned | Entity check (no safety flag) | 86.9% | 49.1% | 7.7% |
| LoRA fine-tuned | Entity check + confidence (top 90%) | 79.2% | 53.6% | 7.7% |
| LoRA fine-tuned | AUROC of low confidence for errors | 0.603 |  | (for checker flags: 0.613) |
| Full fine-tuned | No guardrail | 100.0% | 0.0% | 12.4% |
| Full fine-tuned | Confidence filter (90% coverage) | 90.0% | 8.5% | 12.6% |
| Full fine-tuned | Confidence filter (80% coverage) | 80.0% | 24.6% | 11.7% |
| Full fine-tuned | Entity check (no safety flag) | 86.2% | 51.1% | 7.0% |
| Full fine-tuned | Entity check + confidence (top 90%) | 77.3% | 54.9% | 7.2% |
| Full fine-tuned | AUROC of low confidence for errors | 0.589 |  | (for checker flags: 0.599) |

## 5. Resource profile (cuda NVIDIA A100-PCIE-40GB)

Inference uses greedy decoding; batched = batches of 64 reports sorted by length.

| System | Parameters | Trainable | Train time (min) | Peak train VRAM (MB) | Train tokens/s | Checkpoint (MB) | ms/report (batched) | ms/report (single) | Generated tokens/s | Peak inference VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot FLAN-T5 | 247,577,856 | 0 (no training) | - | - | - | - | 73.7 | 1277 | 649 | 2792 |
| LoRA fine-tuned | 249,347,328 | 1,769,472 | 36.2 | 13589 | 22645 | 6.8 | 62.2 | 1553 | 857 | 2802 |
| Full fine-tuned | 247,577,856 | 247,577,856 | 40.6 | 17252 | 23800 | 1132.8 | 50.6 | 1230 | 1070 | 3101 |

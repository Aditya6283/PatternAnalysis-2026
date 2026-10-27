"""
utils.py - Shared helpers: seeding, devices, ROUGE, bootstrap intervals, files and plots.

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).

ROUGE [1] is computed with Google's `rouge-score` package (the implementation used by the
BioLaySumm organisers and Hugging Face `evaluate`), with Porter stemming. For ROUGE-Lsum
each text is split into sentences (one per line) with safety.split_sentences, which needs
no downloaded NLTK models (Rangpur compute nodes may be offline).

References:
    [1] C.-Y. Lin. ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop on
        Text Summarization Branches Out, 2004.
    [2] B. Efron, R. Tibshirani. An Introduction to the Bootstrap. Chapman & Hall, 1993.
"""

import csv
import json
import os
import platform
import random
import time
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # Rangpur compute nodes have no display
import matplotlib.pyplot as plt
import numpy as np
import torch

from safety import split_sentences

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


# ---------------------------------------------------------------------------
# Reproducibility, devices, timing
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def autocast_ctx(device: torch.device, enabled: bool = True):
    """bfloat16 mixed precision on CUDA (T5 is stable in bf16, unlike fp16); no-op on CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    """Peak memory allocated by PyTorch tensors since the last reset (MB)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    return float("nan")


class Timer:
    """Wall-clock timer that synchronises CUDA before reading the clock."""

    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self.start


def library_versions() -> Dict[str, str]:
    """Versions for the README's reproducibility table."""
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": str(torch.version.cuda), "numpy": np.__version__}
    for name in ("transformers", "tokenizers", "rouge_score", "pandas", "pyarrow", "sklearn",
                 "matplotlib", "huggingface_hub"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "installed")
        except ImportError:
            out[name] = "not installed"
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


# ---------------------------------------------------------------------------
# ROUGE and statistics
# ---------------------------------------------------------------------------
_SCORER = None


def _scorer():
    global _SCORER
    if _SCORER is None:
        from rouge_score import rouge_scorer
        _SCORER = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    return _SCORER


def _lsum_format(text: str) -> str:
    return "\n".join(split_sentences(text)) if text.strip() else ""


def rouge_per_example(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, np.ndarray]:
    """Per-example ROUGE F1 for rouge1/2/L/Lsum (arrays of length N, values in [0, 1])."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references differ in length")
    scorer = _scorer()
    out = {k: np.zeros(len(predictions)) for k in ROUGE_TYPES}
    for i, (p, r) in enumerate(zip(predictions, references)):
        p, r = str(p), str(r)
        s = scorer.score(r, p) if p.strip() else None
        s_lsum = scorer.score(_lsum_format(r), _lsum_format(p)) if p.strip() else None
        for k in ("rouge1", "rouge2", "rougeL"):
            out[k][i] = s[k].fmeasure if s else 0.0
        out["rougeLsum"][i] = s_lsum["rougeLsum"].fmeasure if s_lsum else 0.0
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap confidence interval of the mean [2]."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Mean of (a - b) on the same examples, its 95% CI, and the share of resamples with a <= b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), \
        float((diffs <= 0).mean())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) formula; NaN if one class."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # average ranks for ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        obj = obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = list(rows[0].keys())
    for row in rows[1:]:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def md_table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_figure(fig, path: str, title: Optional[str] = None) -> None:
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")      # keeps legends placed outside the axes
    plt.close(fig)


def plot_training(steps: List[Dict], evals: List[Dict], path: str, title: str) -> None:
    """Training curves: per-step train loss (raw + moving average) with dev loss, dev ROUGE,
    and the learning-rate schedule, all against (fractional) epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    if steps:
        ep = np.array([s["epoch"] for s in steps])
        loss = np.array([s["train_loss"] for s in steps])
        axes[0].plot(ep, loss, color="tab:blue", alpha=0.25, linewidth=0.8, label="train (per log step)")
        k = max(1, len(loss) // 40)
        if len(loss) >= k > 1:
            smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
            axes[0].plot(ep[k - 1:], smooth, color="tab:blue", label=f"train (moving avg, {k})")
        axes[2].plot(ep, [s["lr"] for s in steps], color="tab:green")
    if evals:
        ee = [e["epoch"] for e in evals]
        axes[0].plot(ee, [e["dev_loss"] for e in evals], "o-", color="tab:orange", label="dev")
        for key, colour in (("dev_rouge1", "tab:purple"), ("dev_rougeL", "tab:red"),
                            ("dev_rougeLsum", "tab:brown")):
            if key in evals[0]:
                axes[1].plot(ee, [e[key] for e in evals], "o-", color=colour, label=key.replace("dev_", ""))
    axes[0].set_ylabel("cross-entropy (nats / token)")
    axes[1].set_ylabel("ROUGE F1 on dev subset")
    axes[2].set_ylabel("learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    save_figure(fig, path, title)


def bar_chart(groups: Sequence[str], series: Dict[str, Sequence[float]], path: str, ylabel: str,
              title: str, errors: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
              hline: Optional[float] = None, hline_label: str = "") -> None:
    """Grouped bars: one group per entry of `groups`, one bar per series."""
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(groups) * max(1, len(series)) / 2), 4))
    width = 0.8 / max(1, len(series))
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        err = None
        if errors and name in errors:
            lo_hi = np.asarray(errors[name], dtype=float)
            err = np.abs(lo_hi.T - np.asarray(vals, dtype=float))
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=name, yerr=err, capsize=3)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1, label=hline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, path, title)

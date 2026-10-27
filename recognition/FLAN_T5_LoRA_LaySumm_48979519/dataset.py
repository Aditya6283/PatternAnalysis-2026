"""
dataset.py - Loading, auditing, leakage-aware splitting and batching of BioLaySumm 2025.

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).

Data: BioLaySumm 2025, Subtask 2.1 LaymanRRG, open-source track [1]: chest X-ray reports
(PadChest, BIMCV-COVID19, Open-i) paired with a plain-language rewrite. Columns:
source, images_path, radiology_report, layman_report. Official splits: train (~150k),
validation (~10k), test (~10.5k, *no* lay references published). `download.py` stores
them as parquet files in DEFAULT_DATA_DIR.

How the splits are used (no patient ids are provided, so identical report TEXT is the
unit of leakage):
    train       official train, exact duplicate (report, lay) pairs removed;
    dev         ~1,000 rows carved out of train for early stopping / model selection,
                grouped by normalised report text so no report string is in both;
    validation  official validation, held out and used ONCE for the final benchmark
                (the task sheet evaluates on it because test labels are hidden). Each row
                is marked `seen_in_train` if its normalised report text also occurs in the
                training split (45% of them do), so results can be split seen vs. unseen;
    test        official test, unlabelled; its reports never occur in train, so it is
                used for the reference-free clinical-safety audit on genuinely new reports.

Run `python dataset.py --audit` for the data audit used in the feasibility review.

References:
    [1] C. Xiao et al. Overview of the BioLaySumm 2025 Shared Task on Lay Summarization of
        Biomedical Research Articles and Radiology Reports. BioNLP @ ACL 2025.
        Data: https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track
"""

import argparse
import os
import random
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from config import DEFAULT_DATA_DIR, DEFAULT_MODEL, MODEL_REVISIONS
from modules import IGNORE_INDEX, format_prompt

TEXT_COLUMNS = ("radiology_report", "layman_report")


# ---------------------------------------------------------------------------
# Reading and normalising
# ---------------------------------------------------------------------------
def split_path(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"{split}.parquet")


def read_split(data_dir: str, split: str) -> pd.DataFrame:
    """One official split as a DataFrame with a stable `uid` column ('<split>-<row>')."""
    path = split_path(data_dir, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python download.py --data-dir {data_dir}` "
                                "on a node with internet access first.")
    df = pd.read_parquet(path)
    for col in TEXT_COLUMNS:
        if col in df:
            df[col] = df[col].fillna("").astype(str).map(lambda t: " ".join(t.split()))
    df.insert(0, "uid", [f"{split}-{i}" for i in range(len(df))])
    return df.reset_index(drop=True)


_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def normalise_report(text: str) -> str:
    """Key used to detect the same report written twice: lower case, no punctuation,
    single spaces. 'No significant findings.' == 'no significant findings'."""
    return " ".join(_PUNCT_RE.sub(" ", str(text).lower()).split())


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def group_dev_split(train: pd.DataFrame, dev_size: int, seed: int,
                    max_group: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole groups of identical (normalised) reports until ~dev_size rows.

    Grouping guarantees no report text appears in both train and dev (the text
    analogue of a patient-level split). Very common boiler-plate reports (> max_group
    rows, e.g. 'No significant findings.') stay in training so one string cannot fill
    the dev set.
    """
    keys = train["radiology_report"].map(normalise_report)
    counts = keys.value_counts()
    candidates = sorted(k for k, c in counts.items() if c <= max_group)
    random.Random(seed).shuffle(candidates)
    chosen, total = set(), 0
    for k in candidates:
        if total >= dev_size:
            break
        chosen.add(k)
        total += int(counts[k])
    is_dev = keys.isin(chosen)
    return train[~is_dev].reset_index(drop=True), train[is_dev].reset_index(drop=True)


def prepare_splits(data_dir: str = DEFAULT_DATA_DIR, dev_size: int = 1000, seed: int = 42,
                   dedup: bool = True, max_train: Optional[int] = None,
                   verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Return {'train', 'dev', 'validation', 'test'} DataFrames (see module docstring).

    Deterministic for a given (dev_size, seed, dedup, max_train), so train.py and
    evaluate.py rebuild exactly the same training set when marking `seen_in_train`.
    """
    train = read_split(data_dir, "train")
    validation = read_split(data_dir, "validation")
    test = read_split(data_dir, "test")
    n_raw = len(train)
    if dedup:
        train = train.drop_duplicates(list(TEXT_COLUMNS)).reset_index(drop=True)
    n_dedup = len(train)
    train, dev = group_dev_split(train, dev_size, seed)
    if max_train and max_train < len(train):
        train = train.sample(n=max_train, random_state=seed).reset_index(drop=True)

    train_keys = set(train["radiology_report"].map(normalise_report))
    dev_keys = set(dev["radiology_report"].map(normalise_report))
    assert not (train_keys & dev_keys), "a report text appears in both train and dev"
    for df in (validation, test):
        df["seen_in_train"] = df["radiology_report"].map(normalise_report).isin(train_keys)
    test["has_reference"] = test["layman_report"].str.len() > 0
    if verbose:
        print(f"[data] train {n_raw} rows -> {n_dedup} after removing duplicate pairs -> "
              f"{len(train)} train + {len(dev)} dev (grouped by report text)", flush=True)
        print(f"[data] validation {len(validation)} rows ({validation['seen_in_train'].mean():.1%} "
              f"of reports also in train); test {len(test)} rows "
              f"({test['seen_in_train'].mean():.1%} in train, "
              f"{test['has_reference'].mean():.0%} with references)", flush=True)
    return {"train": train, "dev": dev, "validation": validation, "test": test}


# ---------------------------------------------------------------------------
# Tokenised dataset, collator and length-grouped batching
# ---------------------------------------------------------------------------
def load_tokenizer(model_name: str = DEFAULT_MODEL, revision: Optional[str] = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision or MODEL_REVISIONS.get(model_name))
    # Lengths are clipped explicitly (LaySummDataset / summarise), so silence the tokenizer's
    # generic "longer than 512 tokens" warning on the few very long reports.
    tok.model_max_length = 100_000
    return tok


class LaySummDataset(Dataset):
    """Pre-tokenised (prompt, lay summary) pairs.

    Inputs are `format_prompt(report)` truncated to max_source_len tokens; targets are
    the lay summaries truncated to max_target_len tokens (both end with </s>).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_source_len: int = 384,
                 max_target_len: int = 320, with_targets: bool = True, batch: int = 2048):
        self.uids = df["uid"].tolist()
        prompts = [format_prompt(r) for r in df["radiology_report"]]
        self.input_ids: List[List[int]] = []
        self.labels: List[List[int]] = []
        self.truncated_sources = 0
        self.truncated_targets = 0
        for i in range(0, len(prompts), batch):
            enc = tokenizer(prompts[i:i + batch], truncation=False)["input_ids"]
            self.truncated_sources += sum(len(e) > max_source_len for e in enc)
            self.input_ids += [self._clip(e, max_source_len) for e in enc]
        if with_targets:
            targets = df["layman_report"].tolist()
            for i in range(0, len(targets), batch):
                enc = tokenizer(targets[i:i + batch], truncation=False)["input_ids"]
                self.truncated_targets += sum(len(e) > max_target_len for e in enc)
                self.labels += [self._clip(e, max_target_len) for e in enc]

    @staticmethod
    def _clip(ids: List[int], max_len: int) -> List[int]:
        """Truncate but keep the final </s> token so the model still learns to stop."""
        return ids if len(ids) <= max_len else ids[:max_len - 1] + ids[-1:]

    def __len__(self) -> int:
        return len(self.input_ids)

    def lengths(self) -> List[int]:
        if self.labels:
            return [len(a) + len(b) for a, b in zip(self.input_ids, self.labels)]
        return [len(a) for a in self.input_ids]

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        item = {"input_ids": self.input_ids[i]}
        if self.labels:
            item["labels"] = self.labels[i]
        return item


class Seq2SeqCollator:
    """Pad a batch: inputs with the pad token (+ attention mask), labels with -100 so
    padded positions do not count in the loss."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        n = len(batch)
        src_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((n, src_len), self.pad, dtype=torch.long)
        attention = torch.zeros((n, src_len), dtype=torch.long)
        for i, b in enumerate(batch):
            input_ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            attention[i, :len(b["input_ids"])] = 1
        out = {"input_ids": input_ids, "attention_mask": attention}
        if "labels" in batch[0]:
            tgt_len = max(len(b["labels"]) for b in batch)
            labels = torch.full((n, tgt_len), IGNORE_INDEX, dtype=torch.long)
            for i, b in enumerate(batch):
                labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
            out["labels"] = labels
        return out


class LengthGroupedBatchSampler(Sampler):
    """Random batches of similar-length examples (less padding, ~2x faster training).

    Each epoch: shuffle all indices, cut them into 'mega-batches' of
    batch_size * mega_mult, sort each mega-batch by length, slice it into batches and
    shuffle the batches. A batch holds at most `batch_size` examples AND at most
    `max_tokens` padded tokens (examples x longest example), so the few very long reports
    are trained in smaller batches and GPU memory stays bounded. For multi-GPU (torchrun)
    every rank takes a disjoint, equally long share of the batches.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int = 0,
                 mega_mult: int = 50, rank: int = 0, world_size: int = 1,
                 max_tokens: Optional[int] = None):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.seed = seed
        self.mega = batch_size * mega_mult
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> List[List[int]]:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        batches = []
        for s in range(0, len(idx), self.mega):
            chunk = sorted(idx[s:s + self.mega], key=lambda i: self.lengths[i], reverse=True)
            batches += token_budget_batches(chunk, self.lengths, self.batch_size, self.max_tokens)
        order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[o] for o in order]
        per_rank = len(batches) // self.world_size
        return batches[self.rank::self.world_size][:per_rank]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


def token_budget_batches(order: Sequence[int], lengths: Sequence[int], batch_size: int,
                         max_tokens: Optional[int] = None) -> List[List[int]]:
    """Cut `order` (sorted longest first) into batches of <= batch_size examples and
    <= max_tokens padded tokens, where padded tokens = examples x longest example."""
    batches, current, longest = [], [], 0
    for i in order:
        new_longest = max(longest, lengths[i])
        full = len(current) >= batch_size or (max_tokens and (len(current) + 1) * new_longest > max_tokens)
        if current and full:
            batches.append(current)
            current, new_longest = [], lengths[i]
        current.append(i)
        longest = new_longest
    if current:
        batches.append(current)
    return batches


def make_train_loader(ds: LaySummDataset, batch_size: int, pad_token_id: int, seed: int,
                      rank: int = 0, world_size: int = 1, num_workers: int = 0,
                      max_tokens: Optional[int] = None) -> DataLoader:
    sampler = LengthGroupedBatchSampler(ds.lengths(), batch_size, seed=seed, rank=rank,
                                        world_size=world_size, max_tokens=max_tokens)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=Seq2SeqCollator(pad_token_id),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def make_eval_loader(ds: LaySummDataset, batch_size: int, pad_token_id: int,
                     max_tokens: Optional[int] = None) -> DataLoader:
    """Deterministic batches (longest first) for loss evaluation."""
    lengths = ds.lengths()
    order = sorted(range(len(ds)), key=lambda i: lengths[i], reverse=True)
    batches = token_budget_batches(order, lengths, batch_size, max_tokens)
    return DataLoader(ds, batch_sampler=batches, collate_fn=Seq2SeqCollator(pad_token_id))


# ---------------------------------------------------------------------------
# Audit (feasibility review)
# ---------------------------------------------------------------------------
def _pct(values: Sequence[int], qs=(0.5, 0.9, 0.99, 1.0)) -> str:
    s = pd.Series(list(values))
    return " / ".join(f"{int(s.quantile(q))}" for q in qs)


def audit_text(data_dir: str = DEFAULT_DATA_DIR, tokenizer=None, sample: int = 20000,
               seed: int = 0) -> str:
    """Markdown summary: rows, sources, lengths, duplication and train/eval overlap."""
    raw = {s: read_split(data_dir, s) for s in ("train", "validation", "test")}
    lines = ["| Split | Rows | PadChest | BIMCV-COVID19 | Open-i | Lay references | "
             "Report words (median/p90/p99/max) | Lay words (median/p90/p99/max) |",
             "|---|---|---|---|---|---|---|---|"]
    for s, df in raw.items():
        src = df["source"].value_counts()
        has_ref = (df["layman_report"].str.len() > 0).mean()
        rw = df["radiology_report"].str.split().str.len()
        lw = df["layman_report"].str.split().str.len()
        lines.append(f"| {s} | {len(df):,} | {src.get('PadChest', 0):,} | {src.get('BIMCV-COVID19', 0):,} | "
                     f"{src.get('OpenI', 0):,} | {has_ref:.0%} | {_pct(rw)} | "
                     f"{_pct(lw) if has_ref else 'n/a'} |")
    tr = raw["train"]
    keys_tr = set(tr["radiology_report"].map(normalise_report))
    n_pairs = len(tr.drop_duplicates(list(TEXT_COLUMNS)))
    lines += ["",
              f"- Distinct report texts in train: {tr['radiology_report'].nunique():,} of {len(tr):,} rows; "
              f"distinct (report, lay) pairs: {n_pairs:,} (duplicates removed before training).",
              f"- Most frequent training report: \"{tr['radiology_report'].value_counts().index[0]}\" "
              f"({tr['radiology_report'].value_counts().iloc[0]:,} rows).",
              f"- Validation reports whose normalised text also occurs in train: "
              f"{raw['validation']['radiology_report'].map(normalise_report).isin(keys_tr).mean():.1%}; "
              f"test: {raw['test']['radiology_report'].map(normalise_report).isin(keys_tr).mean():.1%}."]
    if tokenizer is not None:
        smp = tr.sample(min(sample, len(tr)), random_state=seed)
        src_len = [len(x) for x in tokenizer([format_prompt(r) for r in smp["radiology_report"]])["input_ids"]]
        tgt_len = [len(x) for x in tokenizer(smp["layman_report"].tolist())["input_ids"]]
        lines.append(f"- Tokens on a {len(smp):,}-row sample (median/p90/p99/max): prompt+report "
                     f"{_pct(src_len)}, lay summary {_pct(tgt_len)}. Limits used: source 384, target 320 "
                     f"(truncates {sum(l > 384 for l in src_len) / len(src_len):.2%} / "
                     f"{sum(l > 320 for l in tgt_len) / len(tgt_len):.2%}).")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="BioLaySumm data audit")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--audit", action="store_true", help="print the data audit table")
    p.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer used for token-length statistics")
    args = p.parse_args()
    splits = prepare_splits(args.data_dir)
    if args.audit:
        print(audit_text(args.data_dir, load_tokenizer(args.model)))
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()

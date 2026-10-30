"""
evaluate.py - The investigation: does fine-tuning preserve clinical facts, or only raise ROUGE?

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).

Compares, on exactly the same held-out reports:
    Copy source        output the expert report unchanged (what ROUGE gives for zero effort)
    Retrieval          lay summary of the most similar TRAINING report (TF-IDF cosine; exploits
                       the duplicated reports; fluent and well written, but about another patient)
    Zero-shot FLAN-T5  the pre-trained model with the same prompt (the task's baseline)
    LoRA fine-tuned    runs/lora/best.pt
    Full fine-tuned    runs/full/best.pt
and writes the tables / figures for README Section 7:
  1. ROUGE-1/2/L/Lsum on the official validation split (bootstrap 95% CI), split into reports
     whose text also occurs in training ("seen", 45%) vs. new reports ("unseen"), plus the
     "clean" reports that are new in text AND come from an X-ray study never seen in training,
     and by source; paired bootstrap tests between systems.
  2. Readability: Flesch-Kincaid grade, jargon rate, length, empty outputs, near-copies.
  3. Clinical-safety audit (safety.py) against the SOURCE report: omissions, hallucinations,
     polarity flips, certainty changes, false reassurance - on unseen validation reports and on
     the official test reports (new report text, no references needed). The dataset's reference
     summaries (written by GPT-4o) get the same audit: their flag rate is the checker's
     false-alarm floor.
  4. Guardrails: does the model's confidence (mean token log-probability) predict errors?
     Errors are labelled offline by disagreement with the reference summary on any critical
     finding; deployable guardrails (confidence filter, entity check, both) are scored by
     coverage (share auto-approved), errors caught and residual error rate, and the
     confidence ranking is drawn as a risk-coverage curve [1].
  5. Resource profile: parameters, trainable parameters, training time / VRAM (from
     runs/*/summary.json), inference latency, throughput and VRAM, checkpoint size.
  6. Autopsy candidates: real failure cases per error type, from the unseen validation reports
     and from the official test reports -> results/autopsy_candidates.md.

Generated summaries are cached in results/generations/ (use --regenerate to redo them).

Usage
    python evaluate.py --lora runs/lora/best.pt --full runs/full/best.pt

References:
    [1] Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017
        (risk-coverage curves). ROUGE and the bootstrap are cited in utils.py, NegEx and the
        Flesch-Kincaid grade in safety.py.
"""

import argparse
import os
import re
import shutil
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from dataset import DEFAULT_DATA_DIR, DEFAULT_MODEL, load_tokenizer, normalise_report, prepare_splits
from modules import LaySummariser, count_parameters, load_summariser
from safety import (CRITICAL, FLAG_TYPES, build_jargon_vocabulary, compare, disagrees, fkgl,
                    jargon_rate, words)
from utils import (ROUGE_TYPES, Timer, auroc, bar_chart, bootstrap_ci, get_device, load_json,
                   md_table, paired_bootstrap_diff, peak_memory_mb, reset_peak_memory,
                   rouge_per_example, save_figure, save_json, set_seed, write_csv)

# The dataset's lay summaries were written by GPT-4o (the LaymanRRG framework), not by people:
# a useful reference, but silver labels rather than ground truth.
REFERENCE = "Reference (GPT-4o)"


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark and safety audit of the lay summarisers")
    p.add_argument("--lora", default=None, help="LoRA checkpoint (runs/lora/best.pt)")
    p.add_argument("--full", default=None, help="full fine-tuning checkpoint (runs/full/best.pt)")
    p.add_argument("--extra", nargs="*", default=[], help="more checkpoints as NAME=PATH")
    p.add_argument("--model", default=DEFAULT_MODEL, help="backbone for the zero-shot baseline")
    p.add_argument("--no-zeroshot", action="store_true")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--main", default=None, help="system used for the autopsy (default: LoRA, else Full)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--dev-size", type=int, default=1000, help="must match training")
    p.add_argument("--seed", type=int, default=42, help="must match training")
    p.add_argument("--max-train-samples", type=int, default=None, help="must match training")
    p.add_argument("--no-dedup", action="store_true", help="must match training")
    p.add_argument("--val-limit", type=int, default=0, help="evaluate on a random subset (0 = all 10,000)")
    p.add_argument("--test-limit", type=int, default=2000, help="test reports for the reference-free audit")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--profile-reports", type=int, default=256, help="reports used to time inference")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--out", default="results")
    p.add_argument("--figures", default="figures")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu"])
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def pct(x, nd=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


# ---------------------------------------------------------------------------
# Baselines without a neural model
# ---------------------------------------------------------------------------
def retrieval_baseline(train: pd.DataFrame, queries, batch: int = 256):
    """Lay summary of the most similar training report (TF-IDF, cosine similarity).

    Identical reports map to their most common lay version, so 'seen' reports get the
    reference summary of the same text; new reports get the summary of a DIFFERENT
    patient's report, which can read perfectly and still be clinically wrong.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    pairs = (train.assign(key=train["radiology_report"].map(normalise_report))
             .groupby(["key", "layman_report"]).size().reset_index(name="n")
             .sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key"))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.3, sublinear_tf=True)
    docs = vec.fit_transform(pairs["key"])
    q = vec.transform([normalise_report(x) for x in queries])
    best = []
    for i in range(0, q.shape[0], batch):
        best.extend(np.asarray((q[i:i + batch] @ docs.T).argmax(axis=1)).ravel().tolist())
    targets = pairs["layman_report"].tolist()
    return [targets[b] for b in best]


def unigram_f1(a: str, b: str) -> float:
    """Bag-of-words F1 between two texts (used to spot near-copies of the source)."""
    ca, cb = Counter(w.lower() for w in words(a)), Counter(w.lower() for w in words(b))
    if not ca or not cb:
        return 0.0
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    p, r = common / sum(ca.values()), common / sum(cb.values())
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Neural systems: cached generation with an inference profile
# ---------------------------------------------------------------------------
def generate_cached(model, tokenizer, df: pd.DataFrame, path: str, args, device) -> pd.DataFrame:
    """Summaries (+ confidence) for every row of df, cached as CSV keyed by uid."""
    if os.path.exists(path) and not args.regenerate:
        cached = pd.read_csv(path, keep_default_na=False)
        if list(cached["uid"]) == list(df["uid"]):
            cached["mean_logprob"] = pd.to_numeric(cached["mean_logprob"], errors="coerce")
            return cached
    outs = model.summarise(df["radiology_report"].tolist(), tokenizer, device,
                           batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                           num_beams=args.num_beams, amp=not args.no_amp)
    gen = pd.DataFrame({"uid": df["uid"].tolist(), **{k: [o[k] for o in outs] for k in outs[0]}})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gen.to_csv(path, index=False)
    return gen


def inference_profile(model, tokenizer, reports, args, device, n_batched: int = 256,
                      n_single: int = 20) -> dict:
    """Speed and memory on a fixed sample: batched throughput and one-report-at-a-time latency
    (a patient-portal request)."""
    reports = list(reports)
    reset_peak_memory(device)
    with Timer(device) as t:
        outs = model.summarise(reports[:n_batched], tokenizer, device, batch_size=args.batch_size,
                               max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                               amp=not args.no_amp)
    n = max(1, len(outs))
    profile = {"ms_per_report_batched": 1000 * t.seconds / n,
               "generated_tokens_per_s": sum(o["n_tokens"] for o in outs) / max(t.seconds, 1e-9),
               "peak_inference_vram_mb": peak_memory_mb(device)}
    times = []
    for r in reports[:n_single]:
        with Timer(device) as t1:
            model.summarise([r], tokenizer, device, batch_size=1, max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams, amp=not args.no_amp)
        times.append(t1.seconds)
    profile["latency_ms_single_report"] = 1000 * float(np.median(times)) if times else float("nan")
    total, trainable = count_parameters(model)
    profile.update(parameters_total=total, parameters_trainable=trainable)
    return profile


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------
def text_metrics(summaries, sources, jargon) -> pd.DataFrame:
    rows = []
    for s, src in zip(summaries, sources):
        s = str(s)
        rows.append({"words": len(words(s)), "grade": fkgl(s), "jargon": jargon_rate(s, jargon),
                     "empty": not s.strip(),
                     "near_copy": normalise_report(s) == normalise_report(src) or unigram_f1(s, src) >= 0.9})
    return pd.DataFrame(rows)


def safety_metrics(summaries, sources) -> pd.DataFrame:
    rows = []
    for s, src in zip(summaries, sources):
        f = compare(str(src), str(s))
        rows.append({**{k: bool(f[k]) for k in FLAG_TYPES}, "any_flag": f["any"],
                     "flag_detail": "; ".join(f"{k}: {','.join(f[k])}" for k in FLAG_TYPES[:-1] if f[k])
                     + ("; false_reassurance" if f["false_reassurance"] else "")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(0)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    report, metrics = [], {}

    splits = prepare_splits(args.data_dir, dev_size=args.dev_size, seed=args.seed,
                            dedup=not args.no_dedup, max_train=args.max_train_samples)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    if args.val_limit and args.val_limit < len(val):
        val = val.sample(n=args.val_limit, random_state=0).reset_index(drop=True)
    test = test.sample(n=min(args.test_limit, len(test)), random_state=0).reset_index(drop=True) \
        if args.test_limit else test.iloc[:0]
    jargon = build_jargon_vocabulary(train["radiology_report"], train["layman_report"])
    n_clean = int((~val["seen_in_train"] & ~val["study_seen_in_train"]).sum())
    report.append(f"# Evaluation summary\n\nHeld-out official validation split: {len(val):,} reports "
                  f"({val['seen_in_train'].mean():.1%} also occur verbatim in the training split; "
                  f"{(~val['seen_in_train']).sum():,} unseen; {val['study_seen_in_train'].mean():.1%} share an "
                  f"X-ray study with a training row; {n_clean:,} are new in both). Reference-free safety audit on "
                  f"{len(test):,} official test reports ({test['seen_in_train'].mean():.1%} of report texts and "
                  f"{test['study_seen_in_train'].mean():.1%} of X-ray studies seen in training). The reference "
                  "summaries were written by GPT-4o (LaymanRRG), so they are silver labels.")

    # ---- systems ----------------------------------------------------------------------------
    systems = OrderedDict()
    systems["Copy source"] = {"kind": "copy"}
    if not args.no_retrieval:
        systems["Retrieval (nearest train report)"] = {"kind": "retrieval"}
    if not args.no_zeroshot:
        systems["Zero-shot FLAN-T5"] = {"kind": "model", "path": None}
    if args.lora:
        systems["LoRA fine-tuned"] = {"kind": "model", "path": args.lora}
    if args.full:
        systems["Full fine-tuned"] = {"kind": "model", "path": args.full}
    for spec in args.extra:
        name, path = spec.rsplit("=", 1)
        systems[name] = {"kind": "model", "path": path}
    main_name = args.main or next((n for n in ("LoRA fine-tuned", "Full fine-tuned") if n in systems),
                                  list(systems)[-1])

    outputs = {}       # name -> {"validation": DataFrame, "test": DataFrame}
    profiles = {}      # name -> inference profile
    ckpt_info = {}     # name -> what the checkpoint itself says (used if its run's summary.json is missing)
    for name, spec in list(systems.items()):
        print(f"[eval] {name}", flush=True)
        if spec["kind"] == "copy":
            outputs[name] = {s: pd.DataFrame({"uid": d["uid"], "summary": d["radiology_report"],
                                              "mean_logprob": np.nan})
                             for s, d in (("validation", val), ("test", test))}
            continue
        if spec["kind"] == "retrieval":
            try:
                outputs[name] = {}
                for s, d in (("validation", val), ("test", test)):
                    summ = retrieval_baseline(train, d["radiology_report"]) if len(d) else []
                    outputs[name][s] = pd.DataFrame({"uid": d["uid"], "summary": summ, "mean_logprob": np.nan})
            except ImportError:
                print("[eval] scikit-learn not installed: skipping the retrieval baseline", flush=True)
                systems.pop(name)
                outputs.pop(name, None)
            continue
        if spec["path"]:
            model, ckpt = load_summariser(spec["path"], device)
            model_name = ckpt["config"]["model_name"]
            data = ckpt.get("data") or {}
            if data and (data.get("dev_size"), data.get("seed"), data.get("max_train")) != \
                    (args.dev_size, args.seed, args.max_train_samples):
                print(f"[eval] WARNING: {name} was trained with data settings {data}; pass the same "
                      "--dev-size/--seed/--max-train-samples so 'seen' marks the right reports", flush=True)
            # LoRA checkpoints hold only the adapters; full ones every weight (all of them trained).
            ckpt_info[name] = {"parameters_trainable": sum(v.numel() for v in ckpt["state_dict"].values())
                               if ckpt["mode"] == "lora" else count_parameters(model)[0],
                               "checkpoint_mb": os.path.getsize(spec["path"]) / 2 ** 20}
        else:
            model, model_name = LaySummariser(args.model, mode="zeroshot").to(device).eval(), args.model
        tokenizer = load_tokenizer(model_name)
        outputs[name] = {}
        for split, df in (("validation", val), ("test", test)):
            if not len(df):
                outputs[name][split] = pd.DataFrame({"uid": [], "summary": [], "mean_logprob": []})
                continue
            outputs[name][split] = generate_cached(model, tokenizer, df, os.path.join(
                args.out, "generations", f"{slug(name)}_{split}.csv"), args, device)
        profiles[name] = inference_profile(model, tokenizer, val["radiology_report"], args, device,
                                           n_batched=args.profile_reports, n_single=min(20, args.profile_reports))
        if not spec["path"]:
            profiles[name]["parameters_trainable"] = 0
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    names = list(systems)
    refs, srcs = val["layman_report"].tolist(), val["radiology_report"].tolist()
    seen = val["seen_in_train"].to_numpy()
    # 'clean' = new report text AND an X-ray study that never appears in training (dataset.study_key)
    clean = ~seen & ~val["study_seen_in_train"].to_numpy()

    # ---- 1. ROUGE -----------------------------------------------------------------------------
    rouge = {n: rouge_per_example(outputs[n]["validation"]["summary"].astype(str).tolist(), refs) for n in names}
    rows = []
    for n in names:
        r = rouge[n]
        lo, hi = bootstrap_ci(r["rougeLsum"])
        rows.append([n] + [fmt(r[k].mean()) for k in ROUGE_TYPES] + [f"[{lo:.3f}, {hi:.3f}]",
                    fmt(r["rougeLsum"][seen].mean()) if seen.any() else "n/a",
                    fmt(r["rougeLsum"][~seen].mean()) if (~seen).any() else "n/a",
                    fmt(r["rougeLsum"][clean].mean()) if clean.any() else "n/a"])
        metrics.setdefault("rouge", {})[n] = {**{k: float(r[k].mean()) for k in ROUGE_TYPES},
                                              "rougeLsum_ci95": [lo, hi],
                                              "rougeLsum_seen": float(r["rougeLsum"][seen].mean()) if seen.any() else None,
                                              "rougeLsum_unseen": float(r["rougeLsum"][~seen].mean()) if (~seen).any() else None,
                                              "rougeLsum_clean": float(r["rougeLsum"][clean].mean()) if clean.any() else None}
    report.append("## 1. ROUGE on the held-out validation split\n\nF1 scores in [0, 1] with Porter stemming; "
                  "ROUGE-Lsum 95% bootstrap CI. *Seen* = the report text also occurs in the training split; "
                  "*clean* = new report text and an X-ray study that never occurs in training.\n\n"
                  + md_table(["System", "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum", "ROUGE-Lsum 95% CI",
                              "ROUGE-Lsum seen", "ROUGE-Lsum unseen", "ROUGE-Lsum clean"], rows))
    # by source
    src_rows = []
    for source in sorted(val["source"].unique()):
        m = (val["source"] == source).to_numpy()
        src_rows.append([source, f"{m.sum():,}"] + [fmt(rouge[n]["rougeLsum"][m].mean()) for n in names])
    report.append("ROUGE-Lsum by data source:\n\n" + md_table(["Source", "Reports"] + names, src_rows))
    # paired comparisons
    pairs = [(a, b) for a, b in (("LoRA fine-tuned", "Full fine-tuned"), ("LoRA fine-tuned", "Zero-shot FLAN-T5"),
                                 ("Full fine-tuned", "Zero-shot FLAN-T5"), (main_name, "Retrieval (nearest train report)"))
             if a in rouge and b in rouge and a != b]
    if pairs:
        prow = []
        for a, b in pairs:
            for subset, mask in (("all", np.ones(len(val), bool)), ("unseen", ~seen)):
                if mask.any():
                    d, lo, hi, p = paired_bootstrap_diff(rouge[a]["rougeLsum"][mask], rouge[b]["rougeLsum"][mask])
                    prow.append([f"{a} - {b}", subset, f"{d:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]", f"{p:.3f}"])
                    metrics.setdefault("paired_bootstrap", {})[f"{a} - {b} ({subset})"] = [d, lo, hi, p]
        report.append("Paired bootstrap on ROUGE-Lsum (same reports; 1,000 resamples). The last column is the "
                      "share of resamples where the first system is not better (a one-sided p-value).\n\n"
                      + md_table(["Comparison", "Reports", "Mean difference", "95% CI", "P(diff <= 0)"], prow))
    bar_chart(names, {k: [rouge[n][k].mean() for n in names] for k in ROUGE_TYPES},
              os.path.join(args.figures, "rouge_systems.png"), "F1", "ROUGE on the held-out validation split",
              errors={"rougeLsum": [bootstrap_ci(rouge[n]["rougeLsum"]) for n in names]})
    if seen.any() and (~seen).any():
        series = {"seen in training": [rouge[n]["rougeLsum"][seen].mean() for n in names],
                  "unseen (new report text)": [rouge[n]["rougeLsum"][~seen].mean() for n in names]}
        if clean.any():
            series["clean (new text and new X-ray study)"] = [rouge[n]["rougeLsum"][clean].mean() for n in names]
        bar_chart(names, series, os.path.join(args.figures, "rouge_seen_unseen.png"), "ROUGE-Lsum F1",
                  "Seen vs. unseen reports: how much of the score is memorisation?")
    # ROUGE vs. report length
    lengths = np.array([len(words(s)) for s in srcs])
    edges = np.unique(np.quantile(lengths, [0, .2, .4, .6, .8, 1]))
    bins = np.clip(np.digitize(lengths, edges[1:-1], right=True), 0, len(edges) - 2)
    fig, ax = plt.subplots(figsize=(9, 4))
    # bin 0 is [edges[0], edges[1]], bin i > 0 is (edges[i], edges[i+1]] (digitize with right=True)
    labels = [f"{int(np.floor(edges[i])) + (1 if i else 0)}-{int(np.floor(edges[i + 1]))}"
              for i in range(len(edges) - 1)]
    for n in names:
        ax.plot(labels, [rouge[n]["rougeLsum"][bins == b].mean() for b in range(len(labels))], "o-", label=n)
    ax.set_xlabel("report length (words, quintiles)")
    ax.set_ylabel("ROUGE-Lsum F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))   # outside: never hides points
    save_figure(fig, os.path.join(args.figures, "rouge_by_length.png"), "Where do summaries degrade?")

    # ---- 2. readability ----------------------------------------------------------------------
    textm = {n: text_metrics(outputs[n]["validation"]["summary"].astype(str), srcs, jargon) for n in names}
    textm[REFERENCE] = text_metrics(refs, srcs, jargon)
    textm["Expert report (input)"] = text_metrics(srcs, srcs, jargon)
    rrows = []
    for n, t in textm.items():
        g = t["grade"].dropna()
        rrows.append([n, fmt(g.mean(), 1), fmt(g.median(), 1), pct((g <= 8).mean()), pct(t["jargon"].mean()),
                      fmt(t["words"].mean(), 1), pct(t["empty"].mean()), pct(t["near_copy"].mean())])
        metrics.setdefault("readability", {})[n] = {"grade_mean": float(g.mean()), "grade_median": float(g.median()),
                                                    "grade_le_8": float((g <= 8).mean()), "jargon_rate": float(t["jargon"].mean()),
                                                    "words": float(t["words"].mean()), "empty": float(t["empty"].mean()),
                                                    "near_copy": float(t["near_copy"].mean())}
    report.append("## 2. Readability (validation)\n\nFlesch-Kincaid grade (patient material targets about grade 6-8); "
                  "jargon = share of words typical of expert reports but avoided in the reference lay summaries "
                  "(learned from the training split); near-copy = the output is (almost) the input report.\n\n"
                  + md_table(["Text", "Grade (mean)", "Grade (median)", "Grade <= 8", "Jargon words", "Words",
                              "Empty", "Near-copy of report"], rrows))
    fig, ax = plt.subplots(figsize=(8, 4))
    order = list(textm)
    ax.boxplot([textm[n]["grade"].dropna().clip(-5, 30) for n in order], showfliers=False)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=15, ha="right", fontsize=8)
    ax.axhspan(6, 8, color="green", alpha=0.12, label="patient-material target")
    ax.set_ylabel("Flesch-Kincaid grade")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, os.path.join(args.figures, "readability.png"), "Readability of each system's output")

    # ---- 3. clinical safety audit -------------------------------------------------------------
    safe_val = {n: safety_metrics(outputs[n]["validation"]["summary"].astype(str), srcs) for n in names}
    safe_val[REFERENCE] = safety_metrics(refs, srcs)
    safe_test = {n: safety_metrics(outputs[n]["test"]["summary"].astype(str), test["radiology_report"])
                 for n in names if len(test)}
    srows = []
    for n, s in safe_val.items():
        unseen = s[~seen]
        row = [n, pct(unseen["any_flag"].mean())] + [pct(unseen[k].mean()) for k in FLAG_TYPES]
        row.append(pct(safe_test[n]["any_flag"].mean()) if n in safe_test else "n/a")
        srows.append(row)
        metrics.setdefault("safety_unseen_validation", {})[n] = {k: float(unseen[k].mean()) for k in ("any_flag",) + FLAG_TYPES}
        if n in safe_test:
            metrics.setdefault("safety_test", {})[n] = {k: float(safe_test[n][k].mean()) for k in ("any_flag",) + FLAG_TYPES}
    report.append("## 3. Clinical-safety audit against the source report\n\nShare of summaries with at least one "
                  f"flag on a critical finding ({', '.join(CRITICAL)}), on the {int((~seen).sum()):,} UNSEEN validation "
                  f"reports; last column: same on the official test reports (new report text). The *{REFERENCE}* row "
                  "is the checker's false-alarm floor (the reference summaries paraphrase in ways the rules miss, and "
                  "some are genuinely incomplete); judge models relative to it.\n\n"
                  + md_table(["Text", "Any flag", "Omission", "Hallucination", "Polarity flip", "Certainty upgrade",
                              "Certainty downgrade", "Left/right swap", "False reassurance", "Any flag (test)"], srows))
    bar_chart(list(safe_val), {k.replace("_", " "): [safe_val[n][~seen][k].mean() for n in safe_val]
                               for k in ("omission", "hallucination", "polarity_flip", "certainty_upgrade",
                                         "laterality_swap")},
              os.path.join(args.figures, "safety_flags.png"), "share of summaries",
              "Automatic safety flags on unseen validation reports",
              hline=float(safe_val[REFERENCE][~seen]["any_flag"].mean()),
              hline_label="reference summaries: any flag")

    # ---- 4. confidence and guardrails ---------------------------------------------------------
    err_ref = {n: np.array([disagrees(s, r) for s, r in zip(outputs[n]["validation"]["summary"].astype(str), refs)])
               for n in names}
    grows, rc_curves = [], {}
    for n in names:
        conf = outputs[n]["validation"]["mean_logprob"].to_numpy(dtype=float)
        if not np.isfinite(conf).any():
            continue
        err = err_ref[n]
        flag = safe_val[n]["any_flag"].to_numpy()
        a_err, a_flag = auroc(-conf, err), auroc(-conf, flag)
        metrics.setdefault("confidence_auroc", {})[n] = {"reference_disagreement": a_err, "checker_flag": a_flag}
        order_c = np.argsort(-conf)                        # most confident first
        cum_err = np.cumsum(err[order_c]) / np.arange(1, len(err) + 1)
        rc_curves[n] = (np.arange(1, len(err) + 1) / len(err), cum_err)
        policies = OrderedDict([("No guardrail", np.ones(len(err), bool))])
        for cov in (0.9, 0.8):
            thr = np.quantile(conf, 1 - cov)
            policies[f"Confidence filter ({int(cov * 100)}% coverage)"] = conf >= thr
        policies["Entity check (no safety flag)"] = ~flag
        thr90 = np.quantile(conf, 0.1)
        policies["Entity check + confidence (top 90%)"] = (~flag) & (conf >= thr90)
        for pol, approved in policies.items():
            caught = err[~approved].sum() / max(1, err.sum())
            residual = err[approved].mean() if approved.any() else float("nan")
            grows.append([n, pol, pct(approved.mean()), pct(caught), pct(residual)])
            metrics.setdefault("guardrails", {}).setdefault(n, {})[pol] = {
                "auto_approved": float(approved.mean()), "errors_caught": float(caught), "residual_error": float(residual)}
        grows.append([n, "AUROC of low confidence for errors", fmt(a_err), "", f"(for checker flags: {fmt(a_flag)})"])
    if grows:
        report.append("## 4. Confidence and guardrails (validation)\n\nError label (offline audit): the summary "
                      "disagrees with the reference summary on whether a critical finding is present. *Auto-approved* "
                      "= released to the patient without review; *errors caught* = share of all errors routed to a "
                      "clinician; *residual error* = error rate among auto-approved summaries.\n\n"
                      + md_table(["System", "Policy", "Auto-approved", "Errors caught", "Residual error"], grows))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for n, (cov, risk) in rc_curves.items():
            ax.plot(cov, risk, label=n)
        ax.set_xlabel("coverage (share of summaries auto-approved, most confident first)")
        ax.set_ylabel("error rate among approved")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save_figure(fig, os.path.join(args.figures, "risk_coverage.png"), "Risk-coverage: is confidence useful?")

    # ---- 5. resource profile -------------------------------------------------------------------
    prow = []
    for n in names:
        spec = systems[n]
        if spec["kind"] != "model":
            continue
        run = {}
        if spec["path"]:
            sj = os.path.join(os.path.dirname(os.path.abspath(spec["path"])), "summary.json")
            run = {**ckpt_info.get(n, {}), **(load_json(sj) if os.path.exists(sj) else {})}
        pr = profiles.get(n, {})
        prow.append([n, f"{pr.get('parameters_total', float('nan')):,}" if pr.get("parameters_total") else "n/a",
                     f"{run.get('parameters_trainable', 0):,}" if run else "0 (no training)",
                     fmt(run.get("train_time_min"), 1) if run else "-", fmt(run.get("peak_train_vram_mb"), 0) if run else "-",
                     fmt(run.get("train_tokens_per_s"), 0) if run else "-", fmt(run.get("checkpoint_mb"), 1) if run else "-",
                     fmt(pr.get("ms_per_report_batched"), 1), fmt(pr.get("latency_ms_single_report"), 0),
                     fmt(pr.get("generated_tokens_per_s"), 0), fmt(pr.get("peak_inference_vram_mb"), 0)])
        metrics.setdefault("resources", {})[n] = {"train": {k: run.get(k) for k in ("parameters_total", "parameters_trainable",
                                                  "train_time_min", "peak_train_vram_mb", "train_tokens_per_s", "checkpoint_mb",
                                                  "best_epoch", "epochs_run")}, "inference": pr}
    if prow:
        report.append(f"## 5. Resource profile ({device}{' ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''})"
                      "\n\nInference uses greedy decoding; batched = batches of "
                      f"{args.batch_size} reports sorted by length.\n\n"
                      + md_table(["System", "Parameters", "Trainable", "Train time (min)", "Peak train VRAM (MB)",
                                  "Train tokens/s", "Checkpoint (MB)", "ms/report (batched)", "ms/report (single)",
                                  "Generated tokens/s", "Peak inference VRAM (MB)"], prow))

    # ---- 6. per-example table and autopsy candidates ------------------------------------------------
    per = val[["uid", "source", "seen_in_train", "radiology_report", "layman_report"]].copy()
    for n in names:
        k = slug(n)
        per[f"{k}_summary"] = outputs[n]["validation"]["summary"].astype(str).to_numpy()
        per[f"{k}_rougeLsum"] = rouge[n]["rougeLsum"]
        per[f"{k}_flags"] = safe_val[n]["flag_detail"].to_numpy()
        per[f"{k}_mean_logprob"] = outputs[n]["validation"]["mean_logprob"].to_numpy()
    per["reference_flags"] = safe_val[REFERENCE]["flag_detail"].to_numpy()
    per.to_csv(os.path.join(args.out, "per_example_validation.csv"), index=False)

    if main_name in outputs:
        mk = slug(main_name)
        sm = safe_val[main_name]
        conf = outputs[main_name]["validation"]["mean_logprob"].to_numpy(dtype=float)
        md = [f"# Autopsy candidates ({main_name})", "",
              "Real failure cases picked automatically from the UNSEEN validation reports. Read them, check that the "
              "flag is a genuine error (the checker can be wrong), then pick 3-5 for README Section 7.7 and explain "
              "the trigger and the clinical consequence in your own words.", ""]
        categories = [("Polarity flip (finding present <-> absent)", sm["polarity_flip"].to_numpy()),
                      ("Omission of a critical finding", sm["omission"].to_numpy()),
                      ("Hallucinated finding", sm["hallucination"].to_numpy()),
                      ("Certainty upgrade (suspected -> definite)", sm["certainty_upgrade"].to_numpy()),
                      ("Left/right swap", sm["laterality_swap"].to_numpy()),
                      ("False reassurance", sm["false_reassurance"].to_numpy()),
                      ("Near-copy of the expert report (not lay language)", textm[main_name]["near_copy"].to_numpy()),
                      ("Lowest ROUGE-Lsum among long reports",
                       (lengths >= np.quantile(lengths, 0.9)) & (rouge[main_name]["rougeLsum"] <= np.quantile(rouge[main_name]["rougeLsum"], 0.1)))]
        cand_rows = []
        for title, mask in categories:
            idx = np.where(mask & ~seen)[0]
            idx = idx[np.argsort(-np.nan_to_num(conf[idx], nan=-1e9))][:3]  # most confident errors first
            md += [f"## {title}: {int((mask & ~seen).sum())} unseen cases", ""]
            for i in idx:
                row = per.iloc[i]
                md += [f"### {row['uid']} ({row['source']})", f"- **Report:** {row['radiology_report']}",
                       f"- **{REFERENCE}:** {row['layman_report']}"]
                for n in names:
                    if systems[n]["kind"] == "model":
                        md.append(f"- **{n}:** {row[f'{slug(n)}_summary']}")
                md += [f"- Flags ({main_name}): {row[f'{mk}_flags'] or 'none'} | confidence {fmt(conf[i])} | "
                       f"ROUGE-Lsum {fmt(row[f'{mk}_rougeLsum'])}", ""]
                cand_rows.append({"category": title, "uid": row["uid"], "flags": row[f"{mk}_flags"],
                                  "mean_logprob": conf[i], "rougeLsum": row[f"{mk}_rougeLsum"]})
        # The same failure types on the official test split. No reference summaries are published for
        # it, so the cases are flagged against the report itself and there is no ROUGE.
        if len(test) and main_name in safe_test:
            st = safe_test[main_name]
            tconf = outputs[main_name]["test"]["mean_logprob"].to_numpy(dtype=float)
            md += ["# Test-split candidates", "",
                   f"Official test reports ({len(test):,} sampled; no references), checked against the report "
                   "itself. The task sheet asks for the safety audit on test cases, so 1-2 of these belong in "
                   "README Section 7.7 next to the validation cases.", ""]
            for title, key in (("Polarity flip", "polarity_flip"), ("Omission of a critical finding", "omission"),
                               ("Hallucinated finding", "hallucination"), ("Certainty upgrade", "certainty_upgrade"),
                               ("False reassurance", "false_reassurance")):
                mask = st[key].to_numpy()
                idx = np.where(mask)[0]
                idx = idx[np.argsort(-np.nan_to_num(tconf[idx], nan=-1e9))][:3]   # most confident errors first
                md += [f"## Test: {title}: {int(mask.sum())} of {len(test):,} reports", ""]
                for i in idx:
                    md += [f"### {test['uid'].iloc[i]} ({test['source'].iloc[i]})",
                           f"- **Report:** {test['radiology_report'].iloc[i]}"]
                    for n in names:
                        if systems[n]["kind"] == "model":
                            md.append(f"- **{n}:** {outputs[n]['test']['summary'].astype(str).iloc[i]}")
                    md += [f"- Flags ({main_name}): {st['flag_detail'].iloc[i] or 'none'} | confidence {fmt(tconf[i])}", ""]
                    cand_rows.append({"category": f"test: {title}", "uid": test["uid"].iloc[i],
                                      "flags": st["flag_detail"].iloc[i], "mean_logprob": tconf[i],
                                      "rougeLsum": float("nan")})
        with open(os.path.join(args.out, "autopsy_candidates.md"), "w") as f:
            f.write("\n".join(md) + "\n")
        write_csv(cand_rows, os.path.join(args.out, "autopsy_candidates.csv"))

    # ---- copy training curves for the README -------------------------------------------------
    for n in names:
        path = systems[n].get("path")
        if path:
            curves = os.path.join(os.path.dirname(os.path.abspath(path)), "curves.png")
            if os.path.exists(curves):
                shutil.copy(curves, os.path.join(args.figures, f"curves_{slug(n)}.png"))

    text = "\n\n".join(report) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    save_json(metrics, os.path.join(args.out, "metrics.json"))
    print("\n" + text, flush=True)
    print(f"Tables -> {os.path.join(args.out, 'summary.md')}; per-example table -> "
          f"{os.path.join(args.out, 'per_example_validation.csv')}; figures -> {args.figures}/", flush=True)


if __name__ == "__main__":
    main()

"""
predict.py - Example usage of a trained summariser (the driver / demo script).

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).

What it does
    1. loads a checkpoint written by train.py (LoRA adapters or full weights), or the
       untrained FLAN-T5 with --zeroshot;
    2. summarises reports: --n random reports from the held-out validation split (default),
       your own text with --text "...", or one report per line from --input file.txt;
    3. prints, for every report: the source, the human reference (if any), the generated lay
       summary, its confidence (mean token log-probability), the automatic safety flags
       (omission / hallucination / polarity flip / certainty change / false reassurance) and the
       Flesch-Kincaid grade; plus ROUGE when references exist;
    4. writes predictions/predictions.csv and a readable predictions/examples.md;
    5. with --export-merged DIR, folds the LoRA adapters into the weights and saves a plain
       Hugging Face model that anyone can load with AutoModelForSeq2SeqLM (deployment).

Usage
    python predict.py --checkpoint runs/lora/best.pt --n 5
    python predict.py --checkpoint runs/full/best.pt --text "No pneumothorax. Small left pleural effusion."
    python predict.py --zeroshot --n 5
    python predict.py --checkpoint runs/lora/best.pt --export-merged exported/flan_t5_laysumm
"""

import argparse
import os

import numpy as np
import pandas as pd

from dataset import DEFAULT_DATA_DIR, DEFAULT_MODEL, load_tokenizer, prepare_splits
from modules import LaySummariser, load_summariser
from safety import compare, fkgl
from utils import Timer, get_device, rouge_per_example, set_seed, write_csv


def parse_args():
    p = argparse.ArgumentParser(description="Generate lay summaries with a trained FLAN-T5")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="best.pt written by train.py")
    src.add_argument("--zeroshot", action="store_true", help="use the untrained FLAN-T5 (baseline)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="backbone for --zeroshot")
    p.add_argument("--text", nargs="+", default=None, help="report text(s) to summarise")
    p.add_argument("--input", default=None, help="text file with one report per line")
    p.add_argument("--n", type=int, default=5, help="validation reports to show when no text is given")
    p.add_argument("--split", default="validation", choices=["validation", "test", "dev"])
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="predictions")
    p.add_argument("--export-merged", default=None, help="save LoRA-merged Hugging Face model here")
    p.add_argument("--device", default="auto", choices=["auto", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)

    if args.zeroshot:
        model = LaySummariser(args.model, mode="zeroshot").to(device).eval()
        name, model_name = "zero-shot FLAN-T5", args.model
    else:
        model, ckpt = load_summariser(args.checkpoint, device)
        name, model_name = f"{ckpt['mode']} fine-tuned (step {ckpt.get('step')})", ckpt["config"]["model_name"]
    tokenizer = load_tokenizer(model_name)
    print(f"Loaded {name} [{model_name}] on {device}")

    if args.export_merged:
        if model.mode == "lora":
            print(f"Merged {model.merge_adapters()} LoRA layers into the backbone weights")
        model.t5.save_pretrained(args.export_merged)
        tokenizer.save_pretrained(args.export_merged)
        print(f"Saved a plain Hugging Face model to {args.export_merged} "
              f"(load with AutoModelForSeq2SeqLM.from_pretrained)")
        return
    os.makedirs(args.out, exist_ok=True)

    # ---- inputs ------------------------------------------------------------------
    if args.text or args.input:
        reports = list(args.text or [])
        if args.input:
            with open(args.input) as f:
                reports += [line.strip() for line in f if line.strip()]
        rows = pd.DataFrame({"uid": [f"input-{i}" for i in range(len(reports))],
                             "radiology_report": reports, "layman_report": [""] * len(reports)})
    else:
        split = prepare_splits(args.data_dir, verbose=False)[args.split]
        rows = split.sample(n=min(args.n, len(split)), random_state=args.seed).reset_index(drop=True)

    # ---- generation ----------------------------------------------------------------
    with Timer(device) as t:
        outs = model.summarise(rows["radiology_report"].tolist(), tokenizer, device,
                               max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
    print(f"Summarised {len(rows)} report(s) in {t.seconds:.2f}s ({1000 * t.seconds / len(rows):.0f} ms each)\n")

    has_ref = rows["layman_report"].str.len().gt(0).all()
    scores = rouge_per_example([o["summary"] for o in outs], rows["layman_report"].tolist()) if has_ref else None
    records, md = [], [f"# Example lay summaries: {name}\n"]
    for i, (row, out) in enumerate(zip(rows.itertuples(), outs)):
        flags = compare(row.radiology_report, out["summary"])
        raised = {k: v for k, v in flags.items() if k != "any" and v}
        rec = {"uid": row.uid, "report": row.radiology_report, "reference": row.layman_report,
               "summary": out["summary"], "mean_logprob": round(out["mean_logprob"], 4),
               "grade_level": round(fkgl(out["summary"]), 1), "safety_flags": raised or "none"}
        if scores is not None:
            rec.update({k: round(float(scores[k][i]), 4) for k in ("rouge1", "rouge2", "rougeLsum")})
        records.append(rec)
        print(f"--- [{i + 1}] {row.uid}")
        print(f"REPORT    : {row.radiology_report}")
        if row.layman_report:
            print(f"REFERENCE : {row.layman_report}")
        print(f"SUMMARY   : {out['summary']}")
        print(f"confidence (mean log-prob) {out['mean_logprob']:.3f} | grade level {rec['grade_level']} | "
              f"safety flags: {raised or 'none'}"
              + (f" | ROUGE-1 {rec['rouge1']:.3f} ROUGE-Lsum {rec['rougeLsum']:.3f}" if scores is not None else ""))
        md += [f"## {i + 1}. {row.uid}", f"**Report:** {row.radiology_report}", ""]
        if row.layman_report:
            md += [f"**Human lay reference:** {row.layman_report}", ""]
        md += [f"**Model summary:** {out['summary']}", "",
               f"Confidence {out['mean_logprob']:.3f} · grade level {rec['grade_level']} · "
               f"safety flags: {raised or 'none'}", ""]
    if scores is not None:
        print("\nMean ROUGE over these examples: " +
              ", ".join(f"{k} {np.mean(v):.4f}" for k, v in scores.items()))
    write_csv(records, os.path.join(args.out, "predictions.csv"))
    with open(os.path.join(args.out, "examples.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nSaved {os.path.join(args.out, 'predictions.csv')} and {os.path.join(args.out, 'examples.md')}")


if __name__ == "__main__":
    main()

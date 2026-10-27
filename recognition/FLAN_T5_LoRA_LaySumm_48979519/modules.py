"""
modules.py - Model components: LoRA adapters and the FLAN-T5 lay summariser.

Project : Expert-to-layperson translation of radiology reports with FLAN-T5,
          LoRA vs. full fine-tuning (COMP3710 Pattern Analysis, Project 2.6, Hard).

Contents (PyTorch only, no NumPy; the pre-trained FLAN-T5 backbone is loaded with
Hugging Face `transformers`, which the task requires in order to fine-tune it):
    format_prompt                     wraps a report in the instruction (config.PROMPT_TEMPLATE)
    LoRALinear                        frozen nn.Linear + trainable low-rank update [3]
    inject_lora / merge_lora          add adapters to chosen layers / fold them back in
    lora_state_dict / count_parameters
    LaySummariser                     FLAN-T5 [1, 2] in 'zeroshot', 'lora' or 'full' mode,
                                      with teacher-forced loss, generation and a per-summary
                                      confidence score (mean token log-probability)
    build_summariser / load_summariser  rebuild a model from a checkpoint

References:
    [1] C. Raffel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text
        Transformer. JMLR 21(140), 2020 (T5).
    [2] H. W. Chung et al. Scaling Instruction-Finetuned Language Models. arXiv:2210.11416,
        2022 (FLAN-T5).
    [3] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_REVISIONS, PROMPT_TEMPLATE

IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss (padding)


def format_prompt(report: str, template: str = PROMPT_TEMPLATE) -> str:
    """Wrap a radiology report in the instruction the model is trained and tested with."""
    return template.format(report=" ".join(str(report).split()))


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update ([3], Sec. 4.1).

        h = W x + (alpha / r) * B A x,     A: r x d_in,  B: d_out x r

    W stays frozen. Only A and B are trained, i.e. r * (d_in + d_out) numbers instead
    of d_in * d_out. B starts at zero, so at initialisation the layer computes exactly
    what the pre-trained layer computed (training starts from the pre-trained model).
    Dropout is applied to the input of the low-rank branch only.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=dtype))
        # Same initialisation as the LoRA reference code: A ~ Kaiming-uniform, B = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        """The frozen base weight (some library code reads `.weight.dtype`)."""
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scaling * update

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """A plain nn.Linear with W + (alpha / r) B A folded in (no extra inference cost)."""
        base = self.base
        merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                           device=base.weight.device, dtype=base.weight.dtype)
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        merged.weight.copy_((base.weight.float() + delta).to(base.weight.dtype))
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        return merged

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


def inject_lora(model: nn.Module, targets: Sequence[str] = ("q", "v"), r: int = 16,
                alpha: float = 32.0, dropout: float = 0.05) -> List[str]:
    """Replace every nn.Linear child whose attribute name is in `targets` by a LoRALinear.

    In T5 the attention projections are called q, k, v, o and the feed-forward layers
    wi_0, wi_1, wo. The default (q, v) follows the original LoRA paper ([3], Sec. 7.1).
    Afterwards only LoRA parameters require gradients. Returns the replaced module names.
    """
    targets = set(targets)
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        if isinstance(parent, LoRALinear):
            continue
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    if not replaced:
        raise ValueError(f"no nn.Linear layers named {sorted(targets)} were found")
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    return replaced


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every LoRALinear back into a plain nn.Linear, in place. Returns how many."""
    count = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, child_name, child.merged_linear())
                count += 1
    return count


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Only the adapter weights (a few MB), which is all a LoRA run needs to save."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# FLAN-T5 lay summariser
# ---------------------------------------------------------------------------
class LaySummariser(nn.Module):
    """FLAN-T5 encoder-decoder that rewrites a radiology report for a layperson.

    mode = 'zeroshot'  pre-trained weights, nothing trained (baseline)
           'lora'      backbone frozen, LoRA adapters trained
           'full'      every weight trained (full fine-tuning)

    Training objective: cross-entropy of the reference lay summary given the prompted
    report, with teacher forcing (the decoder sees the true previous tokens).
    """

    def __init__(self, model_name: str = "google/flan-t5-base", mode: str = "lora",
                 lora_r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05,
                 lora_targets: Sequence[str] = ("q", "v"), revision: Optional[str] = None,
                 backbone: Optional[nn.Module] = None):
        """`backbone` lets tests pass a tiny randomly initialised T5 instead of downloading one."""
        super().__init__()
        if mode not in ("zeroshot", "lora", "full"):
            raise ValueError(f"mode must be zeroshot, lora or full, got {mode}")
        revision = revision or MODEL_REVISIONS.get(model_name)
        if backbone is None:
            from transformers import AutoModelForSeq2SeqLM  # weights come from the Hugging Face cache
            backbone = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
        self.t5 = backbone
        self.mode = mode
        self.config = dict(model_name=model_name, mode=mode, lora_r=lora_r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, lora_targets=list(lora_targets),
                           revision=revision)
        self.lora_modules: List[str] = []
        if mode == "lora":
            self.lora_modules = inject_lora(self.t5, lora_targets, lora_r, lora_alpha, lora_dropout)
        elif mode == "zeroshot":
            for p in self.t5.parameters():
                p.requires_grad_(False)

    @property
    def pad_token_id(self) -> int:
        return self.t5.config.pad_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        """Teacher-forced pass. With labels, `.loss` is the mean token cross-entropy."""
        return self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # -- checkpoints ------------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """What a checkpoint must store: adapters only (LoRA) or every weight (full)."""
        if self.mode == "lora":
            return lora_state_dict(self)
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise KeyError(f"unexpected keys in checkpoint: {unexpected[:5]}")
        if self.mode == "lora":
            missing_lora = [k for k in missing if "lora_" in k]
            if missing_lora:
                raise KeyError(f"checkpoint is missing LoRA weights: {missing_lora[:5]}")

    def merge_adapters(self) -> int:
        """Fold LoRA into the backbone so the model can be shared as a plain FLAN-T5."""
        return merge_lora(self.t5)

    # -- inference ---------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int = 320, num_beams: int = 1) -> torch.Tensor:
        """Decode summaries (greedy by default). Returns token ids incl. the start token."""
        return self.t5.generate(input_ids=input_ids, attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens, num_beams=num_beams,
                                do_sample=False)

    @torch.no_grad()
    def sequence_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          sequences: torch.Tensor, chunk: int = 8) -> Dict[str, torch.Tensor]:
        """Confidence of each generated summary, by re-scoring it with teacher forcing.

        For every output token y_t we take log p(y_t | y_<t, report), including the
        end-of-sequence token, and ignore padding. Returns per-summary
            mean_logprob  average token log-probability (closer to 0 = more confident)
            min_logprob   the least confident token
            n_tokens      number of scored tokens
        One extra forward pass is much cheaper than storing all generation scores. It runs
        in chunks of `chunk` summaries, each cut to its longest summary, because the full
        (summaries x length x 32,128-word vocabulary) probability table would need GBs.
        """
        outs = {"mean_logprob": [], "min_logprob": [], "n_tokens": []}
        for s in range(0, sequences.shape[0], chunk):
            seq = sequences[s:s + chunk]                   # [start, y_1, ..., y_T, pad, pad, ...]
            targets = seq[:, 1:]                           # the tokens the model chose
            mask = targets != self.pad_token_id            # padding after </s> is not part of a summary
            length = max(1, int(mask.sum(1).max()))       # generated tokens come first, then padding
            targets, mask = targets[:, :length], mask[:, :length]
            # Teacher forcing: the decoder reads [start, y_1, ..., y_{T-1}] and predicts y_1 ... y_T.
            # (Passing decoder inputs instead of labels also avoids the model computing a loss.)
            logits = self.t5(input_ids=input_ids[s:s + chunk], attention_mask=attention_mask[s:s + chunk],
                             decoder_input_ids=seq[:, :length].contiguous()).logits
            logp = F.log_softmax(logits.float(), dim=-1)
            token_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
            n = mask.sum(1)
            outs["mean_logprob"].append(token_lp.sum(1) / n.clamp(min=1))
            outs["min_logprob"].append(token_lp.masked_fill(~mask, float("inf")).min(1).values)
            outs["n_tokens"].append(n)
        return {k: torch.cat(v) for k, v in outs.items()}

    @torch.no_grad()
    def summarise(self, reports: Sequence[str], tokenizer, device: torch.device,
                  batch_size: int = 32, max_source_len: int = 384, max_new_tokens: int = 320,
                  num_beams: int = 1, amp: bool = True,
                  prompt_template: str = PROMPT_TEMPLATE) -> List[Dict]:
        """Summarise a list of reports. Returns one dict per report, in input order:
        {'summary', 'mean_logprob', 'min_logprob', 'n_tokens'}.

        Reports are processed longest-first so each batch holds similar lengths
        (less padding, faster generation); results are put back in the original order.
        """
        was_training = self.training
        self.eval()
        prompts = [format_prompt(r, prompt_template) for r in reports]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        results: List[Optional[Dict]] = [None] * len(prompts)
        use_amp = amp and device.type == "cuda"
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_source_len).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                seqs = self.generate(enc["input_ids"], enc["attention_mask"], max_new_tokens, num_beams)
                conf = self.sequence_logprobs(enc["input_ids"], enc["attention_mask"], seqs)
            texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
            for j, i in enumerate(idx):
                results[i] = {"summary": " ".join(texts[j].split()),
                              "mean_logprob": float(conf["mean_logprob"][j]),
                              "min_logprob": float(conf["min_logprob"][j]),
                              "n_tokens": int(conf["n_tokens"][j])}
        if was_training:
            self.train()
        return results


def build_summariser(config: Dict) -> LaySummariser:
    """Rebuild the architecture stored in a checkpoint's `config` (backbone from the Hub cache)."""
    cfg = dict(config)
    cfg["lora_targets"] = tuple(cfg.get("lora_targets", ("q", "v")))
    return LaySummariser(**cfg)


def load_summariser(path: str, device: torch.device) -> Tuple[LaySummariser, Dict]:
    """Load a checkpoint written by train.py (adapters or full weights) onto `device`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_summariser(ckpt["config"])
    model.load_trainable_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt

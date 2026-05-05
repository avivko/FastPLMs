#!/usr/bin/env python3
"""Fine-tune FastPLMs sequence models (classification, regression, or MLM) from CSV (Docker/Singularity)."""

from __future__ import annotations

import argparse
import os
import sys

# Trainer import otherwise pulls TensorFlow/Keras; image has Keras 3 without tf-keras shim.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from drorlab_fastplms.cli_common import (
    apply_attn_backend_after_load,
    configure_hf_token,
    is_e1_config,
    is_esmc_config,
    model_config_with_attn,
    resolve_torch_dtype,
    try_entrypoint_setup,
)
from drorlab_fastplms.e1_context import (
    build_e1_row_strings,
    normalize_e1_multiseq_string,
    prepare_e1_inputs_for_runtime,
    validate_e1_embed_inputs,
)


def reinit_classifier_head_if_nonfinite(model: torch.nn.Module) -> None:
    """Re-init the HF sequence head if any parameter is NaN/Inf (can happen with ``ignore_mismatched_sizes``)."""
    clf = getattr(model, "classifier", None)
    if clf is None:
        return
    parts = [p.detach().flatten() for p in clf.parameters() if p is not None and p.numel() > 0]
    if not parts or torch.isfinite(torch.cat(parts)).all():
        return

    def _reset(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    clf.apply(_reset)


class SequenceTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_fn: Callable[[Any], str], label_col: str):
        self.df = df.reset_index(drop=True)
        self.text_fn = text_fn
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {"text": self.text_fn(row), "label": row[self.label_col]}


class TokenizerCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        regression: bool,
        add_sequence_id: bool,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.regression = regression
        self.add_sequence_id = add_sequence_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        raw_labels = [f["label"] for f in features]
        labels = torch.tensor(raw_labels)
        if self.regression:
            labels = labels.float()
        else:
            labels = labels.long()
        batch = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.add_sequence_id:
            batch["sequence_id"] = batch["attention_mask"].to(dtype=torch.bool)
        batch["labels"] = labels
        return batch


class E1Collator:
    def __init__(self, preparer: Any, regression: bool):
        self.preparer = preparer
        self.regression = regression

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f["text"] for f in features]
        raw_labels = [f["label"] for f in features]
        labels = torch.tensor(raw_labels)
        if self.regression:
            labels = labels.float()
        else:
            labels = labels.long()
        batch = self.preparer.get_batch_kwargs(texts, device=torch.device("cpu"))
        batch.pop("labels", None)
        batch["labels"] = labels
        return batch


BASE_TRAINER_KWARGS = {
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "save_strategy": "steps",
    "save_steps": 200,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "none",
    "label_names": ["labels"],
}


def _uses_tensorboard(report_to: Union[str, List[str]]) -> bool:
    if report_to == "all":
        return True
    if isinstance(report_to, list):
        return "tensorboard" in report_to
    return report_to == "tensorboard"


def _resolve_report_to(name: str) -> Union[str, List[str]]:
    n = name.strip().lower()
    if n in ("none", ""):
        return "none"
    if n == "both":
        return ["wandb", "tensorboard"]
    if n in ("all", "wandb", "tensorboard"):
        return n
    raise ValueError(f"Unknown report backend {name!r}; use none, wandb, tensorboard, both, or all")


def build_trainer_kw(
    args: argparse.Namespace,
    steps_per_epoch: int,
    total_train_steps: int,
    *,
    for_mlm: bool,
) -> Dict[str, Any]:
    """HF ``TrainingArguments`` kwargs shared by MLM and seq-classification paths."""
    tr = dict(BASE_TRAINER_KWARGS)
    if for_mlm:
        tr.pop("label_names", None)
    tr["report_to"] = _resolve_report_to(getattr(args, "report_to", "none"))

    ev = getattr(args, "eval_strategy", "steps")
    if ev == "epoch":
        tr["eval_strategy"] = "epoch"
        tr["save_strategy"] = "epoch"
        tr.pop("eval_steps", None)
        tr.pop("save_steps", None)
        if steps_per_epoch < 200:
            tr["logging_steps"] = 1
    else:
        if steps_per_epoch < 200:
            tr["logging_steps"] = 1
            tr["eval_steps"] = max(1, min(tr["eval_steps"], steps_per_epoch))
            tr["save_steps"] = max(1, min(tr["save_steps"], steps_per_epoch))
        elif tr["report_to"] != "none" and steps_per_epoch >= 200:
            tr["logging_steps"] = min(tr["logging_steps"], max(50, steps_per_epoch // 200))

    if tr.get("warmup_steps", 0) > total_train_steps:
        tr["warmup_steps"] = 0
    return tr


def compute_metrics_regression(p: EvalPrediction) -> Dict[str, float]:
    pred = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    pred = pred.flatten()
    lab = p.label_ids.flatten()
    rho, _ = spearmanr(pred, lab)
    return {"spearman": float(rho) if rho == rho else 0.0}


def get_hf_tokenizer(model: Any) -> PreTrainedTokenizerBase:
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        return model.tokenizer
    bm = getattr(model, "base_model", None)
    if bm is not None and getattr(bm, "tokenizer", None) is not None:
        return bm.tokenizer
    inner = getattr(bm, "model", None) if bm is not None else None
    if inner is not None and getattr(inner, "tokenizer", None) is not None:
        return inner.tokenizer
    raise RuntimeError("Could not find a HuggingFace tokenizer on this model (expected non-E1).")


def compute_metrics_classification(p: EvalPrediction) -> Dict[str, float]:
    pred = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    yhat = np.argmax(pred, axis=-1)
    lab = p.label_ids
    return {"accuracy": float(accuracy_score(lab, yhat))}


def _maybe_subsample(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.iloc[:max_rows].copy()


def load_train_val_dfs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train CSV, optional val CSV, 10% holdout if val omitted, then optional row caps."""
    train_df = pd.read_csv(args.train_csv)
    if args.val_csv:
        val_df = pd.read_csv(args.val_csv)
    else:
        n = max(1, len(train_df) // 10)
        if len(train_df) > n:
            val_df = train_df.tail(n).copy()
            train_df = train_df.iloc[:-n].copy()
        else:
            val_df = train_df.copy()
    train_df = _maybe_subsample(train_df, args.max_train_rows)
    val_df = _maybe_subsample(val_df, args.max_val_rows)
    return train_df, val_df


class SequenceMLMDataset(Dataset):
    """Single-sequence rows for MLM (no labels; masking happens in the collator)."""

    def __init__(self, df: pd.DataFrame, text_fn: Callable[[Any], str]):
        self.df = df.reset_index(drop=True)
        self.text_fn = text_fn

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {"text": self.text_fn(row)}


class MLMCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        mlm_probability: float,
        add_sequence_id: bool,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_sequence_id = add_sequence_id
        self._mlm = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        per_example: List[Dict[str, Any]] = [
            {"input_ids": encoded["input_ids"][i], "attention_mask": encoded["attention_mask"][i]}
            for i in range(len(texts))
        ]
        batch = self._mlm(per_example)
        if self.add_sequence_id:
            batch["sequence_id"] = batch["attention_mask"].to(dtype=torch.bool)
        return batch


def run_mlm(args: argparse.Namespace) -> int:
    """Masked LM fine-tuning for tokenizer-backed models (e.g. ESM2, ESM++, DPLM, ANKH)."""
    if not args.no_entrypoint_setup:
        try_entrypoint_setup()
    configure_hf_token()

    cfg = model_config_with_attn(args.model, args.attn_backend)
    if is_e1_config(cfg):
        raise ValueError(
            "MLM fine-tuning is only supported for tokenizer models (e.g. Synthyra/ESM2-8M). "
            "E1 is not supported here."
        )
    dtype = resolve_torch_dtype(args.dtype)

    train_df, val_df = load_train_val_dfs(args)
    if args.seq_col not in train_df.columns:
        raise ValueError(f"Missing sequence column {args.seq_col!r}")

    def text_fn(row: pd.Series) -> str:
        return str(row[args.seq_col])[: args.max_length]

    train_ds = SequenceMLMDataset(train_df, text_fn)
    val_ds = SequenceMLMDataset(val_df, text_fn)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {args.model} for MLM (train={len(train_ds)}, val={len(val_ds)}) ...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        config=cfg,
        trust_remote_code=True,
        dtype=dtype,
    )
    apply_attn_backend_after_load(model, args.attn_backend, cfg)

    if not args.no_lora:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora)
        for n, p in model.named_parameters():
            if "lm_head" in n or "lm_out" in n:
                p.requires_grad = True

    tok = get_hf_tokenizer(model)
    collator = MLMCollator(
        tok,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        add_sequence_id=is_esmc_config(cfg),
    )

    use_bf16 = dtype == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = dtype == torch.float16 and torch.cuda.is_available()

    steps_per_epoch = max(1, (len(train_ds) + args.batch_size - 1) // args.batch_size)
    total_train_steps = steps_per_epoch * args.epochs
    tr_kw = build_trainer_kw(args, steps_per_epoch, total_train_steps, for_mlm=True)
    t_extra: Dict[str, Any] = {}
    if _uses_tensorboard(tr_kw["report_to"]):
        t_extra["logging_dir"] = os.path.join(args.output_dir, "tb")
    if getattr(args, "run_name", None):
        t_extra["run_name"] = args.run_name

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        remove_unused_columns=False,
        bf16=use_bf16,
        fp16=use_fp16,
        **tr_kw,
        **t_extra,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"Done. Checkpoints under {args.output_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune FastPLMs for seq classification / regression / MLM.")
    ap.add_argument("--model", required=True, help="HF model id, e.g. Synthyra/ESM2-150M")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", default=None)
    ap.add_argument("--seq-col", required=True, help="CSV column name for sequences")
    ap.add_argument("--label-col", default="label", help="Ignored when --task mlm")
    ap.add_argument("--task", choices=("regression", "classification", "mlm"), required=True)
    ap.add_argument("--num-labels", type=int, default=None, help="Classification: number of classes (required if task=classification)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument(
        "--attn-backend",
        default="auto",
        help="auto | sdpa | flex | kernels_flash (default: auto; use sdpa for strict reproducibility)",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-entrypoint-setup", action="store_true")
    ap.add_argument("--e1-combined-col", default=None)
    ap.add_argument("--e1-context-cols", nargs="*", default=None)
    ap.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="MLM mask probability (only for --task mlm)",
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="After train/val split, keep at most this many training rows (large CSVs)",
    )
    ap.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="After split, cap validation rows",
    )
    ap.add_argument(
        "--report-to",
        default="none",
        choices=("none", "wandb", "tensorboard", "both", "all"),
        help="Log train/eval loss to W&B and/or TensorBoard (set WANDB_API_KEY or wandb login for cloud)",
    )
    ap.add_argument("--run-name", default=None, help="Run name for Weights & Biases and HF Trainer")
    ap.add_argument(
        "--eval-strategy",
        default="steps",
        choices=("steps", "epoch"),
        help="Run validation every N steps (default) or once per epoch (clearer epoch-wise curves)",
    )
    args = ap.parse_args(argv)

    if args.task == "mlm":
        return run_mlm(args)

    if not args.no_entrypoint_setup:
        try_entrypoint_setup()
    configure_hf_token()

    if args.task == "classification" and args.num_labels is None:
        print("--num-labels is required for classification", file=sys.stderr)
        return 2

    num_labels = 1 if args.task == "regression" else int(args.num_labels)
    regression = args.task == "regression"

    cfg = model_config_with_attn(args.model, args.attn_backend)
    cfg.num_labels = num_labels
    cfg.problem_type = "regression" if regression else "single_label_classification"
    e1 = is_e1_config(cfg)
    dtype = resolve_torch_dtype(args.dtype)

    train_df, val_df = load_train_val_dfs(args)

    if args.label_col not in train_df.columns:
        raise ValueError(f"Missing label column {args.label_col!r}")
    if args.seq_col not in train_df.columns:
        raise ValueError(f"Missing sequence column {args.seq_col!r}")

    use_e1_multiseq = e1 and bool(
        args.e1_combined_col or (args.e1_context_cols and len(args.e1_context_cols) > 0)
    )

    if use_e1_multiseq:

        def text_fn(row: pd.Series) -> str:
            # Do not slice here: multiseq context can exceed --max-length characters; E1
            # enforces limits in prep_tokens (max positions / sequences), not raw str len.
            normalized = normalize_e1_multiseq_string(
                build_e1_row_strings(
                    combined_col=args.e1_combined_col,
                    context_cols=args.e1_context_cols or [],
                    query_col=args.seq_col,
                    row=row.to_dict(),
                )
            )
            return prepare_e1_inputs_for_runtime([normalized], truncate=False, max_len=args.max_length)[0]

    else:

        def text_fn(row: pd.Series) -> str:
            raw = str(row[args.seq_col])
            if e1:
                normalized = normalize_e1_multiseq_string(raw)
                return prepare_e1_inputs_for_runtime([normalized], truncate=False, max_len=args.max_length)[0]
            return raw[: args.max_length]

    if e1:
        train_texts = [text_fn(train_df.iloc[i]) for i in range(len(train_df))]
        val_texts = [text_fn(val_df.iloc[i]) for i in range(len(val_df))]
        try:
            validate_e1_embed_inputs(
                train_texts + val_texts,
                row_labels=[f"train:{i}" for i in range(len(train_texts))]
                + [f"val:{i}" for i in range(len(val_texts))],
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2

    train_ds = SequenceTextDataset(train_df, text_fn, args.label_col)
    val_ds = SequenceTextDataset(val_df, text_fn, args.label_col)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model} (e1={e1}, num_labels={num_labels}) ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=cfg,
        trust_remote_code=True,
        dtype=dtype,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    apply_attn_backend_after_load(model, args.attn_backend, cfg)
    reinit_classifier_head_if_nonfinite(model)

    if not args.no_lora:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora)
        for n, p in model.named_parameters():
            if "classifier" in n:
                p.requires_grad = True

    if e1:
        collator = E1Collator(model.model.prep_tokens, regression=regression)
    else:
        tok = get_hf_tokenizer(model)
        collator = TokenizerCollator(
            tok,
            max_length=args.max_length,
            regression=regression,
            add_sequence_id=is_esmc_config(cfg),
        )

    metrics_fn = compute_metrics_regression if regression else compute_metrics_classification

    use_bf16 = dtype == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = dtype == torch.float16 and torch.cuda.is_available()

    steps_per_epoch = max(1, (len(train_ds) + args.batch_size - 1) // args.batch_size)
    total_train_steps = steps_per_epoch * args.epochs
    tr_kw = build_trainer_kw(args, steps_per_epoch, total_train_steps, for_mlm=False)
    t_extra: Dict[str, Any] = {}
    if _uses_tensorboard(tr_kw["report_to"]):
        t_extra["logging_dir"] = os.path.join(args.output_dir, "tb")
    if getattr(args, "run_name", None):
        t_extra["run_name"] = args.run_name

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        remove_unused_columns=False,
        bf16=use_bf16,
        fp16=use_fp16,
        **tr_kw,
        **t_extra,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"Done. Checkpoints under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

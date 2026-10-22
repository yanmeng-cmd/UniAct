#!/usr/bin/env python3
"""Inference-only comparison of standard GELU and the activation golden model on GLUE dev sets.

Every task loads an already fine-tuned sequence-classification checkpoint. This
program never trains, fine-tunes, optimizes, or saves model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from uniact import make_activation_replacement

TASKS = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli")
CHECKPOINTS = {
    "cola": "yoshitomo-matsubara/bert-large-uncased-cola",
    "sst2": "yoshitomo-matsubara/bert-large-uncased-sst2",
    "mrpc": "yoshitomo-matsubara/bert-large-uncased-mrpc",
    "stsb": "yoshitomo-matsubara/bert-large-uncased-stsb",
    "qqp": "yoshitomo-matsubara/bert-large-uncased-qqp",
    "mnli": "yoshitomo-matsubara/bert-large-uncased-mnli",
    "qnli": "yoshitomo-matsubara/bert-large-uncased-qnli",
    "rte": "yoshitomo-matsubara/bert-large-uncased-rte",
    "wnli": "yoshitomo-matsubara/bert-large-uncased-wnli",
}
TEXT_COLUMNS = {
    "cola": ("sentence",), "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2"), "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"), "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def tokenize_dataset(dataset, tokenizer, task: str, max_length: int):
    def tokenize(batch):
        return tokenizer(*(batch[column] for column in TEXT_COLUMNS[task]),
                         truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    keep = set(tokenizer.model_input_names) | {"label"}
    drop = [column for column in tokenized.column_names if column not in keep]
    if drop:
        tokenized = tokenized.remove_columns(drop)
    return tokenized


def task_metrics(task: str, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if task == "stsb":
        pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
        gold = np.asarray(labels, dtype=np.float64).reshape(-1)
        pearson = float(pearsonr(pred, gold).statistic)
        spearman = float(spearmanr(pred, gold).statistic)
        return {"pearson": pearson, "spearman": spearman,
                "glue_score": (pearson + spearman) / 2.0}

    pred = np.asarray(predictions).argmax(axis=-1)
    labels = np.asarray(labels)
    result = {"accuracy": float(accuracy_score(labels, pred))}
    if task in {"mrpc", "qqp"}:
        result["f1"] = float(f1_score(labels, pred, zero_division=0))
    if task == "cola":
        result["matthews_correlation"] = float(matthews_corrcoef(labels, pred))
        result["glue_score"] = result["matthews_correlation"]
    elif task in {"mrpc", "qqp"}:
        result["glue_score"] = (result["accuracy"] + result["f1"]) / 2.0
    else:
        result["glue_score"] = result["accuracy"]
    return result


def run_prediction(model, dataset, tokenizer, task: str, batch_size: int,
                   device: torch.device, workers: int, fp16: bool) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    all_logits, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            if fp16 and device.type == "cuda":
                inputs = {key: value if not value.is_floating_point() else value.half()
                          for key, value in inputs.items()}
            logits = model(**inputs).logits
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return task_metrics(task, np.concatenate(all_logits), np.concatenate(all_labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.cpu and args.fp16:
        parser.error("--fp16 is supported only for CUDA inference")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.set_num_threads(max(1, min(32, args.workers * 2)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "inference_only",
        "training_performed": False,
        "model": "BERT-large",
        "activation": "Q4.4/Q0.8 golden model with signed Q8.8 external interface",
        "max_length": args.max_length,
        "tasks": {},
    }

    for task in args.tasks:
        checkpoint_id = CHECKPOINTS[task]
        print(f"[BERT-large/{task}] loading fine-tuned checkpoint {checkpoint_id}",
              flush=True)
        raw = load_dataset("glue", task, cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, use_fast=True,
                                                  cache_dir=args.cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_id, cache_dir=args.cache_dir
        )
        if args.fp16 and device.type == "cuda":
            model = model.half()
        model.to(device)

        split_names = (("validation_matched", "validation_mismatched")
                       if task == "mnli" else ("validation",))
        task_result = {}
        for split in split_names:
            data = tokenize_dataset(raw[split], tokenizer, task, args.max_length)
            exact_metrics = run_prediction(
                model, data, tokenizer, task, args.eval_batch_size, device,
                args.workers, args.fp16,
            )
            gelu_sites = [
                (module, module.intermediate_act_fn)
                for module in model.modules()
                if hasattr(module, "intermediate_act_fn")
            ]
            if len(gelu_sites) != model.config.num_hidden_layers:
                raise RuntimeError(
                    f"{checkpoint_id}: expected {model.config.num_hidden_layers} GELU "
                    f"sites, found {len(gelu_sites)}"
                )
            try:
                for module, original in gelu_sites:
                    module.intermediate_act_fn = make_activation_replacement(
                        original, "gelu_logistic")
                approx_metrics = run_prediction(
                    model, data, tokenizer, task, args.eval_batch_size, device,
                    args.workers, args.fp16,
                )
            finally:
                for module, original in gelu_sites:
                    module.intermediate_act_fn = original

            task_result[split] = {
                "checkpoint": checkpoint_id,
                "exact_gelu": exact_metrics,
                "activation_golden_model": approx_metrics,
                "delta": {key: approx_metrics[key] - exact_metrics[key]
                          for key in exact_metrics if key in approx_metrics},
            }
            print(f"[BERT-large/{task}/{split}] "
                  f"exact={exact_metrics} approx={approx_metrics}", flush=True)

        report["tasks"][task] = task_result
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        del model, raw, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    exact_scores, approx_scores = [], []
    for splits in report["tasks"].values():
        exact_scores.append(float(np.mean(
            [part["exact_gelu"]["glue_score"] for part in splits.values()]
        )))
        approx_scores.append(float(np.mean(
            [part["activation_golden_model"]["glue_score"] for part in splits.values()]
        )))
    if exact_scores:
        report["mean_glue_score"] = {
            "exact_gelu": float(np.mean(exact_scores)),
            "activation_golden_model": float(np.mean(approx_scores)),
            "delta": float(np.mean(approx_scores) - np.mean(exact_scores)),
            "tasks_in_average": list(report["tasks"]),
        }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

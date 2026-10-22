#!/usr/bin/env python3
"""Inference-only perplexity comparison for TinyLlama on WikiText-2."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from uniact import make_activation_replacement

MODEL_ID = "TinyLlama/TinyLlama_v1.1"


def install_silu_approximator(model: torch.nn.Module) -> tuple[list[tuple[object, object]], int]:
    """Replace each Llama MLP's callable SiLU with the fixed-point golden model."""
    changed = []
    for module in model.modules():
        if hasattr(module, "act_fn"):
            original = module.act_fn
            if "silu" in getattr(original, "__name__", type(original).__name__).lower():
                changed.append((module, original))
                module.act_fn = make_activation_replacement(original, "silu")
    expected = int(model.config.num_hidden_layers)
    if len(changed) != expected:
        raise RuntimeError(
            f"expected one SiLU gate per decoder layer ({expected}), found {len(changed)}"
        )
    return changed, len(changed)


@torch.inference_mode()
def perplexity(model, token_ids: torch.Tensor, device: torch.device,
               context: int, stride: int, max_tokens: int = 0) -> dict:
    model.eval()
    if max_tokens > 0:
        token_ids = token_ids[:, :max_tokens]
    total_tokens = token_ids.shape[1]
    if total_tokens < 2:
        raise ValueError("WikiText-2 test split produced fewer than two tokens")

    total_nll = 0.0
    counted = 0
    previous_end = 0
    start_time = time.perf_counter()
    for begin in range(0, total_tokens, stride):
        end = min(begin + context, total_tokens)
        new_tokens = end - previous_end
        if new_tokens <= 0:
            break
        input_ids = token_ids[:, begin:end].to(device)
        labels = input_ids.clone()
        labels[:, :-new_tokens] = -100
        result = model(input_ids=input_ids, labels=labels)
        # Causal-LM loss predicts the next token; each newly covered target is
        # counted once even though context windows overlap.
        target_count = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(result.loss) * target_count
        counted += target_count
        previous_end = end
        if end == total_tokens:
            break
        if end - begin < context:
            break
        if counted and counted % 100_000 < context:
            print(f"perplexity pass: {counted}/{total_tokens} tokens", flush=True)

    if counted == 0:
        raise RuntimeError("no next-token targets were scored")
    mean_nll = total_nll / counted
    return {
        "perplexity": math.exp(mean_nll),
        "mean_negative_log_likelihood": mean_nll,
        "scored_tokens": counted,
        "seconds": time.perf_counter() - start_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Perplexity uses one continuous stream; retained for report clarity.")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Optional smoke-test cap; 0 evaluates the complete test split.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("continuous-stream perplexity currently requires --batch-size 1")
    if args.stride <= 0 or args.threads <= 0 or args.max_tokens < 0:
        parser.error("stride/threads must be positive and max-tokens nonnegative")
    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        parser.error("--fp16 requires a CUDA device")
    torch.set_num_threads(args.threads)

    print(f"loading inference checkpoint {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    ).to(device).eval()
    context = int(model.config.max_position_embeddings)
    if args.stride > context:
        raise ValueError(f"stride {args.stride} exceeds model context {context}")
    stride = args.stride

    print("loading WikiText-2 raw test split", flush=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                           split="test", cache_dir=args.cache_dir)
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids

    exact = perplexity(model, token_ids, device, context, stride, args.max_tokens)
    changed, replaced = install_silu_approximator(model)
    try:
        approximate = perplexity(model, token_ids, device, context, stride, args.max_tokens)
    finally:
        for module, original in changed:
            module.act_fn = original

    report = {
        "mode": "inference_only",
        "training_performed": False,
        "model_id": args.model_id,
        "dataset": "Salesforce/wikitext, wikitext-2-raw-v1, test split",
        "dataset_examples": len(dataset),
        "input_tokens": int(token_ids.shape[1]),
        "device": str(device),
        "dtype": "float16" if args.fp16 else "float32",
        "context_length": context,
        "stride": stride,
        "max_tokens": args.max_tokens,
        "activation_replacement": "all TinyLlama gated-MLP SiLU sites",
        "activation_modules_replaced": replaced,
        "exact": exact,
        "activation_golden_model": approximate,
        "delta_perplexity": approximate["perplexity"] - exact["perplexity"],
        "delta_perplexity_percent": 100.0 * (
            approximate["perplexity"] / exact["perplexity"] - 1.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

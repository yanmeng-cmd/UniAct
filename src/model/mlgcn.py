#!/usr/bin/env python3
"""Inference-only ResNet-101 + ML-GCN evaluation on COCO 2014 validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from golden import Mode
from uniact import TorchActivationFunction


def load_upstream_model(repo: Path, checkpoint: Path, device: torch.device):
    """Load the upstream CVPR'19 implementation without invoking its trainer."""
    repo = repo.resolve()
    if not (repo / "models.py").is_file() or not (repo / "coco.py").is_file():
        raise FileNotFoundError(f"{repo} is not a clone of megvii-research/ML-GCN")
    sys.path.insert(0, str(repo))
    # The upstream repository predates NumPy 1.24 and references np.int.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    from models import gcn_resnet101  # pylint: disable=import-outside-toplevel

    adj_file = repo / "data" / "coco" / "coco_adj.pkl"
    if not adj_file.is_file():
        raise FileNotFoundError(f"missing upstream COCO graph file: {adj_file}")
    model = gcn_resnet101(num_classes=80, t=0.4, pretrained=False,
                          adj_file=str(adj_file))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value
                      for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def evaluate(model, loader, device: torch.device, use_approx: bool,
             threshold: float) -> dict:
    sigmoid = TorchActivationFunction(Mode.SIGMOID) if use_approx else torch.sigmoid
    scores, targets = [], []
    with torch.inference_mode():
        for step, ((images, _names, embedding_batch), target) in enumerate(loader, 1):
            model_dtype = next(model.parameters()).dtype
            images = images.to(device=device, dtype=model_dtype, non_blocking=True)
            if torch.is_tensor(embedding_batch):
                embeddings = embedding_batch.to(device=device, dtype=model_dtype, non_blocking=True)
            else:
                embeddings = torch.as_tensor(np.asarray(embedding_batch), device=device,
                                             dtype=model_dtype)
            logits = model(images, embeddings)
            probabilities = sigmoid(logits.float()).clamp(0.0, 1.0)
            scores.append(probabilities.cpu().numpy())
            targets.append(target.numpy())
            if step % 100 == 0:
                print(f"COCO validation: {step * loader.batch_size} images", flush=True)

    probability = np.concatenate(scores, axis=0)
    signed_target = np.concatenate(targets, axis=0)
    labels = (signed_target == 1).astype(np.int32)
    class_ap = average_precision_score(labels, probability, average=None)
    predicted = probability >= threshold
    return {
        "examples": int(labels.shape[0]),
        "classes": int(labels.shape[1]),
        "map": float(np.mean(class_ap)),
        "micro_f1_at_fixed_threshold": float(
            f1_score(labels.reshape(-1), predicted.reshape(-1), zero_division=0)
        ),
        "macro_f1_at_fixed_threshold": float(
            f1_score(labels, predicted, average="macro", zero_division=0)
        ),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlgcn-repo", type=Path, required=True,
                        help="Official megvii-research/ML-GCN checkout")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Official COCO checkpoint file")
    parser.add_argument("--coco-root", type=Path, required=True,
                        help="ML-GCN COCO root; official loader prepares val images/annotations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or not 0.0 < args.threshold < 1.0:
        parser.error("batch-size must be positive, workers nonnegative, threshold in (0,1)")
    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        parser.error("--fp16 requires CUDA")
    model = load_upstream_model(args.mlgcn_repo, args.checkpoint, device)
    from coco import COCO2014  # pylint: disable=import-outside-toplevel
    from util import Warp  # pylint: disable=import-outside-toplevel

    embedding_path = args.mlgcn_repo / "data" / "coco" / "coco_glove_word2vec.pkl"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"missing upstream COCO label embeddings: {embedding_path}")
    transform = transforms.Compose([
        Warp(448), transforms.ToTensor(),
        transforms.Normalize(mean=model.image_normalization_mean,
                             std=model.image_normalization_std),
    ])
    dataset = COCO2014(str(args.coco_root), transform=transform, phase="val",
                       inp_name=str(embedding_path))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    if args.fp16:
        model.half()

    exact = evaluate(model, loader, device, use_approx=False, threshold=args.threshold)
    approx = evaluate(model, loader, device, use_approx=True, threshold=args.threshold)
    report = {
        "mode": "inference_only",
        "training_performed": False,
        "model": "ResNet-101 + ML-GCN (official COCO 2014 checkpoint)",
        "dataset": "MS-COCO 2014 validation",
        "activation_replacement": "output sigmoid only",
        "device": str(device),
        "dtype": "float16" if args.fp16 else "float32",
        "results": {
            "exact_sigmoid": exact,
            "activation_golden_model": approx,
            "delta_map": approx["map"] - exact["map"],
            "delta_micro_f1": (approx["micro_f1_at_fixed_threshold"] -
                               exact["micro_f1_at_fixed_threshold"]),
            "delta_macro_f1": (approx["macro_f1_at_fixed_threshold"] -
                               exact["macro_f1_at_fixed_threshold"]),
        },
        "notes": [
            "The standard ML-GCN checkpoint and upstream architecture are used unchanged.",
            "The project evaluator replaces only the final score sigmoid; no weights are updated.",
            "mAP uses per-class average precision over all 80 classes; F1 uses the fixed probability threshold.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

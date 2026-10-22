#!/usr/bin/env python3
"""Inference-only ImageNet-1K evaluation for EfficientNet-B0 and DeiT-Tiny.

EfficientNet-B0 reports isolated SiLU-only, isolated Sigmoid-only, and joint
SiLU+Sigmoid replacement. DeiT-Tiny reports GELU replacement.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from timm import create_model
from timm.data import create_transform, resolve_model_data_config

from uniact import replace_vision_activations


def build_model(name: str):
    if name == "efficientnet-b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        return model, weights.transforms(), "efficientnet_b0", {
            "weights": "torchvision EfficientNet_B0_Weights.IMAGENET1K_V1",
            "activation_modes": ["silu", "sigmoid"],
            "variants": {
                "silu_only": ("silu",),
                "sigmoid_only": ("sigmoid",),
                "silu_and_sigmoid": ("silu", "sigmoid"),
            },
        }
    model = create_model("deit_tiny_patch16_224.fb_in1k", pretrained=True)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform, "deit_tiny", {
        "weights": "timm deit_tiny_patch16_224.fb_in1k",
        "activation_modes": ["gelu_logistic"],
        "variants": {"gelu_logistic": ("gelu_logistic",)},
    }


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, label: str) -> dict[str, float]:
    model.eval()
    total = top1 = top5 = 0
    for step, (images, targets) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        topk = logits.topk(5, dim=1).indices
        top1 += int((topk[:, 0] == targets).sum().item())
        top5 += int((topk == targets[:, None]).any(dim=1).sum().item())
        total += targets.numel()
        if step % 100 == 0:
            print(f"{label}: {total} images", flush=True)
    return {
        "examples": total,
        "top1_accuracy": top1 / total,
        "top5_accuracy": top5 / total,
    }


def main(model_name=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("efficientnet-b0", "deit-tiny"),
                        default=model_name, required=model_name is None)
    parser.add_argument("--imagenet-val", type=Path, required=True,
                        help="ImageFolder layout: val/<synset>/*.jpg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")

    model, transform, model_kind, metadata = build_model(args.model)
    dataset = datasets.ImageFolder(str(args.imagenet_val), transform=transform)
    if len(dataset.classes) != 1000:
        raise RuntimeError(f"expected 1000 ImageNet class directories, found {len(dataset.classes)}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    device = torch.device(args.device)
    model.to(device).eval()

    results = {"exact": evaluate(model, loader, device, "exact"),
               "activation_golden_models": {}}
    for variant, modes in metadata["variants"].items():
        approx_model = copy.deepcopy(model)
        replaced = replace_vision_activations(approx_model, model_kind, modes=modes)
        approx_model.to(device).eval()
        approx_metrics = evaluate(approx_model, loader, device, variant)
        results["activation_golden_models"][variant] = {
            "activation_modules_replaced": replaced,
            **approx_metrics,
            "delta_top1_percentage_points": 100.0 * (
                approx_metrics["top1_accuracy"] - results["exact"]["top1_accuracy"]
            ),
            "delta_top5_percentage_points": 100.0 * (
                approx_metrics["top5_accuracy"] - results["exact"]["top5_accuracy"]
            ),
        }
        del approx_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "mode": "inference_only",
        "training_performed": False,
        "model": args.model, **metadata,
        "dataset": "ImageNet-1K validation",
        "dataset_path": str(args.imagenet_val.resolve()),
        "dataset_examples": len(dataset),
        "classes": len(dataset.classes), "device": str(device),
        "batch_size": args.batch_size, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

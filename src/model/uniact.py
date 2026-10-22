"""UniAct activation functions for PyTorch inference.

All datapath operations use torch integer tensors and work on CPU or CUDA.
The external model interface is signed Q8.8; the core follows the Q4.4/Q0.8
integer reference in golden.py.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn

from golden import Mode, load_current_lut


@lru_cache(maxsize=None)
def _tables_for_device(device_type: str, device_index: int | None):
    device = torch.device(device_type, device_index)
    lut = load_current_lut()
    return (
        torch.tensor(lut.low_2bit, dtype=torch.int64, device=device),
        torch.tensor(lut.high_2bit, dtype=torch.int64, device=device),
    )


def _apply_fixed(x: torch.Tensor, mode: Mode) -> torch.Tensor:
    original_dtype = x.dtype
    device = x.device
    lut_low, lut_high = _tables_for_device(device.type, device.index)

    def round_away(value: torch.Tensor) -> torch.Tensor:
        magnitude = torch.floor(torch.abs(value) + 0.5).to(torch.int64)
        return torch.where(value >= 0, magnitude, -magnitude)

    def round_shift_away(value: torch.Tensor, shift: int) -> torch.Tensor:
        magnitude = (torch.abs(value) + (1 << (shift - 1))) >> shift
        return torch.where(value >= 0, magnitude, -magnitude)

    # Input quantization and the two outer range bypasses match Q8.8 software API.
    external_code = round_away(x.to(torch.float32) * 256.0)
    negative_tail = external_code <= -2048
    positive_tail = external_code >= 2032
    X = round_shift_away(external_code, 4)
    U = torch.abs(X)

    K = 628 if mode is Mode.GELU_LOGISTIC else 369
    product_code = U * K
    n = product_code >> 12
    T = product_code & 4095

    # Exponent lookup: fine address j_e = T[11:6].
    j_e = T >> 6
    aL_e = (j_e & 7) | ((j_e >> 4) << 3)
    aV_e = j_e >> 1
    C_e = lut_low[aL_e] + (lut_high[aV_e] << 2)
    b_e = (T >> 6) & 1
    E13 = 8192 - T - (C_e << 5) - (b_e << 4)

    # Q0.8 exponent code Z with round-to-nearest, ties-away-from-zero.
    shift = n + 5
    rounding_bias = torch.bitwise_left_shift(torch.ones_like(n), n + 4)
    Z = torch.bitwise_right_shift(E13 + rounding_bias, shift)
    Z = torch.clamp(Z, 0, 255)
    Z = torch.where(n >= 9, torch.zeros_like(Z), Z)

    # Reciprocal lookup and its Z<8 bypass.
    j_r = Z >> 2
    aL_r = (j_r & 7) | ((j_r >> 4) << 3)
    aV_r = j_r >> 1
    C_r = lut_low[aL_r] + (lut_high[aV_r] << 2)
    # C_r[2] is both a coefficient bit and the fitted direction selector.
    a0 = (C_r >> 2) & 1
    D = torch.where(Z < 8, Z, (C_r << 2) + (a0 << 1))
    S = Z + D

    M = torch.full_like(U, 16) if mode is Mode.SIGMOID else U
    positive_result = 512 * M - M * S
    negative_result = M * S if mode is Mode.SIGMOID else -M * S
    P_out = torch.where(X >= 0, positive_result, negative_result)

    # P_out has scale 2**13. Round to signed external Q8.8.
    output_code = round_shift_away(P_out, 5)
    output_code = torch.where(negative_tail, torch.zeros_like(output_code), output_code)
    positive_value = torch.full_like(output_code, 256) if mode is Mode.SIGMOID else external_code
    output_code = torch.where(positive_tail, positive_value, output_code)
    return (output_code.to(torch.float32) / 256.0).to(original_dtype)


class TorchFixedActivation(nn.Module):
    """nn.Module wrapper for registered activation children in any model."""

    def __init__(self, mode: Mode | str):
        super().__init__()
        self.mode = mode if isinstance(mode, Mode) else Mode(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_fixed(x, self.mode)


class TorchActivationFunction:
    """Wrapper for activation attributes stored as plain functions."""

    def __init__(self, mode: Mode | str):
        self.mode = mode if isinstance(mode, Mode) else Mode(mode)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_fixed(x, self.mode)


def make_activation_replacement(original, mode: Mode | str):
    """Keep the original attribute's module/function kind so restoration works.

    Transformers versions may expose either an nn.Module or a plain callable.
    Assigning a callable to a registered module raises TypeError; assigning a
    module to a function also registers it and prevents restoring that function.
    """
    if isinstance(original, nn.Module):
        return TorchFixedActivation(mode).train(original.training)
    return TorchActivationFunction(mode)


def replace_transformer_gelu(model: nn.Module, expected_layers: int | None = None) -> int:
    """Replace BERT intermediate GELU function attributes in-place."""
    count = 0
    for module in model.modules():
        if hasattr(module, "intermediate_act_fn"):
            module.intermediate_act_fn = make_activation_replacement(
                module.intermediate_act_fn, Mode.GELU_LOGISTIC)
            count += 1
    if count == 0:
        raise RuntimeError("no Transformer intermediate_act_fn GELU sites were found")
    if expected_layers is not None and count != expected_layers:
        raise RuntimeError(f"expected {expected_layers} GELU sites, replaced {count}")
    return count


def replace_vision_activations(
    model: nn.Module,
    model_kind: str,
    modes: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Replace selected supported vision activations in-place.

    modes enables isolated SiLU, isolated Sigmoid, or joint replacement on
    EfficientNet-B0. DeiT-Tiny supports GELU replacement.
    """
    counts = {"gelu_logistic": 0, "silu": 0, "sigmoid": 0}
    targets: dict[str, tuple[type[nn.Module], Mode]] = {
        "gelu_logistic": (nn.GELU, Mode.GELU_LOGISTIC),
        "silu": (nn.SiLU, Mode.SILU),
        "sigmoid": (nn.Sigmoid, Mode.SIGMOID),
    }
    defaults = {"efficientnet_b0": ("silu", "sigmoid"),
                "deit_tiny": ("gelu_logistic",)}
    if model_kind not in defaults:
        raise ValueError(f"unknown vision model kind: {model_kind}")
    selected_modes = tuple(modes) if modes is not None else defaults[model_kind]
    if not selected_modes or any(name not in defaults[model_kind] for name in selected_modes):
        raise ValueError(f"unsupported activation selection for {model_kind}: {selected_modes}")

    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            selected = None
            for key in selected_modes:
                module_type, mode = targets[key]
                matches = isinstance(child, module_type)
                if key == "gelu_logistic" and child.__class__.__name__.lower() == "gelu":
                    matches = True
                if matches:
                    selected = (key, mode)
                    break
            if selected is None:
                continue
            key, mode = selected
            setattr(parent, child_name, TorchFixedActivation(mode))
            counts[key] += 1

    missing = [name for name in selected_modes if counts[name] == 0]
    if missing:
        raise RuntimeError(f"{model_kind}: expected activation modules were not found: {missing}")
    return counts

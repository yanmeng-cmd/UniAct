"""Bit-accurate UniAct v3 golden model.

The core consumes signed Q4.4 integer codes. External adapters consume signed
fixed-point codes in an explicit format; Q8.8 is the reference example.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

COEFFICIENT_FILE = Path(__file__).with_name("coefficients.json")

class Mode(str, Enum):
    SIGMOID = "sigmoid"
    SILU = "silu"
    GELU_LOGISTIC = "gelu_logistic"

def _mode(value: Mode | str) -> Mode:
    return value if isinstance(value, Mode) else Mode(value)

def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer code")
    return value

def round_shift_away(value: int, shift: int) -> int:
    """Nearest integer for value / 2**shift; exact ties move away from zero."""
    _integer(value, "value")
    if shift < 0:
        raise ValueError("shift must be nonnegative")
    if shift == 0:
        return value
    magnitude = (abs(value) + (1 << (shift - 1))) >> shift
    return magnitude if value >= 0 else -magnitude

@dataclass(frozen=True, slots=True)
class SignedFixed:
    total_bits: int
    frac_bits: int

    def __post_init__(self) -> None:
        if not 2 <= self.total_bits <= 32:
            raise ValueError("total_bits must be in 2..32")
        if not 0 <= self.frac_bits <= self.total_bits - 2:
            raise ValueError("external format must represent +1 and have a sign bit")

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def minimum_code(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def maximum_code(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    def validate_code(self, code: int) -> int:
        _integer(code, "external code")
        if not self.minimum_code <= code <= self.maximum_code:
            raise ValueError(f"external code {code} outside signed {self.total_bits} bit range")
        return code

    def quantize_real(self, value: float) -> int:
        """Explicit test adapter; the hardware-facing API uses integer codes."""
        if not math.isfinite(value):
            raise ValueError("external real input must be finite")
        scaled = value * self.scale
        if not math.isfinite(scaled):
            raise ValueError("scaled external input is not finite")
        code = math.floor(scaled + 0.5) if scaled >= 0 else -math.floor(-scaled + 0.5)
        return self.validate_code(code)

    def decode(self, code: int) -> float:
        return self.validate_code(code) / self.scale

Q8_8 = SignedFixed(16, 8)

@dataclass(frozen=True, slots=True)
class SplitLUT:
    low_2bit: tuple[int, ...]
    high_2bit: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.low_2bit) != 32 or len(self.high_2bit) != 32:
            raise ValueError("split LUT requires two arrays of 32 entries")
        if any(not 0 <= v <= 3 for v in self.low_2bit + self.high_2bit):
            raise ValueError("each LUT word must be a 2 bit unsigned code")
        if any(not 1 <= self.coefficient(j) <= 15 for j in range(64)):
            raise ValueError("each combined coefficient must be in 1..15")

    @staticmethod
    def addresses(j: int) -> tuple[int, int]:
        _integer(j, "fine address")
        if not 0 <= j < 64:
            raise ValueError("fine address must be 0..63")
        # a_L = {j[5:4], j[2:0]}; a_V = j[5:1].
        return (j & 0b111) | ((j >> 4) << 3), j >> 1

    def coefficient(self, j: int) -> int:
        low_address, high_address = self.addresses(j)
        return self.low_2bit[low_address] + 4 * self.high_2bit[high_address]

    def read_dual(self, exponent_j: int, reciprocal_j: int) -> tuple[int, int]:
        """Two independent same-cycle table reads, with identical content."""
        return self.coefficient(exponent_j), self.coefficient(reciprocal_j)

@lru_cache(maxsize=1)
def load_current_lut() -> SplitLUT:
    data: dict[str, Any] = json.loads(COEFFICIENT_FILE.read_text(encoding="utf-8"))
    if (data["low_address_omit_bit"], data["high_address_omit_bit"]) != (3, 0):
        raise ValueError("coefficient file does not match the design contract")
    return SplitLUT(tuple(data["low_2bit"]), tuple(data["high_2bit"]))

@dataclass(frozen=True, slots=True)
class CoreTrace:
    mode: Mode
    X: int
    U: int
    K: int
    P_code: int
    n: int
    T: int
    j_e: int
    a_L_e: int
    a_V_e: int
    C_e: int
    b_e: int
    E13: int
    Z: int
    j_r: int
    a_L_r: int
    a_V_r: int
    C_r: int
    a0: int
    D: int
    S: int
    M: int
    P_out: int

    @property
    def output_value(self) -> float:
        return self.P_out / 8192

    @property
    def gate_value(self) -> float:
        return 1 - self.S / 512

def run_core(X: int, mode: Mode | str, lut: SplitLUT | None = None) -> CoreTrace:
    """Run every integer stage of the Q4.4 core, including nominal code -128."""
    _integer(X, "Q4.4 core code")
    if not -128 <= X <= 127:
        raise ValueError("Q4.4 core code must be -128..127")
    mode = _mode(mode)
    lut = load_current_lut() if lut is None else lut
    U = abs(X)
    K = 628 if mode is Mode.GELU_LOGISTIC else 369
    P_code = U * K
    n = P_code >> 12
    T = P_code & 4095
    j_e = T >> 6
    a_L_e, a_V_e = lut.addresses(j_e)
    C_e = lut.coefficient(j_e)
    b_e = (T >> 6) & 1
    E13 = 8192 - T - 32 * C_e - 16 * b_e
    if not 0 < E13 < 8192:
        raise AssertionError("exponent accumulator outside documented 13 bit range")
    Z = 0 if n >= 9 else max(0, min(255, (E13 + (1 << (n + 4))) >> (n + 5)))
    j_r = Z >> 2
    a_L_r, a_V_r = lut.addresses(j_r)
    C_r = lut.coefficient(j_r)
    # Reuse C_r[2] (the low bit of the high-coefficient table). The table was
    # fitted so this coefficient bit also softly encodes correction direction;
    # run time reads it directly without a direction comparator or extra bit.
    a0 = (C_r >> 2) & 1
    D = Z if Z < 8 else 4 * C_r + 2 * a0
    S = Z + D
    M = 16 if mode is Mode.SIGMOID else U
    P_out = (512 * M - M * S if X >= 0 else
             M * S if mode is Mode.SIGMOID else -M * S)
    return CoreTrace(mode, X, U, K, P_code, n, T, j_e, a_L_e, a_V_e,
                     C_e, b_e, E13, Z, j_r, a_L_r, a_V_r, C_r, a0,
                     D, S, M, P_out)

@dataclass(frozen=True, slots=True)
class ExternalTrace:
    mode: Mode
    input_code: int
    format: SignedFixed
    branch: str
    core: CoreTrace | None
    output_code: int

    @property
    def output_value(self) -> float:
        return self.output_code / self.format.scale

def _q44_code_from_external(code: int, fmt: SignedFixed) -> int:
    if fmt.frac_bits >= 4:
        return round_shift_away(code, fmt.frac_bits - 4)
    return code << (4 - fmt.frac_bits)

def _external_code_from_pout(pout: int, fmt: SignedFixed) -> int:
    # P_out encodes a real value with scale 8192 = 2**13.
    if fmt.frac_bits <= 13:
        return round_shift_away(pout, 13 - fmt.frac_bits)
    return pout << (fmt.frac_bits - 13)

def run_external_code(code: int, mode: Mode | str,
                      fmt: SignedFixed = Q8_8,
                      lut: SplitLUT | None = None) -> ExternalTrace:
    """Bit-true fixed-point wrapper. Compare external codes before Q4.4 rounding."""
    code = fmt.validate_code(code)
    mode = _mode(mode)
    if code <= -8 * fmt.scale:
        return ExternalTrace(mode, code, fmt, "negative_tail", None, 0)
    if 16 * code >= 127 * fmt.scale:
        output = fmt.scale if mode is Mode.SIGMOID else code
        return ExternalTrace(mode, code, fmt, "positive_tail", None,
                             fmt.validate_code(output))
    X = _q44_code_from_external(code, fmt)
    core = run_core(X, mode, lut)
    output = fmt.validate_code(_external_code_from_pout(core.P_out, fmt))
    return ExternalTrace(mode, code, fmt, "core", core, output)

def _round_shift_away_array(values: Any, shift: int) -> Any:
    import numpy as np
    if shift == 0:
        return values
    magnitude = (np.abs(values) + (1 << (shift - 1))) >> shift
    return np.where(values >= 0, magnitude, -magnitude)

def run_core_array(codes: Any, mode: Mode | str,
                   lut: SplitLUT | None = None,
                   return_stages: bool = False) -> Any:
    """Vectorized exact integer core; returns P_out codes with scale 8192.

    With return_stages=True, returns a dictionary of stage arrays including P_out.
    The scalar run_core remains the detailed RTL reference for one element.
    """
    import numpy as np
    mode = _mode(mode)
    lut = load_current_lut() if lut is None else lut
    X = np.asarray(codes)
    if X.dtype.kind not in "iu":
        raise TypeError("core codes must be an integer array")
    if X.dtype.kind == "u" and np.any(X > 127):
        raise ValueError("Q4.4 core codes must be -128..127")
    X = X.astype(np.int64, copy=False)
    if np.any((X < -128) | (X > 127)):
        raise ValueError("Q4.4 core codes must be -128..127")
    U = np.abs(X)
    K = 628 if mode is Mode.GELU_LOGISTIC else 369
    P_code = U * K
    n = P_code >> 12
    T = P_code & 4095
    j_e = T >> 6
    low = np.asarray(lut.low_2bit, dtype=np.int64)
    high = np.asarray(lut.high_2bit, dtype=np.int64)
    a_L_e = (j_e & 7) | ((j_e >> 4) << 3)
    a_V_e = j_e >> 1
    C_e = low[a_L_e] + 4 * high[a_V_e]
    b_e = (T >> 6) & 1
    E13 = 8192 - T - 32 * C_e - 16 * b_e
    Z = np.clip(np.right_shift(E13 + np.left_shift(1, n + 4), n + 5), 0, 255)
    Z = np.where(n >= 9, 0, Z)
    j_r = Z >> 2
    a_L_r = (j_r & 7) | ((j_r >> 4) << 3)
    a_V_r = j_r >> 1
    C_r = low[a_L_r] + 4 * high[a_V_r]
    a0 = (C_r >> 2) & 1
    D = np.where(Z < 8, Z, 4 * C_r + 2 * a0)
    S = Z + D
    M = 16 if mode is Mode.SIGMOID else U
    P_out = np.where(X >= 0, 512 * M - M * S,
                     np.where(mode is Mode.SIGMOID, M * S, -M * S))
    if not return_stages:
        return P_out
    return dict(X=X, U=U, K=K, P_code=P_code, n=n, T=T,
                j_e=j_e, a_L_e=a_L_e, a_V_e=a_V_e, C_e=C_e, b_e=b_e,
                E13=E13, Z=Z, j_r=j_r, a_L_r=a_L_r, a_V_r=a_V_r,
                C_r=C_r, a0=a0, D=D, S=S, M=M, P_out=P_out)

def run_external_array(codes: Any, mode: Mode | str,
                       fmt: SignedFixed = Q8_8,
                       lut: SplitLUT | None = None) -> Any:
    """Vectorized fixed-point wrapper; output uses the same external format."""
    import numpy as np
    mode = _mode(mode)
    input_array = np.asarray(codes)
    if input_array.dtype.kind not in "iu":
        raise TypeError("external codes must be an integer array")
    original_shape = input_array.shape
    if input_array.dtype.kind == "u" and np.any(input_array > fmt.maximum_code):
        raise ValueError("an external code is outside the declared signed width")
    flat = input_array.astype(np.int64, copy=False).reshape(-1)
    if np.any((flat < fmt.minimum_code) | (flat > fmt.maximum_code)):
        raise ValueError("an external code is outside the declared signed width")
    negative = flat <= -8 * fmt.scale
    positive = 16 * flat >= 127 * fmt.scale
    middle = ~(negative | positive)
    output = np.empty_like(flat)
    output[negative] = 0
    output[positive] = fmt.scale if mode is Mode.SIGMOID else flat[positive]
    if np.any(middle):
        inside = flat[middle]
        X = (_round_shift_away_array(inside, fmt.frac_bits - 4)
             if fmt.frac_bits >= 4 else inside << (4 - fmt.frac_bits))
        P_out = run_core_array(X, mode, lut)
        output[middle] = (_round_shift_away_array(P_out, 13 - fmt.frac_bits)
                          if fmt.frac_bits <= 13 else P_out << (fmt.frac_bits - 13))
    if np.any((output < fmt.minimum_code) | (output > fmt.maximum_code)):
        raise AssertionError("model output exceeds the declared external format")
    return output.reshape(original_shape)

def apply_float_array(values: Any, mode: Mode | str,
                      fmt: SignedFixed = Q8_8) -> Any:
    """Convenience adapter: quantize reals to fmt, run integer model, decode.

    This adapter adds external input/output quantization. Hardware comparisons
    should use run_external_array with already encoded integer inputs.
    """
    import numpy as np
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("external real input must be finite")
    scaled = values.astype(np.float64) * fmt.scale
    if not np.all(np.isfinite(scaled)):
        raise ValueError("scaled external real input is not finite")
    if np.any((scaled < fmt.minimum_code - 0.5) | (scaled > fmt.maximum_code + 0.5)):
        raise ValueError("external real input exceeds declared fixed-point range")
    codes = np.where(scaled >= 0, np.floor(scaled + 0.5),
                     -np.floor(-scaled + 0.5)).astype(np.int64)
    return run_external_array(codes, mode, fmt) / fmt.scale

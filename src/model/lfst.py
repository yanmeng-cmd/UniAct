#!/usr/bin/env python3
"""Fit the shared LUT using discrete multi-start coordinate descent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INITIAL_L = (
    2, 3, 2, 3, 0, 0, 1, 1,
    1, 2, 3, 2, 3, 1, 3, 3,
    2, 1, 2, 3, 0, 1, 3, 1,
    3, 2, 2, 1, 1, 2, 1, 2,
)
INITIAL_H = (
    1, 0, 1, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 1, 2, 1,
    1, 2, 1, 3, 0, 0, 0, 0,
)
SEED, RESTARTS, SWEEPS = 20261012, 8, 16
LAMBDA_PEAK, LAMBDA_DIR, LAMBDA_SYM = 0.05, 3e-6, 3e-6
LOCKS = {0: 2, 30: 1, 32: 1, 63: 0}
MODE_NAMES = ("sigmoid", "silu", "gelu_logistic")

J = np.arange(64, dtype=np.int64)
A_L, A_H = (J & 7) | ((J >> 4) << 3), J >> 1
X = np.arange(-128, 128, dtype=np.int64)
MODE = np.arange(3)[:, None]
M = np.where(MODE == 0, 16, np.abs(X)[None, :])
P = np.abs(X)[None, :] * np.array([369, 369, 628])[:, None]
I, F = P >> 12, P & 4095
GATE = 1 / (1 + np.exp(-np.array([1., 1., 1.702])[:, None] * X[None, :] / 16))
REFERENCE = np.where(MODE == 0, GATE, X[None, :] / 16 * GATE)


def delta(w: float) -> float:
    return (1 - w / 2 - 1 / (1 + w)) - 2 * (1 - w / 2 - 2 ** (-w))


Z_DIR = np.arange(8, 256, dtype=np.int64)
RESIDUAL = np.array([delta(float(z / 256)) for z in Z_DIR])
DIR_WEIGHT = np.abs(RESIDUAL)
# Preserve the direction polarity used to generate the released v3 tables.
DIRECTION = RESIDUAL < 0
MIRROR = np.rint((2 * 0.4522 - Z_DIR / 256) * 256).astype(np.int64)
PAIR = (MIRROR >= 8) & (MIRROR < 256) & (Z_DIR < MIRROR)
Z_LEFT, Z_RIGHT = Z_DIR[PAIR], MIRROR[PAIR]
D_LEFT = np.array([delta(float(z / 256)) for z in Z_LEFT])
D_RIGHT = np.array([delta(float(z / 256)) for z in Z_RIGHT])
PAIR_WEIGHT = np.minimum(np.abs(D_LEFT), np.abs(D_RIGHT)) * np.clip(
    1 - np.abs(D_LEFT + D_RIGHT) / (np.abs(D_LEFT) + np.abs(D_RIGHT) + 1e-12), 0, 1)


def compose(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low[A_L] + 4 * high[A_H]


def core_outputs(c: np.ndarray) -> np.ndarray:
    ef = 8192 - F - 32 * c[F >> 6] - 16 * ((F >> 6) & 1)
    z = np.clip(np.right_shift(ef + np.left_shift(1, I + 4), I + 5), 0, 255)
    z = np.where(I >= 9, 0, z)
    cr = c[z >> 2]
    d = np.where(z < 8, z, 4 * cr + 2 * ((cr >> 2) & 1))
    s = z + d
    return np.where(X[None, :] >= 0, M * (512 - s),
                    np.where(MODE == 0, M * s, -M * s))


def evaluate(c: np.ndarray) -> dict | None:
    if np.any((c < 1) | (c > 15)):
        return None
    error = core_outputs(c) / 8192 - REFERENCE
    mse = np.mean(error * error, axis=1)
    peak = np.max(np.abs(error), axis=1)
    selector = (c >> 2) & 1
    direction = float(np.sum(DIR_WEIGHT * (selector[Z_DIR >> 2] != DIRECTION)) /
                      max(float(np.sum(DIR_WEIGHT)), 1e-30))
    symmetry = float(np.sum(PAIR_WEIGHT * (selector[Z_LEFT >> 2] ==
                                         selector[Z_RIGHT >> 2])) /
                     max(float(np.sum(PAIR_WEIGHT)), 1e-30))
    objective = float(sum((1 / 3) * (mse[i] + LAMBDA_PEAK * peak[i] ** 2)
                          for i in range(3)) + error[0, 128] ** 2)
    return {"objective": objective + LAMBDA_DIR * direction + LAMBDA_SYM * symmetry,
            "mse": dict(zip(MODE_NAMES, mse.tolist())),
            "max_abs_error": dict(zip(MODE_NAMES, peak.tolist())),
            "direction_loss": direction, "symmetry_loss": symmetry}


def search() -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(SEED)
    low0, high0 = np.array(INITIAL_L), np.array(INITIAL_H)
    starts = [(low0.copy(), high0.copy())]
    for _ in range(RESTARTS):
        starts.append((np.clip(low0 + rng.integers(-1, 2, size=32), 0, 3),
                       np.clip(high0 + rng.integers(-1, 2, size=32), 0, 3)))
    best = None
    for start, (low, high) in enumerate(starts):
        for item, value in LOCKS.items():
            (low if item < 32 else high)[item % 32] = value
        current = evaluate(compose(low, high))
        if current is None:
            continue
        for sweep in range(SWEEPS):
            changed = False
            order = range(64) if sweep % 2 == 0 else range(63, -1, -1)
            for item in order:
                if item in LOCKS:
                    continue
                table, index = (low if item < 32 else high), item % 32
                old = winner = int(table[index])
                winner_metrics = current
                for value in range(4):
                    if value == old:
                        continue
                    table[index] = value
                    trial = evaluate(compose(low, high))
                    if trial is not None and trial["objective"] < winner_metrics["objective"] - 1e-16:
                        winner, winner_metrics = value, trial
                table[index] = winner
                if winner != old:
                    current, changed = winner_metrics, True
            if not changed:
                break
        print(f"Start {start}: J={current['objective']:.12g}", flush=True)
        if best is None or current["objective"] < best[2]["objective"]:
            best = low.copy(), high.copy(), current
    if best is None:
        raise RuntimeError("No feasible coefficient table")
    return best


def lut_init(table: np.ndarray) -> str:
    value = sum(((int(v) & 1) << i) | (((int(v) >> 1) & 1) << (32 + i))
                for i, v in enumerate(table))
    return f"64'h{value:016X}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "work/lut/coefficients.json")
    args = parser.parse_args()
    low, high, metrics = search()
    result = {"version": "c2_soft_direction_v3", "low_address_omit_bit": 3,
              "high_address_omit_bit": 0, "low_2bit": low.tolist(),
              "high_2bit": high.tolist(),
              "lut6_2_init": {"low_pair": lut_init(low), "high_pair": lut_init(high),
                              "replicate_for_two_independent_ports": True}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Coefficients: {args.output}")


if __name__ == "__main__":
    main()

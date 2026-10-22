#!/usr/bin/env python3
"""Run the five inference-only accuracy evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = ("bert-large", "deit-tiny", "efficientnet-b0", "mlgcn-coco", "tinyllama")
SCRIPTS = dict(zip(MODELS, ("bert.py", "deit.py", "efficientnet.py",
                           "mlgcn.py", "tinyllama.py")))


def environment(work: Path, offline: bool) -> dict[str, str]:
    env = os.environ.copy()
    assets = work / "assets"
    hf = assets / "huggingface"
    env.update({
        "HF_HOME": str(hf), "HF_HUB_CACHE": str(hf / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf / "hub"),
        "HF_DATASETS_CACHE": str(hf / "datasets"),
        "TORCH_HOME": str(assets / "torch"),
        "TMPDIR": str(work / "tmp"), "XDG_CACHE_HOME": str(work / "cache"),
        "HF_HUB_OFFLINE": str(int(offline)),
        "HF_DATASETS_OFFLINE": str(int(offline)),
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    env.pop("TRANSFORMERS_CACHE", None)
    return env


def command(model: str, work: Path, output: Path, args) -> list[str]:
    cmd = [sys.executable, str(HERE / SCRIPTS[model])]
    assets = work / "assets"
    if model == "bert-large":
        cmd += ["--output-dir", str(output / model), "--max-length", "128"]
        if args.device == "cpu":
            cmd += ["--cpu"]
        if args.batch_size:
            cmd += ["--eval-batch-size", str(args.batch_size)]
    else:
        cmd += ["--output", str(output / (model + ".json"))]
        if args.device != "auto":
            cmd += ["--device", args.device]
        if args.batch_size and model != "tinyllama":
            cmd += ["--batch-size", str(args.batch_size)]
    if model in ("deit-tiny", "efficientnet-b0"):
        cmd += ["--imagenet-val", str(assets / "imagenet/val")]
    elif model == "mlgcn-coco":
        cmd += ["--mlgcn-repo", str(assets / "ML-GCN"),
                "--checkpoint", str(assets / "coco_checkpoint.pth.tar"),
                "--coco-root", str(assets / "coco2014")]
    if args.workers is not None and model != "tinyllama":
        cmd += ["--workers", str(args.workers)]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", nargs="?", choices=("all", *MODELS), default="all")
    p.add_argument("--work-dir", type=Path, default=ROOT / "work")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--online", action="store_true",
                   help="allow online cache lookup; default evaluates downloaded assets offline")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        p.error("batch-size must be positive")
    if args.workers is not None and args.workers < 0:
        p.error("workers must be nonnegative")
    work = args.work_dir.expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = work / "results" / run_id
    selected = MODELS if args.model == "all" else (args.model,)
    commands = [(name, command(name, work, output, args)) for name in selected]
    if args.dry_run:
        for name, cmd in commands:
            print(f"[{name}] {shlex.join(cmd)}")
        return 0
    if not (work / "assets").is_dir():
        p.error("assets are missing; run make download first")
    for name in ("tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id, "status": "running", "training_performed": False,
        "source_sha256": {q.name: hashlib.sha256(q.read_bytes()).hexdigest()
                          for q in sorted(HERE.glob("*.py"))},
        "coefficient_sha256": hashlib.sha256(
            (HERE / "coefficients.json").read_bytes()).hexdigest(),
        "runs": [],
    }
    summary_path = output / "summary.json"

    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    failed = 0
    for name, cmd in commands:
        print(f"\n=== {name}; log: {logs / (name + '.log')} ===", flush=True)
        with (logs / (name + ".log")).open("w") as log:
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=environment(work, not args.online),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                summary["status"] = "interrupted"
                summary["interrupted_model"] = name
                save()
                return 130
        failed += code != 0
        summary["runs"].append({"model": name, "exit_code": code, "command": cmd})
        save()
        if code and args.fail_fast:
            break
    summary["status"] = "failed" if failed else "passed"
    save()
    print(f"\n{summary['status']}; summary: {summary_path}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())

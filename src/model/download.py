#!/usr/bin/env python3
"""Download checkpoints and validation datasets for the five UniAct evaluations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli")
ASSETS = ("bert-large", "glue", "deit-tiny", "efficientnet-b0",
          "tinyllama", "mlgcn-coco", "imagenet")
MLGCN_URL = "https://github.com/megvii-research/ML-GCN.git"
MLGCN_REVISION = "5517d6ff8012ead45c26a07b3a9d960cdaa0b9d1"
MLGCN_CHECKPOINT_ID = "1ivLi1Rc-dCUmN1ProcMk76zxF1DSvlIk"
IMAGENET_URL = "https://image-net.org/data/ILSVRC/blurred/val_blurred.tar.gz"
COCO_URLS = {
    "val2014.zip": "https://images.cocodataset.org/zips/val2014.zip",
    "annotations_trainval2014.zip":
        "https://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


def download_file(url: str, destination: Path) -> Path:
    """Resume a partial download; rename only a completed response."""
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "UniAct"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {destination.name}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = offset > 0 and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"):
            raise RuntimeError("server returned an inconsistent resume offset")
        remaining = response.headers.get("Content-Length")
        received = 0
        with partial.open("ab" if resumed else "wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
                received += len(block)
        if remaining is not None and received != int(remaining):
            raise RuntimeError(f"incomplete download: {destination.name}; rerun to resume")
    if not partial.stat().st_size:
        raise RuntimeError(f"empty download: {destination.name}")
    partial.replace(destination)
    return destination


def extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe archive member")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("archive symlinks are not supported")
        z.extractall(target)


def validate_imagenet(path: Path) -> int:
    classes = sorted(q for q in path.iterdir()
                     if q.is_dir() and re.fullmatch(r"n\d{8}", q.name))
    if len(classes) != 1000:
        raise ValueError(f"expected 1000 ImageNet synset directories at {path}")
    count = 0
    for directory in classes:
        n = sum(q.is_file() and q.suffix.lower() in (".jpg", ".jpeg", ".png")
                for q in directory.iterdir())
        if not n:
            raise ValueError(f"empty ImageNet class: {directory.name}")
        count += n
    return count


def prepare_imagenet(assets: Path, args) -> None:
    root = assets / "imagenet"
    target = root / "val"
    root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"ImageNet ready: {validate_imagenet(target)} images")
        return
    if args.imagenet_val:
        source = args.imagenet_val.expanduser().resolve()
        count = validate_imagenet(source)
        target.symlink_to(source, target_is_directory=True)
        print(f"Using existing ImageNet split: {count} images")
        return
    archive = args.imagenet_archive
    if archive:
        archive = archive.expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
    else:
        archive = download_file(args.imagenet_url, root / "downloads/val_blurred.tar.gz")
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=root))
    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if ".." in parts or member.name.startswith("/"):
                    raise ValueError("unsafe ImageNet archive member")
                index = next((i for i, part in enumerate(parts)
                              if re.fullmatch(r"n\d{8}", part)), None)
                if index is None or len(parts) != index + 2:
                    continue
                if Path(parts[-1]).suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                dest = staging / parts[index] / parts[-1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, dest.open("wb") as output:
                    shutil.copyfileobj(source, output)
        count = validate_imagenet(staging)
        staging.rename(target)
        print(f"ImageNet ready: {count} images")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare_mlgcn(assets: Path, checkpoint_source: Path | None) -> None:
    repo = assets / "ML-GCN"
    if not (repo / "models.py").exists():
        if repo.exists():
            raise FileExistsError(f"incomplete ML-GCN directory: {repo}")
        subprocess.run(["git", "clone", "--no-checkout", MLGCN_URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach",
                        MLGCN_REVISION], check=True)
    for name in ("coco_adj.pkl", "coco_glove_word2vec.pkl"):
        if not (repo / "data/coco" / name).is_file():
            raise FileNotFoundError(f"upstream ML-GCN asset missing: {name}")
    checkpoint = assets / "coco_checkpoint.pth.tar"
    if not checkpoint.is_file():
        partial = checkpoint.with_name(checkpoint.name + ".part")
        if checkpoint_source:
            shutil.copy2(checkpoint_source.expanduser().resolve(), partial)
        else:
            import gdown
            result = gdown.download(id=MLGCN_CHECKPOINT_ID,
                                    output=str(partial), quiet=False)
            if not result:
                raise RuntimeError("ML-GCN checkpoint unavailable; use --mlgcn-checkpoint")
        if partial.stat().st_size < 1024 * 1024:
            raise RuntimeError("ML-GCN checkpoint is incomplete")
        partial.replace(checkpoint)
    if checkpoint.stat().st_size < 1024 * 1024:
        raise RuntimeError("ML-GCN checkpoint is incomplete")
    root = assets / "coco2014"
    for name, url in COCO_URLS.items():
        archive = download_file(url, root / "tmp" / name)
        if name == "val2014.zip":
            ready = root / "data/val2014"
            complete = ready.is_dir() and sum(1 for _ in ready.glob("*.jpg")) == 40504
        else:
            complete = (root / "data/annotations/instances_val2014.json").is_file()
        if not complete:
            extract_zip(archive, root / "data")
    # The original loader defines label order and builds its validation indexes.
    sys.path.insert(0, str(repo))
    from coco import COCO2014
    dataset = COCO2014(str(root), phase="val",
                       inp_name=str(repo / "data/coco/coco_glove_word2vec.pkl"))
    # Upstream ML-GCN excludes 367 validation images without object labels.
    if len(dataset) != 40137:
        raise RuntimeError(f"expected 40137 labeled COCO validation images, got {len(dataset)}")
    print(f"ML-GCN and COCO ready: {len(dataset)} images")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", nargs="+", choices=("all", *ASSETS), default=("all",))
    p.add_argument("--work-dir", type=Path, default=ROOT / "work")
    p.add_argument("--imagenet-val", type=Path)
    p.add_argument("--imagenet-archive", type=Path)
    p.add_argument("--imagenet-url", default=IMAGENET_URL)
    p.add_argument("--mlgcn-checkpoint", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    selected = set(ASSETS if "all" in args.assets else args.assets)
    if "bert-large" in selected:
        selected.add("glue")
    work = args.work_dir.expanduser().resolve()
    if args.dry_run:
        for name in ASSETS:
            if name in selected:
                print(f"{name}: {work / 'assets'}")
        return
    assets = work / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    hf = assets / "huggingface"
    os.environ.update({
        "HF_HOME": str(hf), "HF_HUB_CACHE": str(hf / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf / "hub"),
        "HF_DATASETS_CACHE": str(hf / "datasets"),
        "TORCH_HOME": str(assets / "torch"),
        "TMPDIR": str(work / "tmp"), "XDG_CACHE_HOME": str(work / "cache"),
        "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0",
    })
    os.environ.pop("TRANSFORMERS_CACHE", None)
    if selected & {"bert-large", "tinyllama"}:
        from huggingface_hub import snapshot_download
        models = (["yoshitomo-matsubara/bert-large-uncased-" + task for task in TASKS]
                  if "bert-large" in selected else [])
        if "tinyllama" in selected:
            models.append("TinyLlama/TinyLlama_v1.1")
        for model in models:
            print(f"Checkpoint: {model}", flush=True)
            snapshot_download(model, cache_dir=str(hf / "hub"))
    if selected & {"glue", "tinyllama"}:
        from datasets import load_dataset
        if "glue" in selected:
            for task in TASKS:
                load_dataset("glue", task, cache_dir=str(hf / "datasets"))
        if "tinyllama" in selected:
            load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                         split="test", cache_dir=str(hf / "datasets"))
    if "efficientnet-b0" in selected:
        from torchvision.models import EfficientNet_B0_Weights
        EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    if "deit-tiny" in selected:
        import timm
        timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=True)
    if "mlgcn-coco" in selected:
        prepare_mlgcn(assets, args.mlgcn_checkpoint)
    if "imagenet" in selected:
        prepare_imagenet(assets, args)
    print("Download complete. No training or fine-tuning was performed.")


if __name__ == "__main__":
    main()

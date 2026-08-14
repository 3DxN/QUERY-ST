from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hescape_mpl_config")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from common_functions import load_samples, match_and_align_trimodal  # noqa: E402
from training_functions import load_trimodal_inputs  # noqa: E402
from trimodal_encoder import TriModalEncoder  # noqa: E402


SERVER_LUNG_DIR = Path("/well/rittscher/users/mju725/trimodal_alignment_objects/lung")
MAC_LUNG_DIR = Path("/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/lung")


@dataclass
class ScalabilityConfig:
    lung_dir: str
    output_dir: str
    sizes: str
    batch_sizes: str
    gene_variant: str
    text_level: str
    epochs: int
    val_size: int
    repeats: int
    seed: int
    lambda_rna_text: float
    lambda_img_text: float
    device: str


def default_lung_dir() -> str:
    return str(SERVER_LUNG_DIR if SERVER_LUNG_DIR.exists() else MAC_LUNG_DIR)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb(device: str) -> float | None:
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1024**2


def max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def parse_sizes(raw_sizes: str, n_total: int) -> list[tuple[str, int]]:
    sizes = []
    for raw in raw_sizes.split(","):
        label = raw.strip()
        if not label:
            continue
        if label.lower() in {"full", "all"}:
            n = n_total
        else:
            n = min(int(label), n_total)
        if n <= 0:
            raise ValueError(f"Invalid size: {label}")
        sizes.append((label, n))
    return sizes


def parse_int_list(raw_values: str) -> list[int]:
    values = []
    for raw in raw_values.split(","):
        value = raw.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"Invalid positive integer value: {value}")
        values.append(parsed)
    if not values:
        raise ValueError("Expected at least one positive integer value.")
    return values


def load_lung_arrays(lung_dir: Path, gene_variant: str, text_level: str):
    samples = load_samples(str(lung_dir))
    uni_embeddings, _, master_gene_vectors, master_text_embeddings, _ = load_trimodal_inputs(samples, str(lung_dir))

    X, Y, Z, aligned_keys = match_and_align_trimodal(
        uni_embeddings,
        master_gene_vectors[gene_variant],
        master_text_embeddings[text_level],
        key_order=list(uni_embeddings.keys()),
        return_keys=True,
    )
    if X is None:
        raise RuntimeError("No aligned lung patches found.")
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32), aligned_keys


def run_one_size(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    selected_indices: np.ndarray,
    size_label: str,
    batch_size: int,
    repeat: int,
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> dict:
    n_selected = len(selected_indices)
    n_val = min(args.val_size, max(2, n_selected // 10))
    n_train = n_selected - n_val
    if n_train < batch_size:
        raise ValueError(f"Need at least batch_size train patches; got n_train={n_train}, batch_size={batch_size}")

    X_sub = X[selected_indices]
    Y_sub = Y[selected_indices]
    Z_sub = Z[selected_indices]

    indices_val = np.arange(n_val)
    indices_train = np.arange(n_val, n_selected)

    run_dir = output_dir / "model_artifacts" / f"n_{n_selected}_batch_{batch_size}_repeat_{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_seed = args.seed + repeat + batch_size * 1000
    torch.manual_seed(run_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    reset_peak_memory(device)

    model = TriModalEncoder(
        img_dim=X_sub.shape[1],
        rna_dim=Y_sub.shape[1],
        text_dim=Z_sub.shape[1],
        proj_dim=128,
        hidden_dim=2048,
        dropout=0.4,
        device=device,
    )

    sync_device(device)
    start = time.perf_counter()
    model.fit(
        X_sub,
        Y_sub,
        Z_sub,
        indices_train,
        indices_val,
        out_dir=str(run_dir),
        lambda_rna_text=args.lambda_rna_text,
        lambda_img_text=args.lambda_img_text,
        batch_size=batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.epochs + 1,
        verbose=False,
    )
    sync_device(device)
    elapsed = time.perf_counter() - start

    row = {
        "size_label": size_label,
        "n_patches": n_selected,
        "n_train": n_train,
        "n_val": n_val,
        "repeat": repeat,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "train_seconds": elapsed,
        "seconds_per_epoch": elapsed / args.epochs,
        "train_patches_per_second": (n_train * args.epochs) / elapsed,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
        "max_cpu_rss_mb": max_rss_mb(),
        "device": device,
    }

    del model, X_sub, Y_sub, Z_sub
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def run_benchmark(args: argparse.Namespace) -> Path:
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    X, Y, Z, aligned_keys = load_lung_arrays(Path(args.lung_dir), args.gene_variant, args.text_level)
    load_seconds = time.perf_counter() - load_start

    sizes = parse_sizes(args.sizes, len(aligned_keys))
    batch_sizes = parse_int_list(args.batch_sizes)
    rng = np.random.default_rng(args.seed)
    global_order = rng.permutation(len(aligned_keys))

    config = ScalabilityConfig(
        lung_dir=args.lung_dir,
        output_dir=str(output_dir),
        sizes=args.sizes,
        batch_sizes=args.batch_sizes,
        gene_variant=args.gene_variant,
        text_level=args.text_level,
        epochs=args.epochs,
        val_size=args.val_size,
        repeats=args.repeats,
        seed=args.seed,
        lambda_rna_text=args.lambda_rna_text,
        lambda_img_text=args.lambda_img_text,
        device=device,
    )
    with (output_dir / "lung_scalability_config.json").open("w") as fh:
        json.dump(asdict(config), fh, indent=2)

    rows = []
    for size_label, n_patches in sizes:
        selected = global_order[:n_patches]
        for batch_size in batch_sizes:
            for repeat in range(args.repeats):
                print(
                    f"\nTiming lung scalability: n={n_patches}, batch_size={batch_size}, "
                    f"repeat={repeat + 1}/{args.repeats}"
                )
                row = run_one_size(X, Y, Z, selected, size_label, batch_size, repeat, args, device, output_dir)
                row["load_seconds"] = load_seconds
                row["n_aligned_total"] = len(aligned_keys)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_dir / "lung_scalability_results.csv", index=False)
                print(
                    f"n={n_patches}, batch_size={batch_size}: "
                    f"{row['seconds_per_epoch']:.2f}s/epoch, "
                    f"{row['train_patches_per_second']:.1f} train patches/s, "
                    f"peak_gpu={row['peak_gpu_memory_mb']} MB"
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["n_patches", "batch_size"], as_index=False)
        .agg(
            train_seconds_mean=("train_seconds", "mean"),
            seconds_per_epoch_mean=("seconds_per_epoch", "mean"),
            train_patches_per_second_mean=("train_patches_per_second", "mean"),
            peak_gpu_memory_mb_max=("peak_gpu_memory_mb", "max"),
            max_cpu_rss_mb_max=("max_cpu_rss_mb", "max"),
        )
        .sort_values(["n_patches", "batch_size"])
    )
    summary.to_csv(output_dir / "lung_scalability_summary.csv", index=False)
    print(f"\nWrote results to {output_dir}")
    print(summary)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple lung scalability timing for the existing TriModalEncoder.fit path.")
    parser.add_argument("--lung-dir", default=default_lung_dir())
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "benchmark" / "outputs" / "lung_scalability"))
    parser.add_argument("--sizes", default="1000,2500,5000,10000,20000,full")
    parser.add_argument("--gene-variant", default="full")
    parser.add_argument("--text-level", default="level_1_2_3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated batch sizes to sweep, e.g. 32,64,128,256. Defaults to --batch-size.",
    )
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-rna-text", type=float, default=0.4)
    parser.add_argument("--lambda-img-text", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args(argv)
    if args.batch_sizes is None:
        args.batch_sizes = str(args.batch_size)
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())

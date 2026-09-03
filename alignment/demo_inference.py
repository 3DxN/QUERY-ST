"""Stand-alone inference demo for a trained TriModalEncoder.

Assumes the H&E patches and their UNI2-h embeddings already exist, i.e. steps 2
and 3 of the pipeline have been run. Neither RNA vectors nor text embeddings are
needed: only the image and text branches of the checkpoint are exercised, so the
script runs against any dataset directory that contains

    <DATASET_DIR>/he_size_<N>_microns_patches_<TISSUE>/
        uni2h_<TISSUE>.pkl
        patch_x<X>_y<Y>.png

Two query modes:

  --text "..."     natural-language query -> ranked H&E patches (needs OPENAI_API_KEY)
  --patch <KEY>    an existing patch      -> visually/semantically similar patches

Examples
--------
    # What does the model retrieve for a description?
    python -u alignment/demo_inference.py \
        --checkpoint /path/to/models/lung_final/best.pt \
        --dataset-dir /path/to/outputs/lung \
        --text "dense immune infiltrate with lymphocytes adjacent to capillaries" \
        --save-fig /tmp/text_query.png

    # Find patches similar to one we already have
    python -u alignment/demo_inference.py \
        --checkpoint /path/to/models/lung_final/best.pt \
        --dataset-dir /path/to/outputs/lung \
        --patch TILD049MA__patch_x1344_y2016

    # Just see which patch keys are available
    python -u alignment/demo_inference.py \
        --checkpoint ... --dataset-dir ... --list 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib

# Chosen before retrieval_functions imports pyplot, so the script works headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_functions import search_by_text, search_cross_modal
from trimodal_encoder import TriModalEncoder


def load_uni_embeddings(dataset_dir):
    """Load every uni2h_<TISSUE>.pkl under dataset_dir.

    Mirrors the key convention of training_functions.load_trimodal_inputs:
    the tissue name is the text after the last underscore of the sample
    directory, and each patch key becomes "<TISSUE>__<patch>". Unlike that
    loader, skipped samples are reported rather than passed over in silence.
    """
    embeddings = {}
    tissue_dirs = {}

    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"dataset directory does not exist: {dataset_dir}")

    for sample_id in sorted(os.listdir(dataset_dir)):
        sample_dir = os.path.join(dataset_dir, sample_id)
        if sample_id.startswith(".") or not os.path.isdir(sample_dir):
            continue

        tissue = sample_id.rsplit("_", 1)[-1]
        uni_path = os.path.join(sample_dir, f"uni2h_{tissue}.pkl")

        if not os.path.exists(uni_path):
            print(f"  skipped {sample_id}: no uni2h_{tissue}.pkl")
            continue

        with open(uni_path, "rb") as fh:
            sample_embeddings = pickle.load(fh)

        for key, vector in sample_embeddings.items():
            embeddings[f"{tissue}__{key.removesuffix('.png')}"] = vector

        tissue_dirs[tissue] = sample_dir
        print(f"  loaded  {sample_id}: {len(sample_embeddings)} patches")

    if not embeddings:
        raise SystemExit(
            f"no uni2h_*.pkl found under {dataset_dir}. Expected sample directories "
            "named he_size_<N>_microns_patches_<TISSUE>, each holding uni2h_<TISSUE>.pkl."
        )

    return embeddings, tissue_dirs


def embed_patches(model, embeddings):
    """Project raw UNI2-h vectors into the shared 128-d latent space."""
    keys = sorted(embeddings)
    X = np.stack([np.asarray(embeddings[k]).reshape(-1) for k in keys]).astype(np.float32)

    expected = model.dims["img_dim"]
    if X.shape[1] != expected:
        raise SystemExit(
            f"image embedding dimension mismatch: patches are {X.shape[1]}-d but the "
            f"checkpoint expects {expected}-d. Were these produced by a different encoder?"
        )

    latents = model.embed_batch(X, branch="img")
    print(f"\nProjected {latents.shape[0]} patches into a {latents.shape[1]}-d latent space.")
    return keys, latents


def patch_image_path(key, tissue_dirs):
    tissue, patch_name = key.split("__", 1)
    return os.path.join(tissue_dirs[tissue], f"{patch_name}.png")


def load_patch_image(key, tissue_dirs):
    path = patch_image_path(key, tissue_dirs)
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (224, 224), color="gainsboro")


def print_results(results, tissue_dirs):
    width = max(len(r["key"]) for r in results)
    print(f"\n{'rank':<5} {'score':<8} {'patch':<{width}}  image")
    print("-" * (5 + 8 + width + 8))
    for rank, result in enumerate(results, start=1):
        path = patch_image_path(result["key"], tissue_dirs)
        mark = "" if os.path.exists(path) else "  (missing)"
        print(f"{rank:<5} {result['score']:<8.4f} {result['key']:<{width}}  {path}{mark}")


def save_contact_sheet(results, tissue_dirs, out_path, query_title, query_key=None):
    """Write the query panel plus the top-k retrieved patches to a single figure."""
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.8))

    axes[0].axis("off")
    axes[0].text(
        0.5, 0.5, query_title,
        ha="center", va="center", fontsize=13, fontweight="bold", wrap=True,
    )

    for i, result in enumerate(results):
        ax = axes[i + 1]
        ax.imshow(load_patch_image(result["key"], tissue_dirs))
        ax.set_title(f"Rank {i + 1}\nscore {result['score']:.3f}", fontsize=11)
        ax.axis("off")

        # A patch retrieving itself is the expected sanity check, so mark it.
        if query_key is not None and result["key"] == query_key:
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("mediumseagreen")
                spine.set_linewidth(4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


def main(args):
    print(f"Loading UNI2-h embeddings from {args.dataset_dir}")
    embeddings, tissue_dirs = load_uni_embeddings(args.dataset_dir)

    if args.list:
        print(f"\nFirst {args.list} of {len(embeddings)} patch keys:")
        for key in sorted(embeddings)[: args.list]:
            print(f"  {key}")
        return

    print(f"\nLoading checkpoint {args.checkpoint}")
    model = TriModalEncoder.load(args.checkpoint, device=args.device)
    model.eval()

    keys, latents = embed_patches(model, embeddings)

    if args.text:
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("the openai package is required for --text queries")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; required to embed the text query")

        print(f"\nText query: {args.text!r}")
        results = search_by_text(OpenAI(), model, args.text, latents, keys, k=args.top_k)
        print_results(results, tissue_dirs)

        if args.save_fig:
            save_contact_sheet(results, tissue_dirs, args.save_fig, f"QUERY\n\n{args.text}")
        return

    if args.patch not in embeddings:
        raise SystemExit(
            f"patch key {args.patch!r} not found. Re-run with --list to see available keys."
        )

    query_idx = keys.index(args.patch)
    print(f"\nPatch query: {args.patch}")
    results = search_cross_modal(query_idx, latents, latents, keys, k=args.top_k)
    print_results(results, tissue_dirs)

    if args.save_fig:
        save_contact_sheet(
            results, tissue_dirs, args.save_fig,
            f"QUERY\n\n{args.patch}", query_key=args.patch,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-modal retrieval demo for a trained TriModalEncoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", help="path to best.pt (not needed with --list)")
    parser.add_argument("--dataset-dir", required=True, help="directory of patch sample folders")

    query = parser.add_mutually_exclusive_group()
    query.add_argument("--text", help="natural-language query (needs OPENAI_API_KEY)")
    query.add_argument("--patch", help="patch key to use as the query, e.g. TISSUE__patch_x0_y0")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-fig", help="write the query and its top-k matches to this PNG")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--list", type=int, metavar="N", default=0,
        help="print the first N available patch keys and exit",
    )

    args = parser.parse_args()
    if not args.list:
        if not args.text and not args.patch:
            parser.error("give one of --text, --patch, or --list")
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --list is used")
    return args


if __name__ == "__main__":
    main(parse_args())

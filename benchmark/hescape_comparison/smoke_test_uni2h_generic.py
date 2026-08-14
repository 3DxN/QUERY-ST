from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the vendored HESCAPE uni2h image encoder branch with "
            "the generic/MLP gene encoder."
        )
    )
    parser.add_argument("--image-weights-root", default="")
    parser.add_argument("--gene-dim", type=int, default=343)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--img-proj", choices=["linear", "mlp", "transformer"], default="linear")
    parser.add_argument("--gene-proj", choices=["linear", "mlp", "identity"], default="linear")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument(
        "--real-image-model",
        action="store_true",
        help="Instantiate the real UNI2-h trunk. Default uses a tiny stub so the smoke test is fast.",
    )
    parser.add_argument(
        "--force-random-image-init",
        action="store_true",
        help="Instantiate UNI2-h with random weights. Use only for mechanics checks.",
    )
    return parser.parse_args()


def choose_device(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def patch_tiny_uni2h_trunk(num_features: int = 32) -> None:
    """Replace only the UNI2-h trunk with a tiny module for fast wiring tests."""
    import timm
    import torch
    import torch.nn as nn

    original_create_model = timm.create_model

    class TinyUni2HTrunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_features = num_features
            self.proj = nn.Linear(3, num_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pooled = x.mean(dim=(2, 3))
            return self.proj(pooled)

    def create_model_with_tiny_uni2h(model_name: str, *model_args, **model_kwargs):
        if model_name == "vit_giant_patch14_224":
            return TinyUni2HTrunk()
        return original_create_model(model_name, *model_args, **model_kwargs)

    timm.create_model = create_model_with_tiny_uni2h


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hescape_mpl_config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hescape_xdg_cache"))

    import torch

    from official_hescape_lung_lopo import (
        import_official_hescape,
        instantiate_official_module,
        parse_args as parse_runner_args,
        patch_timm_pretrained_for_smoke,
        resolve_uni2h_checkpoint_root,
    )

    if not args.real_image_model:
        patch_tiny_uni2h_trunk()
        args.force_random_image_init = True
        print("Using tiny UNI2-h trunk stub for fast configuration smoke test.")
    elif args.force_random_image_init:
        patch_timm_pretrained_for_smoke()
    else:
        checkpoint_root = resolve_uni2h_checkpoint_root(args.image_weights_root)
        if checkpoint_root is None:
            raise FileNotFoundError(
                "Could not find UNI2-h weights. Either pass --image-weights-root "
                "or use --force-random-image-init for a mechanics-only smoke test."
            )
        print(f"UNI2-h checkpoint root: {checkpoint_root}")

    official = import_official_hescape(allow_drvi=False, allow_conch=False)
    transforms = official["TRANSFORMS"]
    if "uni2h" not in transforms:
        raise RuntimeError("Vendored HESCAPE TRANSFORMS does not expose a uni2h transform.")

    runner_args = parse_runner_args([])
    runner_args.image_encoder = "uni2h"
    runner_args.image_weights_root = args.image_weights_root
    runner_args.gene_encoder = "generic"
    runner_args.embed_dim = args.embed_dim
    runner_args.img_proj = args.img_proj
    runner_args.gene_proj = args.gene_proj
    runner_args.force_random_image_init = args.force_random_image_init

    with tempfile.TemporaryDirectory(prefix="hescape_uni2h_generic_") as tmp_dir:
        if not args.real_image_model:
            runner_args.image_weights_root = tmp_dir
        module = instantiate_official_module(runner_args, args.gene_dim, Path(tmp_dir))

        print("Instantiated HESCAPE module:")
        print(f"  image_encoder={runner_args.image_encoder}")
        print("  gene_encoder=generic (HESCAPE MLP)")
        print(f"  gene_dim={args.gene_dim}")
        print(f"  embed_dim={args.embed_dim}")

        if args.run_forward:
            device = choose_device(args.device)
            module.eval().to(device)
            batch = {
                official["DatasetEnum"].IMG: torch.randn(args.batch_size, 3, 224, 224, device=device),
                official["DatasetEnum"].GEXP: torch.randn(args.batch_size, args.gene_dim, device=device),
            }
            with torch.inference_mode():
                img_embed, gene_embed, logit_scale = module.model(batch, norm=True)
            print("Forward pass succeeded:")
            print(f"  img_embed={tuple(img_embed.shape)}")
            print(f"  gene_embed={tuple(gene_embed.shape)}")
            print(f"  logit_scale={float(logit_scale.detach().cpu()):.4f}")

    print("UNI2-h + generic/MLP smoke test passed.")


if __name__ == "__main__":
    main()

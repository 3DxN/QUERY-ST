from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import re
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate


DEFAULT_LUNG_DIR = "/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/lung"
HERE = Path(__file__).resolve().parent
OFFICIAL_HESCAPE_SRC = HERE / "external" / "hescape" / "src"
PATCH_FOLDER_PREFIX_RE = re.compile(r"^he_size_\d+_microns_patches_")
_ORIGINAL_METADATA_VERSION = importlib.metadata.version


@dataclass
class OfficialHescapeRunConfig:
    lung_dir: str
    output_dir: str
    suffix: str
    gene_variant: str
    alignment_key_space: str
    text_level: str
    image_encoder: str
    precomputed_image_embeddings: bool
    image_weights_root: str
    gene_encoder: str
    gene_weights_root: str
    drvi_model_dir: str
    embed_dim: int
    loss: str
    img_proj: str
    gene_proj: str
    img_finetune: bool
    gene_finetune: bool
    gene_batch_layout: str
    temperature: float
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    val_frac: float
    split_seed: int
    seed: int
    accelerator: str
    devices: str
    num_workers: int
    max_train_patches: int | None
    max_val_patches: int | None
    max_test_patches: int | None
    patient: str | None
    force_random_image_init: bool
    disable_checkpoints: bool
    skip_completed_patients: bool


def install_unused_optional_stubs(allow_drvi: bool = False, allow_conch: bool = False) -> None:
    """Let official HESCAPE import when unused optional encoders are absent.

    HESCAPE imports DRVI and CONCH modules at import time, even when the run
    uses `gene_enc_name="generic"` and an image encoder that is not CONCH. These
    stubs raise only if those unused branches are selected.
    """
    if not allow_drvi and "drvi" not in sys.modules:
        drvi_mod = types.ModuleType("drvi")
        drvi_model_mod = types.ModuleType("drvi.model")

        class _MissingDRVI:
            @staticmethod
            def load(*_: Any, **__: Any) -> Any:
                raise RuntimeError("drvi is not installed; choose gene_enc_name='generic' or install drvi-py.")

        drvi_model_mod.DRVI = _MissingDRVI
        drvi_mod._hescape_missing_stub = True
        drvi_mod.model = drvi_model_mod
        sys.modules["drvi"] = drvi_mod
        sys.modules["drvi.model"] = drvi_model_mod

    if allow_drvi and getattr(sys.modules.get("drvi"), "_hescape_missing_stub", False):
        del sys.modules["drvi"]
        sys.modules.pop("drvi.model", None)

    if not allow_conch and "conch" not in sys.modules:
        conch_mod = types.ModuleType("conch")
        open_clip_custom_mod = types.ModuleType("conch.open_clip_custom")

        def _missing_conch(*_: Any, **__: Any) -> Any:
            raise RuntimeError("conch is not installed; choose another image encoder or install CONCH.")

        open_clip_custom_mod.create_model_from_pretrained = _missing_conch
        conch_mod._hescape_missing_stub = True
        conch_mod.open_clip_custom = open_clip_custom_mod
        sys.modules["conch"] = conch_mod
        sys.modules["conch.open_clip_custom"] = open_clip_custom_mod

    if allow_conch and getattr(sys.modules.get("conch"), "_hescape_missing_stub", False):
        del sys.modules["conch"]
        sys.modules.pop("conch.open_clip_custom", None)


def install_hescape_metadata_stub() -> None:
    """Let the vendored source import even when HESCAPE is not pip-installed."""
    try:
        _ORIGINAL_METADATA_VERSION("HESCAPE")
        return
    except importlib.metadata.PackageNotFoundError:
        pass

    def version_with_local_hescape(distribution_name: str) -> str:
        if distribution_name.lower() == "hescape":
            return "0.0.1-local"
        return _ORIGINAL_METADATA_VERSION(distribution_name)

    importlib.metadata.version = version_with_local_hescape


def import_official_hescape(allow_drvi: bool = False, allow_conch: bool = False) -> dict[str, Any]:
    if OFFICIAL_HESCAPE_SRC.exists():
        sys.path.insert(0, str(OFFICIAL_HESCAPE_SRC))
    install_hescape_metadata_stub()
    install_unused_optional_stubs(allow_drvi=allow_drvi, allow_conch=allow_conch)

    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from hescape.constants import DatasetEnum
    from hescape.data_modules.image_gexp_dataset import TRANSFORMS
    from hescape.modules.pretrain_module import ClampCallback, PretrainModule, get_clip_metrics

    return {
        "pl": pl,
        "OmegaConf": OmegaConf,
        "EarlyStopping": EarlyStopping,
        "ModelCheckpoint": ModelCheckpoint,
        "CSVLogger": CSVLogger,
        "DatasetEnum": DatasetEnum,
        "TRANSFORMS": TRANSFORMS,
        "ClampCallback": ClampCallback,
        "PretrainModule": PretrainModule,
        "get_clip_metrics": get_clip_metrics,
    }


def patch_timm_pretrained_for_smoke() -> None:
    """Avoid network downloads in the smoke test while preserving HESCAPE code flow."""
    import timm

    original_create_model = timm.create_model

    def create_model_without_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_without_pretrained


def _resolve_hf_snapshot_file(repo_dir: Path, filename: str) -> Path | None:
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        candidate = repo_dir / "snapshots" / revision / filename
        if candidate.exists():
            return candidate

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        for candidate in snapshots_dir.glob(f"*/{filename}"):
            if candidate.exists():
                return candidate
    return None


def resolve_uni2h_checkpoint_root(image_weights_root: str | Path | None) -> Path | None:
    """Find a local UNI2-h checkpoint root compatible with the vendored encoder."""
    roots: list[Path] = []
    if image_weights_root:
        roots.append(Path(image_weights_root))

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        roots.append(Path(hf_hub_cache))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")

    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root in seen:
            continue
        seen.add(root)

        candidates = [
            root / "uni2h" / "pytorch_model.bin",
            root / "UNI2-h" / "pytorch_model.bin",
            root / "pytorch_model.bin",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.parent

        repo_dir = root
        if repo_dir.name != "models--MahmoodLab--UNI2-h":
            repo_dir = root / "models--MahmoodLab--UNI2-h"
        if repo_dir.exists():
            checkpoint = _resolve_hf_snapshot_file(repo_dir, "pytorch_model.bin")
            if checkpoint is not None:
                return checkpoint.parent

    return None


def configure_uni2h_weights(args: argparse.Namespace) -> None:
    if args.image_encoder != "uni2h" or args.force_random_image_init or args.precomputed_image_embeddings:
        return

    checkpoint_root = resolve_uni2h_checkpoint_root(args.image_weights_root)
    if checkpoint_root is None:
        raise FileNotFoundError(
            "Could not find UNI2-h weights. Pass --image-weights-root pointing to either "
            "a HESCAPE-style folder with uni2h/pytorch_model.bin or the Hugging Face cache "
            "folder containing models--MahmoodLab--UNI2-h. For a mechanics-only smoke test, "
            "use --force-random-image-init."
        )

    if not args.image_weights_root:
        args.image_weights_root = str(checkpoint_root)


def parse_patch_xy(key: str) -> tuple[int | None, int | None]:
    match = re.search(r"patch_x(\d+)_y(\d+)", key)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def clean_patch_key(tissue: str, raw_key: str) -> str:
    return f"{tissue}__{raw_key.removesuffix('.png')}"


def iter_lung_sample_dirs(lung_dir: Path) -> list[Path]:
    """Mirror common_functions.load_samples ordering for final_models_compayl."""
    return [
        lung_dir / name
        for name in os.listdir(lung_dir)
        if (lung_dir / name).is_dir() and not name.startswith(".")
    ]


def tissue_from_patch_folder(folder_name: str) -> str:
    stripped = PATCH_FOLDER_PREFIX_RE.sub("", folder_name)
    if stripped != folder_name:
        return stripped
    return folder_name.rsplit("_", 1)[-1]


def load_optional_key_order(lung_dir: Path, kind: str, text_level: str) -> list[str]:
    if kind == "text":
        text_path = lung_dir / "text-descriptions-all-samples-levels" / f"openai_text_embeddings_{text_level}.pkl"
        if not text_path.exists():
            raise FileNotFoundError(f"Could not find text embeddings for key matching: {text_path}")
        return list(load_pickle(text_path).keys())

    keys: list[str] = []
    for sample_dir in iter_lung_sample_dirs(lung_dir):
        tissue = tissue_from_patch_folder(sample_dir.name)
        if kind == "cell":
            path = sample_dir / "patch_vectors.pkl"
        elif kind == "uni":
            path = sample_dir / f"uni2h_{tissue}.pkl"
        else:
            raise ValueError(f"Unknown key-order kind: {kind}")
        if not path.exists():
            continue
        keys.extend(clean_patch_key(tissue, raw_key) for raw_key in load_pickle(path).keys())
    return keys


def load_optional_key_set(lung_dir: Path, kind: str, text_level: str) -> set[str]:
    return set(load_optional_key_order(lung_dir, kind, text_level))


def load_full_trimodal_key_order(lung_dir: Path, text_level: str) -> list[str]:
    uni_key_order = load_optional_key_order(lung_dir, "uni", text_level)
    cell_keys = load_optional_key_set(lung_dir, "cell", text_level)
    text_keys = load_optional_key_set(lung_dir, "text", text_level)
    required_keys = set(uni_key_order) & cell_keys & text_keys
    return [key for key in uni_key_order if key in required_keys]


def load_lung_records(
    lung_dir: Path,
    gene_variant: str,
    alignment_key_space: str,
    text_level: str,
    precomputed_image_embeddings: bool = False,
) -> list[dict[str, Any]]:
    metadata_path = lung_dir / "metadata_lung.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find patient metadata: {metadata_path}")
    lung_meta = pd.read_csv(metadata_path)
    sample_to_patient = dict(zip(lung_meta["sample"], lung_meta["patient"], strict=False))

    aligned_key_order: list[str] | None = None
    required_keys: set[str] | None = None
    if alignment_key_space == "full_trimodal":
        aligned_key_order = load_full_trimodal_key_order(lung_dir, text_level)
        required_keys = set(aligned_key_order)
    elif alignment_key_space != "image_gene":
        raise ValueError(f"Unknown alignment key space: {alignment_key_space}")

    records: list[dict[str, Any]] = []
    for sample_dir in iter_lung_sample_dirs(lung_dir):
        tissue = tissue_from_patch_folder(sample_dir.name)
        patient = sample_to_patient.get(tissue)
        if patient is None:
            continue

        image_embeddings = None
        if precomputed_image_embeddings:
            image_embeddings_path = sample_dir / f"uni2h_{tissue}.pkl"
            if not image_embeddings_path.exists():
                continue
            image_embeddings = load_pickle(image_embeddings_path)

        genes_path = sample_dir / f"patch_genes_{gene_variant}.pkl"
        if not genes_path.exists():
            continue
        gene_vectors = load_pickle(genes_path)
        for raw_key, vector in gene_vectors.items():
            patch_name = raw_key if raw_key.endswith(".png") else f"{raw_key}.png"
            unique_key = clean_patch_key(tissue, patch_name)
            if required_keys is not None and unique_key not in required_keys:
                continue
            image_path = sample_dir / patch_name
            if not image_path.exists():
                continue

            image_embedding = None
            if image_embeddings is not None:
                embedding_key = raw_key.removesuffix(".png")
                if embedding_key not in image_embeddings:
                    continue
                image_embedding = np.asarray(image_embeddings[embedding_key]).reshape(-1).astype(np.float32)

            x_coord, y_coord = parse_patch_xy(unique_key)
            record = {
                "unique_key": unique_key,
                "tissue": tissue,
                "patient": patient,
                "x": x_coord,
                "y": y_coord,
                "image_path": str(image_path),
                "gene": np.nan_to_num(np.asarray(vector).reshape(-1), nan=0.0).astype(np.float32),
            }
            if image_embedding is not None:
                record["image_embedding"] = image_embedding
            records.append(record)

    if aligned_key_order is not None:
        records_by_key = {record["unique_key"]: record for record in records}
        missing_keys = [key for key in aligned_key_order if key not in records_by_key]
        if missing_keys:
            raise ValueError(
                f"{len(missing_keys)} full-trimodal patches are missing from "
                f"patch_genes_{gene_variant}.pkl or their image files. Example: {missing_keys[0]}"
            )
        records = [records_by_key[key] for key in aligned_key_order if key in records_by_key]
    else:
        records.sort(key=lambda item: item["unique_key"])

    if not records:
        raise ValueError(f"No image/gene records found in {lung_dir}")

    gene_dims = {record["gene"].shape[0] for record in records}
    if len(gene_dims) != 1:
        raise ValueError(f"Found inconsistent gene dimensions: {sorted(gene_dims)}")

    if precomputed_image_embeddings:
        image_dims = {record["image_embedding"].shape[0] for record in records}
        if image_dims != {1536}:
            raise ValueError(f"Expected 1536-dimensional UNI2-h features, found: {sorted(image_dims)}")

    print(f"Loaded {len(records)} image/gene patch records ({next(iter(gene_dims))} genes).")
    return records


class LungPatchGeneDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        image_transform: Any,
        dataset_enum: Any,
        gene_batch_layout: str = "flat",
    ) -> None:
        self.records = records
        self.image_transform = image_transform
        self.DatasetEnum = dataset_enum
        self.gene_batch_layout = gene_batch_layout

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if "image_embedding" in record:
            image = torch.from_numpy(record["image_embedding"]).float()
        else:
            image = self.image_transform(Image.open(record["image_path"]).convert("RGB"))
        gene = torch.from_numpy(record["gene"]).float()
        if self.gene_batch_layout == "channel":
            gene = gene.unsqueeze(0)
        return {
            self.DatasetEnum.NAME: record["unique_key"],
            self.DatasetEnum.IMG: image,
            self.DatasetEnum.GEXP: gene,
            self.DatasetEnum.SOURCE: "lung",
            self.DatasetEnum.TISSUE: record["tissue"],
        }


class HescapeGeneCollator:
    """Apply the official HESCAPE gene transform to each collated batch."""

    def __init__(self, dataset_enum: Any, gene_transform: Any = None) -> None:
        self.DatasetEnum = dataset_enum
        self.gene_transform = gene_transform

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = default_collate(samples)
        if self.gene_transform is not None:
            genes = batch[self.DatasetEnum.GEXP]
            if genes.ndim == 2:
                genes = genes.unsqueeze(1)
            batch[self.DatasetEnum.GEXP] = self.gene_transform(genes).contiguous()
        return batch


def split_keys_to_indices(
    aligned_keys: list[str],
    train_keys: set[str],
    val_keys: set[str],
    test_keys: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(train_keys & val_keys) == 0, "Train/val key overlap"
    assert len(train_keys & test_keys) == 0, "Train/test key overlap"
    assert len(val_keys & test_keys) == 0, "Val/test key overlap"

    indices_train = np.array([i for i, key in enumerate(aligned_keys) if key in train_keys], dtype=int)
    indices_val = np.array([i for i, key in enumerate(aligned_keys) if key in val_keys], dtype=int)
    indices_test = np.array([i for i, key in enumerate(aligned_keys) if key in test_keys], dtype=int)

    n_total = len(indices_train) + len(indices_val) + len(indices_test)
    assert n_total == len(aligned_keys), (
        f"Only {n_total}/{len(aligned_keys)} aligned keys assigned to a split. "
        "This means some aligned keys are missing from train/val/test split keys."
    )

    return indices_train, indices_val, indices_test


def keys_to_indices(aligned_keys: list[str], key_set: set[str]) -> np.ndarray:
    indices = np.array([i for i, key in enumerate(aligned_keys) if key in key_set], dtype=int)
    missing_keys = key_set - set(aligned_keys)
    if missing_keys:
        raise ValueError(f"{len(missing_keys)} keys were not found in aligned keys. Example: {next(iter(missing_keys))}")
    return indices


def split_lopo_indices(
    records: list[dict[str, Any]],
    patient: str,
    rng: np.random.Generator,
    val_frac: float = 0.1,
    max_train_patches: int | None = None,
    max_val_patches: int | None = None,
    max_test_patches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Match alignment/final_models_compayl.ipynb lopo_model splitting."""
    aligned_keys = [record["unique_key"] for record in records]
    patient_records = [record for record in records if record["patient"] == patient]
    if not patient_records:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    trainval_keys = [record["unique_key"] for record in records if record["patient"] != patient]
    test_keys_all = {record["unique_key"] for record in patient_records}

    rng.shuffle(trainval_keys)
    split_point = int(len(trainval_keys) * (1.0 - val_frac))

    train_keys = set(trainval_keys[:split_point])
    val_keys = set(trainval_keys[split_point:])

    indices_train, indices_val, _ = split_keys_to_indices(
        aligned_keys,
        train_keys,
        val_keys,
        test_keys_all,
    )

    indices_train = cap_indices(indices_train, max_train_patches, rng)
    indices_val = cap_indices(indices_val, max_val_patches, rng)

    patient_tissues = list(dict.fromkeys(record["tissue"] for record in patient_records))
    test_by_tissue = {}
    for tissue in patient_tissues:
        sample_test_keys = {
            record["unique_key"]
            for record in patient_records
            if record["tissue"] == tissue
        }
        indices_test = keys_to_indices(aligned_keys, sample_test_keys)
        test_by_tissue[tissue] = cap_indices(indices_test, max_test_patches, rng)

    return indices_train, indices_val, test_by_tissue


def cap_indices(indices: np.ndarray, max_items: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_items is None or max_items <= 0 or len(indices) <= max_items:
        return indices
    # Keep at least two rows because official HESCAPE's generic gene encoder
    # squeezes singleton batches before BatchNorm.
    max_items = max(2, int(max_items))
    return np.sort(rng.choice(indices, size=max_items, replace=False))


def subset_records(records: list[dict[str, Any]], indices: np.ndarray) -> list[dict[str, Any]]:
    return [records[int(i)] for i in indices]


def make_loader(
    records: list[dict[str, Any]],
    image_transform: Any,
    dataset_enum: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
    gene_batch_layout: str = "flat",
    gene_transform: Any = None,
) -> DataLoader:
    if len(records) == 1:
        raise ValueError(
            "Official HESCAPE generic gene encoder cannot process singleton batches "
            "because it squeezes the batch dimension before BatchNorm."
        )
    return DataLoader(
        LungPatchGeneDataset(records, image_transform, dataset_enum, gene_batch_layout=gene_batch_layout),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=drop_last,
        collate_fn=HescapeGeneCollator(dataset_enum, gene_transform),
    )


def should_drop_singleton_remainder(n_records: int, batch_size: int) -> bool:
    return n_records > batch_size and (n_records % batch_size) == 1


def make_official_cfg(args: argparse.Namespace, output_dir: Path, strategy: str = "auto") -> Any:
    official = import_official_hescape(
        allow_drvi=args.gene_encoder == "drvi",
        allow_conch=args.image_encoder == "conch",
    )
    OmegaConf = official["OmegaConf"]
    return OmegaConf.create(
        {
            "paths": {
                "pretrain_weights": {
                    "img_enc_path": args.image_weights_root,
                    "gene_enc_path": args.gene_weights_root,
                },
                "anatomy": {
                    "output": str(output_dir),
                    "pretrain_weights": {"drvi_model_dir": args.drvi_model_dir},
                },
            },
            "training": {
                "lightning": {"trainer": {"strategy": strategy}},
                "evaluations": {"batch_key": "name", "label_key": "tissue"},
            },
        }
    )


def instantiate_official_module(args: argparse.Namespace, gene_dim: int, output_dir: Path) -> Any:
    configure_uni2h_weights(args)
    official = import_official_hescape(
        allow_drvi=args.gene_encoder == "drvi",
        allow_conch=args.image_encoder == "conch",
    )
    PretrainModule = official["PretrainModule"]
    cfg = make_official_cfg(args, output_dir)
    image_encoder = "precomputed_uni2h" if args.precomputed_image_embeddings else args.image_encoder
    return PretrainModule(
        input_genes=gene_dim,
        embed_dim=args.embed_dim,
        img_enc_name=image_encoder,
        gene_enc_name=args.gene_encoder,
        loss=args.loss,
        img_finetune=args.img_finetune,
        gene_finetune=args.gene_finetune,
        img_proj=args.img_proj,
        gene_proj=args.gene_proj,
        n_tissue=None,
        n_region=None,
        image_size=224,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cfg=cfg,
        lambda_scheduler=None,
    )


@torch.inference_mode()
def evaluate_global_official(
    pl_module: Any,
    loader: DataLoader,
    get_clip_metrics: Any,
    device: torch.device,
) -> dict[str, Any]:
    pl_module.eval()
    pl_module.to(device)
    img_parts: list[torch.Tensor] = []
    gene_parts: list[torch.Tensor] = []
    for batch in loader:
        batch_on_device = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        img_embed, gene_embed, _ = pl_module.model(batch_on_device, norm=False)
        img_parts.append(img_embed.detach().cpu())
        gene_parts.append(gene_embed.detach().cpu())

    img_all = torch.cat(img_parts, dim=0)
    gene_all = torch.cat(gene_parts, dim=0)
    logit_scale = pl_module.model.logit_scale.exp().detach().cpu()
    metrics = get_clip_metrics(img_all, gene_all, logit_scale=logit_scale, stage="test")
    metrics = {key.removeprefix("test/"): float(value.detach().cpu()) for key, value in metrics.items()}
    return {
        "n_test": int(img_all.shape[0]),
        "image_to_gene": _extract_direction(metrics, "image_to_gene"),
        "gene_to_image": _extract_direction(metrics, "gene_to_image"),
    }


def _extract_direction(metrics: dict[str, float], direction: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_name, value in metrics.items():
        if raw_name.startswith(direction):
            out[raw_name.removeprefix(f"{direction}_")] = value
    return out


def flatten_results(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_key, result in results.items():
        for direction in ["image_to_gene", "gene_to_image"]:
            metrics = result.get(direction)
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "method": "official_hescape",
                    "split_key": split_key,
                    "direction": direction,
                    "n_test": result.get("n_test"),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def aggregate_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    metric_cols = [
        col
        for col in [
            "R@1",
            "R@5",
            "R@10",
            "mean_rank",
            "median_rank",
        ]
        if col in rows.columns
    ]
    grouped = rows.groupby(["method", "direction"], dropna=False)
    summary = grouped[metric_cols].mean().reset_index()
    summary["n_splits"] = grouped.size().to_numpy()
    summary["n_test_total"] = grouped["n_test"].sum().to_numpy()
    return summary


def save_patient_results(
    output_dir: Path,
    patient: str,
    patient_results: dict[str, Any],
    gpu_memory: dict[str, float],
) -> Path:
    patient_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(patient))
    patient_dir = output_dir / "patients" / patient_name
    patient_dir.mkdir(parents=True, exist_ok=True)

    with (patient_dir / "official_hescape_results.pkl").open("wb") as fh:
        pickle.dump(patient_results, fh)

    rows = flatten_results(patient_results)
    rows.to_csv(patient_dir / "official_hescape_by_split.csv", index=False)
    aggregate_summary(rows).to_csv(patient_dir / "official_hescape_aggregate.csv", index=False)

    with (patient_dir / "gpu_memory.json").open("w") as fh:
        json.dump(gpu_memory, fh, indent=2)

    with (patient_dir / "completed.json").open("w") as fh:
        json.dump({"patient": str(patient), "split_keys": list(patient_results)}, fh, indent=2)

    return patient_dir


def run_official_hescape_lopo(args: argparse.Namespace) -> Path:
    if args.force_random_image_init:
        patch_timm_pretrained_for_smoke()
    configure_uni2h_weights(args)

    official = import_official_hescape(
        allow_drvi=args.gene_encoder == "drvi",
        allow_conch=args.image_encoder == "conch",
    )
    pl = official["pl"]
    DatasetEnum = official["DatasetEnum"]
    TRANSFORMS = official["TRANSFORMS"]
    ClampCallback = official["ClampCallback"]
    CSVLogger = official["CSVLogger"]
    EarlyStopping = official["EarlyStopping"]
    ModelCheckpoint = official["ModelCheckpoint"]
    get_clip_metrics = official["get_clip_metrics"]

    pl.seed_everything(args.seed, workers=True)

    lung_dir = Path(args.lung_dir)
    output_dir = Path(args.output_dir) / args.suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_lung_records(
        lung_dir,
        gene_variant=args.gene_variant,
        alignment_key_space=args.alignment_key_space,
        text_level=args.text_level,
        precomputed_image_embeddings=args.precomputed_image_embeddings,
    )
    metadata = pd.DataFrame(
        [{k: v for k, v in record.items() if k not in {"gene", "image_embedding"}} for record in records]
    )
    metadata.to_csv(output_dir / "official_hescape_aligned_lung_metadata.csv", index=False)

    config = OfficialHescapeRunConfig(**{field: getattr(args, field) for field in OfficialHescapeRunConfig.__dataclass_fields__})
    with (output_dir / "official_hescape_config.json").open("w") as fh:
        json.dump(asdict(config), fh, indent=2)

    image_transform = TRANSFORMS.get(args.image_encoder, TRANSFORMS["default"])
    if args.precomputed_image_embeddings:
        print("Image input: cached 1536-dimensional UNI2-h patch features; frozen UNI2-h forward is bypassed.")
    gene_transform = None
    if args.gene_variant == "raw_counts":
        gene_transform = TRANSFORMS["default_gene"] if args.gene_encoder == "generic" else TRANSFORMS["log1p_only"]
        print(f"Gene input: patch_genes_raw_counts.pkl with official HESCAPE {args.gene_encoder} preprocessing.")
    gene_batch_layout = args.gene_batch_layout
    if gene_batch_layout == "auto":
        gene_batch_layout = "channel" if args.gene_encoder == "drvi" else "flat"
    gene_dim = int(records[0]["gene"].shape[0])
    patients = metadata["patient"].dropna().unique().tolist()
    if args.patient is not None:
        if args.patient not in patients:
            raise ValueError(
                f"Requested patient {args.patient!r} was not found. "
                f"Available patients: {patients}"
            )
        patients = [args.patient]
    elif args.limit_patients is not None:
        patients = patients[: args.limit_patients]
    split_rng = np.random.default_rng(args.split_seed)

    device = torch.device("cpu")
    if args.accelerator == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.accelerator in {"auto", "mps"} and torch.backends.mps.is_available():
        device = torch.device("mps")

    results: dict[str, Any] = {}
    for patient in patients:
        print("\n========================================")
        print(f"HOLD OUT PATIENT: {patient}")
        indices_train, indices_val, test_by_tissue = split_lopo_indices(
            records,
            patient,
            split_rng,
            val_frac=args.val_frac,
            max_train_patches=args.max_train_patches,
            max_val_patches=args.max_val_patches,
            max_test_patches=args.max_test_patches,
        )

        patient_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(patient))
        patient_dir = output_dir / "patients" / patient_name
        patient_results_path = patient_dir / "official_hescape_results.pkl"
        if args.skip_completed_patients and (patient_dir / "completed.json").exists() and patient_results_path.exists():
            completed_results = load_pickle(patient_results_path)
            results.update(completed_results)
            print(f"Skipping completed patient {patient}; loaded results from {patient_dir}")
            continue

        if len(indices_train) == 0 or len(indices_val) == 0:
            print(f"Skipping {patient}: empty train or validation split.")
            continue
        print(f"Length training: {len(indices_train)}. Length val: {len(indices_val)}.")
        if args.max_train_patches or args.max_val_patches or args.max_test_patches:
            print(
                "Patch caps: "
                f"train={args.max_train_patches}, val={args.max_val_patches}, test={args.max_test_patches}"
            )

        fold_output = output_dir / "checkpoints" / re.sub(r"[^A-Za-z0-9_.-]+", "_", str(patient))
        fold_output.mkdir(parents=True, exist_ok=True)
        model = instantiate_official_module(args, gene_dim, fold_output)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_loader = make_loader(
            train_records := subset_records(records, indices_train),
            image_transform,
            DatasetEnum,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            drop_last=should_drop_singleton_remainder(len(train_records), args.batch_size),
            gene_batch_layout=gene_batch_layout,
            gene_transform=gene_transform,
        )
        val_loader = make_loader(
            val_records := subset_records(records, indices_val),
            image_transform,
            DatasetEnum,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            drop_last=should_drop_singleton_remainder(len(val_records), args.batch_size),
            gene_batch_layout=gene_batch_layout,
            gene_transform=gene_transform,
        )

        callbacks = [
            ClampCallback(),
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        ]
        if not args.disable_checkpoints:
            callbacks.append(
                ModelCheckpoint(dirpath=str(fold_output), monitor="val_loss", mode="min", save_top_k=1, save_last=True)
            )
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            logger=CSVLogger(save_dir=str(fold_output / "csv_logger"), name=""),
            callbacks=callbacks,
            enable_checkpointing=not args.disable_checkpoints,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        patient_results: dict[str, Any] = {}
        for tissue, indices_test in test_by_tissue.items():
            key_name = str(patient) if len(test_by_tissue) == 1 else f"{patient}__{tissue}"
            print(f"-> Evaluating sample: {tissue} ({len(indices_test)} patches)")
            test_loader = make_loader(
                test_records := subset_records(records, indices_test),
                image_transform,
                DatasetEnum,
                min(args.batch_size, len(indices_test)),
                args.num_workers,
                shuffle=False,
                drop_last=should_drop_singleton_remainder(len(test_records), min(args.batch_size, len(indices_test))),
                gene_batch_layout=gene_batch_layout,
                gene_transform=gene_transform,
            )
            patient_results[key_name] = evaluate_global_official(model, test_loader, get_clip_metrics, device)

        gpu_memory: dict[str, float] = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gpu_memory = {
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
            }

        results.update(patient_results)
        saved_patient_dir = save_patient_results(output_dir, str(patient), patient_results, gpu_memory)
        print(f"Saved completed patient results to {saved_patient_dir}")
        if gpu_memory:
            print(
                "Peak GPU memory: "
                f"allocated={gpu_memory['peak_allocated_mb']:.1f} MB, "
                f"reserved={gpu_memory['peak_reserved_mb']:.1f} MB"
            )

    with (output_dir / f"{args.suffix}_official_hescape_results.pkl").open("wb") as fh:
        pickle.dump(results, fh)
    rows = flatten_results(results)
    rows.to_csv(output_dir / f"{args.suffix}_official_hescape_by_split.csv", index=False)
    aggregate_summary(rows).to_csv(output_dir / f"{args.suffix}_official_hescape_aggregate.csv", index=False)
    print(f"Done. Official HESCAPE outputs written to {output_dir}")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official HESCAPE code on local lung patch PNGs and gene counts.")
    parser.add_argument("--lung-dir", default=DEFAULT_LUNG_DIR)
    parser.add_argument("--output-dir", default=str(HERE / "outputs" / "official_hescape"))
    parser.add_argument("--suffix", default="official_hescape_lung_full_lopo")
    parser.add_argument("--gene-variant", default="raw_counts")
    parser.add_argument("--alignment-key-space", choices=["full_trimodal", "image_gene"], default="full_trimodal")
    parser.add_argument("--text-level", default="level_1_2_3")
    parser.add_argument("--image-encoder", default="densenet")
    parser.add_argument("--precomputed-image-embeddings", action="store_true")
    parser.add_argument("--image-weights-root", default="")
    parser.add_argument("--gene-encoder", choices=["generic", "drvi"], default="generic")
    parser.add_argument("--gene-weights-root", default="")
    parser.add_argument("--drvi-model-dir", default="")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--loss", choices=["CLIP", "SIGLIP"], default="CLIP")
    parser.add_argument("--img-proj", choices=["linear", "mlp", "transformer"], default="linear")
    parser.add_argument("--gene-proj", choices=["linear", "mlp", "identity"], default="linear")
    parser.add_argument("--img-finetune", action="store_true")
    parser.add_argument("--gene-finetune", action="store_true")
    parser.add_argument("--gene-batch-layout", choices=["auto", "flat", "channel"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=24442)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--patient", default=None, help="Run one specific held-out patient.")
    parser.add_argument("--max-train-patches", type=int, default=None)
    parser.add_argument("--max-val-patches", type=int, default=None)
    parser.add_argument("--max-test-patches", type=int, default=None)
    parser.add_argument("--force-random-image-init", action="store_true")
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--skip-completed-patients", action="store_true")
    parser.add_argument("--enable-progress-bar", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_official_hescape_lopo(parse_args())

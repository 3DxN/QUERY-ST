import argparse
import gc
import os
import sys
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations_functions import (
    abl_study_cell_type_model,
    abl_study_lambda,
    abl_study_reports_model,
)
from common_functions import (
    create_multisample_metadata,
    fuse_gene_cell_type_vectors,
    load_samples,
    split_checkerboard_xenium,
)
from training_functions import load_trimodal_inputs


DEFAULT_LUNG_DIR = "/well/rittscher/users/mju725/trimodal_alignment_objects/lung"
DEFAULT_PROSTATE_DIR = "/well/rittscher/users/mju725/trimodal_alignment_objects/prostate"
DEFAULT_MODELS_DIR = "/well/rittscher/users/mju725/trimodal_alignment_objects/models"


def load_dataset(dataset, dataset_dir):
    print("\n" + "=" * 70)
    print(f"Loading {dataset} inputs from {dataset_dir}")
    print("=" * 70)

    samples = load_samples(str(dataset_dir))
    uni_embeddings, cell_vectors, gene_vectors, text_embeddings, _ = (
        load_trimodal_inputs(samples, str(dataset_dir))
    )

    metadata = create_multisample_metadata(list(uni_embeddings.keys()))
    train_keys, val_keys, test_keys = split_checkerboard_xenium(
        metadata,
        n_tiles_row=5,
        n_tiles_col=5,
        val_frac=0.1,
        test_frac=0.1,
        return_keys=True,
        random_state=42,
    )

    common_fused_keys = sorted(set(cell_vectors) & set(gene_vectors["full"]))
    fused_vectors = fuse_gene_cell_type_vectors(
        common_fused_keys,
        cell_vectors,
        gene_vectors["full"],
    )

    return {
        "name": dataset,
        "dir": str(dataset_dir),
        "uni": uni_embeddings,
        "cell": cell_vectors,
        "gene": gene_vectors["full"],
        "fused": fused_vectors,
        "text": text_embeddings,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "test_keys": test_keys,
    }


def run_ablation_1(data, models_dir, run_checks):
    """Compare cell-type vectors with fused gene and cell-type vectors."""
    dataset = data["name"]
    print(f"\nAblation 1: transcriptomic representation ({dataset})")

    return abl_study_cell_type_model(
        uni_embeddings=data["uni"],
        cell_embeddings=data["cell"],
        gene_embeddings=data["gene"],
        text_embeddings=data["text"]["level_1_2_3"],
        train_keys=data["train_keys"],
        val_keys=data["val_keys"],
        test_keys=data["test_keys"],
        models_output_dir=str(models_dir),
        suffix_out_dir=f"abl_transcriptomic_representation_{dataset}",
        objects_output_dir=data["dir"],
        run_checks=run_checks,
    )


def run_ablation_2(data, models_dir, run_checks):
    """Sweep the RNA-text and image-text loss weights."""
    dataset = data["name"]
    print(f"\nAblation 2: loss-weight sensitivity ({dataset})")

    return abl_study_lambda(
        lambda1_rna_text_values=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        lambda2_img_text_values=[0, 0.1, 0.25, 0.5],
        uni_embeddings=data["uni"],
        target_embeddings=data["fused"],
        text_embeddings=data["text"]["level_1_2_3"],
        train_keys=data["train_keys"],
        val_keys=data["val_keys"],
        test_keys=data["test_keys"],
        models_output_dir=str(models_dir),
        suffix_out_dir=f"abl_lambda_sweep_{dataset}",
        objects_output_dir=data["dir"],
        run_checks=run_checks,
    )


def run_ablation_3(data, models_dir, run_checks):
    """Compare the available pathology-report granularity levels."""
    dataset = data["name"]
    print(f"\nAblation 3: report granularity ({dataset})")

    results, _ = abl_study_reports_model(
        uni_embeddings=data["uni"],
        fused_vector=data["fused"],
        master_text_embeddings=data["text"],
        train_keys=data["train_keys"],
        val_keys=data["val_keys"],
        test_keys=data["test_keys"],
        models_output_dir=str(models_dir),
        suffix_out_dir=f"abl_reports_{dataset}",
        objects_output_dir=data["dir"],
        lambda_rna_text=0.4,
        lambda_img_text=0.1,
        run_checks=run_checks,
    )
    return results


def main(args):
    dataset_dirs = {
        "lung": Path(args.lung_dir),
        "prostate": Path(args.prostate_dir),
    }
    models_dir = Path(args.models_dir) / "abl_studies_compayl"
    models_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        1: run_ablation_1,
        2: run_ablation_2,
        3: run_ablation_3,
    }

    print(f"Datasets: {args.datasets}")
    print(f"Ablations: {args.ablations}")
    print(f"Model output directory: {models_dir}")
    print(f"Alignment checks enabled: {not args.skip_checks}")

    for dataset in args.datasets:
        data = load_dataset(dataset, dataset_dirs[dataset])

        for ablation in args.ablations:
            runners[ablation](data, models_dir, run_checks=not args.skip_checks)

        del data
        gc.collect()

    print("\nAll requested ablations completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lung and prostate ablation studies 1-3."
    )
    parser.add_argument("--lung-dir", default=DEFAULT_LUNG_DIR)
    parser.add_argument("--prostate-dir", default=DEFAULT_PROSTATE_DIR)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("lung", "prostate"),
        default=["lung", "prostate"],
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip alignment sanity checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

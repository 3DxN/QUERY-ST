from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import glob
from common_functions import *
from trimodal_encoder import TriModalEncoder

def load_trimodal_inputs(samples, input_dir):
    uni_embeddings = {}
    cell_vectors = {}
    master_gene_vectors = defaultdict(dict)
    master_text_embeddings = defaultdict(dict)
    master_openai_dict_gt = defaultdict(dict)

    def clean_dict(input_dict, output_dict):
        for k, v in input_dict.items():
            clean_k = k.removesuffix(".png")
            new_key = f"{tissue_name}__{clean_k}"
            output_dict[new_key] = v
        return output_dict

    for sample_id in tqdm(samples, disable=True):
        tissue_name = sample_id.rsplit("_", 1)[-1]
        patch_dir = os.path.join(input_dir, sample_id)

        # UNI
        uni_path = os.path.join(patch_dir, f"uni2h_{tissue_name}.pkl")
        if not os.path.exists(uni_path):
            continue

        with open(uni_path, "rb") as f:
            uni_emb = pickle.load(f)

        uni_embeddings = clean_dict(uni_emb, uni_embeddings)

        # Cell types
        cell_path = os.path.join(patch_dir, "patch_vectors.pkl")
        if not os.path.exists(cell_path):
            continue

        with open(cell_path, "rb") as f:
            cell_vec = pickle.load(f)

        cell_vectors = clean_dict(cell_vec, cell_vectors)

        genes_paths = glob.glob(os.path.join(patch_dir, "patch_genes*.pkl"))

        for path in genes_paths:
            variant = (
                os.path.basename(path)
                .removesuffix(".pkl")
                .removeprefix("patch_genes_")
            )

            with open(path, "rb") as f:
                gene_vec = pickle.load(f)

            for k, v in gene_vec.items():
                clean_k = k.removesuffix(".png")
                new_key = f"{tissue_name}__{clean_k}"
                master_gene_vectors[variant][new_key] = v

    # Text embeddings (multiple levels)
    text_dir = os.path.join(
        input_dir,
        "text-descriptions-all-samples-levels",
    )

    text_paths = glob.glob(
        os.path.join(text_dir, "openai_text_embeddings_*.pkl")
    )

    for path in text_paths:
        level = (
            os.path.basename(path)
            .removesuffix(".pkl")
            .removeprefix("openai_text_embeddings_")
        )

        with open(path, "rb") as f:
            text_emb = pickle.load(f)

        master_text_embeddings[level] = text_emb

    descriptions_gt_paths = glob.glob(
        os.path.join(text_dir, "openai_descriptions_dict_*.pkl")
    )

    for path in descriptions_gt_paths:
        level = (
            os.path.basename(path)
            .removesuffix(".pkl")
            .removeprefix("openai_descriptions_dict_")
        )

        with open(path, "rb") as f:
            descriptions_dict = pickle.load(f)

        master_openai_dict_gt[level] = descriptions_dict

    return (
        uni_embeddings,
        cell_vectors,
        dict(master_gene_vectors),
        dict(master_text_embeddings),
        dict(master_openai_dict_gt)
    )

def train_and_test_model(X, Y, Z, indices_train, indices_val, indices_test, models_output_dir, suffix_out_dir, output_model = False, verbose=False, lambda_rna_text=0.5, lambda_img_text=0.25):
    evaluate_text = lambda_rna_text > 0 or lambda_img_text > 0

    trimodal_model = TriModalEncoder(
        img_dim=1536,
        rna_dim=Y.shape[1],
        text_dim=1536,
        proj_dim=128,
        hidden_dim=2048,
        dropout=0.4,
    )

    print(f"{models_output_dir}/{suffix_out_dir}")

    trimodal_model.fit(
        X,
        Y,
        Z,
        indices_train,
        indices_val,
        out_dir=f"{models_output_dir}/{suffix_out_dir}",
        lambda_rna_text=lambda_rna_text,
        lambda_img_text=lambda_img_text,
        epochs=50,
        lr=1e-4,
        patience=10,
        verbose=verbose
    )

    results = evaluate_test_trimodal(
        trimodal_model,
        X,
        Y,
        Z,
        indices_test,
        evaluate_text = evaluate_text
    )
    print(results)

    if output_model:
        return trimodal_model, results

    return results

def train_only_model(
    X,
    Y,
    Z,
    indices_train,
    indices_val,
    models_output_dir,
    suffix_out_dir,
    output_model=False,
    lambda_rna_text=0.5,
    lambda_img_text=0.25,
    hidden_dim=2048,
):
    trimodal_model = TriModalEncoder(
        img_dim=1536,
        rna_dim=Y.shape[1],
        text_dim=1536,
        proj_dim=128,
        hidden_dim=hidden_dim,
        dropout=0.4,
    )

    if lambda_rna_text == 0 and lambda_img_text == 0:
        trimodal_model.text_proj.requires_grad_(False)

    print(f"{models_output_dir}/{suffix_out_dir}")

    trimodal_model.fit(
        X,
        Y,
        Z,
        indices_train,
        indices_val,
        out_dir=f"{models_output_dir}/{suffix_out_dir}",
        lambda_rna_text=lambda_rna_text,
        lambda_img_text=lambda_img_text,
        epochs=50,
        lr=1e-4,
        patience=10,
        verbose=False
    )
    
    return trimodal_model

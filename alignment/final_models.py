import pickle
import os

import sys
from pathlib import Path

project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))

from common_functions import *
from training_functions import *
from retrieval_functions import *

import numpy as np

def lopo_model(
    meta_df,
    unique_patients,
    patient_string,
    uni_embeddings,
    fused_vector,
    text_embeddings,
    models_output_dir,
    suffix_out_dir,
    objects_output_dir,
    lambda_rna_text=0.5,
    lambda_img_text=0.25,
    val_frac=0.1,
    seed=42,
    run_checks=True,
    hidden_dim=2048
):
    """
    Leave-one-patient-out evaluation.
    """
    results_leave_one_out = {}
    models_leave_one_out = {}

    os.makedirs(f"{objects_output_dir}/lopo_objects", exist_ok=True)

    meta_df = meta_df.copy()

    assert "unique_key" in meta_df.columns, "meta_df must contain unique_key"
    assert "tissue" in meta_df.columns, "meta_df must contain tissue"
    assert patient_string in meta_df.columns, f"meta_df must contain {patient_string}"

    if meta_df[patient_string].isna().any():
        missing_tissues = meta_df.loc[meta_df[patient_string].isna(), "tissue"].unique()
        raise ValueError(f"Some tissues have no patient mapping: {missing_tissues}")

    X_loo, Y_loo, Z_loo, aligned_keys = match_and_align_trimodal(
        uni_embeddings,
        fused_vector,
        text_embeddings,
        key_order=list(uni_embeddings.keys()),
        return_keys=True,
    )

    aligned_key_set = set(aligned_keys)

    meta_aligned = meta_df[meta_df["unique_key"].isin(aligned_key_set)].copy()

    if len(meta_aligned) != len(aligned_keys):
        raise ValueError(
            f"Metadata/alignment mismatch: {len(meta_aligned)} metadata rows "
            f"but {len(aligned_keys)} aligned keys."
        )

    rng = np.random.default_rng(seed)

    for patient in unique_patients:
        print("\n========================================")
        print(f"HOLD OUT PATIENT: {patient}")

        patient_meta = meta_aligned[meta_aligned[patient_string] == patient].copy()

        if len(patient_meta) == 0:
            print(f"Skipping {patient}: no aligned patches found.")
            continue

        trainval_meta = meta_aligned[meta_aligned[patient_string] != patient].copy()

        trainval_keys = trainval_meta["unique_key"].tolist()
        test_keys_all = set(patient_meta["unique_key"].tolist())

        rng.shuffle(trainval_keys)

        split_point = int(len(trainval_keys) * (1.0 - val_frac))

        train_keys = set(trainval_keys[:split_point])
        val_keys = set(trainval_keys[split_point:])

        indices_train_loo, indices_val_loo, indices_test_patient = split_keys_to_indices(
            aligned_keys,
            train_keys,
            val_keys,
            test_keys_all,
        )

        if run_checks:
            run_alignment_tests(
                indices_train=indices_train_loo,
                indices_val=indices_val_loo,
                indices_test=indices_test_patient,
                X=X_loo,
                Y=Y_loo,
                Z=Z_loo,
                aligned_keys=aligned_keys,
                train_keys=train_keys,
                val_keys=val_keys,
                test_keys=test_keys_all,
                uni_embeddings=uni_embeddings,
                c2s_embeddings=fused_vector,
                text_embeddings=text_embeddings,
            )

        patient_samples = patient_meta["tissue"].unique()

        print("Held-out samples:", patient_samples)
        print(
            f"Length training: {len(indices_train_loo)}. "
            f"Length val: {len(indices_val_loo)}. "
            f"Length patient test: {len(indices_test_patient)}"
        )

        model = train_only_model(
            X_loo,
            Y_loo,
            Z_loo,
            indices_train_loo,
            indices_val_loo,
            models_output_dir,
            f"lopo/lopo_{suffix_out_dir}_{patient}",
            output_model=True,
            lambda_rna_text=lambda_rna_text,
            lambda_img_text=lambda_img_text,
            hidden_dim=hidden_dim
        )

        models_leave_one_out[patient] = model

        for sample_id in patient_samples:
            sample_test_keys = set(
                patient_meta.loc[
                    patient_meta["tissue"] == sample_id,
                    "unique_key",
                ].tolist()
            )

            indices_test_sample = keys_to_indices(aligned_keys, sample_test_keys)

            if len(indices_test_sample) == 0:
                raise ValueError(
                    f"No aligned test indices found for patient={patient}, sample={sample_id}"
                )

            print(f"-> Evaluating sample: {sample_id} ({len(indices_test_sample)} patches)")

            if len(patient_samples) == 1:
                key_name = patient
            else:
                key_name = f"{patient}__{sample_id}"

            results_leave_one_out[key_name] = evaluate_test_trimodal(
                model,
                X_loo,
                Y_loo,
                Z_loo,
                indices_test_sample,
                evaluate_text=(lambda_rna_text > 0 or lambda_img_text > 0),
            )

    with open(f"{objects_output_dir}/lopo_objects/lopo_{suffix_out_dir}_models.pkl", "wb") as f:
        pickle.dump(models_leave_one_out, f)

    with open(f"{objects_output_dir}/lopo_objects/lopo_{suffix_out_dir}_results.pkl", "wb") as f:
        pickle.dump(results_leave_one_out, f)

    return results_leave_one_out, models_leave_one_out


def run_spatial_kfold(
    meta_df,
    uni_emb,
    fused_vec,
    text_emb,
    models_out_dir,
    base_suffix,
    objects_output_dir,
    n_folds=5,
    n_tiles=20,
    lambda_rna_text=0.5,
    lambda_img_text=0.25,
    run_checks=True,
):
    """
    Spatial K-fold CV.

    Important:
    - Folds are generated as patch keys.
    - Keys are converted to row indices after alignment.
    - This avoids row-order mismatch between meta_df and X/Y/Z.
    """

    print(f"--- Starting {n_folds}-Fold Spatial CV for {base_suffix} ---")

    os.makedirs(f"{objects_output_dir}/cv_objects", exist_ok=True)

    meta_df = meta_df.copy()

    assert "unique_key" in meta_df.columns, "meta_df must contain unique_key"
    assert "tissue" in meta_df.columns, "meta_df must contain tissue"
    assert "x" in meta_df.columns, "meta_df must contain x"
    assert "y" in meta_df.columns, "meta_df must contain y"

    X, Y, Z, aligned_keys = match_and_align_trimodal(
        uni_emb,
        fused_vec,
        text_emb,
        key_order=list(uni_emb.keys()),
        return_keys=True,
    )

    aligned_key_set = set(aligned_keys)

    meta_aligned = meta_df[meta_df["unique_key"].isin(aligned_key_set)].copy()

    if len(meta_aligned) != len(aligned_keys):
        raise ValueError(
            f"Metadata/alignment mismatch: {len(meta_aligned)} metadata rows "
            f"but {len(aligned_keys)} aligned keys."
        )

    folds = split_checkerboard_xenium(
        meta_aligned,
        n_folds=n_folds,
        n_tiles_row=n_tiles,
        n_tiles_col=n_tiles,
        random_state=42,
        return_keys=True,
    )

    kfold_results = {}
    kfold_models = {}

    for i, (train_keys, val_keys, test_keys) in enumerate(folds):
        fold_id = i + 1

        print(f"\n=== Running Fold {fold_id} / {n_folds} ===")

        train_keys = set(train_keys)
        val_keys = set(val_keys)
        test_keys = set(test_keys)

        idx_train, idx_val, idx_test = split_keys_to_indices(
            aligned_keys,
            train_keys,
            val_keys,
            test_keys,
        )

        if run_checks:
            run_alignment_tests(
                indices_train=idx_train,
                indices_val=idx_val,
                indices_test=idx_test,
                X=X,
                Y=Y,
                Z=Z,
                aligned_keys=aligned_keys,
                train_keys=train_keys,
                val_keys=val_keys,
                test_keys=test_keys,
                uni_embeddings=uni_emb,
                c2s_embeddings=fused_vec,
                text_embeddings=text_emb,
            )

        print(
            f"Fold {fold_id} sizes: "
            f"train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}"
        )

        fold_suffix = f"{base_suffix}/fold{fold_id}"

        model, results = train_and_test_model(
            X,
            Y,
            Z,
            idx_train,
            idx_val,
            idx_test,
            models_out_dir,
            fold_suffix,
            output_model=True,
            lambda_rna_text=lambda_rna_text,
            lambda_img_text=lambda_img_text,
        )

        kfold_results[f"fold_{fold_id}"] = results
        kfold_models[f"fold_{fold_id}"] = model

    with open(f"{objects_output_dir}/cv_objects/cv_{base_suffix}_models.pkl", "wb") as f:
        pickle.dump(kfold_models, f)

    with open(f"{objects_output_dir}/cv_objects/cv_{base_suffix}_results.pkl", "wb") as f:
        pickle.dump(kfold_results, f)

    return kfold_results, kfold_models

def main():
    # Lung
    lung_dir =  "/well/rittscher/users/mju725/trimodal_alignment_objects/lung"
    models_output_dir = "/well/rittscher/users/mju725/trimodal_alignment_objects/models"

    samples_lung = load_samples(lung_dir)

    uni_embeddings_lung, cell_vectors_lung, master_gene_vectors_lung, master_text_embeddings_lung, ground_truth_openai_dict_lung = load_trimodal_inputs(samples_lung, lung_dir)

    meta_df_lung = create_multisample_metadata(list(uni_embeddings_lung.keys()))

    lung_full_meta = pd.read_csv(f"{lung_dir}/metadata_lung.csv")

    sample_to_patient = dict(zip(lung_full_meta["sample"], lung_full_meta["patient"]))
    meta_df_lung["patient"] = meta_df_lung["tissue"].map(sample_to_patient)

    unique_patients_lung = meta_df_lung["patient"].dropna().unique().tolist()

    common_keys_fused_lung = sorted(
        list(set(cell_vectors_lung.keys()) & set(master_gene_vectors_lung["full"].keys()))
    )

    master_fused_vectors_lung = fuse_gene_cell_type_vectors(
        common_keys_fused_lung,
        cell_vectors_lung,
        master_gene_vectors_lung["full"],
    )

    results_lopo_lung, models_lopo_lung = lopo_model(
        meta_df=meta_df_lung,
        unique_patients=unique_patients_lung,
        patient_string="patient",
        uni_embeddings=uni_embeddings_lung,
        fused_vector=master_fused_vectors_lung,
        text_embeddings=master_text_embeddings_lung["level_1_2_3"],
        models_output_dir=models_output_dir,
        suffix_out_dir="lung_final",
        objects_output_dir=lung_dir,
        lambda_rna_text=0.4,
        lambda_img_text=0.1,
        run_checks=True,
    )

    results_lopo_lung_bimodal, models_lopo_lung_bimodal = lopo_model(
        meta_df=meta_df_lung,
        unique_patients=unique_patients_lung,
        patient_string="patient",
        uni_embeddings=uni_embeddings_lung,
        fused_vector=master_fused_vectors_lung,
        text_embeddings=master_text_embeddings_lung["level_1_2_3"],
        models_output_dir=models_output_dir,
        suffix_out_dir="lung_final_bimodal",
        objects_output_dir=lung_dir,
        lambda_rna_text=0,
        lambda_img_text=0,
        run_checks=True,
    )

    results_lopo_lung, models_lopo_lung = lopo_model(
        meta_df=meta_df_lung,
        unique_patients=unique_patients_lung,
        patient_string="patient",
        uni_embeddings=uni_embeddings_lung,
        fused_vector=master_gene_vectors_lung["full"],
        text_embeddings=master_text_embeddings_lung["level_1_2_3"],
        models_output_dir=models_output_dir,
        suffix_out_dir="lung_final_gene_vecs_only",
        objects_output_dir=lung_dir,
        lambda_rna_text=0.4,
        lambda_img_text=0.1,
        run_checks=True,
    )

    results_lopo_lung_bimodal, models_lopo_lung_bimodal = lopo_model(
        meta_df=meta_df_lung,
        unique_patients=unique_patients_lung,
        patient_string="patient",
        uni_embeddings=uni_embeddings_lung,
        fused_vector=master_gene_vectors_lung["full"],
        text_embeddings=master_text_embeddings_lung["level_1_2_3"],
        models_output_dir=models_output_dir,
        suffix_out_dir="lung_final_gene_vecs_only_bimodal",
        objects_output_dir=lung_dir,
        lambda_rna_text=0,
        lambda_img_text=0,
        run_checks=True,
    )
            
    # matched scale (for hescape comparison)
    results_leave_one_out_matched_bimodal, models_leave_one_out_matched_bimodal = lopo_model(
        meta_df=meta_df_lung,
        unique_patients=unique_patients_lung,
        patient_string="patient",
        uni_embeddings=uni_embeddings_lung,
        fused_vector=master_gene_vectors_lung["full"],
        text_embeddings=master_text_embeddings_lung["level_1_2_3"],
        models_output_dir=models_output_dir,
        suffix_out_dir="lung_matched_bimodal_hidden128",
        objects_output_dir=lung_dir,
        lambda_rna_text=0,
        lambda_img_text=0,
        seed=42,
        run_checks=True,
        hidden_dim=128
    )


    ## Prostate
    prostate_dir = "/well/rittscher/users/mju725/trimodal_alignment_objects/prostate"
    samples_prostate = load_samples(prostate_dir)

    uni_embeddings_prostate, cell_vectors_prostate, master_gene_vectors_prostate, master_text_embeddings_prostate, ground_truth_openai_dict_prostate = load_trimodal_inputs(samples_prostate, prostate_dir)

    meta_df_prostate = create_multisample_metadata(list(uni_embeddings_prostate.keys()))

    common_keys_fused_prostate = sorted(
        list(set(cell_vectors_prostate.keys()) & set(master_gene_vectors_prostate['full'].keys()))
    )

    master_fused_vectors_prostate = fuse_gene_cell_type_vectors(
        common_keys_fused_prostate, 
        cell_vectors_prostate, 
        master_gene_vectors_prostate['full']
    )


    kfold_results_prostate_5tiles, kfold_models_prostate_5tiles = run_spatial_kfold(
        meta_df_prostate, 
        uni_embeddings_prostate, 
        master_fused_vectors_prostate,
        master_text_embeddings_prostate['level_1_2_3'], 
        models_output_dir, 
        "prostate_final_spatial_kfold_5tiles",
        prostate_dir,
        n_folds=10,
        n_tiles=5,
        lambda_rna_text=0.4,
        lambda_img_text=0.1
    )

    kfold_results_prostate_5tiles_bimodal, kfold_models_prostate_5tiles_bimodal = run_spatial_kfold(
        meta_df_prostate, 
        uni_embeddings_prostate, 
        master_fused_vectors_prostate,
        master_text_embeddings_prostate['level_1_2_3'], 
        models_output_dir, 
        "prostate_final_spatial_kfold_5tiles_bimodal",
        prostate_dir,
        n_folds=10,
        n_tiles=5,
        lambda_rna_text=0,
        lambda_img_text=0
    )


    kfold_results_prostate_5tiles, kfold_models_prostate_5tiles = run_spatial_kfold(
        meta_df_prostate, 
        uni_embeddings_prostate, 
        master_gene_vectors_prostate['full'],
        master_text_embeddings_prostate['level_1_2_3'], 
        models_output_dir, 
        "prostate_final_spatial_kfold_5tiles_gene_vecs_only",
        prostate_dir,
        n_folds=10,
        n_tiles=5,
        lambda_rna_text=0.4,
        lambda_img_text=0.1
    )

    kfold_results_prostate_5tiles_bimodal, kfold_models_prostate_5tiles_bimodal = run_spatial_kfold(
        meta_df_prostate, 
        uni_embeddings_prostate, 
        master_gene_vectors_prostate['full'],
        master_text_embeddings_prostate['level_1_2_3'], 
        models_output_dir, 
        "prostate_final_spatial_kfold_5tiles_bimodal_gene_vecs_only",
        prostate_dir,
        n_folds=10,
        n_tiles=5,
        lambda_rna_text=0,
        lambda_img_text=0
    )

if __name__ == "__main__":
    main()
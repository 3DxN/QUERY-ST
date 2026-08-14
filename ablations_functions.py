
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pickle
import textwrap
from matplotlib.lines import Line2D
from tqdm import tqdm
from collections import defaultdict, Counter

from common_functions import match_and_align_trimodal, split_keys_to_indices, run_alignment_tests, fuse_gene_cell_type_vectors
from training_functions import train_and_test_model

def prepare_aligned_ablation_data(
    uni_embeddings,
    target_embeddings,
    text_embeddings,
    train_keys,
    val_keys,
    test_keys,
    run_checks=True,
):
    """
    Align X/Y/Z, convert fixed split keys to row indices, and optionally run sanity checks.

    This keeps the same train/val/test split across ablations while avoiding row-order bugs.
    """

    X, Y, Z, aligned_keys = match_and_align_trimodal(
        uni_embeddings,
        target_embeddings,
        text_embeddings,
        key_order=list(uni_embeddings.keys()),
        return_keys=True,
    )

    aligned_key_set = set(aligned_keys)

    train_keys_aligned = set(train_keys) & aligned_key_set
    val_keys_aligned = set(val_keys) & aligned_key_set
    test_keys_aligned = set(test_keys) & aligned_key_set

    n_requested = len(set(train_keys)) + len(set(val_keys)) + len(set(test_keys))
    n_available = (
        len(train_keys_aligned)
        + len(val_keys_aligned)
        + len(test_keys_aligned)
    )

    if n_available != len(aligned_keys):
        raise ValueError(
            f"Split/alignment mismatch: {n_available} split keys assigned, "
            f"but {len(aligned_keys)} aligned keys exist."
        )

    if n_available < n_requested:
        print(
            f"Note: using {n_available}/{n_requested} split keys for this ablation "
            f"because some keys are missing from one or more modalities."
        )

    indices_train, indices_val, indices_test = split_keys_to_indices(
        aligned_keys,
        train_keys_aligned,
        val_keys_aligned,
        test_keys_aligned,
    )

    if run_checks:
        run_alignment_tests(
            indices_train=indices_train,
            indices_val=indices_val,
            indices_test=indices_test,
            X=X,
            Y=Y,
            Z=Z,
            aligned_keys=aligned_keys,
            train_keys=train_keys_aligned,
            val_keys=val_keys_aligned,
            test_keys=test_keys_aligned,
            uni_embeddings=uni_embeddings,
            c2s_embeddings=target_embeddings,
            text_embeddings=text_embeddings,
        )

    return X, Y, Z, aligned_keys, indices_train, indices_val, indices_test


def save_ablation_results(all_results, objects_output_dir, suffix_out_dir):
    out_dir = f"{objects_output_dir}/ablation_objects"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/abl_{suffix_out_dir}.pkl", "wb") as f:
        pickle.dump(all_results, f)
        
def abl_study_lambda(
    lambda1_rna_text_values,
    lambda2_img_text_values,
    uni_embeddings,
    target_embeddings,
    text_embeddings,
    train_keys,
    val_keys,
    test_keys,
    models_output_dir,
    suffix_out_dir,
    objects_output_dir,
    run_checks=True,
):
    all_results = {}

    X_l, Y_l, Z_l, aligned_keys, indices_train, indices_val, indices_test = (
        prepare_aligned_ablation_data(
            uni_embeddings=uni_embeddings,
            target_embeddings=target_embeddings,
            text_embeddings=text_embeddings,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
            run_checks=run_checks,
        )
    )

    for l1 in lambda1_rna_text_values:
        for l2 in lambda2_img_text_values:
            print(f"l1: {l1}, l2: {l2}")

            results_lambda_val = train_and_test_model(
                X_l,
                Y_l,
                Z_l,
                indices_train,
                indices_val,
                indices_test,
                models_output_dir,
                f"ablation_studies/lambda_l1_{l1}_l2_{l2}_{suffix_out_dir}",
                lambda_rna_text=l1,
                lambda_img_text=l2,
                verbose=False,
            )

            all_results[f"lambda_l1_{l1}_l2_{l2}"] = results_lambda_val
            print("*" * 50)

    save_ablation_results(all_results, objects_output_dir, suffix_out_dir)

    return all_results

def abl_study_cell_type_model(
    uni_embeddings,
    cell_embeddings,
    gene_embeddings,
    text_embeddings,
    train_keys,
    val_keys,
    test_keys,
    models_output_dir,
    suffix_out_dir,
    objects_output_dir,
    lambda_rna_text=0.4,
    lambda_img_text=0.1,
    run_checks=True,
):
    all_results = {}

    print("Cell type only experiment")

    X_cell_type, Y_cell_type, Z_cell_type, aligned_keys, indices_train, indices_val, indices_test = (
        prepare_aligned_ablation_data(
            uni_embeddings=uni_embeddings,
            target_embeddings=cell_embeddings,
            text_embeddings=text_embeddings,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
            run_checks=run_checks,
        )
    )

    results_cell_type = train_and_test_model(
        X_cell_type,
        Y_cell_type,
        Z_cell_type,
        indices_train,
        indices_val,
        indices_test,
        models_output_dir,
        f"ablation_studies/cell_type_{suffix_out_dir}",
        lambda_rna_text=lambda_rna_text,
        lambda_img_text=lambda_img_text,
    )

    all_results["cell_type"] = results_cell_type
    
    print("Gene vectors experiment")
    X_genes, Y_genes, Z_genes, aligned_keys, indices_train, indices_val, indices_test = (
        prepare_aligned_ablation_data(
            uni_embeddings=uni_embeddings,
            target_embeddings=gene_embeddings,
            text_embeddings=text_embeddings,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
            run_checks=run_checks,
        )
    )

    results_genes = train_and_test_model(
        X_genes,
        Y_genes,
        Z_genes,
        indices_train,
        indices_val,
        indices_test,
        models_output_dir,
        f"ablation_studies/gene_vecs_{suffix_out_dir}",
        lambda_rna_text=lambda_rna_text,
        lambda_img_text=lambda_img_text,
    )

    all_results["gene_vecs"] = results_genes
    
    print("*" * 50)
    print("Fusing the cell type and gene set...")

    common_keys_fused = sorted(
        list(set(cell_embeddings.keys()) & set(gene_embeddings.keys()))
    )

    fused_vector = fuse_gene_cell_type_vectors(
        common_keys_fused,
        cell_embeddings,
        gene_embeddings,
    )

    X_fused, Y_fused, Z_fused, aligned_keys, indices_train, indices_val, indices_test = (
        prepare_aligned_ablation_data(
            uni_embeddings=uni_embeddings,
            target_embeddings=fused_vector,
            text_embeddings=text_embeddings,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
            run_checks=run_checks,
        )
    )

    results_fused = train_and_test_model(
        X_fused,
        Y_fused,
        Z_fused,
        indices_train,
        indices_val,
        indices_test,
        models_output_dir,
        f"ablation_studies/fused_{suffix_out_dir}",
        lambda_rna_text=lambda_rna_text,
        lambda_img_text=lambda_img_text,
    )

    all_results["fused"] = results_fused

    save_ablation_results(all_results, objects_output_dir, suffix_out_dir)

    return all_results

def abl_study_reports_model(
    uni_embeddings,
    fused_vector,
    master_text_embeddings,
    train_keys,
    val_keys,
    test_keys,
    models_output_dir,
    suffix_out_dir,
    objects_output_dir,
    lambda_rna_text=0.4,
    lambda_img_text=0.1,
    run_checks=True,
):
    all_results = {}
    all_models = {}

    for report_level in master_text_embeddings.keys():
        print(f"Report level: {report_level}")

        X_report, Y_report, Z_report, aligned_keys, indices_train, indices_val, indices_test = (
            prepare_aligned_ablation_data(
                uni_embeddings=uni_embeddings,
                target_embeddings=fused_vector,
                text_embeddings=master_text_embeddings[report_level],
                train_keys=train_keys,
                val_keys=val_keys,
                test_keys=test_keys,
                run_checks=run_checks,
            )
        )

        model, results_report_level = train_and_test_model(
            X_report,
            Y_report,
            Z_report,
            indices_train,
            indices_val,
            indices_test,
            models_output_dir,
            f"ablation_studies/{report_level}_{suffix_out_dir}",
            output_model=True,
            lambda_rna_text=lambda_rna_text,
            lambda_img_text=lambda_img_text,
        )

        all_results[report_level] = results_report_level
        all_models[report_level] = model

        print("*" * 50)

    save_ablation_results(all_results, objects_output_dir, suffix_out_dir)

    return all_results, all_models


def plot_abl_study_lambda_heatmaps(
    results_dict,
    figures_dir,
    out_name,
    metric="R@10",
    chosen_l1=0.4,
    chosen_l2=0.1,
    cmap="Oranges",
    show_values=True,
    show_colorbars=False,
    share_scale=False,
):
    def parse_data(results_dict):
        rows = []
        missing = []

        task_map = {
            "image_rna": "Image-RNA",
            "rna_text": "RNA-Text",
            "image_text": "Image-Text",
        }

        for key, metrics_dict in results_dict.items():
            match = re.match(r"lambda_l1_([0-9.]+)_l2_([0-9.]+)", key)
            if match is None:
                missing.append((key, "could not parse lambda values"))
                continue

            l1 = float(match.group(1))
            l2 = float(match.group(2))

            row = {"l1": l1, "l2": l2}

            for task_key, task_name in task_map.items():
                task_metrics = metrics_dict.get(task_key, {})
                row[task_name] = task_metrics.get(metric, np.nan)

                if task_key not in metrics_dict or metric not in task_metrics:
                    missing.append((key, f"{task_key}.{metric}"))

            rows.append(row)

        df = pd.DataFrame(rows)

        return df

    def make_pivot(df, task_name):
        return (
            df.pivot(index="l2", columns="l1", values=task_name)
              .sort_index(ascending=False)
              .sort_index(axis=1)
        )

    def add_chosen_marker(ax, pivot, chosen_l1, chosen_l2):
        cols = list(pivot.columns)
        rows = list(pivot.index)

        if chosen_l1 not in cols or chosen_l2 not in rows:
            return

        x = cols.index(chosen_l1)
        y = rows.index(chosen_l2)

        ax.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.8,
                zorder=6,
                clip_on=False,
            )
        )

    df = parse_data(results_dict)

    task_order = [
        "Image-RNA",
        "RNA-Text",
        "Image-Text",
    ]

    pivots = {task: make_pivot(df, task) for task in task_order}

    if share_scale:
        all_values = pd.concat([df[task] for task in task_order]).dropna()
        global_vmin = all_values.min()
        global_vmax = all_values.max()
        scale_by_task = {
            task: (global_vmin, global_vmax)
            for task in task_order
        }
    else:
        scale_by_task = {
            task: (df[task].dropna().min(), df[task].dropna().max())
            for task in task_order
        }

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.8, 2.9),
        sharey=True,
    )

    for i, (ax, task) in enumerate(zip(axes, task_order)):
        pivot = pivots[task]
        vmin, vmax = scale_by_task[task]

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=show_values,
            fmt=".2f",
            annot_kws={"size": 7.5},
            cbar=show_colorbars,
            cbar_kws={"label": metric, "shrink": 0.78},
            linewidths=0.75,
            linecolor="#eeeeee",
            square=True,
            mask=pivot.isna(),
        )
        add_chosen_marker(ax, pivot, chosen_l1, chosen_l2)

        ax.set_title(task, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=0, length=0)
        ax.tick_params(axis="y", rotation=0, length=0)

        if i > 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.supxlabel(r"$\lambda_1$ (weight for RNA-Text)", y=0.03, fontsize=13)
    fig.supylabel(r"$\lambda_2$ (weight for Image-Text)", x=0.02, fontsize=13)

    # increased spacing between panels
    fig.subplots_adjust(wspace=0.22, bottom=0.18, left=0.08, right=0.98, top=0.86)

    os.makedirs(figures_dir, exist_ok=True)

    plt.savefig(f"{figures_dir}/{out_name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{figures_dir}/{out_name}.svg", format="svg", bbox_inches="tight")
    plt.show()

    return df, pivots


##### RNA representation delta tradeoff ####
def _make_rna_representation_df(abl_transcriptomic_representation_lung):
    rep_order = ["cell_type", "gene_vecs", "fused"]

    rep_labels = {
        "cell_type": "Cell types",
        "gene_vecs": "Genes",
        "fused": "Fusion",
    }

    rows = []

    for rep in rep_order:
        d = abl_transcriptomic_representation_lung[rep]

        image_rna_r10 = np.mean([
            d["image_rna"]["R@10"],
            d["rna_image"]["R@10"],
        ])

        image_rna_medr = np.median([
            d["image_rna"]["median_rank"],
            d["rna_image"]["median_rank"],
        ])

        rows.append({
            "rep": rep,
            "label": rep_labels[rep],
            "Image--RNA R@10": image_rna_r10,
            "Image--RNA MedR": image_rna_medr,
            "Image--Text R@10": d["image_text"]["R@10"],
            "Image--Text MedR": d["image_text"]["median_rank"],
            "RNA--Text R@10": d["rna_text"]["R@10"],
            "RNA--Text MedR": d["rna_text"]["median_rank"],
        })

    df = pd.DataFrame(rows)

    reference = df[df["rep"] == "gene_vecs"].iloc[0]

    for metric in ["Image--RNA R@10", "Image--Text R@10", "RNA--Text R@10"]:
        df[f"Delta {metric}"] = (df[metric] - reference[metric]) * 100

    return df

def plot_rna_representation_delta_tradeoff_gene_reference(
    abl_transcriptomic_representation_lung,
    figures_dir,
    out_name="rna_representation_delta_tradeoff_gene_reference",
):
    os.makedirs(figures_dir, exist_ok=True)

    df = _make_rna_representation_df(abl_transcriptomic_representation_lung)

    colors = {
        "cell_type": "#BDBDBD",
        "gene_vecs": "#F4A261",
        "fused": "#D55E00",
    }

    sizes = {
        "cell_type": 190,
        "gene_vecs": 190,
        "fused": 260,
    }

    # Per-label placement
    label_specs = {
        "cell_type": {
            "dx": 0.7, "dy": 0.40,
            "ha": "left", "va": "center"
        },
        "gene_vecs": {
            "dx": 0.65, "dy": 1,
            "ha": "left", "va": "center"
        },
        "fused": {
            "dx": 0.7, "dy": 0.35,
            "ha": "left", "va": "center"
        },
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    for _, row in df.iterrows():
        rep = row["rep"]

        x = row["Delta Image--RNA R@10"]
        y = row["Delta RNA--Text R@10"]

        ax.scatter(
            x,
            y,
            s=sizes[rep],
            color=colors[rep],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

        if rep == "cell_type":
            label = "Cell types"
        elif rep == "gene_vecs":
            label = "Genes\n(reference)"
        elif rep == "fused":
            label = "Fusion\n(selected)"
        else:
            label = row["label"]

        spec = label_specs[rep]

        ax.text(
            x + spec["dx"],
            y + spec["dy"],
            label,
            ha=spec["ha"],
            va=spec["va"],
            fontsize=12,
            fontweight="bold" if rep == "fused" else "normal",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )

    # Reference lines
    ax.axhline(0, color="0.55", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(0, color="0.55", linewidth=0.9, linestyle="--", zorder=1)

    ax.set_xlim(-18.5, 7.0)
    ax.set_ylim(-2.2, 14.0)

    ax.set_xlabel("Δ Image–RNA R@10")
    ax.set_ylabel("Δ RNA–Text R@10")

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    plt.tight_layout()

    fig.savefig(f"{figures_dir}/{out_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{figures_dir}/{out_name}.svg", bbox_inches="tight")
    plt.show()

    return df


#### Effect of including different report levels ####
def plot_text_level_ablation(
    abl_study_results,
    figures_dir,
    out_name,
    metric="median_rank",
    annotate=True,
):
    os.makedirs(figures_dir, exist_ok=True)

    task_map = {
        "Image-RNA": "image_rna",
        "Image-Text": "image_text",
        "RNA-Text": "rna_text",
    }

    display_levels = ["1", "2", "3", "1_2", "1_3", "2_3", "1_2_3"]

    level_name_map = {
        "1": "L1",
        "2": "L2",
        "3": "L3",
        "1_2": "L1+L2",
        "1_3": "L1+L3",
        "2_3": "L2+L3",
        "1_2_3": "Full",
    }
    
    delta_x_map = {
        "Image-RNA": 0.05,
        "Image-Text": 0.6,
        "RNA-Text": 0.25,
    }

    rows = []

    for level in display_levels:
        level_key = f"level_{level}"

        if level_key not in abl_study_results:
            continue

        metrics_dict = abl_study_results[level_key]

        for task_label, task_key in task_map.items():
            if task_key not in metrics_dict:
                continue
            if metric not in metrics_dict[task_key]:
                continue

            rows.append({
                "level": level,
                "level_label": level_name_map[level],
                "task": task_label,
                "value": metrics_dict[task_key][metric],
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No matching levels/tasks found in abl_study_results.")

    level_order = [
        level for level in display_levels
        if level in df["level"].unique()
    ]
    level_labels = [level_name_map[level] for level in level_order]

    task_order = ["Image-RNA", "Image-Text", "RNA-Text"]

    colors = {
        "Image-RNA": "#D55E00",
        "Image-Text": "#E69F00",
        "RNA-Text": "#0072B2",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10, 3),
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )

    y_positions = np.arange(len(level_order))

    for ax, task in zip(axes, task_order):
        print(f"T: {task}")
        sub = df[df["task"] == task].copy()

        sub["y"] = sub["level"].apply(lambda x: level_order.index(x))

        ax.scatter(
            sub["value"] - delta_x_map[task],
            sub["y"],
            s=58,
            color=colors[task],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        if annotate:
            for _, row in sub.iterrows():
                ax.text(
                    row["value"],
                    row["y"],
                    f" {row['value']:.0f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color=colors[task],
                )

        ax.set_title(task, fontsize=14, pad=6)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(level_labels)

        ax.invert_yaxis()
        ax.grid(axis="x", color="0.95", linewidth=0.8)
        ax.set_axisbelow(True)

        # Task-specific x-limits with padding
        xmax = sub["value"].max()
        ax.set_xlim(left=0, right=max(4, xmax * 1.25))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Report level")
    axes[1].set_xlabel("Median rank (lower is better)")

    plt.tight_layout()

    fig.savefig(f"{figures_dir}/{out_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{figures_dir}/{out_name}.svg", format="svg", bbox_inches="tight")

    plt.show()

    return df

##### Effect of interactions ####
def _make_kde_df(
    before_target,
    before_control,
    after_target,
    after_control,
    before_model,
    after_model,
    target_label,
    control_label,
):
    rows = []

    for scores, cohort, model, stage in [
        (before_target, target_label, before_model, "before"),
        (before_control, control_label, before_model, "before"),
        (after_target, target_label, after_model, "after"),
        (after_control, control_label, after_model, "after"),
    ]:
        for score in np.asarray(scores, dtype=float):
            rows.append({
                "Score": score,
                "Cohort": cohort,
                "Model": model,
                "Stage": stage,
            })

    return pd.DataFrame(rows)

def has_target_interaction(interactions, cell_a, cell_b):
    for interaction in interactions:
        pair = interaction[0]

        if len(pair) != 2:
            continue

        x, y = pair

        if {x, y} == {cell_a, cell_b}:
            return True

    return False

def canonical_pair(a, b):
    return tuple(sorted([a, b]))

def build_interaction_stats_table(
    interactions_counts_sorted,
    patches_description,
    density_lookup,
    min_a_count=3,
    min_b_count=3,
    density_threshold=0.90,
):
    stats_rows = []
    all_patch_rows = []

    for interaction_key, n_top2 in interactions_counts_sorted.items():
        cell_a, cell_b = interaction_key.split("____")

        patch_rows = []

        for patch_key, desc in patches_description.items():
            comp = desc["cell_type_composition"]

            a_count = comp.get(cell_a, 0)
            b_count = comp.get(cell_b, 0)
            density = density_lookup[patch_key]["percentile"]

            has_relation = has_target_interaction(
                desc.get("interaction_summary_cell_type", []),
                cell_a,
                cell_b,
            )

            candidate = (a_count >= min_a_count) and (b_count >= min_b_count)
            has_relation_and_candidate = candidate and has_relation
            dense = density >= density_threshold
            positive = candidate and dense and has_relation

            row = {
                "interaction": interaction_key,
                "cell_a": cell_a,
                "cell_b": cell_b,
                "patch": patch_key,
                "a_count": a_count,
                "b_count": b_count,
                "density": density,
                "candidate": candidate,
                "dense": dense,
                "has_relation": has_relation,
                "has_relation_and_candidate": has_relation_and_candidate,
                "positive": positive,
            }

            patch_rows.append(row)
            all_patch_rows.append(row)

        df_tmp = pd.DataFrame(patch_rows)

        stats_rows.append({
            "interaction": interaction_key,
            "cell_a": cell_a,
            "cell_b": cell_b,
            "n_top2_mentions": n_top2,
            "n_candidates": int(df_tmp["candidate"].sum()),
            "n_dense_candidates": int((df_tmp["candidate"] & df_tmp["dense"]).sum()),
            "n_relation": int(df_tmp["has_relation"].sum()),
            "n_relation_and_candidate": int(df_tmp["has_relation_and_candidate"].sum()),
            "n_full_positive": int(df_tmp["positive"].sum()),
            "mean_density_candidates": df_tmp.loc[df_tmp["candidate"], "density"].mean(),
            "mean_density_relation": df_tmp.loc[df_tmp["has_relation"], "density"].mean(),
        })

    df_patches = pd.DataFrame(all_patch_rows)
    df_stats = pd.DataFrame(stats_rows)

    df_stats["full_positive_rate"] = (
        df_stats["n_full_positive"]
        / df_stats["n_dense_candidates"].replace(0, np.nan)
    )

    df_stats = df_stats.sort_values(
        ["n_full_positive", "n_top2_mentions"],
        ascending=False,
    ).reset_index(drop=True)

    df_patches = df_patches.sort_values(
        ["interaction", "positive", "candidate", "dense", "density"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    return df_patches, df_stats

def build_test_patch_descriptions_and_interaction_stats(
    all_samples_metadata,
    test_keys,
    get_interaction_summary_func,
    get_density_description_func,
    build_interaction_stats_table_func,
    canonical_pair_func,
    patch_col="patch_name",
    cell_type_col="cell_type",
    x_col="x_global",
    y_col="y_global",
    k_interactions=2,
    radius=50,
    min_a_count=5,
    min_b_count=5,
    density_threshold=0.85,
    verbose=True,
):
    metadata_test = all_samples_metadata[
        all_samples_metadata[patch_col].isin(test_keys)
    ].copy()

    patch_cell_counts = metadata_test.groupby(patch_col).size()
    patch_density_percentiles = patch_cell_counts.rank(pct=True, method="min")

    density_lookup = pd.DataFrame({
        "count": patch_cell_counts,
        "percentile": patch_density_percentiles.round(3),
    }).to_dict(orient="index")

    patches_description = {}

    grouped_metadata = metadata_test.groupby(patch_col)

    for patch_name, df_cells_patch in tqdm(
        grouped_metadata,
        total=len(grouped_metadata),
        disable=not verbose,
    ):
        interaction_summary_cell_type = get_interaction_summary_func(
            df_cells_patch,
            k_interactions=k_interactions,
            x_col=x_col,
            y_col=y_col,
            type_col=cell_type_col,
            radius=radius,
            get_summary_in_text=False,
        )

        density_info = get_density_description_func(density_lookup, patch_name)

        patches_description[patch_name] = {
            "cell_type_composition": df_cells_patch[cell_type_col].value_counts(),
            "interaction_summary_cell_type": interaction_summary_cell_type,
            "density_info": density_info,
        }

    missing = set(test_keys) - set(patches_description.keys())

    if verbose:
        print("Missing test patches:", len(missing))

    interactions_counts = {}

    for patch in test_keys:
        if patch not in patches_description:
            continue

        interactions = patches_description[patch]["interaction_summary_cell_type"]

        for interaction in interactions:
            a, b = interaction[0]

            if a == b:
                continue

            a_canon, b_canon = canonical_pair_func(a, b)
            interaction_key = f"{a_canon}____{b_canon}"

            interactions_counts[interaction_key] = (
                interactions_counts.get(interaction_key, 0) + 1
            )

    interactions_counts_sorted = dict(
        sorted(
            interactions_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    df_patch_stats, df_interaction_stats = build_interaction_stats_table_func(
        interactions_counts_sorted,
        patches_description,
        density_lookup,
        min_a_count=min_a_count,
        min_b_count=min_b_count,
        density_threshold=density_threshold,
    )

    return {
        "metadata_test": metadata_test,
        "density_lookup": density_lookup,
        "patches_description": patches_description,
        "interactions_counts": interactions_counts,
        "interactions_counts_sorted": interactions_counts_sorted,
        "df_patch_stats": df_patch_stats,
        "df_interaction_stats": df_interaction_stats,
        "missing_patches": missing,
    }
    
def _extract_valid_interaction_pairs(desc, canonical_pair_func):
    interactions = desc.get("interaction_summary_cell_type", [])

    valid_pairs = []
    all_pairs = set()

    for interaction in interactions:
        if len(interaction[0]) != 2:
            continue

        a, b = interaction[0]

        if a == b:
            continue

        a, b = canonical_pair_func(a, b)
        pair_str = f"{a}____{b}"

        count = interaction[1] if len(interaction) > 1 else 1

        valid_pairs.append((pair_str, count))
        all_pairs.add(pair_str)

    return valid_pairs, all_pairs


def build_top_interaction_cohorts(
    patches_description,
    canonical_pair_func,
    min_cell_count=3,
    min_signature_count=5,
    signature=None,
    signature_index=0,
    verbose=True,
):
    """
    Build positive and hard-negative cohorts for the top interaction.
    """
    patch_signatures = {}
    signature_to_patches = defaultdict(list)

    patch_all_pairs = {}
    patch_top_pair = {}

    for patch_key, desc in patches_description.items():
        valid_pairs, all_pairs = _extract_valid_interaction_pairs(
            desc,
            canonical_pair_func=canonical_pair_func,
        )

        patch_all_pairs[patch_key] = all_pairs

        if len(valid_pairs) == 0:
            patch_top_pair[patch_key] = None
            continue

        valid_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
        top_pair = valid_pairs[0][0]

        patch_top_pair[patch_key] = top_pair

        # tuple of length 1, same as your original code
        patch_signature = (top_pair,)

        patch_signatures[patch_key] = patch_signature
        signature_to_patches[patch_signature].append(patch_key)

    sig_counts = Counter(patch_signatures.values())

    top_signatures = [
        sig
        for sig, count in sig_counts.most_common()
        if count >= min_signature_count
    ]

    top_signature_patches = {
        sig: signature_to_patches[sig]
        for sig in top_signatures
    }

    if signature is None:
        if len(top_signatures) == 0:
            raise ValueError(
                f"No top-1 interaction signatures found with >= {min_signature_count} patches."
            )

        if signature_index >= len(top_signatures):
            raise IndexError(
                f"signature_index={signature_index} but only {len(top_signatures)} "
                "top signatures are available."
            )

        signature = top_signatures[signature_index]

    # keep compatibility if user passes "AT2____Capillary" instead of ("AT2____Capillary",)
    if isinstance(signature, str):
        signature = (signature,)

    if len(signature) != 1:
        raise ValueError(
            "This function is for top-1 interactions only. "
            "Expected signature like ('AT2____Capillary',)."
        )

    target_pair = signature[0]
    cell_a, cell_b = target_pair.split("____")
    required_cells = {cell_a, cell_b}

    cohort_a_patches = []
    cohort_b_patches = []

    for patch_key, desc in patches_description.items():
        comp = desc["cell_type_composition"]

        has_required_cells = (
            comp.get(cell_a, 0) >= min_cell_count
            and comp.get(cell_b, 0) >= min_cell_count
        )

        if not has_required_cells:
            continue

        is_top_interaction = patch_top_pair.get(patch_key) == target_pair

        if is_top_interaction:
            cohort_a_patches.append(patch_key)
        else:
            cohort_b_patches.append(patch_key)

    if verbose:
        print(f"Found {len(top_signatures)} top-1 interaction niches with >= {min_signature_count} patches.")
        print("=" * 80)
        print("Target top interaction")
        print("=" * 80)
        print(f"{cell_a} <-> {cell_b}")

        print("\nCohort summary")
        print("-" * 80)
        print(f"Minimum cell count per required cell: {min_cell_count}")
        print(f"True positives, Cohort A: {len(cohort_a_patches)}")
        print(f"Hard negatives, Cohort B: {len(cohort_b_patches)}")

    return {
        "cohort_a_patches": cohort_a_patches,
        "cohort_b_patches": cohort_b_patches,
        "target_pair": target_pair,
        "required_cells": required_cells,
        "signature": signature,
        "patch_signatures": patch_signatures,
        "signature_to_patches": signature_to_patches,
        "sig_counts": sig_counts,
        "top_signatures": top_signatures,
        "top_signature_patches": top_signature_patches,
        "patch_all_pairs": patch_all_pairs,
        "patch_top_pair": patch_top_pair,
    }
    
    
def plot_internalization_density_panel_clean(
    scores_l1_a,
    scores_l1_b,
    scores_l12_a,
    scores_l12_b,
    scores_l12_dense,
    scores_l12_nondense,
    scores_l123_dense,
    scores_l123_nondense,
    figures_dir,
    out_name,
    query_interaction,
    query_density,
    modality="Text-Img",
    spatial_target_label="Interacting",
    spatial_control_label="Non-interacting",
    density_target_label="High density",
    density_control_label="Non-dense",
    target_color="#ff8834",
    control_color="#9CA3AF",
    figsize=(8.2, 6.0),
    xlim=None,
    bw_adjust=1.0,
    query_wrap_width=100,
):
    os.makedirs(figures_dir, exist_ok=True)

    panel_specs = [
        {
            "title": "Spatial relations",
            "query": query_interaction,
            "target_label": spatial_target_label,
            "control_label": spatial_control_label,
            "before_model": "L1",
            "after_model": "L1+L2",
            "before_target": np.asarray(scores_l1_a, dtype=float),
            "before_control": np.asarray(scores_l1_b, dtype=float),
            "after_target": np.asarray(scores_l12_a, dtype=float),
            "after_control": np.asarray(scores_l12_b, dtype=float),
        },
        {
            "title": "Cellular density",
            "query": query_density,
            "target_label": density_target_label,
            "control_label": density_control_label,
            "before_model": "L1+L2",
            "after_model": "L1+L2+L3",
            "before_target": np.asarray(scores_l12_dense, dtype=float),
            "before_control": np.asarray(scores_l12_nondense, dtype=float),
            "after_target": np.asarray(scores_l123_dense, dtype=float),
            "after_control": np.asarray(scores_l123_nondense, dtype=float),
        },
    ]

    if xlim is None:
        all_scores = np.concatenate([
            spec["before_target"] for spec in panel_specs
        ] + [
            spec["before_control"] for spec in panel_specs
        ] + [
            spec["after_target"] for spec in panel_specs
        ] + [
            spec["after_control"] for spec in panel_specs
        ])

        lo = np.nanpercentile(all_scores, 0.5)
        hi = np.nanpercentile(all_scores, 99.5)
        pad = 0.12 * max(hi - lo, 0.1)
        xlim = (lo - pad, hi + pad)

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 15,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 10.5,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        sharey=False,
    )

    for ax, spec in zip(axes, panel_specs):
        df = _make_kde_df(
            before_target=spec["before_target"],
            before_control=spec["before_control"],
            after_target=spec["after_target"],
            after_control=spec["after_control"],
            before_model=spec["before_model"],
            after_model=spec["after_model"],
            target_label=spec["target_label"],
            control_label=spec["control_label"],
        )

        curve_specs = [
            {
                "cohort": spec["target_label"],
                "model": spec["before_model"],
                "color": target_color,
                "linestyle": "--",
                "linewidth": 2.3,
                "fill": False,
                "alpha": 0.00,
            },
            {
                "cohort": spec["control_label"],
                "model": spec["before_model"],
                "color": control_color,
                "linestyle": "--",
                "linewidth": 2.3,
                "fill": False,
                "alpha": 0.00,
            },
            {
                "cohort": spec["target_label"],
                "model": spec["after_model"],
                "color": target_color,
                "linestyle": "-",
                "linewidth": 2.8,
                "fill": True,
                "alpha": 0.24,
            },
            {
                "cohort": spec["control_label"],
                "model": spec["after_model"],
                "color": control_color,
                "linestyle": "-",
                "linewidth": 2.8,
                "fill": True,
                "alpha": 0.20,
            },
        ]

        for curve in curve_specs:
            subset = df[
                (df["Cohort"] == curve["cohort"])
                & (df["Model"] == curve["model"])
            ]

            if subset.shape[0] < 2 or subset["Score"].std() == 0:
                continue

            if curve["fill"]:
                sns.kdeplot(
                    data=subset,
                    x="Score",
                    ax=ax,
                    fill=True,
                    color=curve["color"],
                    alpha=curve["alpha"],
                    linewidth=0,
                    common_norm=False,
                    bw_adjust=bw_adjust,
                )

            sns.kdeplot(
                data=subset,
                x="Score",
                ax=ax,
                fill=False,
                color=curve["color"],
                linestyle=curve["linestyle"],
                linewidth=curve["linewidth"],
                common_norm=False,
                bw_adjust=bw_adjust,
            )

        ax.set_xlim(*xlim)
        ax.set_ylabel("Density")

        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.30)
        ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.12)
        ax.set_axisbelow(True)

        sns.despine(ax=ax)

        wrapped_query = "\n".join(
            textwrap.wrap(f"Query ({modality}): {spec['query']}", width=query_wrap_width)
        )

        ax.set_title(
            spec["title"],
            loc="left",
            fontweight="bold",
            pad=28,
        )

        ax.text(
            0.0,
            1.04,
            wrapped_query,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.2,
            fontweight="normal",
            color="#4B5563",
            style="italic",
        )

    axes[-1].set_xlabel("Retrieval similarity score")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=target_color,
            lw=3,
            linestyle="-",
            label="Query-matching cohort",
        ),
        Line2D(
            [0],
            [0],
            color=control_color,
            lw=3,
            linestyle="-",
            label="Control cohort",
        ),
        Line2D(
            [0],
            [0],
            color="#374151",
            lw=2.4,
            linestyle="--",
            label="Before adding level",
        ),
        Line2D(
            [0],
            [0],
            color="#374151",
            lw=2.4,
            linestyle="-",
            label="After adding level",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=True,
        edgecolor="#D1D5DB",
        fontsize=10.5,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.12,
        top=0.84,
        hspace=0.6,
    )

    fig.savefig(f"{figures_dir}/{out_name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{figures_dir}/{out_name}.svg", format="svg", bbox_inches="tight")

    plt.show()

    return fig, axes

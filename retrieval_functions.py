import numpy as np
import torch
import textwrap
import matplotlib.pyplot as plt
import os
from PIL import Image
import re
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict, Counter
import matplotlib.patches as mpatches
import glob
import geopandas as gpd

from shapely.affinity import scale


def embed_text_to_latent(client, query, model):
    resp = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    q_raw = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    with torch.no_grad():
        q_latent = model.embed_batch(q_raw, branch="text")
        
    return q_latent

def retrieve_latent_neighbors(q_latent, target_db, keys, k=5):
    sims = np.dot(q_latent, target_db.T)[0]
    top_idx = np.argsort(sims)[::-1][:k]

    return [
        {
            "patch": keys[i],
            "score": float(sims[i])
        }
        for i in top_idx
    ]

def search_cross_modal(query_idx, query_emb_matrix, target_emb_matrix, keys, k=5):
    query_vec = query_emb_matrix[query_idx].reshape(1, -1)
    
    # Compute cosine similarity
    sims = np.dot(query_vec, target_emb_matrix.T)[0]
    
    # I deliberately do NOT exclude the query itself - seeing the query retrieved in top retrieved is a good sanity check.
    top_indices = np.argsort(sims)[::-1][:k]
    top_scores = sims[top_indices]
    
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            "original_index": idx,
            "key": keys[idx],
            "score": top_scores[i]
        })
        
    return results

def visualize_search_by_example(
    query_idx, 
    query_modality,
    target_modality,
    query_emb_matrix, 
    target_emb_matrix, 
    keys, 
    meta_df, 
    images_root, 
    sample_id_col="tissue",
    k=5, 
    microns_size=112,
    save_path=None
):
    results = search_cross_modal(query_idx, query_emb_matrix, target_emb_matrix, keys, k=k)

    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.5))
    
    def load_patch_img(key, meta_row):
        fname = key.split("__")[-1] + ".png"
        sample = meta_row[sample_id_col]
        
        full_path = os.path.join(
            images_root, 
            f"he_size_{microns_size}_microns_patches_{sample}", 
            fname
        )
        
        if os.path.exists(full_path):
            return Image.open(full_path)
        else:
            return Image.new("RGB", (112, 112), color="gray")

    query_key = keys[query_idx]
    query_meta = meta_df.iloc[query_idx]
    
    ax_q = axes[0]
    img_q = load_patch_img(query_key, query_meta)
    
    ax_q.imshow(img_q)
    ax_q.set_title(
        f"QUERY ({query_modality.upper()})", 
        fontsize=15, 
        fontweight="bold", 
        color="black"
    )
    ax_q.axis("off")
    
    for i, res in enumerate(results):
        ax = axes[i + 1]
        
        match_idx = res["original_index"]
        match_key = res["key"]
        score = res["score"]
        match_meta = meta_df.iloc[match_idx]
        
        img_m = load_patch_img(match_key, match_meta)
        ax.imshow(img_m)
        
        ax.set_title(
            f"Rank {i+1} ({target_modality.upper()})\n"
            f"Score: {score:.3f}", 
            fontsize=15
        )
        
        # Highlight self-retrieval (Green Box Logic)
        if match_key == query_key:
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_edgecolor("lightgreen")
                spine.set_linewidth(5)
        else:
            ax.axis("off")

        print(f"match_key - {match_key}, query_key - {query_key}")
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_path}.svg", format="svg", bbox_inches="tight")
        
    plt.show()


def compute_spatial_similarity(client, query_text, model, target_emb_matrix, branch="text"):
    """
    Computes similarity between a text query and ALL patches in the target matrix.
    """
    # Embed Text
    resp = client.embeddings.create(input=query_text, model="text-embedding-3-small")
    q_raw = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    # Project Text
    with torch.no_grad():
        q_latent = model.embed_batch(q_raw, branch=branch) # Shape (1, 128)
        
    # Find similar ones using cosine similarity
    scores = np.dot(q_latent, target_emb_matrix.T)[0] # Shape (N_patches,)
    
    return scores

def plot_spatial_heatmap(
    client,
    query_text, 
    model, 
    target_emb_matrix, 
    meta_df, 
    sample_id, 
    target_modality="img",
    cmap="Oranges",      
    point_size=40,
    threshold=None,      
    background_color="#d3d3d3", 
    save_path=None,
    geojson_dir=None,
    annotation_scale=0.2125,
    flip_annotation_y=True,
):
    mask = meta_df["tissue"] == sample_id
    sample_df = meta_df[mask].copy()
    sample_embs = target_emb_matrix[mask]
    
    if len(sample_df) == 0:
        print(f"Error: No patches found for sample {sample_id}")
        return

    print(f"Computing similarity for: '{query_text}'...")
    
    scores = compute_spatial_similarity(
        client,
        query_text,
        model,
        sample_embs,
    )
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    xs = sample_df["x"]
    ys = -sample_df["y"] 

    # Similarity map
    if threshold is not None:
        high_mask = scores >= threshold
        low_mask = scores < threshold
        
        if np.sum(low_mask) > 0:
            ax.scatter(
                xs[low_mask], 
                ys[low_mask],
                c=background_color, 
                s=point_size, 
                alpha=0.5,          
                edgecolors="none",
            )
        
        if np.sum(high_mask) > 0:
            sc = ax.scatter(
                xs[high_mask], 
                ys[high_mask],
                c=scores[high_mask], 
                cmap=cmap, 
                s=point_size, 
                alpha=1.0, 
                edgecolors="none",
                vmin=threshold,     
                vmax=np.max(scores),
            )
        else:
            print(f"Warning: No points exceeded threshold {threshold}")
            plt.close(fig)
            return
            
    else:
        sc = ax.scatter(
            xs,
            ys,
            c=scores, 
            cmap=cmap, 
            s=point_size, 
            alpha=0.9,
            edgecolors="none",
        )

    # Overlay author annotations
    if geojson_dir is not None:
        geojson_files = sorted(
            glob.glob(
                os.path.join(
                    geojson_dir,
                    "*.geojson",
                )
            )
        )
        
        for geojson_file in geojson_files:
            gdf = gpd.read_file(geojson_file)
            
            if gdf.empty:
                continue

            filename = os.path.basename(geojson_file).lower()

            if "mixed_inflammation" in filename:
                color = "#333399"
            elif "severe_fibrosis" in filename:
                color = "#335C33"
            else:
                continue

            y_scale = (
                -annotation_scale
                if flip_annotation_y
                else annotation_scale
            )

            gdf["geometry"] = gdf["geometry"].apply(
                lambda geom: scale(
                    geom,
                    xfact=annotation_scale,
                    yfact=y_scale,
                    origin=(0, 0),
                )
            )

            gdf.boundary.plot(
                ax=ax,
                color=color,
                linewidth=2.5,
                zorder=10,
                aspect="equal",
            )
            
    cbar = plt.colorbar(
        sc,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    if threshold is not None:
        cbar.set_label(
            f"Cosine similarity (>{threshold})",
            rotation=270,
            labelpad=24,
            fontsize=22,
        )
    else:
        cbar.set_label(
            f"Cosine similarity ({target_modality.upper()})",
            rotation=270,
            labelpad=24,
            fontsize=22,
        )

    cbar.ax.tick_params(
        labelsize=22,
    )

    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()
    
    if save_path:
        fig.savefig(
            f"{save_path}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        
        fig.savefig(
            f"{save_path}.svg",
            format="svg",
            bbox_inches="tight",
        )
        
    plt.show()

    return fig, ax

def search_by_text(client, model, query_text, target_emb_matrix, keys, k=5):
    q_latent = embed_text_to_latent(client, query_text, model) 

    sims = np.dot(q_latent, target_emb_matrix.T)[0]

    top_indices = np.argsort(sims)[::-1][:k]
    top_scores = sims[top_indices]
    
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            "original_index": idx,
            "key": keys[idx],
            "score": top_scores[i]
        })
        
    return results


def visualize_text_search(
    client, 
    model, 
    query_text, 
    target_emb_matrix, 
    keys, 
    meta_df, 
    images_root, 
    ground_truth_level,
    sample_id_col="tissue",
    target_modality="img",
    k=5, 
    microns_size=112,
    save_path=None
):
    results = search_by_text(client, model, query_text, target_emb_matrix, keys, k=k)

    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4.5))

    def load_patch_img(key, meta_row):
        fname = key.split("__")[-1] + ".png"
        sample = meta_row[sample_id_col]
        full_path = os.path.join(
            images_root,
            f"he_size_{microns_size}_microns_patches_{sample}",
            fname
        )
        if os.path.exists(full_path):
            return Image.open(full_path)
        else:
            return Image.new("RGB", (112, 112), color="gray")

    # ---- QUERY PANEL ----
    ax_q = axes[0]
    ax_q.axis("off")

    wrapped_query = "\n".join(textwrap.wrap(query_text, width=22))
    ax_q.text(
        0.5, 0.6,
        f"QUERY:\n\n{wrapped_query}",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold"
    )

    ground_truth_dict = {}
    
    # Retrieval panels
    for i, res in enumerate(results):
        ax = axes[i + 1]
        
        match_idx = res["original_index"]
        match_key = res["key"]
        score = res["score"]
        match_meta = meta_df.iloc[match_idx]

        img_m = load_patch_img(match_key, match_meta)
        ax.imshow(img_m)
        ax.axis("off")

        # Retrieve ground truth entry
        gt_entry = (
            ground_truth_level.get(match_key)
            or ground_truth_level.get(match_key.split("__")[-1])
        )

        if gt_entry:
            gt_text = gt_entry.get("text_descr") or gt_entry.get("patch_summary", "")
            ground_truth_dict[match_key] = gt_text

        ax.set_title(
            f"Rank {i+1} ({target_modality.upper()})\n"
            f"Score: {score:.3f}\n\n",
            fontsize=30
        )

    plt.tight_layout()

    if save_path:
        safe_name = "".join([c if c.isalnum() else "_" for c in query_text[:20]])
        plt.savefig(f"{save_path}/{safe_name}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_path}/{safe_name}.svg", format="svg", bbox_inches="tight")

    plt.show()

    return ground_truth_dict

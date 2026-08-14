import numpy as np
from sklearn.decomposition import PCA
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import re
import tifffile
import umap
import scanpy as sc

from sklearn.neighbors import radius_neighbors_graph

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def load_samples(input_dir):
    return [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
    ]

def match_and_align_trimodal(
    uni_embeddings,
    gene_embeddings,
    text_embeddings,
    key_order=None,
    return_keys=True,
):
    """
    Align image, RNA/cell/fused, and text embeddings.
    """

    if key_order is None:
        key_order = list(uni_embeddings.keys())

    keys = [
        k for k in key_order
        if k in uni_embeddings and k in gene_embeddings and k in text_embeddings
    ]

    if len(keys) == 0:
        print("WARNING: No common keys found across Image, RNA, and Text!")
        if return_keys:
            return None, None, None, []
        return None, None, None

    print(f"Found {len(keys)} common keys across all 3 modalities.")
    print("First aligned keys:", keys[:5])

    X = np.stack(
        [np.asarray(uni_embeddings[k]).reshape(-1) for k in keys],
        axis=0,
    )

    Y = np.stack(
        [np.asarray(gene_embeddings[k]).reshape(-1) for k in keys],
        axis=0,
    )

    Z = np.stack(
        [np.asarray(text_embeddings[k]).reshape(-1) for k in keys],
        axis=0,
    )

    print("X shape (Image):", X.shape)
    print("Y shape (RNA):  ", Y.shape)
    print("Z shape (Text): ", Z.shape)

    if return_keys:
        return X, Y, Z, keys

    return X, Y, Z

def split_keys_to_indices(aligned_keys, train_keys, val_keys, test_keys):
    train_keys = set(train_keys)
    val_keys = set(val_keys)
    test_keys = set(test_keys)

    assert len(train_keys & val_keys) == 0, "Train/val key overlap"
    assert len(train_keys & test_keys) == 0, "Train/test key overlap"
    assert len(val_keys & test_keys) == 0, "Val/test key overlap"

    indices_train = np.array(
        [i for i, k in enumerate(aligned_keys) if k in train_keys],
        dtype=int,
    )

    indices_val = np.array(
        [i for i, k in enumerate(aligned_keys) if k in val_keys],
        dtype=int,
    )

    indices_test = np.array(
        [i for i, k in enumerate(aligned_keys) if k in test_keys],
        dtype=int,
    )

    n_total = len(indices_train) + len(indices_val) + len(indices_test)

    assert n_total == len(aligned_keys), (
        f"Only {n_total}/{len(aligned_keys)} aligned keys assigned to a split. "
        "This means some aligned keys are missing from train/val/test split keys."
    )

    return indices_train, indices_val, indices_test

def keys_to_indices(aligned_keys, key_set):
    key_set = set(key_set)

    indices = np.array(
        [i for i, k in enumerate(aligned_keys) if k in key_set],
        dtype=int,
    )

    missing_keys = key_set - set(aligned_keys)

    if len(missing_keys) > 0:
        raise ValueError(
            f"{len(missing_keys)} keys were not found in aligned_keys. "
            f"Examples: {list(missing_keys)[:5]}"
        )

    return indices

def compute_metrics_numpy(Hq, Hdb):
    sims = Hq @ Hdb.T
    ranks = np.argsort(-sims, axis=1)
    Nq = Hq.shape[0]
    rr = []
    ranks_list = []
    for i in range(Nq):
        pos = np.where(ranks[i] == i)[0]
        rpos = (int(pos[0]) + 1) if pos.size else Hdb.shape[0]
        rr.append(1.0 / rpos)
        ranks_list.append(rpos)
    rr = np.array(rr)
    stats = {
        "R@1": float(np.mean(np.array(ranks_list) == 1)),
        "R@5": float(np.mean(np.array(ranks_list) <= 5)),
        "R@10": float(np.mean(np.array(ranks_list) <= 10)),
        "MRR": float(np.mean(rr)),
        "median_rank": int(np.median(ranks_list)),
        "mean_rank": float(np.mean(ranks_list))
    }
    return stats


def _metrics_from_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "R@1": float(np.mean(ranks == 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
        "median_rank": int(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def compute_bidirectional_metrics(Ha, Hb, metric_device=None):
    """Compute A->B and B->A metrics from one similarity matrix.

    Counting scores greater than the paired score avoids a full argsort and
    gives deterministic best-rank handling for exact ties.
    """
    if Ha.shape[0] != Hb.shape[0]:
        raise ValueError(
            f"Paired retrieval requires equal rows, got {Ha.shape[0]} and {Hb.shape[0]}."
        )
    if Ha.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty retrieval set.")

    if metric_device is None:
        metric_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.inference_mode():
        a = torch.as_tensor(Ha, dtype=torch.float32, device=metric_device)
        b = torch.as_tensor(Hb, dtype=torch.float32, device=metric_device)
        similarities = a @ b.T
        positives = torch.diagonal(similarities)

        ranks_ab = 1 + torch.sum(
            similarities > positives[:, None], dim=1
        )
        ranks_ba = 1 + torch.sum(
            similarities > positives[None, :], dim=0
        )

        ranks_ab = ranks_ab.cpu().numpy()
        ranks_ba = ranks_ba.cpu().numpy()

    return _metrics_from_ranks(ranks_ab), _metrics_from_ranks(ranks_ba)

def split_checkerboard_xenium(
    meta_df,
    n_tiles_row=5,
    n_tiles_col=5,
    val_frac=0.1,
    test_frac=0.2,
    n_folds=None,
    random_state=42,
    return_keys=True,
):
    """
    Spatial tile splitter.

    If return_keys=False:
        returns integer indices based on meta_df["original_index"].

    If return_keys=True:
        returns patch keys from meta_df["unique_key"].
        This is safer for ablations because keys survive reordering.
    """

    df = meta_df.copy()
    all_tile_ids = np.zeros(len(df), dtype=int)

    current_tile_offset = 0
    rng = np.random.RandomState(random_state)

    for tissue in df["tissue"].unique():
        mask = df["tissue"] == tissue

        xs = df.loc[mask, "x"].values
        ys = df.loc[mask, "y"].values

        x_edges = np.linspace(xs.min(), xs.max() + 1e-6, n_tiles_col + 1)
        y_edges = np.linspace(ys.min(), ys.max() + 1e-6, n_tiles_row + 1)

        col_idxs = np.digitize(xs, x_edges) - 1
        row_idxs = np.digitize(ys, y_edges) - 1

        col_idxs = np.clip(col_idxs, 0, n_tiles_col - 1)
        row_idxs = np.clip(row_idxs, 0, n_tiles_row - 1)

        local_ids = row_idxs * n_tiles_col + col_idxs
        global_ids = local_ids + current_tile_offset

        all_tile_ids[np.where(mask)[0]] = global_ids

        current_tile_offset += n_tiles_row * n_tiles_col

    unique_tiles = np.unique(all_tile_ids)
    rng.shuffle(unique_tiles)

    if return_keys:
        output_values = df["unique_key"].values
    else:
        output_values = df["original_index"].values

    if n_folds is not None:
        folds_output = []

        tile_chunks = np.array_split(unique_tiles, n_folds)

        tile_to_fold = {}
        for fold_k, tiles in enumerate(tile_chunks):
            for t in tiles:
                tile_to_fold[t] = fold_k

        spot_fold_ids = np.array([tile_to_fold[t] for t in all_tile_ids])

        for k in range(n_folds):
            test_fold = k
            val_fold = (k + 1) % n_folds

            mask_test = spot_fold_ids == test_fold
            mask_val = spot_fold_ids == val_fold
            mask_train = ~(mask_test | mask_val)

            folds_output.append((
                output_values[mask_train],
                output_values[mask_val],
                output_values[mask_test],
            ))

        return folds_output

    n_total = len(unique_tiles)
    n_test = int(np.ceil(n_total * test_frac))
    n_val = int(np.ceil(n_total * val_frac))

    test_tiles = unique_tiles[:n_test]
    val_tiles = unique_tiles[n_test:n_test + n_val]
    train_tiles = unique_tiles[n_test + n_val:]

    mask_test = np.isin(all_tile_ids, test_tiles)
    mask_val = np.isin(all_tile_ids, val_tiles)
    mask_train = np.isin(all_tile_ids, train_tiles)

    return output_values[mask_train], output_values[mask_val], output_values[mask_test]

def visualize_xenium_checkerboard(image, meta_df, 
                                  idx_train, idx_val, idx_test, 
                                  n_tiles_row=10, n_tiles_col=10):
    """
    Visualizes the checkerboard split on the full Xenium H&E image.
    """
    print("Preparing visualization...")
    
    df = meta_df.copy()
    df['split_label'] = 'None'
    
    df.loc[df['original_index'].isin(idx_train), 'split_label'] = 'Train'
    df.loc[df['original_index'].isin(idx_val), 'split_label'] = 'Val'
    df.loc[df['original_index'].isin(idx_test), 'split_label'] = 'Test'
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    
    xs = df['x'].values
    ys = df['y'].values
    
    x_edges = np.linspace(xs.min(), xs.max() + 1, n_tiles_col + 1)
    y_edges = np.linspace(ys.min(), ys.max() + 1, n_tiles_row + 1)
    
    # Iterate through Grid Tiles and Color them
    for r in range(n_tiles_row):
        for c in range(n_tiles_col):
            
            # Define pixel bounds for this tile
            x0 = x_edges[c]
            x1 = x_edges[c+1]
            y0 = y_edges[r]
            y1 = y_edges[r+1]
            
            # Find patches that fall inside this grid tile
            mask_in_tile = (
                (df['x'] >= x0) & (df['x'] < x1) & 
                (df['y'] >= y0) & (df['y'] < y1)
            )
            
            subset = df[mask_in_tile]
            
            if len(subset) == 0:
                continue # Empty background tile
            
            # Determine color based on the split of the patches in this tile
            split_type = subset['split_label'].mode()
            
            if len(split_type) > 0:
                stype = split_type[0]
                if stype == 'Train': color = 'green'
                elif stype == 'Val': color = 'blue'
                elif stype == 'Test': color = 'red'
                else: color = 'gray'
                
                # Draw the colored rectangle
                width = x1 - x0
                height = y1 - y0
                
                rect = patches.Rectangle(
                    (x0, y0), width, height, 
                    linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
                )
                ax.add_patch(rect)
                
                ax.text(x0 + 50, y0 + 150, f"{r},{c}", color="white", fontsize=8, fontweight='bold')

    train_df = df[df['split_label'] == 'Train']
    ax.scatter(train_df['x'], train_df['y'], c='green', s=1, alpha=0.3)
    
    val_df = df[df['split_label'] == 'Val']
    ax.scatter(val_df['x'], val_df['y'], c='blue', s=1, alpha=0.3)
    
    test_df = df[df['split_label'] == 'Test']
    ax.scatter(test_df['x'], test_df['y'], c='red', s=5, alpha=0.8) # Make test spots pop

    legend_elements = [
        patches.Patch(facecolor='green', alpha=0.3, label='Train'),
        patches.Patch(facecolor='blue', alpha=0.3, label='Validation'),
        patches.Patch(facecolor='red', alpha=0.3, label='Test')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f"Xenium Checkerboard Split ({n_tiles_row}x{n_tiles_col})")
    plt.axis('off')
    plt.show()

def plot_joint_latent_space(model, X_norm, Y_norm, indices_test):
    """
    Projects BOTH Image and RNA embeddings into the same 2D space.
    """
    print("Generating Joint UMAP (Images + RNA)...")
    
    # Get Embeddings for Test Set
    model.eval()
    device = next(model.parameters()).device
    

    plot_indices = indices_test
        
    X_sub = torch.from_numpy(X_norm[plot_indices]).float().to(device)
    Y_sub = torch.from_numpy(Y_norm[plot_indices]).float().to(device)
    
    with torch.inference_mode():
        emb_img = torch.nn.functional.normalize(model.img_head(X_sub), dim=1).cpu().numpy()
        emb_rna = torch.nn.functional.normalize(model.rna_head(Y_sub), dim=1).cpu().numpy()
        
    # We stack them: First N are images, Next N are RNA
    combined_emb = np.vstack([emb_img, emb_rna])
    
    modality_labels = np.array([0] * len(emb_img) + [1] * len(emb_rna))
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(combined_emb)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Images (Blue)
    plt.scatter(
        embedding_2d[modality_labels==0, 0], 
        embedding_2d[modality_labels==0, 1],
        c='blue', s=10, alpha=0.5, label='Histology (Image)'
    )
    
    # Plot RNA (Red)
    plt.scatter(
        embedding_2d[modality_labels==1, 0], 
        embedding_2d[modality_labels==1, 1],
        c='red', s=10, alpha=0.5, label='Transcriptomics (RNA)'
    )
    
    plt.legend()
    plt.title("Joint Latent Space Alignment\n(Do the modalities overlap?)")
    plt.show()

def evaluate_test_trimodal(model, X, Y, Z, indices_test, evaluate_text=True):
    """
    Evaluates the Tri-Modal model on the test set and returns metrics as a dict.
    """
    results = {
        "n_test": len(indices_test)
    }

    X_test = X[indices_test]
    Y_test = Y[indices_test]
    Z_test = Z[indices_test]

    # Embeddings
    H_img = model.embed_batch(X_test, branch="img")
    H_rna = model.embed_batch(Y_test, branch="rna")

    # Image <-> RNA
    stats_image_rna, stats_rna_image = compute_bidirectional_metrics(H_img, H_rna)
    print("\n--- [Image -> RNA] ---")
    print(f"R@10: {stats_image_rna['R@10']:.4f} | MRR: {stats_image_rna['MRR']:.4f}")
    results["image_rna"] = stats_image_rna
    print("\n--- [RNA -> Image] ---")
    print(f"R@10: {stats_rna_image['R@10']:.4f} | MRR: {stats_rna_image['MRR']:.4f}")
    results["rna_image"] = stats_rna_image

    # Image <-> Text and RNA <-> Text
    if evaluate_text:
        H_text = model.embed_batch(Z_test, branch="text")

        stats_image_text, stats_text_image = compute_bidirectional_metrics(H_img, H_text)
        stats_rna_text, stats_text_rna = compute_bidirectional_metrics(H_rna, H_text)

        for label, key, metrics in (
            ("Image -> Text", "image_text", stats_image_text),
            ("Text -> Image", "text_image", stats_text_image),
            ("RNA -> Text", "rna_text", stats_rna_text),
            ("Text -> RNA", "text_rna", stats_text_rna),
        ):
            print(f"\n--- [{label}] ---")
            print(f"R@10: {metrics['R@10']:.4f} | MRR: {metrics['MRR']:.4f}")
            results[key] = metrics

    return results


#### Microenvironment characterisation ####
def radius_neighbor_adjacency_by_type(df, x_col, y_col, type_col, radius, include_self=False):
    df = df[df[type_col].notna()]

    coords = df[[x_col, y_col]].values
    cell_types = df[type_col].values
    unique_types = np.unique(cell_types)

    adj = radius_neighbors_graph(
        coords,
        radius=radius,
        mode="connectivity", 
        include_self=include_self
    )
    
    out = pd.DataFrame(index=df.index)

    for ct in unique_types:
        idx = np.where(cell_types == ct)[0]
        vals = adj[:, idx].sum(axis=1) 
        out[f"adj_to_{ct}"] = np.asarray(vals).ravel()

    return out, adj

def top_interactions_description(interaction: pd.DataFrame, top_k: int = 3, get_summary_in_text: bool = True):
    lineages = interaction.index.tolist()
    scores = {}

    for a in lineages:
        for b in lineages:
            val_ab = interaction.loc[a].get(f"adj_to_{b}", 0)
            val_ba = interaction.loc[b].get(f"adj_to_{a}", 0)
            
            score = 0.5 * (val_ab + val_ba)
            
            # Store alphabetically to avoid duplicates (A-B vs B-A)
            key = tuple(sorted([a, b]))
            scores[key] = score

    # Sort descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if get_summary_in_text:
        return [f"{a}<->{b}" for (a, b), _ in ranked[:top_k]]
    else:
        return ranked[:top_k]

def get_interaction_summary(df, k_interactions, x_col, y_col, type_col, radius, get_summary_in_text=True):
    df = df[df[type_col].notna()]
    
    if df.shape[0] < 2:
        return []
    
    features, adj = radius_neighbor_adjacency_by_type(
        df, x_col=x_col, y_col=y_col, type_col=type_col, radius=radius
    )
    df_cells_patch_w_features = pd.concat([df, features], axis=1)
    filtered_columns = [c for c in df_cells_patch_w_features.columns if c.startswith("adj")]
    
    # Sum interactions by type
    df_adj = df_cells_patch_w_features.groupby(type_col)[filtered_columns].sum()
    
    # Normalize interaction density
    interaction = (df_adj / df_adj.sum()).fillna(0)
    
    return top_interactions_description(interaction, top_k=k_interactions, get_summary_in_text=get_summary_in_text)
        

def get_cell_type_composition(df_cells_patch, has_lineage=False):
    
    if has_lineage:
        lines = []

        for lineage in df_cells_patch['lineage'].value_counts().index:
            subset = df_cells_patch[df_cells_patch['lineage'] == lineage]
            vc = subset['cell_type'].value_counts()

            cells = ", ".join([f"{ct}({n})" for ct, n in vc.items()])
            lines.append(f"{lineage.upper()}: {cells}")

        return " | ".join(lines)
    else:
        vc = df_cells_patch['cell_type'].value_counts()
        return ", ".join([f"{ct}({n})" for ct, n in vc.items()])

def lineage_or_cell_type_ranking(df_cells_patch, has_lineage=False):
    variable = 'lineage' if has_lineage else 'cell_type'
    counts = df_cells_patch[variable].value_counts()
    return " > ".join(counts.index.tolist())


def get_density_description(density_lookup, patch_name):
    if patch_name not in density_lookup:
        return "UNKNOWN_DENSITY"
    
    data = density_lookup[patch_name]
    count = int(data['count'])
    pct = data['percentile']
    
    if pct < 0.10:
        category = "VERY_SPARSE (Acellular/Void)"
    elif pct < 0.30:
        category = "SPARSE (Hypocellular)"
    elif pct < 0.70:
        category = "MODERATE (Cellular)"
    elif pct < 0.90:
        category = "DENSE (Hypercellular)"
    else:
        category = "SOLID (Packed/High-Density)"
        
    return f"[{category}: {count} cells, {pct:.2f} percentile]"

def fuse_gene_cell_type_vectors(common_keys, gene_vector, cell_type_vector):
    # concatenate with cell type vectors
    Y_fused_dict = {}
    for k in common_keys:
        vec_rna = cell_type_vector[k]                
        vec_struct = gene_vector[k]
        
        Y_fused_dict[k] = np.concatenate([vec_rna, vec_struct]).astype(np.float32)
    
    return Y_fused_dict


def create_multisample_metadata(keys):
    data = []
    pattern = re.compile(r"(.+)__patch_x(\d+)_y(\d+)")
    
    print(f"Parsing metadata for {len(keys)} patches...")
    
    for i, k in enumerate(keys):
        match = pattern.search(k)
        if match:
            tissue = match.group(1)
            x = int(match.group(2))
            y = int(match.group(3))
            
            data.append({
                "unique_key": k,
                "tissue": tissue, # crucial for split_checkerboard
                "x": x,
                "y": y,
                "original_index": i
            })
        else:
            print(f"Warning: Failed to parse key: {k}")
            
    return pd.DataFrame(data)

def embed_test_ids_cv(model, meta_df, idx_test, uni_embeddings, rna_vectors):
    test_keys_set = set(meta_df.iloc[idx_test]['unique_key'].values)
    mask_test = meta_df['unique_key'].isin(test_keys_set)

    meta_df_target_fold = meta_df[mask_test].copy()

    subset_keys = meta_df_target_fold['unique_key'].tolist()

    X_img_subset = np.stack([uni_embeddings[k] for k in subset_keys]).astype(np.float32)
    X_rna_subset = np.stack([rna_vectors[k] for k in subset_keys]).astype(np.float32)

    with torch.no_grad():
        H_img_target_fold = model.embed_batch(X_img_subset, branch="img")
        H_rna_target_fold = model.embed_batch(X_rna_subset, branch="rna")

    return H_img_target_fold, H_rna_target_fold, meta_df_target_fold
  
def run_alignment_tests(
    indices_train,
    indices_val,
    indices_test,
    X,
    Y,
    Z,
    aligned_keys,
    train_keys,
    val_keys,
    test_keys,
    uni_embeddings,
    c2s_embeddings,
    text_embeddings,
    atol=1e-5,
):
    """
    Runs all split/alignment sanity checks.

    Checks:
    1. train/val/test integer indices are disjoint, in range, and cover X/Y/Z.
    2. X/Y/Z rows match aligned_keys.
    3. integer indices recover the intended train/val/test key sets.

    Raises AssertionError if any check fails.
    """

    print("\n[1/3] Checking basic split integrity...")

    train = set(indices_train)
    val = set(indices_val)
    test = set(indices_test)

    print("N train:", len(train))
    print("N val:  ", len(val))
    print("N test: ", len(test))
    print("N total split:", len(train | val | test))
    print("N arrays:", len(X), len(Y), len(Z))

    assert len(train) > 0, "Train split is empty"
    assert len(val) > 0, "Validation split is empty"
    assert len(test) > 0, "Test split is empty"

    assert len(train & val) == 0, "Leakage: train and val overlap"
    assert len(train & test) == 0, "Leakage: train and test overlap"
    assert len(val & test) == 0, "Leakage: val and test overlap"

    assert len(X) == len(Y) == len(Z), "X, Y, Z have different lengths"
    assert len(aligned_keys) == len(X), "aligned_keys length does not match X/Y/Z"

    max_idx = max(max(train), max(val), max(test))
    min_idx = min(min(train), min(val), min(test))

    assert min_idx >= 0, "Negative index found"
    assert max_idx < len(X), "Split index exceeds aligned array length"

    assert len(train | val | test) == len(X), (
        f"Split does not cover all aligned rows: "
        f"{len(train | val | test)} split rows vs {len(X)} aligned rows"
    )

    print("Basic split checks passed.")

    print("\n[2/3] Checking row order alignment...")

    print("N aligned keys:", len(aligned_keys))
    print("N X/Y/Z:", len(X), len(Y), len(Z))

    missing_uni = [k for k in aligned_keys if k not in uni_embeddings]
    missing_c2s = [k for k in aligned_keys if k not in c2s_embeddings]
    missing_text = [k for k in aligned_keys if k not in text_embeddings]

    print("Missing UNI keys:", len(missing_uni))
    print("Missing RNA/cell/fused keys:", len(missing_c2s))
    print("Missing text keys:", len(missing_text))

    assert len(missing_uni) == 0, f"Missing UNI keys: {missing_uni[:5]}"
    assert len(missing_c2s) == 0, f"Missing RNA/cell/fused keys: {missing_c2s[:5]}"
    assert len(missing_text) == 0, f"Missing text keys: {missing_text[:5]}"

    X_expected = np.stack(
        [np.asarray(uni_embeddings[k]).reshape(-1) for k in aligned_keys],
        axis=0,
    )

    Y_expected = np.stack(
        [np.asarray(c2s_embeddings[k]).reshape(-1) for k in aligned_keys],
        axis=0,
    )

    Z_expected = np.stack(
        [np.asarray(text_embeddings[k]).reshape(-1) for k in aligned_keys],
        axis=0,
    )

    x_ok = np.allclose(X, X_expected, atol=atol)
    y_ok = np.allclose(Y, Y_expected, atol=atol)
    z_ok = np.allclose(Z, Z_expected, atol=atol)

    print("X order correct:", x_ok)
    print("Y order correct:", y_ok)
    print("Z order correct:", z_ok)

    if not x_ok:
        bad = np.where(~np.isclose(X, X_expected, atol=atol).all(axis=1))[0][:10]
        raise AssertionError(f"X is not aligned to aligned_keys. First bad rows: {bad}")

    if not y_ok:
        bad = np.where(~np.isclose(Y, Y_expected, atol=atol).all(axis=1))[0][:10]
        raise AssertionError(f"Y is not aligned to aligned_keys. First bad rows: {bad}")

    if not z_ok:
        bad = np.where(~np.isclose(Z, Z_expected, atol=atol).all(axis=1))[0][:10]
        raise AssertionError(f"Z is not aligned to aligned_keys. First bad rows: {bad}")

    print("Alignment order check passed.")

    print("\n[3/3] Checking key split to index conversion...")

    aligned_keys_arr = np.asarray(aligned_keys)

    recovered_train_keys = set(aligned_keys_arr[indices_train])
    recovered_val_keys = set(aligned_keys_arr[indices_val])
    recovered_test_keys = set(aligned_keys_arr[indices_test])

    train_keys = set(train_keys)
    val_keys = set(val_keys)
    test_keys = set(test_keys)

    assert recovered_train_keys == train_keys, (
        f"Train indices do not recover train keys. "
        f"Missing: {list(train_keys - recovered_train_keys)[:5]}, "
        f"Extra: {list(recovered_train_keys - train_keys)[:5]}"
    )

    assert recovered_val_keys == val_keys, (
        f"Val indices do not recover val keys. "
        f"Missing: {list(val_keys - recovered_val_keys)[:5]}, "
        f"Extra: {list(recovered_val_keys - val_keys)[:5]}"
    )

    assert recovered_test_keys == test_keys, (
        f"Test indices do not recover test keys. "
        f"Missing: {list(test_keys - recovered_test_keys)[:5]}, "
        f"Extra: {list(recovered_test_keys - test_keys)[:5]}"
    )

    print("Key split to index conversion check passed.")

    print("\nAll tests passed. Split and alignment are safe to use.")

    return True

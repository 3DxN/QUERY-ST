
import os
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage.transform import resize

from spatialdata_io import xenium, xenium_aligned_image
from spatialdata import bounding_box_query, transform
import spatialdata as sd
import spatialdata_plot

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_ROOT = "/Volumes/HD_rafael/DPhil/prostate_full"
IMAGES_DIR = "/Volumes/HD_rafael/DPhil/prostate_full/images"
OUTPUT_DIR = "/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/prostate"

SAMPLES = ["PCA007", "PCA008", "PCA009", "PCA010"]

PATCH_SIZE_MICRONS = 224
STRIDE_MICRONS = 224
TARGET_PIXELS = 224

patches_folder_prefix = f"he_size_{PATCH_SIZE_MICRONS}_microns_patches"

def build_global_vocab(samples, input_root):
    print("Building global vocabularies...")
    global_cell_types = set()
    
    for sample in samples:
        annot_path = os.path.join(input_root, f"output_{sample}", f"cell_annots_{sample}.csv")
        if os.path.exists(annot_path):
            df = pd.read_csv(annot_path)
            if 'group' in df.columns:
                global_cell_types.update(df['group'].dropna().unique().astype(str))
    
    sorted_types = sorted(list(global_cell_types))
    
    
    first_sample = samples[0]
    try:
        temp_sdata = xenium(os.path.join(input_root, f"output_{first_sample}"))
        global_genes = temp_sdata.table.var_names.values
    except:
        print("Warning: Could not load first sample to get gene list.")
        global_genes = []

    return sorted_types, global_genes

def visualize_alignment_check(
    sdata,
    sample,
    output_dir,
    box_size=500,
    offset_frac=0.6,
):
    print(f"   > Generating visual alignment check for {sample}...")

    check_dir = os.path.join(output_dir, "alignment_checks")
    os.makedirs(check_dir, exist_ok=True)

    try:
        # tissue center
        x_col = "x_global" if "x_global" in sdata.table.obs else "x_centroid"
        y_col = "y_global" if "y_global" in sdata.table.obs else "y_centroid"

        mid_x = float(sdata.table.obs[x_col].median())
        mid_y = float(sdata.table.obs[y_col].median())

        # crop size
        d = box_size * offset_frac
        centers = [
            (mid_x - d, mid_y - d),
            (mid_x + d, mid_y - d),
            (mid_x - d, mid_y + d),
            (mid_x + d, mid_y + d),
        ]

        # lightweight object for plotting
        sdata_light = sd.SpatialData(
            images=sdata.images,
            labels=sdata.labels,
        )

        fig, axes = plt.subplots(
            nrows=4,
            ncols=2,
            figsize=(18, 24),
            squeeze=False,
        )

        for i, (cx, cy) in enumerate(centers):
            min_c = [cx - box_size / 2, cy - box_size / 2]
            max_c = [cx + box_size / 2, cy + box_size / 2]

            cropped = bounding_box_query(
                sdata_light,
                min_coordinate=min_c,
                max_coordinate=max_c,
                axes=("x", "y"),
                target_coordinate_system="global",
            )

            cropped.pl.render_images("he_image").pl.show(
                ax=axes[i, 0],
                title=f"H&E (crop {i+1})",
                coordinate_systems="global",
            )

            cropped.pl.render_labels("cell_labels").pl.show(
                ax=axes[i, 1],
                title=f"Labels (crop {i+1})",
                coordinate_systems="global",
            )

        plt.tight_layout()
        save_path = os.path.join(check_dir, f"{sample}_alignment_check_4x.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"   > Saved visual check to {save_path}")

    except Exception as e:
        print(f"   ! Warning: Visualization failed: {e}")

def process_sample(sample, global_types, global_genes, metadata_global):
    input_dir = os.path.join(INPUT_ROOT, f"output_{sample}")
    output_sample_dir = os.path.join(OUTPUT_DIR, f"{patches_folder_prefix}_{sample}")
    
    if not os.path.exists(output_sample_dir):
        os.makedirs(output_sample_dir)

    print(f"Processing {sample}...")

    # Load data (SpatialData methodology for unprocessed xenium)
    try:
        sdata = xenium(input_dir)
        
        # Load & Align Image
        image_file = f"{sample}_HE.ome.tif"
        image_path = os.path.join(IMAGES_DIR, image_file)
        alignment_path = os.path.join(input_dir, "alignment/matrix.csv")
        
        # Align image - using alignment path I created in xenium explorer
        aligned_img = xenium_aligned_image(image_path, alignment_path)
        sdata.images["he_image"] = aligned_img
        
        # Load cell type composition/annotations
        annot_path = os.path.join(input_dir, f"cell_annots_{sample}.csv")
        cell_annotations = pd.read_csv(annot_path).set_index("cell_id")
        
        sdata.table.obs = sdata.table.obs.set_index("cell_id", drop=False) 
        sdata.table.obs = sdata.table.obs.join(cell_annotations, how="left")
        
        # Transform to global coordinates - used when creating the patches on the image
        gdf_global = transform(sdata.shapes["cell_circles"], to_coordinate_system="global")
        if hasattr(gdf_global, "compute"):
            gdf_global = gdf_global.compute()
            
        sdata.table.obs.loc[gdf_global.index, "x_global"] = gdf_global.geometry.x
        sdata.table.obs.loc[gdf_global.index, "y_global"] = gdf_global.geometry.y

        visualize_alignment_check(sdata, sample, OUTPUT_DIR)
        
    except Exception as e:
        print(f"Error loading {sample}: {e}")
        return None

    # Pre-process - normalisation
    adata = sdata.table.copy()
    
    if len(global_genes) > 0 and not np.array_equal(adata.var_names, global_genes):
        adata = adata[:, global_genes].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    genes_to_use = list(adata.var_names)

    # Extract gene vectors for each patch 
    obs = sdata.table.obs
    min_x, min_y = obs['x_global'].min(), obs['y_global'].min()
    
    obs['grid_x'] = ((obs['x_global'] - min_x) // STRIDE_MICRONS).astype(int)
    obs['grid_y'] = ((obs['y_global'] - min_y) // STRIDE_MICRONS).astype(int)
    
    # Sync grid info back to adata for gene aggregation
    adata.obs['grid_x'] = obs['grid_x']
    adata.obs['grid_y'] = obs['grid_y']

    # Update global metadata for CSV export
    obs['sample'] = sample
    obs['patch_name'] = (
        sample + "__patch_x" + (min_x + obs['grid_x'] * STRIDE_MICRONS).astype(int).astype(str) + 
        "_y" + (min_y + obs['grid_y'] * STRIDE_MICRONS).astype(int).astype(str)
    )
    # Keep relevant columns
    cols_to_keep = ['x_global', 'y_global', 'group', 'sample', 'patch_name'] 
    metadata_sample = obs[cols_to_keep].copy().rename(columns={'group': 'cell_type'})

    # Cell type maps
    ct_map = {ct: i for i, ct in enumerate(global_types)}
    patch_grps = obs.groupby(['grid_x', 'grid_y', 'group']).size()
    
    gene_df = pd.DataFrame(adata.X.toarray(), index=adata.obs.index, columns=adata.var_names)
    gene_df['gx'] = adata.obs['grid_x']
    gene_df['gy'] = adata.obs['grid_y']
    gene_means = gene_df.groupby(['gx', 'gy']).mean()

    vectors = {"ct": {}, "ct_raw": {}, "full": {}}
    extract_tasks = []

    for (gx, gy) in tqdm(patch_grps.index.droplevel(2).unique(), desc="Building Vectors"):
        x_mic = min_x + (gx * STRIDE_MICRONS)
        y_mic = min_y + (gy * STRIDE_MICRONS)
        pname = f"patch_x{int(x_mic)}_y{int(y_mic)}.png"
        
        # Cell Type Vector
        vec = np.zeros(len(global_types), dtype=np.float32)
        try:
            for ct, n in patch_grps.loc[(gx, gy)].items():
                if ct in ct_map: vec[ct_map[ct]] = n
        except KeyError: pass
        
        vectors["ct_raw"][pname] = vec
        vectors["ct"][pname] = np.log1p(vec)
        
        # Gene Vectors
        if (gx, gy) in gene_means.index:
            row = gene_means.loc[(gx, gy)]
            vectors["full"][pname] = row[genes_to_use].values.astype(np.float32)
            
        extract_tasks.append((x_mic, y_mic, pname))

    # Save Vectors
    with open(os.path.join(output_sample_dir, "patch_vectors_raw.pkl"), "wb") as f: pickle.dump(vectors["ct_raw"], f)
    with open(os.path.join(output_sample_dir, "patch_vectors.pkl"), "wb") as f: pickle.dump(vectors["ct"], f)
    with open(os.path.join(output_sample_dir, "patch_genes_full.pkl"), "wb") as f: pickle.dump(vectors["full"], f)

    pd.Series(global_types).to_csv(os.path.join(output_sample_dir, "cell_type_legend.csv"), index=False)

    # We use bounding box query from spatialdata for this one (samples and tissues are much bigger than lung)
    print(f"Extracting {len(extract_tasks)} images...")
    
    he_img_layer = sdata.images["he_image"]
    
    def _extract_worker(args):
        x_start, y_start, name = args
        min_c = [x_start, y_start]
        max_c = [x_start + PATCH_SIZE_MICRONS, y_start + PATCH_SIZE_MICRONS]
        
        try:
            crop = bounding_box_query(
                he_img_layer,
                axes=("x", "y"),
                min_coordinate=min_c,
                max_coordinate=max_c,
                target_coordinate_system="global"
            )
            
            if crop is None: return None
            img_data = crop.data.compute()
            
            if img_data.size == 0 or np.mean(img_data) > 230: return None
            
            img_np = np.transpose(img_data, (1, 2, 0))
            if img_np.shape[0] != TARGET_PIXELS:
                img_np = resize(img_np, (TARGET_PIXELS, TARGET_PIXELS), 
                                anti_aliasing=True, preserve_range=True).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
                
            plt.imsave(os.path.join(output_sample_dir, name), img_np)
        except Exception:
            pass

    Parallel(n_jobs=multiprocessing.cpu_count()-4, backend="threading")(
        delayed(_extract_worker)(t) for t in tqdm(extract_tasks)
    )
    
    print(f"Finished {sample}")
    return metadata_sample

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Global Vocabs
    g_types, g_genes = build_global_vocab(SAMPLES, INPUT_ROOT)
        
    metadata_accumulator = []
        
    for sample in SAMPLES:
        meta_s = process_sample(sample, g_types, g_genes, None)
        if meta_s is not None:
            metadata_accumulator.append(meta_s)
                
    if metadata_accumulator:
        metadata_global = pd.concat(metadata_accumulator, ignore_index=True)
        metadata_global.to_csv(os.path.join(OUTPUT_DIR, "all_samples_metadata.csv"), index=False)


import os
import re
import pickle
import multiprocessing
import numpy as np
import pandas as pd
import tifffile
import scanpy as sc
import torch
import timm
from scipy import sparse, io, signal, ndimage
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt

INPUT_DIR = "/Volumes/HD_rafael/DPhil/xenium_lung/inputs"
IMAGES_FOLDER = "/Volumes/HD_rafael/DPhil/xenium_lung/all_TMA"
OUTPUT_DIR = "/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/lung_temp_exp"

PIXEL_SCALE = 0.2125
PATCH_SIZE_MICRONS = 112
STRIDE_MICRONS = 112
TARGET_PIXELS = 224
PATCH_SIZE_PX = int(PATCH_SIZE_MICRONS / PIXEL_SCALE)

# We noticed that the following samples need alignment
FFT_SAMPLES = [
    "TILD049MA", "TILD080LA", "TILD111LA", "TILD113LA", "TILD299MA", "TILD315MA",
    "VUHD038", "VUHD090", "VUILD49LA", "VUILD58MA", "VUILD141MA", "VUILD142MA",
    "VUILD48LA1", "VUILD105MA1"
]

patches_folder_prefix = f"he_size_{PATCH_SIZE_MICRONS}_microns_patches"

# %%
def map_images(input_dir, images_folder):
    sample_ids = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")]
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".tif") and not f.startswith(".")]
    image_map = {}
    for sample in sample_ids:
        match = next((img for img in image_files if sample in img), None)
        if match:
            image_map[sample] = os.path.join(images_folder, match)
    return image_map

def coords_to_image(x_coords, y_coords, target_shape, scale_factor):
    x_idx = (x_coords * scale_factor).astype(int)
    y_idx = (y_coords * scale_factor).astype(int)
    grid = np.zeros(target_shape, dtype=np.float32)
    mask = (x_idx >= 0) & (x_idx < target_shape[1]) & (y_idx >= 0) & (y_idx < target_shape[0])
    np.add.at(grid, (y_idx[mask], x_idx[mask]), 1)
    return ndimage.gaussian_filter(grid, sigma=2)

def calculate_offsets(image_map):
    offsets = {}
    print(f"Calculating offsets for {len(image_map)} samples...")
    
    for sample, img_path in tqdm(image_map.items()):
        meta_path = os.path.join(INPUT_DIR, sample, "metadata.csv")
        if not os.path.exists(meta_path): continue

        try:
            df = pd.read_csv(meta_path)
            x_raw = df['x_global'].values
            y_raw = df['y_global'].values
            
            # FFT Alignment for known bad samples
            if sample in FFT_SAMPLES:
                img = tifffile.imread(img_path)
                DS = 20
                img_small = img[::DS, ::DS]
                
                if img_small.ndim == 3: gray = np.mean(img_small, axis=2)
                else: gray = img_small
                
                tissue_map = (255 - gray)
                tissue_map[tissue_map < 30] = 0
                
                x_raw_min, y_raw_min = x_raw.min(), y_raw.min()
                x_local = x_raw - x_raw_min
                y_local = y_raw - y_raw_min
                
                scale_to_small = 1.0 / (PIXEL_SCALE * DS)
                cell_map = coords_to_image(x_local, y_local, tissue_map.shape, scale_to_small)
                
                correlation = signal.fftconvolve(tissue_map, cell_map[::-1, ::-1], mode='same')
                y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
                
                y_center, x_center = np.array(tissue_map.shape) // 2
                shift_x_microns = (x_peak - x_center) * (PIXEL_SCALE * DS)
                shift_y_microns = (y_peak - y_center) * (PIXEL_SCALE * DS)
                
                offsets[sample] = (x_raw_min - shift_x_microns, y_raw_min - shift_y_microns)
            
            # Dynamic Shift for cropped samples
            else:
                img_header = tifffile.TiffFile(img_path)
                w_microns = img_header.pages[0].shape[1] * PIXEL_SCALE
                
                if x_raw.max() > w_microns:
                    offsets[sample] = (x_raw.min(), y_raw.min())
                else:
                    offsets[sample] = (0, 0)
                    
        except Exception as e:
            print(f"Error calculating offset for {sample}: {e}")
            
    return offsets

def build_global_vocab(image_map):
    print("Building global vocabularies...")
    global_cell_types = set()
    
    for sample in image_map:
        meta_path = os.path.join(INPUT_DIR, sample, "metadata.csv")
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path, usecols=["cell_type"])
            global_cell_types.update(df['cell_type'].dropna().unique().astype(str))
            
    sorted_types = sorted(list(global_cell_types))
    
    # Get global gene list from first sample
    first_key = list(image_map.keys())[0]
    genes_path = os.path.join(INPUT_DIR, first_key, "genes.tsv")
    global_genes = pd.read_csv(genes_path, header=None)[0].values
    
    return sorted_types, global_genes


def process_sample(sample, img_path, offset, global_types, global_genes, metadata_global):
    patch_path = os.path.join(INPUT_DIR, sample)
    output_dir = os.path.join(OUTPUT_DIR, f"{patches_folder_prefix}_{sample}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data - extracted from R (seurat object provided by the authors)
    try:
        metadata = pd.read_csv(os.path.join(patch_path, "metadata.csv")).set_index("cell_id")
        counts = io.mmread(os.path.join(patch_path, "counts.mtx")).T.tocsr()
        genes = pd.read_csv(os.path.join(patch_path, "genes.tsv"), header=None)[0].values
        barcodes = pd.read_csv(os.path.join(patch_path, "barcodes.tsv"), header=None)[0].values
        
        adata = sc.AnnData(counts, obs=metadata)
        adata.var_names = genes; adata.obs_names = barcodes
        
        if not np.array_equal(adata.var_names, global_genes):
            adata = adata[:, global_genes].copy()

        adata.layers["raw_counts"] = adata.X.copy()
            
        he_img = tifffile.memmap(img_path)
    except Exception as e:
        print(f"Skipping {sample}: {e}")
        return

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    genes_to_use = list(adata.var_names)

    # 3. Grid & Vectors
    off_x, off_y = offset
    metadata['grid_x'] = ((metadata['x_global'] - off_x) // STRIDE_MICRONS).astype(int)
    metadata['grid_y'] = ((metadata['y_global'] - off_y) // STRIDE_MICRONS).astype(int)
    adata.obs['grid_x'] = metadata['grid_x']
    adata.obs['grid_y'] = metadata['grid_y']

    # Metadata folder
    p_x = off_x + (metadata['grid_x'] * STRIDE_MICRONS)
    p_y = off_y + (metadata['grid_y'] * STRIDE_MICRONS)
    
    metadata['sample'] = sample
    
    metadata['patch_name'] = (
        sample + "__patch_x" + p_x.astype(int).astype(str) + 
        "_y" + p_y.astype(int).astype(str)
    )

    # Save the required CSV with x, y, cell_type, and corresponding patch
    metadata_sample = metadata[['x_global', 'y_global', 'cell_type', 'lineage', 'sample', 'disease_status', 'sample_affect', 'patch_name']]
    
    # .to_csv(
    #     os.path.join(output_dir, "cells_per_patch.csv")
    # )

    # Cell type maps
    ct_map = {ct: i for i, ct in enumerate(global_types)}
    patch_grps = metadata.groupby(['grid_x', 'grid_y', 'cell_type']).size()
    
    gene_df = pd.DataFrame(adata.X.toarray(), index=adata.obs.index, columns=adata.var_names)
    gene_df['gx'] = adata.obs['grid_x']; gene_df['gy'] = adata.obs['grid_y']
    gene_means = gene_df.groupby(['gx', 'gy']).mean()

    raw_gene_df = pd.DataFrame(adata.layers["raw_counts"].toarray(), index=adata.obs.index, columns=adata.var_names)
    raw_gene_df['gx'] = adata.obs['grid_x']; raw_gene_df['gy'] = adata.obs['grid_y']
    raw_gene_sums = raw_gene_df.groupby(['gx', 'gy']).sum()
    
    vectors = {"ct": {}, "ct_raw": {}, "full": {}, "genes_raw": {}}
    extract_tasks = []
    
    for (gx, gy) in patch_grps.index.droplevel(2).unique():
        x_mic = off_x + (gx * STRIDE_MICRONS)
        y_mic = off_y + (gy * STRIDE_MICRONS)
        pname = f"patch_x{int(x_mic)}_y{int(y_mic)}.png"
        
        # CT Vector
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
            vectors["genes_raw"][pname] = raw_gene_sums.loc[(gx, gy), genes_to_use].values.astype(np.float32)
            
        extract_tasks.append((x_mic, y_mic, off_x, off_y, pname))

    # Save Vectors
    with open(os.path.join(output_dir, "patch_vectors_raw.pkl"), "wb") as f: pickle.dump(vectors["ct_raw"], f)
    with open(os.path.join(output_dir, "patch_vectors.pkl"), "wb") as f: pickle.dump(vectors["ct"], f)
    with open(os.path.join(output_dir, "patch_genes_full.pkl"), "wb") as f: pickle.dump(vectors["full"], f)
    with open(os.path.join(output_dir, "patch_genes_raw_counts.pkl"), "wb") as f: pickle.dump(vectors["genes_raw"], f)

    pd.Series(global_types).to_csv(os.path.join(output_dir, "cell_type_legend.csv"), index=False)
    
    # Extract Images
    def _extract(args):
        x_glob, y_glob, ox, oy, name = args
        x_px = int((x_glob - ox) / PIXEL_SCALE)
        y_px = int((y_glob - oy) / PIXEL_SCALE)
        
        if x_px < 0 or y_px < 0: return None
        end_x, end_y = x_px + PATCH_SIZE_PX, y_px + PATCH_SIZE_PX
        if end_x > he_img.shape[1] or end_y > he_img.shape[0]: return None
        
        try:
            crop = he_img[y_px:end_y, x_px:end_x]
            if crop.size == 0 or np.mean(crop) > 235: return None
            if crop.shape[0] != TARGET_PIXELS:
                crop = resize(crop, (TARGET_PIXELS, TARGET_PIXELS), anti_aliasing=True, preserve_range=True).astype(np.uint8)
            plt.imsave(os.path.join(output_dir, name), crop)
        except: pass

    Parallel(n_jobs=multiprocessing.cpu_count()-2, backend="threading")(delayed(_extract)(t) for t in extract_tasks)
    print(f"Finished {sample}: {len(vectors['ct'])} patches")

    return metadata_sample

if __name__ == "__main__":
    img_map = map_images(INPUT_DIR, IMAGES_FOLDER)
    print(f"Mapped {len(img_map)} samples.")
        
    # Alignment Calculation
    offsets = calculate_offsets(img_map)
        
    # Global Vocabs
    g_types, g_genes = build_global_vocab(img_map)
        
    metadata_global = pd.DataFrame()

    metadata_accumulator_array = []

    # Process Patches & Vectors
    for sample, img_path in img_map.items():
        if sample in offsets:
            metadata_sample = process_sample(sample, img_path, offsets[sample], g_types, g_genes, metadata_global)
            metadata_accumulator_array.append(metadata_sample)

        
    metadata_global = pd.concat(metadata_accumulator_array, ignore_index=True)
    print(f"Final Global Metadata Shape: {metadata_global.shape}")
    metadata_global.to_csv(os.path.join(OUTPUT_DIR, "all_samples_metadata.csv"), index=False)

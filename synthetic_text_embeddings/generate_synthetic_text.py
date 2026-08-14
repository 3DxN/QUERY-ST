import pickle
import os
from tqdm import tqdm

import pandas as pd

import sys
from pathlib import Path

project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))

from common_functions import *

import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scipy.stats import kruskal
# import scikit_posthocs as sp

from openai import OpenAI

from collections import defaultdict
import time

client = OpenAI()

SYSTEM_PROMPT_REPORT_1 = {
    "lung": """
You are an expert pulmonary pathologist generating concise, ground-truth textual descriptions of lung cell populations for semantic retrieval.

### Inputs
1. **COMPOSITION_DATA**: A list of cell types and their relative abundance.

### Output format (STRICT)
- Maximum **2 sentences total**.
- Plain text only. No lists or markdown.

### Guidelines
- **Sentence 1**: Identify the **dominant biological compartment** and the **specific top 2–3 cell types** based on abundance.
- **Sentence 2**: Describe secondary or minor cell populations that define the specific subtype of this region.
- **Constraint**: Do not infer spatial geometry, density, or interaction patterns (e.g., do not use words like “clusters”, “interface”, “dense”, “sparse”, “contact”). Focus strictly on cellular presence.

### Example

Input:
COMPOSITION_DATA: IMMUNE: Neutrophils(12), Monocytes/MDMs(9), Interstitial Macrophages(9), NK/NKT(4), Alveolar Macrophages(2), Proliferating Myeloid(2), CD4+ T-cells(1), SPP1+ Macrophages(1) | ENDOTHELIAL: Capillary(22), Lymphatic(3), Venous(2) | EPITHELIAL: AT2(13), AT1(5) | MESENCHYMAL: Alveolar FBs(7), Proliferating FBs(4), SMCs/Pericytes(2), Inflammatory FBs(2), Activated Fibrotic FBs(2).

Output:
An immune-dominant region is driven by neutrophils, monocytes/MDMs, and interstitial macrophages, with prominent capillary endothelium and AT2 epithelium. Additional components include NK/NKT cells, alveolar macrophages, lymphatic and venous endothelium, AT1 cells, and a mixed fibroblast compartment including proliferating and inflammatory subtypes.
""",

    "prostate": """
You are an expert genitourinary pathologist generating concise, ground-truth textual descriptions of prostate tissue cell populations for semantic retrieval.

### Inputs
1. **COMPOSITION_DATA**: A list of prostate cell states and their relative abundance.

### Output format (STRICT)
- Maximum **2 sentences total**.
- Plain text only. No lists or markdown.

### Guidelines
- **Sentence 1**: Identify the **dominant biological compartment or tissue program** and the **specific top 2–3 cell states** based on abundance.
- **Sentence 2**: Describe secondary or minor cell populations that define the specific biological context of this region.
- **Constraint**: Do not infer spatial geometry, density, or interaction patterns (e.g., do not use words like “clusters”, “interface”, “dense”, “sparse”, “contact”). Focus strictly on cellular presence.

### Example

Input:
COMPOSITION_DATA: Luminal_epithelia_MMP7_LTF(9), clone_6_PCA007(3), clone_3_PCA007(1), Epi_Immune_mix_CXCR4_CCL5(1), Fibro_stress_NR4A1_ADAMTS1(1).

Output:
An epithelial-dominant region is driven by luminal epithelial cells expressing MMP7 and LTF together with a prominent PCA007 tumor clone. Additional components include minor epithelial clones, a mixed epithelial–immune cell state, and a small stressed fibroblast population.
"""
}


SYSTEM_PROMPT_REPORT_2 = {
    "lung": """
You are an expert pulmonary pathologist describing spatial interaction patterns for semantic retrieval.

### Input
**INTERACTION_DATA**: Lineage-level and cell-type-level interaction pairs.

### Output format (STRICT)
- Maximum **2 sentences total**.
- Plain text only.

### Guidelines
- **Sentence 1**: Summarize the **dominant lineage-level interaction structure** (e.g., Immune–Immune, Immune–Endothelial).
- **Sentence 2**: Describe the **key cell-type contacts explicitly** using the provided interaction pairs.
- **Constraint**: Describe **only interactions**, not abundance, density, dominance, or tissue architecture. Do not infer spatial geometry.

### Example

Input:
INTERACTION_DATA: [DOMINANT_INTERACTIONS: Immune<->Immune, Endothelial<->Immune] [SPECIFIC_CONTACTS: Capillary<->Capillary, CD4+ T-cells<->Capillary, AT2<->Capillary]

Output:
The interaction network is dominated by immune–immune and immune–endothelial connectivity. At the cell-type level, capillaries interact with capillaries, CD4+ T cells interact with capillaries, and AT2 cells interact with capillaries.
""",

    "prostate": """
You are an expert genitourinary pathologist describing spatial interaction patterns for semantic retrieval.

### Input
**INTERACTION_DATA**: Cell-type-level interaction pairs.

### Output format (STRICT)
- Maximum **2 sentences total**.
- Plain text only.

### Guidelines
- **Sentence 1**: Summarize the **dominant interaction structure** at a high level (e.g., epithelial–epithelial, epithelial–stromal, epithelial–immune).
- **Sentence 2**: Describe the **key cell-type contacts explicitly** using the provided interaction pairs.
- **Constraint**: Describe **only interactions**, not abundance, density, dominance, or tissue architecture. Do not infer spatial geometry.

### Example

Input:
INTERACTION_DATA: [SPECIFIC_CONTACTS: Luminal_epithelia_MMP7_LTF<->Luminal_epithelia_MMP7_LTF, clone_6_PCA007<->clone_6_PCA007, Epi_Immune_mix_CXCR4_CCL5<->Luminal_epithelia_MMP7_LTF]

Output:
The interaction structure is dominated by epithelial–epithelial connectivity. Luminal epithelial cells interact with themselves, PCA007 tumor clones interact with the same clone, and mixed epithelial–immune cells interact with luminal epithelial cells.
"""
}

SYSTEM_PROMPT_REPORT_3 = {
    "lung": """
You are an expert pulmonary pathologist describing tissue density and global compartment weighting for semantic retrieval.

### Input
**SPATIAL_ARCHITECTURE**: Cellularity category, cell count, density percentile, and lineage rank.

### Output format (STRICT)
- Maximum **2 sentences total**.
- Plain text only.

### Guidelines
- **Sentence 1**: Describe **tissue cellularity and texture** using the provided density category and percentile.
- **Sentence 2**: Summarize **global compartment dominance** using lineage rank (e.g., immune-dominant, mesenchymal-biased).
- **Constraint**: Do not describe interactions, adjacency, or specific cell-type contacts. Do not infer biological function.

### Example

Input:
SPATIAL_ARCHITECTURE: [DENSE (Hypercellular): 102 cells, 0.88 percentile] [LINEAGE_RANK: Immune > Endothelial > Epithelial > Mesenchymal]

Output:
The tissue region is densely cellular, indicating a compact tissue texture. The cellular landscape is immune dominant, with secondary endothelial and epithelial representation and minimal mesenchymal contribution.
""",

    "prostate": """
You are an expert genitourinary pathologist describing tissue density and global cellular weighting for semantic retrieval.

### Input
**SPATIAL_ARCHITECTURE**: Cellularity category, cell count, density percentile, and cell-type rank.

### Output format (STRICT)
- Maximum **2 sentences total**.
- Plain text only.

### Guidelines
- **Sentence 1**: Describe **tissue cellularity and texture** using the provided density category and percentile.
- **Sentence 2**: Summarize **global cellular dominance** using the cell-type rank (e.g., epithelial-dominant, stromal-biased).
- **Constraint**: Do not describe interactions, adjacency, or specific cell-type contacts. Do not infer biological function.

### Example

Input:
SPATIAL_ARCHITECTURE: [MODERATE (Cellular): 15 cells, 0.46 percentile] [CELL_TYPE_RANK: Luminal_epithelia_MMP7_LTF > clone_6_PCA007 > clone_3_PCA007 > Epi_Immune_mix_CXCR4_CCL5 > Fibro_stress_NR4A1_ADAMTS1]

Output:
The tissue region shows moderate cellularity, consistent with an intermediate-density cellular composition. The cellular landscape is epithelial dominant with secondary tumor-associated epithelial states and minimal stromal or immune-associated representation.
"""
}


def read_legend(legend_csv_path):
    df = pd.read_csv(legend_csv_path)
    return df[df.columns[0]].astype(str).tolist()

# In prostate samples have slightly different cell types (not all in common) - this function creates a global legend
# and then remaps the match vectors of each sample
def build_global_legend_and_remap(input_dir, samples):
    # Build global legend
    global_legend = []
    global_index = {}

    def register(cell_type):
        if cell_type not in global_index:
            global_index[cell_type] = len(global_legend)
            global_legend.append(cell_type)

    legends = {}

    for sample_id in tqdm(samples, desc="Building global legend"):
        patch_dir = os.path.join(input_dir, sample_id)
        legend_path = os.path.join(patch_dir, "cell_type_legend.csv")
        raw_path = os.path.join(patch_dir, "patch_vectors_raw.pkl")

        if not (os.path.exists(legend_path) and os.path.exists(raw_path)):
            print(f"Skipping {sample_id} (missing legend or vectors)")
            continue

        legend = read_legend(legend_path)
        legends[sample_id] = legend

        for ct in legend:
            register(ct)

    # Remap patch vectors
    remapped_patch_vectors = {}

    for sample_id, legend in tqdm(legends.items(), desc="Remapping vectors"):
        patch_dir = os.path.join(input_dir, sample_id)
        raw_path = os.path.join(patch_dir, "patch_vectors_raw.pkl")

        # local idx -> global idx
        idx_map = {i: global_index[ct] for i, ct in enumerate(legend)}

        with open(raw_path, "rb") as f:
            raw_cell_vecs = pickle.load(f)

        for k, v in raw_cell_vecs.items():
            v = np.asarray(v, dtype=float).ravel()
            out = np.zeros(len(global_legend), dtype=float)

            # assign counts into global vector
            for i, count in enumerate(v[: len(legend)]):
                if count > 0:
                    out[idx_map[i]] += count

            clean_k = k.removesuffix(".png")
            new_key = f"{sample_id}__{clean_k}"
            remapped_patch_vectors[new_key] = out

    return global_legend, remapped_patch_vectors


def get_patch_content_summary(input_dir, samples):
    global_legend, master_raw_cell_vectors = build_global_legend_and_remap(
        input_dir, samples
    )
    
    print(f"Total Aggregated Patches: {len(master_raw_cell_vectors)}")

    patch_content_summary = {}

    for patch_name, vector in master_raw_cell_vectors.items():
        active_cells = []

        for idx, count in enumerate(vector):
            if count > 0:
                active_cells.append({
                    "cell_type": global_legend[idx],
                    "count": int(count)
                })

        active_cells.sort(key=lambda x: x["count"], reverse=True)

        patch_content_summary[
            patch_name.removeprefix("he_size_224_microns_patches_")
        ] = active_cells

    return patch_content_summary

def generate_multilevel_report_inputs(density_lookup, df_cells_patch, patch_name, has_lineage=False, k_interactions=2):
    # Lineage ranking and cell type composition
    lineage_or_cell_type_rank = lineage_or_cell_type_ranking(df_cells_patch, has_lineage=has_lineage)
    
    cell_type_composition = get_cell_type_composition(df_cells_patch, has_lineage=has_lineage)
    
    # Microenvironment interactions
    interaction_summary_cell_type = get_interaction_summary(
        df_cells_patch, k_interactions=k_interactions, x_col="x_global", y_col="y_global", type_col="cell_type", radius=50
    )
    spec_cont = ', '.join(interaction_summary_cell_type) if interaction_summary_cell_type else "None (Isolated)"
    
    # Density metrics
    density_info = get_density_description(density_lookup, patch_name)

    # level 1: composition Only
    level_1 = (
        f"COMPOSITION_DATA: {cell_type_composition}."
    )

    # level 2: microenvironment interactions
    if has_lineage:
        interaction_summary_lineage = get_interaction_summary(
            df_cells_patch, k_interactions=2, x_col="x_global", y_col="y_global", type_col="lineage", radius=50
        )

        lineage_int = ', '.join(interaction_summary_lineage) if interaction_summary_lineage else "None (Isolated)"
    
        level_2 = (
            f"MICROENVIRONMENT_DATA: "
            f"[DOMINANT_INTERACTIONS: {lineage_int}] "
            f"[SPECIFIC_CONTACTS: {spec_cont}].\n"
        )
    else:
        level_2 = (
            f"MICROENVIRONMENT_DATA: "
            f"[SPECIFIC_CONTACTS: {spec_cont}].\n"
        )

    # level 3: density information
    level_3 = (
        f"SPATIAL_ARCHITECTURE: "
        f"{density_info}"
        f"[{'LINEAGE_RANK:' if has_lineage else 'CELL_TYPE_RANK:'} {lineage_or_cell_type_rank}]"
    )

    return level_1, level_2, level_3


def generate_reports_per_patch(all_samples_metadata, output_dir, has_lineage, k_interactions=2):
    patch_cell_counts = all_samples_metadata.groupby('patch_name').size()

    patch_density_percentiles = patch_cell_counts.rank(pct=True, method='min')

    density_lookup = pd.DataFrame({
        'count': patch_cell_counts,
        'percentile': patch_density_percentiles.round(3)
    }).to_dict(orient='index')
    
    patch_description = {}

    grouped_metadata = all_samples_metadata.groupby('patch_name')

    target_patches = set(all_samples_metadata['patch_name'])

    print(f"Generating descriptions for {len(target_patches)} patches...")

    for patch_name, df_cells_patch in tqdm(grouped_metadata, total=len(grouped_metadata)):
        
        if patch_name not in target_patches:
            continue

        level_1, level_2, level_3 = generate_multilevel_report_inputs(density_lookup, df_cells_patch, patch_name, has_lineage, k_interactions=k_interactions)

        patch_description[patch_name] = {
            "level_1" : level_1,
            "level_2": level_2,
            "level_3": level_3
        }

    with open(f"{output_dir}/reports_per_patch.pkl", "wb") as f:
        pickle.dump(patch_description, f)
    
    return patch_description

def split_by_tissue(patch_description):
    by_tissue = defaultdict(dict)

    for patch_id, data in patch_description.items():
        tissue = patch_id.split("__", 1)[0]
        by_tissue[tissue][patch_id] = data

    return dict(by_tissue)


def wait_for_batch_completion(batch_id, poll_interval=60):
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status

        print(f"Batch {batch_id} status: {status}")

        if status == "completed":
            return batch

        time.sleep(poll_interval)


def batch_send_openai_requests_per_tissue(
    patch_description,
    prompts,
    levels=("level_1",),
    output_dir=".",
):
    valid_levels = {"level_1", "level_2", "level_3"}
    levels = tuple(levels)

    invalid = set(levels) - valid_levels
    if invalid:
        raise ValueError(f"Invalid levels requested: {invalid}")

    os.makedirs(output_dir, exist_ok=True)

    patches_by_tissue = split_by_tissue(patch_description)

    for tissue, tissue_patches in patches_by_tissue.items():
        print(f"TISSUE: {tissue}")
        batch_output_dir = os.path.join(output_dir, "batch_requests_jsonl")
        os.makedirs(batch_output_dir, exist_ok=True)

        for level in levels:
            samples = [
                (patch_id, data[level])
                for patch_id, data in tissue_patches.items()
                if level in data
            ]

            if not samples:
                continue

            jsonl_file = f"{batch_output_dir}/{tissue}_{level}_batch.jsonl"

            with open(jsonl_file, "w") as f:
                for patch_id, patch_summary in samples:
                    request = {
                        "custom_id": patch_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4.1-nano",
                            "temperature": 0.6,
                            "messages": [
                                {"role": "system", "content": prompts[level]},
                                {"role": "user", "content": patch_summary},
                            ],
                        },
                    }
                    f.write(json.dumps(request) + "\n")

            batch_file = client.files.create(
                file=open(jsonl_file, "rb"),
                purpose="batch",
            )

            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            print(
                f"{tissue} | {level} | "
                f"{len(samples)} patches → Batch ID: {batch.id}"
            )

            wait_for_batch_completion(batch.id)


def batch_openai_by_level(patch_samples, prompts, levels=("level_1",), output_dir=".", output_folder="batch_requests_jsonl"):
    
    for level in levels:

        print(level)
        batch_output_dir = os.path.join(output_dir, output_folder)
        os.makedirs(batch_output_dir, exist_ok=True)

        jsonl_file = f"{batch_output_dir}/{level}_batch.jsonl"
        with open(jsonl_file, "w") as f:
            for patch_id, patch_summary in patch_samples.items():
                
                patch_summary = patch_summary[level]
                request = {
                    "custom_id": patch_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4.1-nano",
                        "temperature": 0.6,
                        "messages": [
                            {"role": "system", "content": prompts[level]},
                            {"role": "user", "content": patch_summary},
                        ],
                    },
                }

                f.write(json.dumps(request) + "\n")

            
        # Upload batch input
        batch_file = client.files.create(
            file=open(jsonl_file, "rb"),
            purpose="batch"
        )

        # Create batch job
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print("Batch ID:", batch.id)

def embed_text_openai(openai_descriptions_dict,level_text, batch_size=100, output_dir=".", output_folder="text-descriptions-all-samples-levels"):
    keys = list(openai_descriptions_dict.keys())
    texts = [openai_descriptions_dict[k]['text_descr'] for k in keys]

    text_embeddings = {}

    # Batch Processing Loop
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch_keys = keys[i : i + batch_size]
        
        response = client.embeddings.create(
            input=batch_texts,
            model="text-embedding-3-small"
        )
        
        for j, data in enumerate(response.data):
            emb_vector = np.array(data.embedding, dtype=np.float32)
            text_embeddings[batch_keys[j].replace(".png", "")] = emb_vector
    
    os.makedirs(f"{output_dir}/{output_folder}", exist_ok=True)
    path_save = f"{output_dir}/{output_folder}/openai_text_embeddings_{level_text}.pkl"
    print(path_save)
    with open(path_save, "wb") as f:
        pickle.dump(text_embeddings, f)
    
    return text_embeddings


def get_results_batch_in_dict(batch_id, patch_description, level_text):
    # Normalize to list
    if not isinstance(batch_id, (list, tuple)):
        batch_id = [batch_id]

    openai_descriptions_dict = {}

    for bid in batch_id:
        batch = client.batches.retrieve(bid)
        result_file_id = batch.output_file_id

        result_file = client.files.content(result_file_id)
        raw_bytes = result_file.read()
        text = raw_bytes.decode("utf-8")
        results = text.splitlines()

        for line in results:
            record = json.loads(line)

            patch_id = record["custom_id"]
            try:
                response_text = record["response"]["body"]["choices"][0]["message"]["content"]
            
            except Exception as e:
                response_text = "No interaction found."
                print(response_text)
                
            openai_descriptions_dict[patch_id] = {
                "patch_summary": patch_description[patch_id][level_text],
                "text_descr": response_text,
            }

    return openai_descriptions_dict


def get_openai_dict_embeddings(batch_id, patch_description, level_text, output_dir=".", output_folder="text-descriptions-all-samples-levels"):

    openai_descriptions_dict = get_results_batch_in_dict(
        batch_id,
        patch_description,
        level_text,
    )

    os.makedirs(
        f"{output_dir}/{output_folder}",
        exist_ok=True,
    )

    with open(
        f"{output_dir}/{output_folder}"
        f"openai_descriptions_dict_{level_text}.pkl",
        "wb",
    ) as f:
        pickle.dump(openai_descriptions_dict, f)

    text_embeddings = embed_text_openai(
        openai_descriptions_dict,
        level_text,
        batch_size=100,
        output_dir=output_dir,
        output_folder=output_folder
    )

    return openai_descriptions_dict, text_embeddings

def concatenate_patch_descriptions(level_a, level_b, multi_level_text, output_dir, output_folder="text-descriptions-all-samples-levels"):
    merged = {}

    shared_patches = level_a.keys() & level_b.keys()

    for patch_id in shared_patches:
        merged[patch_id] = {}

        shared_keys = level_a[patch_id].keys() & level_b[patch_id].keys()

        for k in shared_keys:
            val_a = level_a[patch_id][k]
            val_b = level_b[patch_id][k]

            if isinstance(val_a, str) and isinstance(val_b, str):
                merged[patch_id][k] = f"{val_a} {val_b}".strip()
            else:
                raise TypeError(
                    f"Non-string value encountered for patch '{patch_id}', key '{k}'"
                )
        
    os.makedirs(
        f"{output_dir}/{output_folder}",
        exist_ok=True,
    )

    with open(
        f"{output_dir}/{output_folder}/"
        f"openai_descriptions_dict_{multi_level_text}.pkl",
        "wb",
    ) as f:
        pickle.dump(merged, f)


    return merged


def get_openai_dict_embeddings_concatenated(level_a, level_b, multi_level_text, output_dir=".", output_folder="text-descriptions-all-samples-levels"):
    
    openai_descriptions_dict_multi_levels = concatenate_patch_descriptions(level_a, level_b, multi_level_text, output_dir, output_folder)

    text_embeddings_multi_levels = embed_text_openai(openai_descriptions_dict_multi_levels, multi_level_text, batch_size=100, output_dir=output_dir, output_folder=output_folder)

    return openai_descriptions_dict_multi_levels, text_embeddings_multi_levels


if __name__ == "__main__":
    lung_dir = "/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/lung"
    prostate_dir = "/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/prostate"

    samples_lung = load_samples(lung_dir)
    patch_content_summary_lung = get_patch_content_summary(lung_dir, samples_lung)
    all_samples_metadata_lung = pd.read_csv(f"/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/lung/all_samples_metadata.csv") 

    samples_prostate = load_samples(prostate_dir)
    patch_content_summary_prostate = get_patch_content_summary(prostate_dir, samples_prostate)
    all_samples_metadata_prostate = pd.read_csv(f"/Volumes/HD_rafael/DPhil/MICCAI_2026/outputs/prostate/all_samples_metadata.csv") 

    # Generate reports (all levels) for each patch
    patch_description_lung = generate_reports_per_patch(all_samples_metadata_lung, lung_dir, has_lineage=True)
    patch_description_prostate = generate_reports_per_patch(all_samples_metadata_prostate, prostate_dir)

    prompts_lung = {
        "level_1": SYSTEM_PROMPT_REPORT_1["lung"],
        "level_2": SYSTEM_PROMPT_REPORT_2["lung"],
        "level_3": SYSTEM_PROMPT_REPORT_3["lung"],
    }

    prompts_prostate = {
        "level_1": SYSTEM_PROMPT_REPORT_1["prostate"],
        "level_2": SYSTEM_PROMPT_REPORT_2["prostate"],
        "level_3": SYSTEM_PROMPT_REPORT_3["prostate"],
    }

    # Submission Section - commented out - assuming you already sent the batches
    # batch_openai_by_level(patch_description_lung, prompts_lung, levels=("level_1", "level_2", "level_3",), output_dir=lung_dir)

    # # for prostate I need to separate the batch requests per tissue (too many samples - passing the limit in Batch API)
    # batch_send_openai_requests_per_tissue(
    #     patch_description=patch_description_prostate,
    #     prompts=prompts_prostate,
    #     levels=("level_1", "level_2", "level_3",),
    #     output_dir=prostate_dir,
    # )

    lung_openai_descriptions_dict_level1, lung_text_embeddings_level1 = get_openai_dict_embeddings("batch_69779437b4ac8190b7a589b962d97b57", patch_description_lung, "level_1", lung_dir)
    lung_openai_descriptions_dict_level2, lung_text_embeddings_level2 = get_openai_dict_embeddings("batch_6977a0b5e6c481909448bbccc377ec62",patch_description_lung, "level_2", lung_dir)
    lung_openai_descriptions_dict_level3, lung_text_embeddings_level3 = get_openai_dict_embeddings("batch_6977b754fdf88190b9042e0da7455a2a", patch_description_lung, "level_3", lung_dir)

    prostate_openai_descriptions_dict_level1, prostate_text_embeddings_level1 = get_openai_dict_embeddings(["batch_69891ab6af508190bd4af1d48b5bf993", "batch_69892566d528819094b868126622e2f0", "batch_69892c48a1008190a4754aacbab2a51a", "batch_69892d01646481909cbf828494e8f208"], patch_description_prostate, "level_1", prostate_dir)
    prostate_openai_descriptions_dict_level2, prostate_text_embeddings_level2 = get_openai_dict_embeddings(["batch_6989b5af82508190811a5ad5fa72cdad", "batch_6989b001683c81909b668558e682d7f1", "batch_6989b181aee88190bafab66a2bd1054e", "batch_6989e3ade8388190a52ae6f6667a63aa"], patch_description_prostate, "level_2", prostate_dir)
    prostate_openai_descriptions_dict_level3, prostate_text_embeddings_level3 = get_openai_dict_embeddings(["batch_6989f06c59b48190a763a7489f98061b", "batch_6989f0a826688190b5654c1ba7965ad0", "batch_6989f9428b7481909b4195483e2b77ca", "batch_6989f9acd8908190998fe7712cc28b00"], patch_description_prostate, "level_3", prostate_dir)

    # Concatenate reports (and then embed) to create multi level reports
    lung_openai_descriptions_dict_level1_2, lung_text_embeddings_level1_2 = get_openai_dict_embeddings_concatenated(lung_openai_descriptions_dict_level1, lung_openai_descriptions_dict_level2, "level_1_2", output_dir=lung_dir)
    lung_openai_descriptions_dict_level1_3, lung_text_embeddings_level1_3 = get_openai_dict_embeddings_concatenated(lung_openai_descriptions_dict_level1, lung_openai_descriptions_dict_level3, "level_1_3", output_dir=lung_dir)
    lung_openai_descriptions_dict_level1_2_3, lung_text_embeddings_level1_2_3 = get_openai_dict_embeddings_concatenated(lung_openai_descriptions_dict_level1_2, lung_openai_descriptions_dict_level3, "level_1_2_3", output_dir=lung_dir)
    lung_openai_descriptions_dict_level3_1_2, lung_text_embeddings_level3_1_2 =  get_openai_dict_embeddings_concatenated(lung_openai_descriptions_dict_level3, lung_openai_descriptions_dict_level1_2, "level_3_1_2", output_dir=lung_dir)

    prostate_openai_descriptions_dict_level1_2, prostate_text_embeddings_level1_2 = get_openai_dict_embeddings_concatenated(prostate_openai_descriptions_dict_level1, prostate_openai_descriptions_dict_level2, "level_1_2", output_dir=prostate_dir)
    prostate_openai_descriptions_dict_level1_3, prostate_text_embeddings_level1_3 = get_openai_dict_embeddings_concatenated(prostate_openai_descriptions_dict_level1, prostate_openai_descriptions_dict_level3, "level_1_3", output_dir=prostate_dir)
    prostate_openai_descriptions_dict_level1_2_3, prostate_text_embeddings_level1_2_3 = get_openai_dict_embeddings_concatenated(prostate_openai_descriptions_dict_level1_2, prostate_openai_descriptions_dict_level3, "level_1_2_3", output_dir=prostate_dir)
    prostate_openai_descriptions_dict_level3_1_2, prostate_text_embeddings_level3_1_2 =  get_openai_dict_embeddings_concatenated(prostate_openai_descriptions_dict_level3, prostate_openai_descriptions_dict_level1_2, "level_3_1_2", output_dir=prostate_dir)



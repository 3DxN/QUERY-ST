# QUERY-ST
QUERY-ST is a research framework that maps three views of the same tissue region into a shared embedding space:

- H&E image patches
- spatial transcriptomics (ST) features
- structured natural-language descriptions derived from ST

The shared space supports cross-modal retrieval, including searching measured ST regions or histology patches with a text query. The framework is designed for exploratory spatial pathology research rather than clinical use.

## How the pipeline works

1. Align H&E images with ST coordinates and create matched tissue patches.
2. Build patch-level gene-expression and cell-type vectors.
3. Extract H&E features with UNI2-h.
4. Generate structured text descriptions and text embeddings from the ST summaries.
5. Train lightweight projection heads to align image, RNA, and text representations.
6. Use the aligned space for cross-modal retrieval and natural-language querying.

## Repository structure

| Folder | Purpose |
| --- | --- |
| `patches_creation_img_rna/` | Creates aligned H&E patches and patch-level RNA and cell-type inputs for the Lung and Prostate datasets. |
| `image_uni_embeddings/` | Extracts UNI2-h embeddings from the H&E patches. |
| `synthetic_text_embeddings/` | Generates structured ST-derived descriptions and their text embeddings. |
| `alignment/` | Trains the final models and contains the interactive text-querying notebook. |
| `ablations/` | Contains component ablations, query-refinement analyses, and plotting notebooks. |
| `benchmark/` | Contains scalability experiments and the HESCAPE comparison. |

## Core files

| File | Purpose |
| --- | --- |
| `trimodal_encoder.py` | Defines the image, RNA, and text projection heads and the shared latent space. |
| `training_functions.py` | Loads aligned inputs and provides model-training utilities. |
| `retrieval_functions.py` | Provides cross-modal retrieval, text-querying, and visualization utilities. |
| `common_functions.py` | Shared preprocessing, alignment, splitting, and evaluation helpers. |
| `ablations_functions.py` | Shared utilities used by the ablation experiments. |
| `environment.yml` | Conda environment containing the project dependencies. |

## Main entry points

- Train the final Lung and Prostate models with `python -u alignment/final_models.py`.
- Explore interacting with a trained model in `alignment/interactive_querying.ipynb`.
- See `benchmark/hescape_comparison/README.md` for the HESCAPE benchmark instructions.

## Usage

Run every command from the repository root.

### Input layout

`load_trimodal_inputs` in `training_functions.py` expects each dataset directory to look like this:

```
<DATASET_DIR>/                                  # e.g. .../outputs/lung
├── metadata_lung.csv                           # lung only: `sample` -> `patient` map, used by LOPO
├── all_samples_metadata.csv                    # per-cell table, written by step 2
├── he_size_112_microns_patches_<TISSUE>/       # one directory per sample
│   ├── patch_x<X>_y<Y>.png                     # 224x224 H&E crops
│   ├── uni2h_<TISSUE>.pkl                      # step 3 output
│   ├── patch_vectors.pkl                       # log1p cell-type counts
│   ├── patch_vectors_raw.pkl
│   ├── patch_genes_full.pkl                    # normalised gene means
│   ├── patch_genes_raw_counts.pkl              # raw sums, used by the HESCAPE benchmark
│   └── cell_type_legend.csv
└── text-descriptions-all-samples-levels/
    ├── openai_text_embeddings_<LEVEL>.pkl      # LEVEL: level_1, level_2, level_3, level_1_2,
    └── openai_descriptions_dict_<LEVEL>.pkl    #        level_1_3, level_1_2_3, level_3_1_2
```

Two conventions fail silently when broken:

- The sample directory name is split on its last underscore to derive `<TISSUE>`, which must match
  the `uni2h_<TISSUE>.pkl` filename inside it. Renaming a directory breaks the join.
- A sample directory missing `uni2h_*.pkl` or `patch_vectors.pkl` is skipped without a warning, so
  the only symptom is a lower patch count later.

### 1. Environment and credentials

```bash
conda env create -f environment.yml
conda activate env_xenium

huggingface-cli login            # MahmoodLab/UNI2-h is a gated repository
export OPENAI_API_KEY="sk-..."   # text generation and live text-query embedding
```

### 2. Build patches and RNA vectors

Lung first needs the Seurat object exported to matrix-market form:

```bash
Rscript patches_creation_img_rna/lung_extracting_data_from_seurat.R
```

Both scripts read their paths from module-level constants, so edit those before running:

```bash
# lung_inputs_creation.py: INPUT_DIR, IMAGES_FOLDER, OUTPUT_DIR
python patches_creation_img_rna/lung_inputs_creation.py

# prostate_xenium_inputs_creation.py: INPUT_ROOT, IMAGES_DIR, OUTPUT_DIR
python patches_creation_img_rna/prostate_xenium_inputs_creation.py
```

Lung uses 112 micron patches and prostate 224 micron patches; both are resampled to 224 pixels.
Tiles whose mean intensity exceeds 235 are dropped as background.

### 3. Extract UNI2-h image embeddings

```bash
# Edit the sample lists, input_dir and output_dir in uni_processing_dict first.
# CUDA_VISIBLE_DEVICES is also pinned inside the script.
mkdir -p uni_lung_output uni_prostate_output
python image_uni_embeddings/generate_uni2h_embeddings.py
```

Move each output pickle into its sample directory as `uni2h_<TISSUE>.pkl`.

### 4. Generate descriptions and text embeddings

```bash
export OPENAI_API_KEY="sk-..."
python synthetic_text_embeddings/generate_synthetic_text.py
```

Descriptions are written by `gpt-4.1-nano` at three levels (composition, interaction, density) and
embedded with `text-embedding-3-small`. As committed, the script only retrieves results for the
OpenAI batch jobs of the original run: the submission calls are commented out and the batch IDs are
hardcoded. To process new data, re-enable `batch_openai_by_level` (lung) or
`batch_send_openai_requests_per_tissue` (prostate) and substitute the new batch IDs.

### 5. Train the final models

```bash
python -u alignment/final_models.py
```

Trains five Lung leave-one-patient-out configurations and four Prostate spatial 10-fold
configurations, covering trimodal and bimodal variants of both the fused and the gene-only RNA
representation. Checkpoints go to the configured models directory; results and model pickles go to
`<DATASET_DIR>/lopo_objects/` and `<DATASET_DIR>/cv_objects/`.

### 6. Ablations

```bash
# All three ablations on both datasets
python -u ablations/ablations.py \
  --lung-dir     /path/to/outputs/lung \
  --prostate-dir /path/to/outputs/prostate \
  --models-dir   /path/to/models

# Only the loss-weight sweep on Lung (24 models: 6 lambda_rna_text x 4 lambda_img_text)
python -u ablations/ablations.py \
  --lung-dir /path/to/outputs/lung \
  --models-dir /path/to/models \
  --datasets lung --ablations 2

# Faster smoke run with the split-integrity assertions disabled
python -u ablations/ablations.py --datasets lung --ablations 1 --skip-checks
```

`--ablations 1` compares transcriptomic representations (cell-type, gene, fused), `2` sweeps the
two text loss weights, and `3` compares report-granularity levels.

### 7. Benchmarks

```bash
# Scalability: wall-clock time and peak RSS/GPU memory across dataset and batch sizes
python -u benchmark/lung_scalability_benchmark.py \
  --lung-dir /path/to/outputs/lung \
  --output-dir benchmark/outputs/lung_scalability \
  --sizes 1000,2500,5000,10000,20000,full \
  --batch-sizes 32,64,128,256 \
  --epochs 10 --repeats 3 --device cuda

# HESCAPE configuration smoke test (cheap, needs no data)
conda env create -f benchmark/hescape_comparison/environment.yml
conda activate hescape_comparison
python benchmark/hescape_comparison/smoke_test_uni2h_generic.py \
  --gene-dim 343 --embed-dim 128 --img-proj mlp --gene-proj linear \
  --run-forward --device cpu
```

The full HESCAPE LOPO comparison is documented in `benchmark/hescape_comparison/README.md`.

### 8. Interactive querying

```bash
export OPENAI_API_KEY="sk-..."
jupyter lab alignment/interactive_querying.ipynb
```

The notebook loads a checkpoint with `TriModalEncoder.load(...)` and drives `search_by_text`,
`visualize_text_search` and `plot_spatial_heatmap` from `retrieval_functions.py`.

## Known limitations

- **Hardcoded paths.** The patch-creation, UNI2-h and text-generation scripts, and
  `alignment/final_models.py`, read absolute paths from module constants and accept no arguments;
  the committed values point at the original authors' machines. Only `ablations/ablations.py` and
  `benchmark/lung_scalability_benchmark.py` take `--*-dir` options.
- **`environment.yml` is a macOS/arm64 lockfile.** It pins osx-arm64 builds and a CPU-only
  `torch`, so it will not solve on the Linux cluster the default paths point to. It also pins
  Python 3.10 even though the committed bytecode was produced by Python 3.13. Treat it as a record
  of the original environment rather than a portable specification.
- **Step 4 is not re-runnable as committed.** See the note in that section.
- **Training is all or nothing.** `final_models.py` runs all nine configurations in sequence with
  no way to select one and no resume or skip-if-present logic.
- **Missing inputs are skipped silently** during both patch creation and input loading, rather than
  raising.
- **The OpenAI embedding model name is duplicated** in `synthetic_text_embeddings/generate_synthetic_text.py`
  and `retrieval_functions.py`. Query embeddings must be produced by the same model used at
  training time; because the dimensions match either way, a mismatch degrades retrieval silently
  instead of failing.
- **No test suite or packaging.** Correctness relies on the runtime assertions in
  `run_alignment_tests`, which `--skip-checks` disables.

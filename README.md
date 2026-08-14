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

- Train the final Lung and Prostate models with `alignment/final_models.py`.
- Explore interacting with a trained model in `alignment/interactive_querying.ipynb`.
- See `benchmark/hescape_comparison/README.md` for the HESCAPE benchmark instructions.

Dataset paths are configured inside the data-preparation and experiment scripts and should be updated for the local environment. An OpenAI API key is required for text generation and live text-query embedding.

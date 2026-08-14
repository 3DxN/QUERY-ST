# HESCAPE Lung LOPO Comparison

This folder reproduces the HESCAPE comparison reported in the QUERY-ST paper. It runs the official HESCAPE model path on the same aligned Lung patches, patient-level LOPO splits, and full held-out retrieval candidate sets used for QUERY-ST.

## Contents

- `official_hescape_lung_lopo.py`: Lung LOPO training and evaluation runner.
- `smoke_test_uni2h_generic.py`: configuration check for the UNI2-h and generic gene-encoder branches.
- `environment.yml`: server Conda environment.
- `external/hescape/`: vendored HESCAPE source, extended with cached UNI2-h support.

## Final Configuration

- Cached UNI2-h patch embeddings.
- HESCAPE generic gene encoder with summed raw patch counts.
- MLP image projection and linear gene projection.
- CLIP loss with a 128-dimensional retrieval space.
- Batch size 256, learning rate `1e-4`, weight decay `1e-4`.
- Maximum 50 epochs with patience 10.
- Patient-level LOPO using the full trimodal patch universe.

Required Lung inputs are `metadata_lung.csv` and, within every patch folder, `uni_embeddings.pkl` and `patch_genes_raw_counts.pkl`.

## Environment

```bash
cd benchmark/hescape_comparison
conda env create -f environment.yml
conda activate hescape_comparison
cd ../..
```

## Run

```bash
python -u benchmark/hescape_comparison/official_hescape_lung_lopo.py \
  --lung-dir /path/to/lung \
  --output-dir /path/to/lung/hescape_comparison \
  --suffix hescape_lung_lopo \
  --gene-variant raw_counts \
  --alignment-key-space full_trimodal \
  --text-level level_1_2_3 \
  --image-encoder uni2h \
  --precomputed-image-embeddings \
  --gene-encoder generic \
  --embed-dim 128 \
  --loss CLIP \
  --img-proj mlp \
  --gene-proj linear \
  --temperature 0.07 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --batch-size 256 \
  --max-epochs 50 \
  --patience 10 \
  --val-frac 0.1 \
  --split-seed 42 \
  --seed 42 \
  --accelerator gpu \
  --devices 1 \
  --skip-completed-patients
```

The runner writes per-patient results immediately and also produces aggregate CSV and pickle outputs after all requested patients finish.

HESCAPE repo: <https://github.com/peng-lab/hescape>

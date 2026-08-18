# REST

Official implementation of **REST: Reliability-Guided Spatial Transcriptomics Supervision for Pathology Foundation Models**.

REST adapts H-optimus-1 with Spot-Reliability Contrastive Alignment (SRCA) and Biology-Informed Program Reconstruction (BIPR). Complementary Prediction Fusion (CPF) combines predictions from the original and REST-adapted encoders.

## Setup

```bash
conda create -n rest python=3.10 -y
conda activate rest
pip install -e .
```

Access to [HEST-1k](https://huggingface.co/datasets/MahmoodLab/hest) and [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) must be accepted on Hugging Face before use.

## Data processing

Download selected HEST-1k samples. Use `--all` instead of `--ids` to download the full collection.

```bash
hf auth login
python download_hest.py --output data/hest --ids TENX95 TENX96 TENX99
```

Create paired patch-expression indices. By default, human Visium and Visium HD samples receive a sample-level 90/10 train/validation split; pass `--split-csv` to use a fixed split.

```bash
python prepare_data.py \
  --hest-root data/hest \
  --output data/patch_index.csv
```

Build the biological program bank from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp) Hallmark and KEGG GMT files and the [PanglaoDB](https://panglaodb.se/markers.html) marker table. The paper configuration contains 50 Hallmark, 186 KEGG, and 151 canonical cell-type programs.

```bash
python prepare_programs.py \
  --hallmark-gmt data/programs/hallmark.gmt \
  --kegg-gmt data/programs/kegg.gmt \
  --panglao-tsv data/programs/PanglaoDB_markers_27_Mar_2020.tsv \
  --output-prefix data/programs/rest_programs
```

The HEST files follow the official `patches/<sample>.h5` and `st/<sample>.h5ad` layout. See the [HEST repository](https://github.com/mahmoodlab/HEST) for dataset details.
The bundled Ensembl-to-symbol table is used automatically for assays that do not store gene symbols directly.

## Training

Single GPU:

```bash
python train.py \
  --index data/patch_index.csv \
  --patch-root data/hest/patches \
  --st-root data/hest/st \
  --program-prefix data/programs/rest_programs \
  --output outputs/rest
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --index data/patch_index.csv \
  --patch-root data/hest/patches \
  --st-root data/hest/st \
  --program-prefix data/programs/rest_programs \
  --output outputs/rest
```

The adapted pathology encoder is saved to `outputs/rest/best/encoder.pt`.

## Inference

Extract original and REST-adapted patch embeddings:

```bash
python inference.py extract \
  --input data/hest/patches/TENX95.h5 \
  --output outputs/TENX95_original.npy

python inference.py extract \
  --input data/hest/patches/TENX95.h5 \
  --checkpoint outputs/rest/best/encoder.pt \
  --output outputs/TENX95_rest.npy
```

Fit the same downstream predictor separately on the two embedding sets, then apply CPF to their predictions:

```bash
python inference.py fuse \
  --raw outputs/predictions_original.npy \
  --adapted outputs/predictions_rest.npy \
  --alpha 0.75 \
  --output outputs/predictions_cpf.npy
```

Select `alpha` on validation data and use the same value for all test samples and targets.

## Data license

HEST-1k is distributed separately under its own license. Users are responsible for complying with the licenses of HEST-1k, H-optimus-1, MSigDB, and PanglaoDB.

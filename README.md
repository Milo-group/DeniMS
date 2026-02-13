# DeniMS - *De-novo* Identification of Mass Spectra

This repository accompanies the paper: [Paper_link]

![Overview Figure](Overview%20Figure.jpeg)

## Installation

To install the necessary environment to run ms2mol, run the following commands:

```
conda create -n ms2mol python=3.9
conda activate ms2mol

git clone https://github.com/yonhar/ms2mol.git
cd ms2mol

pip install -r requirements.txt
conda install -c conda-forge rdkit=2023.03.2 graph-tool=2.45
```

## Quick Start

### Preprocessing

We process high-resolution MS datasets using a standardized preparation pipeline that filters invalid entries, annotates fragment-ion formulas, and associates each spectrum with the relevant metadata. The initial processing steps follow our previous work, available in the following repository: https://github.com/Nir-Cohen-2003/HRMS_utils.

The final preprocessing scripts used in this project are provided in the Preprocessing/ folder.

A fully integrated pipeline with step-by-step explanations *will be added soon*.

In the meantime, the preprocessed FragHub Parquet file and the corresponding molecular graph dictionary can be downloaded from [Zenodo](https://zenodo.org/records/18539020).


<!--
#### Process FragHub Dataset

The preprocessing stage filters data, generates molecular graph representations, and creates train/validation/test splits.

```bash
cd Preprocessing
python data_processing.py \
    -input_parquet fraghub/fraghub.parquet \
    -generate_graph_dict \
    -split_type random 
```

This will generate:
- `fraghub/fraghub_filtered.parquet`: Filtered MS spectra data
- `fraghub/fraghub_smiles_canonical.txt`: Canonical SMILES strings
- `fraghub/smiles_dict_fraghub.pt`: A dictionary that contains for each SMILES in the dataset its molecular graph, FP representation, and corresponding indices in fraghub_filtered.parquet
- `fraghub/splits_fraghub_random.pkl`: Train/val/test splits (if split_type is specified)

**Key Arguments:**
- `-input_parquet`: Path to input parquet file
- `-generate_graph_dict`: Generate molecular graph dictionary
- `-split_type`: Split type (`random` or `MCES`)
- `-val_fraction`: Validation set fraction (default: 0.05)
- `-test_fraction`: Test set fraction (default: 0.05)

It is also possible to download the preprocessed filtered dataset and the molecular graph dictionary from [Zenodo](https://zenodo.org/records/18539020)
-->

### Stage 1: Contrastive Training

Train the contrastive model to learn aligned representations between MS spectra and molecular structures. A trained version can also be downloaded from [Zenodo](https://zenodo.org/records/18539020).

#### Basic Training

```bash
python train.py \
    -mode contrastive \
    -data_path Preprocessing/fraghub/fraghub_filtered.parquet \
    -smiles_path Preprocessing/fraghub/smiles_dict_fraghub.pt \
    -split_path Preprocessing/fraghub/splits_fraghub_random.pkl \
    -trainable_temperature \
    -comment "contrastive_fraghub"
```

#### Training Modes

- **`contrastive`**: InfoNCE contrastive loss (default)
- **`fp`**: MSE loss to molecular fingerprints
- **`mixed`**: Combined contrastive and fingerprint losses

#### Key Arguments

- `-mode`: Training mode (`contrastive`, `fp`, or `mixed`)

**Training:**
- `-batch_size`: Batch size (default: 512)
- `-epochs`: Number of training epochs (default: 500)
- `-lr`: Learning rate (default: 4e-4)
- `-trainable_temperature`: Make temperature trainable

**Data:**
- `-data_path`: Path to processed parquet file
- `-smiles_path`: Path to SMILES graph dictionary
- `-split_path`: Path to predefined splits (optional, uses random split if not provided)

#### Evaluation

Evaluate a trained contrastive model:

```bash
python run_evaluation.py \
    -cp_name [cp_name] \
    -mode contrastive \
    -data_path Preprocessing/fraghub/fraghub_filtered.parquet \
    -smiles_path Preprocessing/fraghub/smiles_dict_fraghub.pt \
    -split_path Preprocessing/fraghub/splits_fraghub_random.pkl \
    -total_samples 512
```

**Evaluation Output:**

The evaluation computes metrics between MS spectrum embeddings and molecular graph embeddings:

- **Pairwise matching accuracy**: Measures how often the model correctly matches MS spectra to their corresponding molecular structures (bidirectional top-1 accuracy)
- **Mean Absolute Error (MAE)**: Average L1 distance between MS and molecular embeddings in the shared embedding space
- **Cosine Similarity**: Average cosine similarity between aligned MS and molecular embedding pairs

Additionally, a **t-SNE visualization** is generated (unless `-no_plot` is specified) showing the 2D projection of both MS and molecular embeddings, with corresponding pairs highlighted. The plot is saved to `analysis_outputs/` with a timestamp.

### Diffusion Model Training

The diffusion model generates molecular graphs conditioned on MS embeddings from the contrastive model. Stages 2-3 consist of pretraining, finetuning, inference, and post-analysis.
This diffusion stage is adapted from [DiGress](https://github.com/cvignac/DiGress).

#### Stage 2: Graph2Mol Pretraining

First, pretrain the diffusion model on molecular graphs without MS conditioning (graph2mol). This establishes a strong baseline for unconditional molecular generation:

```bash
cd MS_diffusion/src
python main.py \
    conditioning.embeddings_type=mol2emb \
    conditioning.embedding_model_path=[contrastive_cp_path]
```

This trains the base diffusion model to generate molecules unconditionally. By default, it uses the FragHub dataset, but you can modify the configuration files in `MS_diffusion/configs/` to work with other datasets.

**Conditioning Types:**
- `ms2emb`: MS spectra → embeddings (from contrastive model) - used for MS2Mol inference
- `ms2fp`: MS spectra → molecular fingerprints
- `mol2emb`: Molecular graphs → embeddings - used for Graph2Mol pretraining
- `mol2fp`: Molecular graphs → fingerprints
- `null`: No conditioning (unconditional generation)

**Key Configuration Files:**
- `configs/conditioning/conditioning_default.yaml`: Conditioning settings
- `configs/general/general_default.yaml`: Training settings, GPU and wandb configuration
- `configs/train/train_default.yaml`: Learning rate, epochs, batch size

#### Stage 3a: MS2Mol Finetuning

Finetune the pretrained Graph2Mol model with MS conditioning to enable MS-to-molecule generation:

```bash
cd MS_diffusion/src
python main.py \
    conditioning.embeddings_type=ms2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    general.resume=[graph2mol_cp_path]
```

Note: Change `embeddings_type` from `mol2emb` to `ms2emb` to switch from molecular graph conditioning to MS spectrum conditioning.

#### Stage 3b: MS2Mol Inference

Run inference on the test set using the finetuned MS2Mol model:

```bash
cd MS_diffusion/src
python main.py \
    conditioning.embeddings_type=ms2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    general.test_only=[ms2mol_cp_path] \ 
    general.samples_to_generate='all'
```

This generates molecular structures for each test MS spectrum. By default, the model generates 25 candidate molecules per spectrum and computes metrics using a statistical approach. The inference results are saved in an output folder (typically in `MS_diffusion/outputs/`).

#### Stage 3c: Post-Analysis of Inference Results

Analyze the inference results using the provided Jupyter notebook:

1. Open `MS_diffusion/post_analysis.ipynb`
2. Run the first three cells to load necessary functions and data
3. In the fourth cell, specify the inference output folder path from Stage 3b
4. Execute the remaining cells to generate comprehensive evaluation metrics and visualizations

The notebook computes detailed metrics and generates plots, which are saved to `MS_diffusion/analysis_outputs/`.


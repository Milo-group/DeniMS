# DeniMS - *De-novo* Identification of Mass Spectra

This repository accompanies the paper: [Paper](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000101/v1)

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

## Usage

Use `apply_model` to run the trained MS2Mol diffusion model on experimental MS data.

### Data preparation

Experimental data must be stored in a single Parquet file, where each row corresponds to a single MS/MS spectrum, and spectra from the same compound share the same `Compound_index`.  
The file should contain the following columns:  
1. `Compound_index` – integer ID of the compound  
2. `precursor_formula` – list/array encoding the elemental composition of the precursor, in the same format used for FragHub  
3. `formulas` – list of fragment-ion formulas per spectrum, each represented as a list/array of element counts  
4. `precursor_type` – e.g. `[M+H]+` or `[M-H]-`  
5. `collision_energy_NCE` – collision energy

See `Preprocessing/experimental/experimental.csv`  for a reference example.  
To calculate the formulas from a raw MS file you can use our `HRMS_utils` repository. SIRIUS or other formula annotation programs can also be used.


### Model application

To run our model, you can use the notebook `MS_diffusion/src/apply_model.ipynb`.

You can also run experimental inference directly from the command line:

```bash
cd MS_diffusion/src
python apply_model.py \
  --model_checkpoint [path_to_model_ckpt.ckpt] \
  --experimental_parquet ../../Preprocessing/experimental/experimental.parquet \
  --output_dir ./inference_results \
  --num_repeats 50 \
```

Our trained models (Fraghub_contrastive_random.ckpt, Fraghub_FP_random.ckpt) can be downloaded from [Zenodo](https://zenodo.org/records/18539020).

#### Running `run_inference_experimental` as an ensemble

`run_inference_experimental` supports generating molecules using **multiple diffusion checkpoints** and aggregating the results into an ensemble:

```bash
cd MS_diffusion/src
python apply_model.py \
  --experimental_parquet ../../Preprocessing/experimental/experimental.parquet \
  --output_dir ./inference_ensemble \
  --ensemble_models_dir [path_to_model_ckpt.ckpt] \
  --repeats_per_model 25
```

In ensemble mode, the function:

- Runs model for each checkpoint and write results into a subdirectory.
- Merges all generated molecules and SMILES per compound across models.
- Writes ensemble summary files in `output_dir`, including:
  - `inference_summary_ensemble.txt`
  - `all_compounds_smiles_ensemble.txt`
  - `top3_smiles_per_compound_ensemble.csv`


## Retrain a model

### Preprocessing

We process high-resolution MS datasets using a standardized preparation pipeline that filters invalid entries, annotates fragment-ion formulas, and associates each spectrum with the relevant metadata. The initial processing steps follow our previous work, available in the following repository: https://github.com/Nir-Cohen-2003/HRMS_utils.

The final preprocessing scripts used in this project are provided in the Preprocessing/ folder.

A fully integrated pipeline with step-by-step explanations **will be added soon**.

In the meantime, the preprocessed FragHub Parquet file and the corresponding molecular graph dictionary (FragHub_filtered_smiles_dict.pt, FragHub_filtered.parquet) can be downloaded from [Zenodo](https://zenodo.org/records/18539020).

### Stage 1: Encoder pretraining

Train the MS spectra encoder with respect to molecular structures. The pretrained contrastive and FP_prediction encoderes (Contrastive_FragHub_random.pth, FP_FragHub_random.pth) can also be downloaded from [Zenodo](https://zenodo.org/records/18539020).

#### Basic Training Example

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
- `-split_path`: Path to predefined splits (uses random split if not provided)

#### Evaluation

Evaluate a trained model:

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

First, pretrain the diffusion model on molecular graphs without MS conditioning (graph2mol). For example, to run a diffusion model based on graph embeddings from the contrastive model, run:

```bash
cd MS_diffusion/src
python main.py \
    conditioning.embeddings_type=mol2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    train.finetune_ms_encoder=False
```

By default, it uses the FragHub dataset, but you can modify the configuration files in `MS_diffusion/configs/` to work with other datasets.

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

Finetune the pretrained Graph2Mol model and pretrained MS encoder to enable MS-to-molecule generation:

```bash
cd MS_diffusion/src
python main.py \
    conditioning.embeddings_type=ms2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    general.resume=[graph2mol_cp_path] \
    train.finetune_ms_encoder=True \
    train.lr=0.0001 
```

Note: Change `embeddings_type` from `mol2emb` to `ms2emb` to switch from molecular graph conditioning to MS spectrum conditioning. For FP-based model, change mol2fp to ms2fp.

#### Stage 3b: MS2Mol Inference

Run inference on the test set using the finetuned MS2Mol model:

```bash
cd MS_diffusion/src
python main.py \
    conditioning.embeddings_type=ms2emb \
    conditioning.embedding_model_path=[contrastive_cp_path] \
    general.test_only=[ms2mol_cp_path] \ 
    general.samples_to_generate='all' \
    train.finetune_ms_encoder=True  
```

This generates molecular structures for each test MS spectrum. By default, the model generates 50 candidate molecules per spectrum and computes metrics using a statistical approach. The inference results are saved in an output folder (typically in `MS_diffusion/outputs/`).

Note that train.finetune_ms_encoder should be true just if the checkpoint provide fineruned MS encoder weights. When relying on freezed MS encoder weights, use False.

#### Stage 3c: Post-Analysis of Inference Results

Analyze the inference results using the provided Jupyter notebook:

1. Open `MS_diffusion/post_analysis.ipynb`
2. Run the first three cells to load necessary functions and data
3. In the fourth cell, specify the inference output folder path from Stage 3b
4. Execute the remaining cells to generate comprehensive evaluation metrics and visualizations

The notebook computes detailed metrics and generates plots, which are saved to `MS_diffusion/analysis_outputs/`.


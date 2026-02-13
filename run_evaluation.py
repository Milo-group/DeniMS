
import argparse
import os
import re
import sys
import time
from datetime import datetime

import torch

# Add paths for imports
sys.path.append('MS_diffusion/src')
sys.path.append('..')

# Project imports
from dataloaders import load_data
from model import Contrastive_model
from evaluation_utils import evaluate_with_aggregation


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def extract_layers_from_checkpoint(state_dict):
    """
    Infer number of transformer layers from a saved state_dict.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Number of transformer layers, or None if not found
    """
    layer_pattern = re.compile(r"ms_encoder\.transformer_encoder\.layers\.(\d+)\.")
    max_layer_index = -1
    for key in state_dict:
        m = layer_pattern.search(key)
        if m:
            max_layer_index = max(max_layer_index, int(m.group(1)))
    return max_layer_index + 1 if max_layer_index >= 0 else None


def infer_hidden_dim_from_checkpoint(state_dict):

    # Try layer norm biases first (most reliable)
    probe_keys = [
        "ms_encoder.transformer_encoder.layers.0.norm2.bias",
        "ms_encoder.transformer_encoder.layers.0.norm1.bias",
    ]
    for k in probe_keys:
        if k in state_dict:
            return state_dict[k].shape[0]
    
    # Fallback to encoder embedding size
    for k in state_dict:
        if k.endswith("ms_encoder.embedding.weight"):
            return state_dict[k].shape[1]
    
    raise RuntimeError("Could not infer hidden_dim from checkpoint.")


def load_checkpoint(cp_path, cp_name):
    """
    Load model checkpoint from disk.
    """
    if cp_name is None:
        raise ValueError("Please provide -cp_name for the model checkpoint to evaluate.")
    
    # Reconstruct the folder structure created by train()
    run_dir = cp_name.split(" - ")[0]
    cp_full_path = os.path.join(cp_path, run_dir, f"{cp_name}.pth")
    
    if not os.path.exists(cp_full_path):
        raise FileNotFoundError(f"Checkpoint not found at: {cp_full_path}")
    
    print(f"Loading checkpoint from: {cp_full_path}")
    ckpt = torch.load(cp_full_path, map_location='cpu', weights_only=False)
    
    if 'model' not in ckpt:
        raise KeyError("Checkpoint missing 'model' state_dict.")
    
    return ckpt, cp_full_path


# ============================================================================
# Argument Parser
# ============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description='Evaluate a trained MS encoder (fp or contrastive mode).')

    # Core (aligned with train)
    p.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('-mode', type=str, default='contrastive', choices=['fp', 'contrastive', 'mixed'],
                   help='Training mode: fp (MSE to fingerprint) or contrastive (InfoNCE).')

    # Evaluation parameters
    p.add_argument('-batch_size', type=int, default=512)
    p.add_argument('-total_samples', type=str, default="all", help='Number of samples to evaluate, or "all" to use the entire test dataset.')
    p.add_argument('-ordered_batches', action='store_true')
    p.add_argument('-multi_MS_per_mol', action='store_true',
                   help='Return all MS spectra for each molecule (multiple MS per molecule mode)')
    p.add_argument('-aggregation_method', type=str, default='mean', 
                   choices=['mean', 'weighted', 'topk', 'centroid_closest'],
                   help='Method to aggregate multiple MS spectra per molecule: mean (regular mean), weighted (by info_score), topk (mean of top-k by info_score), or centroid_closest (select closest to centroid, or both if N=2)')
    p.add_argument('-topk', type=int, default=3,
                   help='Number of top spectra to use when aggregation_method is "topk" (default: 3)')
    p.add_argument('-outlier_filter', action='store_true',
                   help='Enable outlier filtering using centroid distance method before aggregation')
    p.add_argument('-outlier_threshold', type=float, default=2.0,
                   help='Z-score threshold for outlier detection (default: 2.0). Samples with distance Z-score > threshold are filtered out.')

    # Data (same defaults/types as train)
    p.add_argument('-data_path', type=str, default='Preprocessing/fraghub/fraghub_filtered.parquet')
    p.add_argument('-smiles_path', type=str, default='Preprocessing/fraghub/smiles_dict_fraghub.pt')
    p.add_argument('-split_path', type=str, default=None,
                   help='Path to predefined split file. If not provided, split will be set to random.')
    p.add_argument('-ordered_sub_batch_size', type=int, default=32,
                   help='Number of similar SMILES grouped into one ordered sub-batch (used when ordered_batches is enabled).')
    p.add_argument('-random_seed', type=int, default=42,
                   help='Random seed used for random train/val/test splits and sampling.')

    # Model arch args (optional overrides; sensible defaults if missing in ckpt)
    p.add_argument('-max_len', type=int, default=129)
    p.add_argument('-hidden_dim', type=int, default=None)
    p.add_argument('-num_transformer_layers', type=int, default=None)
    p.add_argument('-nhead', type=int, default=8)
    p.add_argument('-dropout', type=float, default=0.1)
    p.add_argument('-input_dropout', type=float, default=0.1)
    p.add_argument('-embeddings_dim', type=int, default=None,
                   help='If unset, defaults to 2048 in fp mode and 512 in contrastive mode (to match train).')
    p.add_argument('-fp_length', type=int, default=2048)
    p.add_argument('-temperature', type=float, default=30,
                   help='Used by evaluation_utils if contrastive metrics are computed.')

    # Checkpoint
    p.add_argument('-cp_path', type=str, default='checkpoints')
    p.add_argument('-cp_name', type=str, default=None, help='Checkpoint filename WITHOUT .pth extension')

    # Misc
    p.add_argument('-comment', type=str, default='eval')
    p.add_argument('-no_plot', action='store_true')

    return p


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main(args):
    """Main evaluation function."""
    
    print("\n" + "="*70)
    print("MS2Mol Model Evaluation")
    print("="*70)

    # Determine split type
    if args.split_path and args.split_path.strip():
        args.split = 'predefined'
        print(f"Using predefined split from: {args.split_path}")
    else:
        args.split = 'random'
        print(f"Using random split with seed: {args.random_seed}")

    # Set embeddings_dim default based on mode
    if args.embeddings_dim is None:
        args.embeddings_dim = 2048 if args.mode == 'fp' else 512
        print(f"Using default embeddings_dim: {args.embeddings_dim} (mode: {args.mode})")

    args.plot = not args.no_plot

    # Load data
    print(f"\n{'='*70}")
    print("Loading data...")
    print(f"  Data path: {args.data_path}")
    print(f"  SMILES dict: {args.smiles_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Ordered batches: {args.ordered_batches}")
    print(f"  Multi-MS per molecule: {args.multi_MS_per_mol}")
    print(f"{'='*70}\n")
    
    train_loader, val_loader, test_loader, graph_dict = load_data(
        args.data_path,
        args.smiles_path,
        args.split,
        args.split_path,
        args.batch_size,
        batch=args.multi_MS_per_mol,
        ordered=args.ordered_batches,
        ordered_sub_batch_size=args.ordered_sub_batch_size,
        random_seed=args.random_seed,
    )
    
    print(f"Data loaded successfully!")
    

    if args.total_samples == "all":
        if args.ordered_batches:
            args.total_samples = sum(len(sb) for sb in test_loader.dataset.ordered_smiles)
        else:
            args.total_samples = len(test_loader.dataset)
    else:
        try:
            args.total_samples = int(args.total_samples)
        except ValueError:
            raise ValueError(f"Invalid value for -total_samples: {args.total_samples}. Use an integer or 'all'.")
    
    print(f"  Total test samples: {args.total_samples}")

    # Load checkpoint
    print(f"\n{'='*70}")
    print("Loading checkpoint...")
    ckpt, cp_path = load_checkpoint(args.cp_path, args.cp_name)
    state_dict = ckpt['model']

    # Infer model architecture from checkpoint if not provided
    hidden_dim = args.hidden_dim or infer_hidden_dim_from_checkpoint(state_dict)
    num_layers = args.num_transformer_layers or extract_layers_from_checkpoint(state_dict) or 3
    
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Transformer layers: {num_layers}")
    print(f"  Embeddings dim: {args.embeddings_dim}")
    print(f"{'='*70}\n")

    # Determine model configuration
    use_graph = (args.mode in ['contrastive', 'mixed'])
    fp_pred = (args.mode in ['fp', 'mixed'])
    
    # Initialize model
    print("Initializing model...")
    model = Contrastive_model(
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        max_len=args.max_len,
        hidden_dim=hidden_dim,
        num_transformer_layers=num_layers,
        nhead=args.nhead,
        embeddings_dim=args.embeddings_dim,
        graph=use_graph,
        fp_pred=fp_pred,
        fp_length=args.fp_length,
        trainable_temperature=True
    ).to(args.device)

    # Load weights
    print("Loading model weights...")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully!")

    # Print evaluation configuration
    run_name = datetime.now().strftime("%D %H-%M").replace("/", "-") + "-" + args.comment
    print(f"\n{'='*70}")
    print(f"Evaluation Configuration")
    print(f"{'='*70}")
    print(f"  Run name: {run_name}")
    print(f"  Checkpoint: {cp_path}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Total samples: {args.total_samples}")
    print(f"  Aggregation method: {args.aggregation_method}")
    if args.aggregation_method == "topk":
        print(f"  Top-k: {args.topk}")
    print(f"  Outlier filtering: {args.outlier_filter}")
    if args.outlier_filter:
        print(f"  Outlier threshold: {args.outlier_threshold}")
    print(f"  Generate plots: {args.plot}")
    print(f"{'='*70}\n")

    # Run evaluation
    start = time.time()
    with torch.no_grad():
        results = evaluate_with_aggregation(
            model, test_loader, graph_dict, args, 
            eval_samples=args.total_samples, 
            plot=args.plot,
            aggregation_method=args.aggregation_method
        )

    elapsed = time.time() - start
    
    # Print final results
    print(f"\n{'='*70}")
    print("Final Evaluation Results")
    print(f"{'='*70}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  Evaluation time: {elapsed:.2f}s")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)

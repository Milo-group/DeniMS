
import sys
sys.path.append('MS_diffusion/src')
sys.path.append('..')

import argparse
import os
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import lr_scheduler

# Project imports
from dataloaders import load_data
from model import Contrastive_model
from evaluation_utils import (
    evaluate_during_training, 
    compute_contrastive_logits, 
    batch_graphs_to_padded_data
)

try:
    import wandb
except Exception:
    wandb = None

# ----------------------
# Utilities
# ----------------------
def save_checkpoint(model, optim, cp_path, run_name, epoch, step, temp=False, scheduler=None):
    run_dir = os.path.join(cp_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    save_dict = {
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'epoch': epoch,
        'step': step
    }
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    fname = f"{run_name} - temp.pth" if temp else f"{run_name} - {epoch}.pth"
    torch.save(save_dict, os.path.join(run_dir, fname))

def warm_cosine_schedule(step, warmsteps, total_iterations):
    if step < warmsteps:
        return step / warmsteps  # Linear warm-up
    else:
        return 0.5 * (1 + np.cos(np.pi * (step - warmsteps) / (total_iterations - warmsteps)))  # Cosine decay

def build_scheduler(optim, total_iterations, scheduler_type='WarmCosine', warmsteps=200):
    if scheduler_type == 'WarmCosine':
        return lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: warm_cosine_schedule(s, warmsteps, total_iterations))
    elif scheduler_type == 'Cosine':
        return lr_scheduler.CosineAnnealingLR(optim, T_max=total_iterations, eta_min=1e-7)
    elif scheduler_type == 'Constant':
        return torch.optim.lr_scheduler.ConstantLR(optim)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

# ----------------------
# Training Loops
# ----------------------
def process_info_scores(information_score, device, default_value=0.1):
    """
    Process information scores for weighting loss.
    If information_score is None or contains None/NaN values, replace with default_value.
    Returns a tensor of weights on the specified device.
    """
    if information_score is None:
        # If entire tensor is None, return default weights
        return torch.full((1,), default_value, device=device)
    
    # Convert to tensor if needed and move to device
    # Note: None values in lists/arrays will become NaN when converted to tensor
    if not isinstance(information_score, torch.Tensor):
        try:
            information_score = torch.tensor(information_score, device=device, dtype=torch.float32)
        except (TypeError, ValueError):
            # If conversion fails (e.g., contains None), create tensor with default values
            if hasattr(information_score, '__len__'):
                return torch.full((len(information_score),), default_value, device=device)
            else:
                return torch.full((1,), default_value, device=device)
    else:
        information_score = information_score.to(device).float()
    
    # Flatten to 1D if needed
    if information_score.dim() > 1:
        information_score = information_score.squeeze()
    if information_score.dim() == 0:
        information_score = information_score.unsqueeze(0)
    
    # Replace None/NaN/Inf values with default_value
    weights = information_score.clone()
    invalid_mask = torch.isnan(weights) | torch.isinf(weights) | (weights < 0)
    weights[invalid_mask] = default_value
    
    # Ensure at least one element
    if weights.numel() == 0:
        return torch.full((1,), default_value, device=device)
    
    return weights

def train_epoch_fp(model, train_iter, graph_dict, optim, scheduler, args, step):
    if args.use_info_weights:
        loss_fn = nn.MSELoss(reduction='none')
    else:
        loss_fn = nn.MSELoss()
    sig = nn.Sigmoid()
    total_loss = 0.0
    iters = len(train_iter)
    data_it = iter(train_iter)
    epoch_num = getattr(args, 'run_epoch', 0) + 1
    for it in tqdm(range(iters), desc=f"Epoch {epoch_num}/{args.epochs} [FP]"):
        optim.zero_grad()
        sos, formula_array, mask, smiles, information_score = next(data_it)

        # Labels: fingerprint/embedding vector from dict
        fp_labels = torch.cat([graph_dict[smiles[i]][1] for i in range(sos.shape[0])], dim=0).to(args.device).float()

        preds = sig(model(sos, formula_array, mask=mask))
        
        if args.use_info_weights:
            # Compute per-sample losses
            per_sample_loss = loss_fn(50 * preds, 50 * fp_labels)
            # Average over feature dimension to get per-sample loss
            per_sample_loss = per_sample_loss.mean(dim=1)
            
            # Get weights from information scores
            weights = process_info_scores(information_score, args.device, default_value=0.1)
            # Ensure weights match batch size
            if weights.shape[0] != per_sample_loss.shape[0]:
                if weights.shape[0] == 1:
                    weights = weights.expand(per_sample_loss.shape[0])
                else:
                    weights = weights[:per_sample_loss.shape[0]]
            
            # Weighted loss
            loss = (per_sample_loss * weights).mean()
        else:
            loss = loss_fn(50 * preds, 50 * fp_labels)  # preserve original scale choice

        loss.backward()
        optim.step()
        scheduler.step()
        step += 1

        total_loss += loss.item()

        if args.temp_cp > 0 and step % args.temp_cp == 0:
            save_checkpoint(model, optim, args.cp_path, args.run_name, f"{args.epoch_float:.2f}", step, temp=True, scheduler=scheduler)
    return total_loss / iters, step

def train_epoch_contrastive(model, train_iter, graph_dict, optim, scheduler, args, step):
    if args.use_info_weights:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    iters = len(train_iter)
    data_it = iter(train_iter)
    labels_template = torch.arange(args.batch_size).to(args.device)
    epoch_num = getattr(args, 'run_epoch', 0) + 1

    for it in tqdm(range(iters), desc=f"Epoch {epoch_num}/{args.epochs} [Contrastive]"):
        optim.zero_grad()
        sos, formula_array, mask, smiles, information_score = next(data_it)

        # Build graph batch for contrastive mode
        graph_list = [graph_dict[smiles[i]][0] for i in range(sos.shape[0])]
        graph_batch = batch_graphs_to_padded_data(graph_list, device=args.device)

        labels = labels_template[:sos.shape[0]]
        ms_features, graph_features = model(sos, formula_array, graph=graph_batch, mask=mask)
        temp = 1.0 / model.inv_temperature if hasattr(model, 'inv_temperature') else args.temperature
        logits_per_ms, logits_per_mol = compute_contrastive_logits(ms_features, graph_features, temp)

        if args.use_info_weights:
            # Compute per-sample losses
            per_sample_loss_ms = loss_fn(logits_per_ms, labels)
            per_sample_loss_fp = loss_fn(logits_per_mol, labels)
            per_sample_loss = (per_sample_loss_ms + per_sample_loss_fp) / 2
            
            # Get weights from information scores
            weights = process_info_scores(information_score, args.device, default_value=0.1)
            # Ensure weights match batch size
            if weights.shape[0] != per_sample_loss.shape[0]:
                if weights.shape[0] == 1:
                    weights = weights.expand(per_sample_loss.shape[0])
                else:
                    weights = weights[:per_sample_loss.shape[0]]
            
            # Weighted loss
            loss = (per_sample_loss * weights).mean()
        else:
            loss = (loss_fn(logits_per_ms, labels) + loss_fn(logits_per_mol, labels)) / 2
        loss.backward()
        optim.step()
        if hasattr(model, 'inv_temperature'):
            model.inv_temperature.data.clamp_(min=1.0/200.0)
        scheduler.step()
        step += 1

        total_loss += loss.item()

        if args.temp_cp > 0 and step % args.temp_cp == 0:
            save_checkpoint(model, optim, args.cp_path, args.run_name, f"{args.epoch_float:.2f}", step, temp=True, scheduler=scheduler)
    return total_loss / iters, step

def train_epoch_mixed(model, train_iter, graph_dict, optim, scheduler, args, step):

    if args.use_info_weights:
        mae = nn.L1Loss(reduction='none')
        ce  = nn.CrossEntropyLoss(reduction='none')
    else:
        mae = nn.L1Loss()
        ce  = nn.CrossEntropyLoss()
    sig = nn.Sigmoid()

    total_loss = 0.0

    iters = len(train_iter)
    data_it = iter(train_iter)
    labels_template = torch.arange(args.batch_size, device=args.device)
    epoch_num = getattr(args, 'run_epoch', 0) + 1

    for it in tqdm(range(iters), desc=f"Epoch {epoch_num}/{args.epochs} [Mixed]"):
        optim.zero_grad()
        sos, formula_array, mask, smiles, information_score = next(data_it)

        fp_labels = torch.cat(
            [graph_dict[smiles[i]][1] for i in range(sos.shape[0])],
            dim=0
        ).to(args.device).float()     

        # Contrastive forward
        graph_list = [graph_dict[smiles[i]][0] for i in range(sos.shape[0])]
        graph_batch = batch_graphs_to_padded_data(graph_list, device=args.device)

        contrastive_labels = labels_template[:sos.shape[0]]                           
        ms_features, graph_features, fp_output = model(sos, formula_array, graph=graph_batch, mask=mask)

        fp_preds = sig(fp_output)
        
        if args.use_info_weights:
            # Compute per-sample MSE loss
            per_sample_mse = mae(10 * fp_preds, 10 * fp_labels)
            per_sample_mse = per_sample_mse.mean(dim=1)  # Average over feature dimension
            
            temp = 1.0 / model.inv_temperature if hasattr(model, 'inv_temperature') else args.temperature
            logits_per_ms, logits_per_mol = compute_contrastive_logits(ms_features, graph_features, temp)
            per_sample_con_ms = ce(logits_per_ms, contrastive_labels)
            per_sample_con_fp = ce(logits_per_mol, contrastive_labels)
            per_sample_con = (per_sample_con_ms + per_sample_con_fp) / 2
            
            # Get weights from information scores
            weights = process_info_scores(information_score, args.device, default_value=0.1)
            # Ensure weights match batch size
            batch_size = per_sample_mse.shape[0]
            if weights.shape[0] != batch_size:
                if weights.shape[0] == 1:
                    weights = weights.expand(batch_size)
                else:
                    weights = weights[:batch_size]
            
            # Weighted losses
            mse_loss = (per_sample_mse * weights).mean()
            con_loss = (per_sample_con * weights).mean()
        else:
            mse_loss = mae(10 * fp_preds, 10 * fp_labels)
            temp = 1.0 / model.inv_temperature if hasattr(model, 'inv_temperature') else args.temperature
            logits_per_ms, logits_per_mol = compute_contrastive_logits(ms_features, graph_features, temp)
            con_loss = (ce(logits_per_ms, contrastive_labels) + ce(logits_per_mol, contrastive_labels)) / 2
     
        loss = mse_loss + con_loss

        loss.backward()
        optim.step()
        if hasattr(model, 'inv_temperature'):
            model.inv_temperature.data.clamp_(min=1.0/200.0)
        scheduler.step()
        step += 1

        total_loss += loss.item()

        if args.temp_cp > 0 and step % args.temp_cp == 0:
            save_checkpoint(model, optim, args.cp_path, args.run_name, f"{args.epoch_float:.2f}", step, temp=True, scheduler=scheduler)
    
    return total_loss / iters, step

# ----------------------
#  Main train function
# ----------------------
def train(model, train_loader, val_loader, graph_dict, args, checkpoint=None):
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=args.wd)
    total_iterations = args.epochs * len(train_loader)
    scheduler = build_scheduler(optim, total_iterations, scheduler_type=args.scheduler_type, warmsteps=args.warmsteps)

    step = 1
    start_epoch = 0
    if checkpoint is not None:
        if 'optimizer' in checkpoint:
            optim.load_state_dict(checkpoint['optimizer'])
        if (not getattr(args, 'no_scheduler_load', False)) and ('scheduler' in checkpoint):
            scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                try:
                    start_epoch = int(float(checkpoint['epoch']))
                except Exception:
                    start_epoch = 0

    model.train()
    start_time = time.time()
    temp_time = time.time()
    if args.mode == 'fp':
        train_step_fn = train_epoch_fp
    elif args.mode == 'contrastive': 
        train_step_fn = train_epoch_contrastive
    else:
        train_step_fn = train_epoch_mixed

    print(f"\nStarting training: {args.epochs - start_epoch} epochs remaining")
    print(f"Total iterations per epoch: {len(train_loader)}")
    print(f"Total training iterations: {total_iterations}\n")

    for epoch in range(start_epoch, args.epochs):
        args.epoch_float = epoch
        args.run_epoch = epoch
        
        avg_loss, step = train_step_fn(model, train_loader, graph_dict, optim, scheduler, args, step)

        print(f"Training loss: {avg_loss:.4f}")

        # Validation evaluation
        print(f"\nRunning validation evaluation...")
        temp_val = 1.0 / model.inv_temperature.item() if hasattr(model, 'inv_temperature') else args.temperature
        metrics = {
            'specs/time (hours)': (time.time() - start_time) / 3600,
            'specs/time per epoch (sec)': (time.time() - temp_time),
            'specs/lr': scheduler.get_last_lr()[0],
            'Train/avg_loss': avg_loss,
            'specs/epoch': epoch,
            'specs/temperature': temp_val
        }
        temp_time_eval = time.time()
        with torch.no_grad():
            eval_results = evaluate_during_training(model, val_loader, graph_dict, args=args, epoch=epoch + 1)
            model.train()
        metrics['specs/eval time'] = time.time() - temp_time_eval
        temp_time = time.time()

        # Print metrics
        print(f"\nEpoch {epoch + 1} Metrics:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Loss: {eval_results.get('Val/eval_loss', 0):.4f}")
        print(f"  Val Accuracy: {eval_results.get('Val/eval_acc', 0):.4f}")
        print(f"  Cosine Similarity: {eval_results.get('Val/cosine_similarity', 0):.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Temperature: {temp_val:.2f}")
        print(f"  Epoch Time: {metrics['specs/time per epoch (sec)']:.1f}s")
        print(f"  Total Time: {metrics['specs/time (hours)']:.2f}h")

        if args.wandb and wandb is not None:
            to_log = dict(metrics)
            to_log.update(eval_results)
            wandb.log(to_log, step=epoch + 1)

        metrics.update(eval_results)

        # Save checkpoint
        if args.epochs_cp > 0 and (epoch + 1) % args.epochs_cp == 0:
            save_checkpoint(model, optim, args.cp_path, args.run_name, epoch + 1, step, temp=False, scheduler=scheduler)
            print(f"\nCheckpoint saved at epoch {epoch + 1}")

# ----------------------
# Runner
# ----------------------
def run(args):
    # Data
    train_loader, val_loader, test_loader, graph_dict = load_data(
        args.data_path,
        args.smiles_path,
        args.split,
        args.split_path,
        args.batch_size,
        ordered=True,
        random_seed=args.random_seed,
        ordered_sub_batch_size=args.ordered_sub_batch_size,
    )

    data_it = iter(train_loader)
    sos, formula_array, mask, smiles, _ = next(data_it)

    # Model
    use_graph = (args.mode in ['contrastive', 'mixed'])
    fp_pred = (args.mode in ['fp', 'mixed'])
    
    model = Contrastive_model(
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_transformer_layers,
        nhead=args.nhead,
        embeddings_dim=args.embeddings_dim,
        graph=use_graph,
        fp_pred = fp_pred,
        fp_length=args.fp_length,
        initial_temperature=args.temperature,
        trainable_temperature=args.trainable_temperature
    ).to(args.device)

    print("Model initialized successfully")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # Run name
    args.run_name = datetime.now().strftime("%D %H-%M").replace("/", "-") + "-" + args.comment

    # Checkpoint load (optional)
    checkpoint = None
    if args.cp_name is not None:
        if os.path.isdir(args.cp_path):
            full_dir = args.cp_name.split(" - ")[0]
            cp_full_path = os.path.join(args.cp_path, full_dir, f"{args.cp_name}.pth")
            checkpoint = torch.load(cp_full_path, map_location=torch.device('cpu'), weights_only=False)

            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded checkpoint from {cp_full_path}")
                if 'epoch' in checkpoint:
                    print(f"  Resuming from epoch {checkpoint['epoch']}")

        else:
            raise FileNotFoundError("Checkpoint directory does not exist: " + str(args.cp_path))

    # W&B
    if args.wandb and wandb is not None:
        wandb.login(key=args.wandb_key)
        cfg = {
            "batch_size": args.batch_size,
            "ordered_sub_batch_size": args.ordered_sub_batch_size,
            "random_seed": args.random_seed,
            "dropout": args.dropout,
            "input_dropout": args.input_dropout,
            "wd": args.wd,
            "mode": args.mode,
            "hidden_dim": args.hidden_dim,
            "num_transformer_layers": args.num_transformer_layers,
            "nhead": args.nhead,
            "temperature": args.temperature,
            "split": args.split
        }
        wandb.init(project=args.wandb_project, name=args.run_name, config=cfg)

    # Train
    train(model, train_loader, val_loader, graph_dict, args, checkpoint=checkpoint)

# ----------------------
# CLI
# ----------------------
def build_argparser():
    p = argparse.ArgumentParser(description='Unified trainer for MS encoding (fp regression / contrastive).')
    # Core
    p.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('-mode', type=str, default='contrastive', choices=['fp', 'contrastive', 'mixed'],
                   help='Training mode: fp (MSE to fingerprint) or contrastive (InfoNCE).')

    # Optimization
    p.add_argument('-lr', type=float, default=4e-4)
    p.add_argument('-wd', type=float, default=5e-4, help='Weight decay. (fp mode in original used 1e-5; override if desired)')
    p.add_argument('-dropout', type=float, default=0.1)
    p.add_argument('-input_dropout', type=float, default=0.1)
    p.add_argument('-scheduler_type', type=str, default='WarmCosine', choices=['WarmCosine','Cosine','Constant'])
    p.add_argument('-warmsteps', type=int, default=3000)
    p.add_argument('-epochs', type=int, default=500)
    p.add_argument('-batch_size', type=int, default=512)
    p.add_argument('-epochs_cp', type=int, default=50)
    p.add_argument('-temp_cp', type=int, default=25)
    p.add_argument('-temperature', type=float, default=15, help='Contrastive temperature (initial if trainable). Used only in contrastive mode.')
    p.add_argument('-trainable_temperature', action='store_true', help='Make 1/temperature trainable (starts at 1/args.temperature, clipped to min 1/200).')
    p.add_argument('-use_info_weights', action='store_true', help='Weight loss by information score. If information score is None for a sample, use 0.1 as default weight.')

    # Data
    p.add_argument('-data_path', type=str, default='Preprocessing/fraghub/fraghub_filtered.parquet')
    p.add_argument('-smiles_path', type=str, default='Preprocessing/fraghub/smiles_dict_fraghub.pt')
    p.add_argument('-split_path', type=str, default=None,
                   help='Path to predefined split file. If provided, split will be automatically set to "predefined".')
    p.add_argument('-ordered_sub_batch_size', type=int, default=32,
                   help='Number of similar SMILES grouped into one ordered sub-batch (used when ordered batching is enabled).')
    p.add_argument('-random_seed', type=int, default=42,
                   help='Random seed used for random train/val/test splits and sampling.')

    # Checkpointing
    p.add_argument('-cp_path', type=str, default='checkpoints')
    p.add_argument('-cp_name', type=str, default=None, help='Checkpoint filename (without directory) to load, e.g. "<run dir> - 50".')
    p.add_argument('-no_scheduler_load', action='store_true',
                   help='If set, do NOT load scheduler state from checkpoint (start scheduler fresh).')

    # Logging
    p.add_argument('-wandb', action='store_true', help='Use Weights & Biases')
    p.add_argument('-wandb_key', type=str, default='')
    p.add_argument('-wandb_project', type=str, default='ms_encoding')
    p.add_argument('-comment', type=str, default='unified_train')

    # Model arch
    p.add_argument('-max_len', type=int, default=129)
    p.add_argument('-hidden_dim', type=int, default=512)
    p.add_argument('-num_transformer_layers', type=int, default=3)
    p.add_argument('-nhead', type=int, default=8)
    p.add_argument('-embeddings_dim', type=int, default=None)
    p.add_argument('-fp_length', type=int, default=2048)

    return p

if __name__ == '__main__':
    args = build_argparser().parse_args()

    if args.split_path and args.split_path.strip():
        args.split = 'predefined'
    else:
        args.split = 'random'

    # Postprocess conditional default
    if args.embeddings_dim is None:
        if args.mode == 'fp':
            args.embeddings_dim = 2048
        else:  # contrastive
            args.embeddings_dim = 512

    run(args)

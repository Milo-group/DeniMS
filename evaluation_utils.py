import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm
from openTSNE import TSNE


def batch_graphs_to_padded_data(data_list, device="cuda"):
    """
    Convert a list of PyTorch Geometric Data objects to a single padded batch.
    
    Takes:
        data_list: List of Data objects, each containing node features (x), 
                   edge indices (edge_index), and edge attributes (edge_attr)

    Returns:
        Data object with padded tensors:
        - X: [B, max_nodes, node_feat_dim] node features
        - E: [B, max_nodes, max_nodes, edge_feat_dim] edge features
        - y: [B, 1] dummy target
        - node_mask: [B, max_nodes] boolean mask for valid nodes
    """
    B = len(data_list)
    max_nodes = max(data.x.size(0) for data in data_list)
    node_feat_dim = data_list[0].x.size(1)
    edge_feat_dim = data_list[0].edge_attr.size(1)

    # Allocate padded tensors
    X = torch.zeros(B, max_nodes, node_feat_dim)
    E = torch.zeros(B, max_nodes, max_nodes, edge_feat_dim)
    y = torch.ones(X.shape[0], 1).type_as(X)
    node_mask = torch.zeros(B, max_nodes, dtype=torch.bool)
    atom_counts = torch.zeros(B, dtype=torch.long)

    for i, data in enumerate(data_list):
        n = data.x.size(0)
        X[i, :n] = data.x
        node_mask[i, :n] = 1
        atom_counts[i] = n

        src, dst = data.edge_index
        E[i, src, dst] = data.edge_attr

        # Set zero edges to default edge type
        valid_n = n  
        edge_block = E[i, :valid_n, :valid_n]  
        zero_mask = (edge_block == 0).all(dim=-1)  
        edge_block[zero_mask] = torch.tensor([1, 0, 0, 0, 0], dtype=E.dtype, device=E.device)

    return Data(
        X=X.to(device),
        E=E.to(device),
        y=y.to(device),
        node_mask=node_mask.to(device)
    )


def compute_contrastive_logits(ms_features, mol_features, temperature=15, calc_similarity=False):

    # L2 normalize features
    ms_features = ms_features / ms_features.norm(dim=1, keepdim=True)
    mol_features = mol_features / mol_features.norm(dim=1, keepdim=True)

    # Compute similarity matrix
    logits_per_ms = temperature * ms_features @ mol_features.t()       
    logits_per_mol = logits_per_ms.t()

    if calc_similarity:
        cos_sim = (ms_features * mol_features).sum(dim=1)
        average_cosine_similarity = cos_sim.mean().item()
        return logits_per_ms, logits_per_mol, average_cosine_similarity
    
    return logits_per_ms, logits_per_mol


@torch.no_grad()
def evaluate_during_training(model, val_dataloader, graph_dict, args, eval_part: float = 1.0, epoch=None):

    model.eval()

    mode = getattr(args, "mode", None)

    if mode == "mixed":
        mode = "contrastive"
        mixed = True
    else:
        mixed = False

    if mode == 'contrastive':
        loss_fn = nn.CrossEntropyLoss()
        use_graph = True
        sigmoid = None
    elif mode == 'fp':
        loss_fn = nn.MSELoss()
        sigmoid = nn.Sigmoid()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_batches = len(val_dataloader)
    batches_num = max(1, int(total_batches * float(eval_part)))

    total_samples = 0
    total_acc = 0
    total_loss = 0.0
    total_cosine_similarity = 0.0

    desc = f"Val Epoch {epoch}" if epoch is not None else "Validation evaluation"
    pbar = tqdm(range(batches_num), desc=desc)
    data_iter = iter(val_dataloader)

    for _ in pbar:
        sos, formula_array, mask, smiles, _ = next(data_iter)

        if mode == 'contrastive':
            # Build graph batch
            graph_list = [graph_dict[smi][0] for smi in smiles]
            graph_batch = batch_graphs_to_padded_data(graph_list, device=args.device)
            
            if mixed:
                ms_features, graph_features, _ = model(sos, formula_array, graph=graph_batch, mask=mask)
            else:
                ms_features, graph_features = model(sos, formula_array, graph=graph_batch, mask=mask)          

        else:  # mode == 'fp'
            # Get fingerprint labels
            fp_features = torch.cat(
                [graph_dict[smi][1] for smi in smiles], dim=0
            ).to(args.device).float()

            preds = model(sos, formula_array, mask=mask)
            ms_features = sigmoid(preds)
            graph_features = fp_features

        # Compute logits and similarity
        temp = 1.0 / model.inv_temperature if hasattr(model, 'inv_temperature') else args.temperature
        logits_per_ms, logits_per_mol, avg_cosine = compute_contrastive_logits(
            ms_features, graph_features, temp, calc_similarity=True
        )

        # Top-1 predictions
        ms_class = torch.argmax(logits_per_ms, dim=1)
        mol_class = torch.argmax(logits_per_mol, dim=1)

        # Ground-truth labels (diagonal alignment)
        labels = torch.arange(ms_class.shape[0], device=args.device)

        # Compute loss
        if mode == 'contrastive':
            loss = (loss_fn(logits_per_ms, labels) + loss_fn(logits_per_mol, labels)) / 2.0
        else:  # fp
            loss = loss_fn(ms_features, graph_features)

        # Compute accuracy
        acc_1 = torch.sum(ms_class.eq(labels))
        acc_2 = torch.sum(mol_class.eq(labels))

        batch_size = labels.shape[0]
        total_loss += float(loss.item())
        total_acc += int((acc_1 + acc_2).item())
        total_samples += batch_size * 2  # two directions
        total_cosine_similarity += float(avg_cosine) * batch_size

    metrics = {
        'Val/eval_acc': total_acc / total_samples,
        'Val/eval_loss': total_loss / batches_num,
        'Val/cosine_similarity': 2 * total_cosine_similarity / total_samples,
    }
    return metrics


# ============================================================================
# Aggregation Methods for Multiple Spectra per Molecule
# ============================================================================

def detect_outliers_centroid(ms_features, threshold=2.0, device="cuda"):
    """
    Detect outliers in MS feature vectors using centroid distance method.
    """
    N = ms_features.shape[0]
    
    if N <= 2:  # Need at least 3 samples for outlier detection
        return torch.ones(N, dtype=torch.bool, device=device)
    
    # Compute centroid and distances
    centroid = torch.mean(ms_features, dim=0)
    distances = torch.norm(ms_features - centroid.unsqueeze(0), dim=1)
    
    # Use Z-score on distances
    mean_dist = torch.mean(distances)
    std_dist = torch.std(distances)
    if std_dist > 1e-8:
        z_scores = torch.abs((distances - mean_dist) / std_dist)
        return z_scores <= threshold
    return torch.ones(N, dtype=torch.bool, device=device)


def select_closest_to_centroid(ms_features, device="cuda"):
    """
    Select the spectrum closest to the centroid. If there are exactly 2 samples, pick both.
    """
    N = ms_features.shape[0]
    
    if N <= 2:
        # If 2 or fewer samples, pick all
        return torch.ones(N, dtype=torch.bool, device=device)
    
    # Compute centroid and distances
    centroid = torch.mean(ms_features, dim=0)
    distances = torch.norm(ms_features - centroid.unsqueeze(0), dim=1)
    
    # Find the index of the closest spectrum
    closest_idx = torch.argmin(distances)
    
    # Create mask with only the closest spectrum selected
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[closest_idx] = True
    
    return mask


def aggregate_ms_features(ms_features, info_scores, method="mean", topk=3, device="cuda",
                         outlier_filter=False, outlier_threshold=2.0, centroid_closest=False):
    """
    Aggregate multiple MS spectra features into a single representation with optional outlier filtering.
    """
    # Apply centroid_closest selection if requested (before outlier filtering)
    if centroid_closest:
        closest_mask = select_closest_to_centroid(ms_features, device=device)
        ms_features = ms_features[closest_mask]
        if info_scores is not None:
            info_scores_device = info_scores.device
            closest_mask_info = closest_mask.to(info_scores_device)
            info_scores = info_scores[closest_mask_info]
    
    # Apply outlier filtering if requested
    if outlier_filter:
        outlier_mask = detect_outliers_centroid(ms_features, threshold=outlier_threshold, device=device)
        
        if outlier_mask.sum() == 0:
            # If all samples are outliers, fall back to all samples
            # (Silently use all samples to avoid verbosity)
            outlier_mask = torch.ones(len(ms_features), dtype=torch.bool, device=device)
        
        ms_features = ms_features[outlier_mask]
        if info_scores is not None:
            # Ensure mask is on the same device as info_scores
            info_scores_device = info_scores.device
            outlier_mask_info = outlier_mask.to(info_scores_device)
            info_scores = info_scores[outlier_mask_info]
    
    if method == "mean":
        # Regular mean of all spectra
        return torch.mean(ms_features, dim=0)
        
    elif method == "weighted":
        # Weighted mean by information score
        eps = 1e-8
        selected_info_score = info_scores
        # Handle both 1D and 2D tensors
        if selected_info_score.dim() > 1:
            selected_info_score = selected_info_score.squeeze(-1)
        selected_info_score = selected_info_score.to(device)
        raw_weights = (selected_info_score + eps)
        weights = raw_weights / raw_weights.sum()
        return torch.sum(ms_features * weights.unsqueeze(-1), dim=0).float()
        
    elif method == "topk":
        # Mean of top-k spectra by information score
        selected_info_score = info_scores
        # Handle both 1D and 2D tensors
        if selected_info_score.dim() > 1:
            selected_info_score = selected_info_score.squeeze(-1)
        selected_info_score = selected_info_score.to(device)
        k = min(topk, selected_info_score.shape[0])
        _, topk_idx = torch.topk(selected_info_score, k=k, largest=True)
        return torch.mean(ms_features[topk_idx], dim=0)
    
    elif method == "centroid_closest":
        # Select only the spectrum closest to centroid (or both if N=2)
        # This is handled above, so just return the mean of selected spectra
        return torch.mean(ms_features, dim=0)
        
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Choose from 'mean', 'weighted', 'topk', or 'centroid_closest'")



@torch.no_grad()
def evaluate_with_aggregation(model, val_dataloader, graph_dict, args, eval_samples=512, 
                              plot=True, aggregation_method="mean"):
    
    model.eval()

    data = iter(val_dataloader)

    MAE = nn.L1Loss()
    sigmoid = nn.Sigmoid()
    mode = getattr(args, "mode", None)

    if mode == "mixed":
        mode = "contrastive"
        mixed = True
    else:
        mixed = False

    topk = getattr(args, "topk", 3)

    list_ms_features = []
    list_graph_features = []

    iterations = int(np.ceil(eval_samples / args.batch_size)) 

    print(f"\n{'='*60}")
    print(f"Starting evaluation with aggregation method: {aggregation_method}")
    if aggregation_method == "topk":
        print(f"Using top-{topk} spectra per molecule")
    if aggregation_method == "centroid_closest":
        print(f"Using centroid-closest selection (both if N=2)")
    if getattr(args, 'outlier_filter', False):
        outlier_threshold = getattr(args, 'outlier_threshold', 2.0)
        print(f"Outlier filtering enabled (centroid method, threshold={outlier_threshold})")
    print(f"Evaluating {eval_samples} samples in {iterations} iterations")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")

    for it in tqdm(range(iterations), desc="Processing batches"):
        sos, formula_array, mask, smiles, info_score = next(data)
        
        if it == (iterations - 1) and eval_samples % args.batch_size != 0:
            bs = eval_samples % args.batch_size
        else:
            bs = args.batch_size

        # Get unique SMILES in this batch
        unique_smiles = []
        for smi in smiles:
            if smi not in unique_smiles:
                unique_smiles.append(smi)

        for smi in unique_smiles[:bs]:

            indices = [i for i in range(len(smiles)) if smiles[i] == smi]

            if mode == 'contrastive':
                # Build graph batch
                graph_list = [graph_dict[smiles[i]][0] for i in indices]
                graph_batch = batch_graphs_to_padded_data(graph_list, device=args.device)

                if mixed:
                    ms_features, graph_features, _ = model(
                        sos[indices], formula_array[indices], 
                        graph=graph_batch, mask=mask[indices]
                    )
                else:
                    ms_features, graph_features = model(
                        sos[indices], formula_array[indices], 
                        graph=graph_batch, mask=mask[indices]
                    )
                        
                # Normalize features
                ms_features = ms_features / ms_features.norm(dim=1, keepdim=True)
                graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)

            else:  # mode == 'fp'
                # Get fingerprint features
                fp_features = torch.cat(
                    [graph_dict[smiles[i]][1] for i in indices], dim=0
                ).to(args.device).float()           
                preds = model(sos[indices], formula_array[indices], mask=mask[indices])
                ms_features = sigmoid(preds)
                graph_features = fp_features

            # Aggregate MS features for this molecule
            selected_info_score = info_score[indices]
            centroid_closest = (aggregation_method == "centroid_closest")
            aggregated_ms = aggregate_ms_features(
                ms_features, selected_info_score, 
                method=aggregation_method, topk=topk, device=args.device,
                outlier_filter=getattr(args, 'outlier_filter', False),
                outlier_threshold=getattr(args, 'outlier_threshold', 2.0),
                centroid_closest=centroid_closest
            )

            list_ms_features.append(aggregated_ms)
            list_graph_features.append(graph_features[0])
    
    # Stack all features
    ms_features = torch.stack(list_ms_features)
    graph_features = torch.stack(list_graph_features)

    print(f"\nComputing metrics on {len(list_ms_features)} unique molecules...")

    # Compute logits and similarity
    temp = 1.0 / model.inv_temperature if hasattr(model, 'inv_temperature') else args.temperature
    logits_per_ms, logits_per_mol, similarity = compute_contrastive_logits(
        ms_features, graph_features, temp, calc_similarity=True
    )

    print(f"Average cosine similarity: {similarity:.4f}")

    # Compute MAE loss
    loss = MAE(ms_features, graph_features)

    # Normalize for classification
    ms_features = ms_features / ms_features.norm(dim=1, keepdim=True)
    graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
    
    # Compute pairwise matching accuracy
    ms_class = torch.argmax(logits_per_ms, 1)
    mol_class = torch.argmax(logits_per_mol, 1)

    labels = torch.arange(ms_class.shape[0]).to(args.device)

    acc_1 = torch.sum(torch.eq(ms_class, labels))
    acc_2 = torch.sum(torch.eq(mol_class, labels))

    total_acc = (acc_1 + acc_2) / (2 * ms_class.shape[0])

    # Build metrics dictionary
    metrics_dict = {
        'Pairwise matching accuracy': total_acc.item(), 
        'Eval_loss (MAE)': loss.item(), 
        'cosine_similarity': similarity,
        "multi_MS_per_mol": args.multi_MS_per_mol, 
        "ordered": args.ordered_batches,
        "aggregation_method": aggregation_method
    }
    if aggregation_method == "topk":
        metrics_dict["topk"] = topk
    if getattr(args, 'outlier_filter', False):
        metrics_dict["outlier_filter"] = True
        metrics_dict["outlier_threshold"] = getattr(args, 'outlier_threshold', 2.0)

    print(f"\n{'='*60}")
    print("Evaluation Results:")
    print(f"  Pairwise matching accuracy: {total_acc.item():.4f}")
    print(f"  MAE Loss: {loss.item():.4f}")
    print(f"  Cosine Similarity: {similarity:.4f}")
    print(f"{'='*60}\n")

    # Generate visualization if requested
    if plot:
        print("Generating t-SNE visualization...")
        plot_embeddings_with_metrics(
            ms_features, graph_features, ms_class, metrics_dict, 
            outdir="analysis_outputs",
            reducer_kwargs={"perplexity": 30}
        )

    return metrics_dict


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_embeddings_with_metrics(
    ms_features,
    graph_features,
    ms_class,
    metrics_dict,
    outdir="analysis_outputs",
    reducer="tsne",
    reducer_kwargs=None,
    random_state=42,
    pca_dim=None,
):
    """
    Plot 2D projection of embeddings (MS + graph) with a metrics textbox.
    """
    reducer = reducer.lower()
    reducer_kwargs = reducer_kwargs or {}

    metrics_text = (
        f"Metrics:  "
        f"Acc: {metrics_dict['Pairwise matching accuracy']:.3f}  "
        f"MAE: {metrics_dict['Eval_loss (MAE)']:.3f}  "
        f"Sim: {metrics_dict['cosine_similarity']:.3f}  "
        f"Multi-MS: {metrics_dict['multi_MS_per_mol']}  "
        f"Order: {metrics_dict['ordered']}  "
        f"Agg: {metrics_dict.get('aggregation_method', 'N/A')}"
    )
    if 'topk' in metrics_dict:
        metrics_text += f" (k={metrics_dict['topk']})"

    # Stack embeddings
    all_embeddings = torch.cat([ms_features[:512], graph_features[:512]], dim=0).detach().float().cpu().numpy()

    # Optional PCA pre-reduction
    if pca_dim is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=int(pca_dim), random_state=random_state)
        all_embeddings = pca.fit_transform(all_embeddings)

    N = all_embeddings.shape[0]

    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, **reducer_kwargs)
    emb2d_fp = tsne.fit(all_embeddings[N//2:])
    embeddings_2d = emb2d_fp.transform(all_embeddings)
    
    xlab, ylab, title = "t-SNE 1", "t-SNE 2", "t-SNE of Embeddings"
    fname_prefix = "tsne"

    # Create figure with plot and metrics box
    fig, (ax, ax_box) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={'height_ratios': [6, 1], 'hspace': 0.15}
    )


    # Scatter plot
    ax.scatter(
        embeddings_2d[:N//2, 0],
        embeddings_2d[:N//2, 1],
        color="red", alpha=0.5, s=10, label="MS"
    )
    ax.scatter(
        embeddings_2d[N//2:, 0],
        embeddings_2d[N//2:, 1],
        color="green", alpha=0.5, s=10, label="Graph"
    )

    # Outline corresponding pairs (first 10)
    colors_list = [
        "blue", "orange", "purple", "brown", "pink",
        "gray", "olive", "cyan", "magenta", "yellow"
    ]
    k = min(len(colors_list), N//2, embeddings_2d.shape[0] - N//2)
    for i, color in enumerate(colors_list[:k]):
        ax.scatter(
            embeddings_2d[i:i+1, 0], embeddings_2d[i:i+1, 1],
            facecolors="none", edgecolors=color, linewidths=0.5, s=30
        )
        ax.scatter(
            embeddings_2d[N//2+i:N//2+i+1, 0], embeddings_2d[N//2+i:N//2+i+1, 1],
            facecolors="none", edgecolors=color, linewidths=0.5, s=30
        )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()

    # Metrics box
    ax_box.axis('off')
    ax_box.set_frame_on(True)
    ax_box.set_facecolor('white')
    ax_box.text(
        0.5, 0.5, metrics_text,
        va='center', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.8, alpha=1.0),
        transform=ax_box.transAxes
    )

    # Save plot
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(outdir, f"{fname_prefix}_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {filename}")
    return filename


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Keep old function names for backward compatibility
padded_batch_to_data = batch_graphs_to_padded_data
features_to_logits = compute_contrastive_logits
training_evaluation = evaluate_during_training
evaluation_batches = evaluate_with_aggregation

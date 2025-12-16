import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import scanpy as sc
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
from scib.metrics import silhouette, silhouette_batch, ilisi_graph, pcr, graph_connectivity
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def find_leiden_resolution_for_n_clusters(adata, target_n_clusters, min_res=0.1, max_res=1.0, 
                                          max_iterations=30, tolerance=0, random_state=42):
    """
    Use binary search to find the Leiden resolution parameter that produces exactly the target number of clusters.
    
    Args:
        adata: AnnData object with neighbors already computed
        target_n_clusters: Target number of clusters (default: 12)
        min_res: Minimum resolution to search (default: 0.4)
        max_res: Maximum resolution to search (default: 0.7)
        max_iterations: Maximum number of binary search iterations (default: 20)
        tolerance: Acceptable difference from target (default: 0, meaning exact match)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (optimal_resolution, actual_n_clusters)
    """
    low, high = min_res, max_res
    best_resolution = None
    best_n_clusters = None
    best_diff = float('inf')
    
    print(f"Binary search for resolution to get {target_n_clusters} clusters...")
    
    for iteration in range(max_iterations):
        mid_res = (low + high) / 2.0
        
        # Perform Leiden clustering with current resolution
        sc.tl.leiden(adata, resolution=mid_res, key_added='leiden', random_state=random_state)
        n_clusters = len(adata.obs['leiden'].unique())
        
        print(f"  Iteration {iteration + 1}: resolution={mid_res:.4f}, n_clusters={n_clusters}")
        
        # Track the best result so far
        diff = abs(n_clusters - target_n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_resolution = mid_res
            best_n_clusters = n_clusters
        
        # Check if we've found the target
        if abs(n_clusters - target_n_clusters) <= tolerance:
            print(f"  ✓ Found exact match: resolution={mid_res:.4f}, n_clusters={n_clusters}")
            return mid_res, n_clusters
        
        # Adjust search range
        # Higher resolution typically means more clusters
        if n_clusters < target_n_clusters:
            low = mid_res
        else:
            high = mid_res
        
        # If the search range becomes too small, break
        if high - low < 0.0001:
            print(f"  Search range too small, stopping. Best: resolution={best_resolution:.4f}, n_clusters={best_n_clusters}")
            break
    
    # If exact match not found, use the best result
    print(f"  Using best result: resolution={best_resolution:.4f}, n_clusters={best_n_clusters} (diff={best_diff})")
    
    # Re-run with the best resolution to ensure adata has the correct clustering
    sc.tl.leiden(adata, resolution=best_resolution, key_added='leiden', random_state=random_state)
    
    return best_resolution, best_n_clusters


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', 
                        type=str, 
                        default='Kanemaru2023_full',
                        choices=['Kanemaru2023_full', 'Li2023b_full'],
                        help='Dataset name')
    parser.add_argument('--embeddings_path', 
                        type=str, 
                        default=None,
                        help='Path to ChromFound embeddings h5ad file (optional, will use default based on dataset_name if not provided)')
    parser.add_argument('--batch_key', 
                        type=str, 
                        default="Batch (HSC)",
                        help='Batch key column name in obs (optional, will try to detect if not provided)')
    parser.add_argument('--root', 
                        type=str, 
                        default='/scratch/wkim/project-2-team-1/ChromFound-Parallel/results-copy/chromfound',
                        help='Root directory for ChromFound embeddings')
    parser.add_argument('--seed', type=int, default=42)
    return parser


parser = get_args_parser()
args = parser.parse_args()

assert args.embeddings_path is not None, "Embeddings path is required"
embeddings_path = Path(args.embeddings_path)
# if args.embeddings_path is None:
#     root = Path(args.root)
#     if args.dataset_name == 'Kanemaru2023_downsampled':
#         embeddings_path = root / 'Kanemaru2023_downsampled' / 'merge10' / 'embeddings_pca.h5ad'
#     elif args.dataset_name == 'Buenrostro2018-bone_marrow_tissue':
#         embeddings_path = root / 'Buenrostro2018-bone_marrow_tissue' / 'merge10' / 'embeddings.h5ad'
#     elif args.dataset_name == 'Li2023b_downsampled':
#         embeddings_path = root / 'Li2023b_downsampled' / 'merge10' / 'embeddings_pca.h5ad'
#     else:
#         raise ValueError(f"Dataset {args.dataset_name} not supported")
# else:
#     embeddings_path = Path(args.embeddings_path)

# Load ChromFound embeddings
print(f"Loading ChromFound embeddings from {embeddings_path}...")
adata = sc.read_h5ad(embeddings_path)

# Extract embeddings from obsm (different datasets use different keys)
embedding_key_map = {
    # 'Kanemaru2023_downsampled': 'X_pca',
    # 'Buenrostro2018-bone_marrow_tissue': 'X_embedding',
    # 'Li2023b_downsampled': 'X_pca',
    'Kanemaru2023_full': 'X_pca',
    'Li2023b_full': 'X_pca',
}

if args.dataset_name in embedding_key_map:
    embedding_key = embedding_key_map[args.dataset_name]
else:
    # Try common keys
    embedding_key = None
    for key in ['X_pca', 'X_embedding']:
        if key in adata.obsm:
            embedding_key = key
            break

if embedding_key is None or embedding_key not in adata.obsm:
    available_keys = list(adata.obsm.keys())
    raise ValueError(f"No suitable embedding key found in adata.obsm. Available keys: {available_keys}")

cell_embeddings = adata.obsm[embedding_key]
print(f"Loaded embeddings from '{embedding_key}' with shape: {cell_embeddings.shape}")

# Check for cell_type column
if 'cell_type' not in adata.obs.columns:
    if 'celltype' in adata.obs.columns:
        adata.obs['cell_type'] = adata.obs['celltype']
        print("Using 'celltype' column as 'cell_type'")
    else:
        raise ValueError("No 'cell_type' or 'celltype' column found in adata.obs. Check your h5ad file.")

num_cell_types = len(adata.obs['cell_type'].unique())
print(f"Number of cell types in the dataset: {num_cell_types}")

# Set default batch key based on dataset if not provided
if args.batch_key is None:
    # Set default batch keys for each dataset (matching the original script)
    default_batch_keys = {
        'Kanemaru2023_downsampled': 'batch_key',
        'Buenrostro2018-bone_marrow_tissue': 'Batch (HSC)',
        'Li2023b_downsampled': 'Batch (HSC)'
    }
    if args.dataset_name in default_batch_keys:
        default_key = default_batch_keys[args.dataset_name]
        if default_key in adata.obs.columns:
            args.batch_key = default_key
            print(f"Using default batch key for {args.dataset_name}: {args.batch_key}")
        else:
            print(f"Warning: Default batch key '{default_key}' not found for {args.dataset_name}.")
            args.batch_key = None
    else:
        args.batch_key = None

# If still None, try to detect batch key
if args.batch_key is None:
    # Common batch key names to check
    possible_batch_keys = ['batch', 'Batch', 'batch_key', 'Batch (HSC)', 'sample', 'Sample']
    batch_key = None
    for key in possible_batch_keys:
        if key in adata.obs.columns:
            batch_key = key
            print(f"Detected batch key: {batch_key}")
            break
    
    if batch_key is None:
        print("Warning: No batch key found. Batch-related metrics will be skipped.")
        print(f"Available columns: {list(adata.obs.columns)}")
        args.batch_key = None
    else:
        args.batch_key = batch_key
else:
    if args.batch_key not in adata.obs.columns:
        print(f"Warning: Specified batch key '{args.batch_key}' not found in obs columns.")
        print(f"Available columns: {list(adata.obs.columns)}")
        args.batch_key = None

# Assign embeddings to the AnnData object
adata.obsm['cell_embeddings_zero_shot'] = cell_embeddings

# UMAP visualization
print("Visualizing UMAP...")
sc.pp.neighbors(adata, use_rep='cell_embeddings_zero_shot')
sc.tl.umap(adata)

# Plot UMAP with original cell types and capture the figure
fig = sc.pl.umap(adata, color='cell_type', return_fig=True, show=True, title='Cell embeddings (true labels)')

# Customize legend to show "cell type" as description
if fig is not None:
    axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('Cell type')

output_dir = Path(f'./zero_shot_feature_extraction_chromfound_{args.dataset_name}')
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'umap_cell_types_true_labels.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("UMAP visualization saved")

# Plot UMAP with batch labels if batch key exists
if args.batch_key is not None:
    fig = sc.pl.umap(adata, color=args.batch_key, return_fig=True, show=True, title='Cell embeddings (batch labels)')
    if fig is not None:
        axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title('Batch')
    plt.savefig(output_dir / 'umap_batch_labels.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("UMAP visualization with batch labels saved")

n_true_cell_types = len(adata.obs['cell_type'].unique())
print(f"Number of cell types: {n_true_cell_types}")

# Perform leiden clustering with binary search to get exactly num_cell_types clusters
# Note: neighbors are already computed using the embeddings
optimal_resolution, n_clusters = find_leiden_resolution_for_n_clusters(
    adata, 
    target_n_clusters=num_cell_types, 
    min_res=0.1, 
    max_res=2.0, 
    random_state=42
)

print(f"Optimal resolution: {optimal_resolution:.4f}")
print(f"Number of Leiden clusters: {n_clusters}")

print("Visualizing UMAP with leiden clustering...")
fig = sc.pl.umap(adata, color='leiden', legend_loc='on data', title='Leiden Clustering (ChromFound)', return_fig=True, show=True)
plt.savefig(output_dir / 'umap_leiden_clustering.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("UMAP visualization with leiden clustering saved")

# Calculate NMI and ARI scores
print("Calculating NMI and ARI scores...")
# Get the true labels (cell types) and predicted labels (Leiden clusters)
true_labels = adata.obs['cell_type'].values
predicted_labels = adata.obs['leiden'].values

# Calculate ARI (Adjusted Rand Index)
ari_score = adjusted_rand_score(true_labels, predicted_labels)

# Calculate NMI (Normalized Mutual Information)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

# Calculate Silhouette scores
silhouette_score = silhouette(adata, label_key='cell_type', embed='cell_embeddings_zero_shot')
if args.batch_key is not None:
    silhouette_batch_score = silhouette_batch(adata, label_key='cell_type', batch_key=args.batch_key, embed='cell_embeddings_zero_shot')
else:
    silhouette_batch_score = None
    print("Skipping silhouette_batch_score (no batch key available)")

# Calculate mean silhouette per group
cell_embeddings = adata.obsm['cell_embeddings_zero_shot']
cell_types = adata.obs['cell_type'].values
silhouette_samples_scores = silhouette_samples(cell_embeddings, cell_types)
mean_silhouette_per_group = pd.Series(silhouette_samples_scores, index=cell_types).groupby(cell_types).mean().mean()

# Calculate graph connectivity (biological conservation metric - measures if same cell types stay connected)
# Higher is better (1 = fully connected, 0 = disconnected)
print("Calculating graph connectivity...")
graph_connectivity_score = graph_connectivity(adata, label_key='cell_type')
print(f"  Graph connectivity score: {graph_connectivity_score:.4f}")

# Calculate batch integration metrics (ilisi, pcr_batch)
if args.batch_key is not None:
    print("Calculating batch integration metrics...")
    
    # Calculate ilisi (Integration Local Inverse Simpson's Index)
    # Higher is better (more batch mixing)
    print("  Calculating ilisi...")
    ilisi_score = ilisi_graph(adata, batch_key=args.batch_key, type_='embed', use_rep='cell_embeddings_zero_shot')
    
    # Calculate PCR batch (Principal Component Regression)
    # Lower is better (less batch effect), so we'll store 1-pcr for "higher is better" interpretation
    print("  Calculating PCR batch...")
    pcr_batch_score_raw = pcr(adata, covariate=args.batch_key, embed='cell_embeddings_zero_shot', recompute_pca=True)
    pcr_batch_score = 1 - pcr_batch_score_raw  # Invert so higher is better
    
    print(f"  ilisi score: {ilisi_score:.4f} (higher is better)")
    print(f"  PCR batch score (1-PCR): {pcr_batch_score:.4f} (higher is better, original PCR: {pcr_batch_score_raw:.4f})")
else:
    ilisi_score = None
    pcr_batch_score = None
    print("Skipping batch integration metrics (no batch key available)")

# Train linear probe on cell embeddings using cell type labels
print("Training linear probe on embeddings...")
cell_types = adata.obs['cell_type'].values

X_train, X_val, y_train, y_val = train_test_split(
    cell_embeddings,
    cell_types,
    test_size=0.2,
    stratify=cell_types,
    random_state=SEED
)

print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples...")
linear_probe = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
    multi_class='multinomial',
    solver='lbfgs',
    verbose=1  # Show convergence progress (prints iteration info)
)
with tqdm(desc="Training linear probe", total=100, ncols=80) as pbar:
    linear_probe.fit(X_train, y_train)
    pbar.update(100)
y_val_pred = linear_probe.predict(X_val)
linear_probe_accuracy = accuracy_score(y_val, y_val_pred)
linear_probe_f1_macro = f1_score(y_val, y_val_pred, average='macro')
linear_probe_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')

print(f"Linear probe accuracy: {linear_probe_accuracy:.4f}")
print(f"Linear probe F1 score (macro): {linear_probe_f1_macro:.4f}")
print(f"Linear probe F1 score (weighted): {linear_probe_f1_weighted:.4f}")

# Train linear probe on embeddings using batch labels (if available)
if args.batch_key is not None:
    print("\nTraining linear probe on embeddings (batch labels)...")
    batch_labels = adata.obs[args.batch_key].values
    X_train_batch, X_val_batch, y_train_batch, y_val_batch = train_test_split(
        cell_embeddings,
        batch_labels,
        test_size=0.2,
        stratify=batch_labels,
        random_state=SEED
    )

    print(f"Training on {len(X_train_batch)} samples, validating on {len(X_val_batch)} samples...")
    linear_probe_batch = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        multi_class='multinomial',
        solver='lbfgs',
        verbose=1  # Show convergence progress (prints iteration info)
    )
    # Use tqdm to show training is in progress
    with tqdm(desc="Training linear probe (batch)", total=100, ncols=80) as pbar:
        linear_probe_batch.fit(X_train_batch, y_train_batch)
        pbar.update(100)  # Complete the progress bar when done

    # Evaluate on validation set
    y_val_pred_batch = linear_probe_batch.predict(X_val_batch)
    linear_probe_batch_accuracy = accuracy_score(y_val_batch, y_val_pred_batch)
    linear_probe_batch_f1_macro = f1_score(y_val_batch, y_val_pred_batch, average='macro')
    linear_probe_batch_f1_weighted = f1_score(y_val_batch, y_val_pred_batch, average='weighted')

    print(f"Linear probe accuracy (batch): {linear_probe_batch_accuracy:.4f}")
    print(f"Linear probe F1 score (macro, batch): {linear_probe_batch_f1_macro:.4f}")
    print(f"Linear probe F1 score (weighted, batch): {linear_probe_batch_f1_weighted:.4f}")
else:
    linear_probe_batch_accuracy = None
    linear_probe_batch_f1_macro = None
    linear_probe_batch_f1_weighted = None
    print("\nSkipping batch linear probe (no batch key available)")

print(f"\nAdjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Silhouette score: {silhouette_score:.4f}")
if silhouette_batch_score is not None:
    print(f"Silhouette batch score: {silhouette_batch_score:.4f}")
print(f"Mean silhouette per group: {mean_silhouette_per_group:.4f}")
print(f"Graph connectivity score: {graph_connectivity_score:.4f}")
if ilisi_score is not None:
    print(f"ilisi score: {ilisi_score:.4f}")
if pcr_batch_score is not None:
    print(f"PCR batch score (1-PCR): {pcr_batch_score:.4f}")
print(f"Linear probe accuracy: {linear_probe_accuracy:.4f}")
print(f"Linear probe F1 score (macro): {linear_probe_f1_macro:.4f}")
print(f"Linear probe F1 score (weighted): {linear_probe_f1_weighted:.4f}")
if linear_probe_batch_accuracy is not None:
    print(f"Linear probe batch accuracy: {linear_probe_batch_accuracy:.4f}")
    print(f"Linear probe batch F1 score (macro): {linear_probe_batch_f1_macro:.4f}")
    print(f"Linear probe batch F1 score (weighted): {linear_probe_batch_f1_weighted:.4f}")
print(f"\nNumber of true cell types: {len(adata.obs['cell_type'].unique())}")
print(f"Number of predicted clusters: {len(adata.obs['leiden'].unique())}")

# Save results to CSV
results_data = {
    'Metric': [
        'Adjusted Rand Index', 
        'Normalized Mutual Information', 
        'Silhouette score', 
        'Mean silhouette per group',
        'Cell type Linear probe accuracy',
        'Cell type Linear probe F1 score (macro)',
        'Cell type Linear probe F1 score (weighted)',
    ],
    'Value': [
        ari_score, 
        nmi_score, 
        silhouette_score, 
        mean_silhouette_per_group,
        linear_probe_accuracy,
        linear_probe_f1_macro,
        linear_probe_f1_weighted,
    ]
}

if silhouette_batch_score is not None:
    results_data['Metric'].insert(3, 'Silhouette batch score')
    results_data['Value'].insert(3, silhouette_batch_score)

# Add graph connectivity (biological conservation metric)
results_data['Metric'].append('Graph connectivity score')
results_data['Value'].append(graph_connectivity_score)

# Add batch integration metrics
if ilisi_score is not None:
    results_data['Metric'].append('ilisi score')
    results_data['Value'].append(ilisi_score)

if pcr_batch_score is not None:
    results_data['Metric'].append('PCR batch score (1-PCR)')
    results_data['Value'].append(pcr_batch_score)

if linear_probe_batch_accuracy is not None:
    results_data['Metric'].extend([
        'Batch label Linear probe accuracy',
        'Batch label Linear probe F1 score (macro)',
        'Batch label Linear probe F1 score (weighted)'
    ])
    results_data['Value'].extend([
        linear_probe_batch_accuracy,
        linear_probe_batch_f1_macro,
        linear_probe_batch_f1_weighted,
    ])

results_df = pd.DataFrame(results_data)
results_file = output_dir / 'results.csv'
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

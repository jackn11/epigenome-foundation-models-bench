import random
import argparse
import numpy as np
import torch
from pathlib import Path
import scanpy as sc
import os
from scib.metrics import ilisi_graph, pcr
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
                        help='Path to ChromFound embeddings h5ad file (required)')
    parser.add_argument('--batch_key', 
                        type=str, 
                        default="Batch (HSC)",
                        help='Batch key column name in obs (optional, will try to detect if not provided)')
    parser.add_argument('--seed', type=int, default=42)
    return parser


parser = get_args_parser()
args = parser.parse_args()

assert args.embeddings_path is not None, "Embeddings path is required"
embeddings_path = Path(args.embeddings_path)

# Load ChromFound embeddings
print(f"Loading ChromFound embeddings from {embeddings_path}...")
adata = sc.read_h5ad(embeddings_path)

# Extract embeddings from obsm (different datasets use different keys)
embedding_key_map = {
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

# Assign embeddings to the AnnData object
adata.obsm['cell_embeddings_zero_shot'] = cell_embeddings

# Set default batch key based on dataset if not provided
if args.batch_key is None:
    # Set default batch keys for each dataset
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
        raise ValueError("No batch key found. Batch integration metrics require a batch key.")
        print(f"Available columns: {list(adata.obs.columns)}")
    else:
        args.batch_key = batch_key
else:
    if args.batch_key not in adata.obs.columns:
        raise ValueError(f"Specified batch key '{args.batch_key}' not found in obs columns.")
        print(f"Available columns: {list(adata.obs.columns)}")

# Compute neighbors (required for ilisi_graph)
print("Computing neighbors for ilisi calculation...")
sc.pp.neighbors(adata, use_rep='cell_embeddings_zero_shot')

# Calculate batch integration metrics (ilisi, pcr_batch)
print("Calculating batch integration metrics...")

# Calculate ilisi (Integration Local Inverse Simpson's Index)
# Higher is better (more batch mixing)
print("  Calculating ilisi...")
ilisi_score = ilisi_graph(adata, batch_key=args.batch_key, type_='embed', use_rep='cell_embeddings_zero_shot', n_cores=128)

# Calculate PCR batch (Principal Component Regression)
# Lower is better (less batch effect), so we'll store 1-pcr for "higher is better" interpretation
print("  Calculating PCR batch...")
pcr_batch_score_raw = pcr(adata, covariate=args.batch_key, embed='cell_embeddings_zero_shot', recompute_pca=True)
pcr_batch_score = 1 - pcr_batch_score_raw  # Invert so higher is better

print(f"  ilisi score: {ilisi_score:.4f} (higher is better)")
print(f"  PCR batch score (1-PCR): {pcr_batch_score:.4f} (higher is better, original PCR: {pcr_batch_score_raw:.4f})")

# Save results to CSV
output_dir = Path(f'./zero_shot_batch_integration_metrics_chromfound_{args.dataset_name}')
output_dir.mkdir(exist_ok=True)

results_data = {
    'Metric': [
        'ilisi score',
        'PCR batch score (1-PCR)',
        'PCR batch score (raw)',
    ],
    'Value': [
        ilisi_score,
        pcr_batch_score,
        pcr_batch_score_raw,
    ]
}

results_df = pd.DataFrame(results_data)
results_file = output_dir / 'results.csv'
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

print("\n=== Batch Integration Metrics Summary ===")
print(f"ilisi score: {ilisi_score:.4f} (higher is better)")
print(f"PCR batch score (1-PCR): {pcr_batch_score:.4f} (higher is better)")
print(f"PCR batch score (raw): {pcr_batch_score_raw:.4f} (lower is better)")

import random
import argparse
import numpy as np
import torch
from pathlib import Path
import scanpy as sc
import os
from scib.metrics import ilisi_graph, pcr
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def ilisi_graph_custom(
    adata,
    batch_key: str,
    connectivity_key: str = "connectivities",
    use_rep: str | None = None,          # unused: kept for API compatibility
    type_: str = "embed",                # unused: kept for API compatibility
    n_cores: int = 1,
    subsample: int | None = None,
    random_state: int = 0,
    exclude_self: bool = True,
    eps: float = 1e-12,
) -> float:
    """
    Compute iLISI (integration Local Inverse Simpson's Index) using a precomputed kNN graph.

    iLISI per cell i:
        p_b = (sum of neighbor weights with batch=b) / (sum of all neighbor weights)
        iLISI_i = 1 / sum_b p_b^2

    Returns:
        Mean iLISI across (subsampled) cells. Higher = better batch mixing.
    """
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key='{batch_key}' not found in adata.obs.columns")

    if adata.obsp is None or connectivity_key not in adata.obsp.keys():
        raise ValueError(
            f"Could not find adata.obsp['{connectivity_key}']. "
            f"Run sc.pp.neighbors(adata, ...) first."
        )

    G = adata.obsp[connectivity_key]
    if not sparse.issparse(G):
        G = sparse.csr_matrix(G)
    else:
        G = G.tocsr()

    # Encode batch labels as integers 0..B-1
    batch_codes, batch_uniques = pd.factorize(adata.obs[batch_key].astype("category"), sort=True)
    batch_codes = batch_codes.astype(np.int32)
    n = adata.n_obs

    # Choose which cells to score (optional subsample)
    rng = np.random.default_rng(random_state)
    if subsample is not None and subsample < n:
        idx = rng.choice(n, size=subsample, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n, dtype=np.int64)

    def ilisi_one(i: int) -> float:
        start, end = G.indptr[i], G.indptr[i + 1]
        neigh = G.indices[start:end]
        w = G.data[start:end].astype(np.float64, copy=False)

        if neigh.size == 0:
            return np.nan

        if exclude_self:
            mask = neigh != i
            neigh = neigh[mask]
            w = w[mask]

        if neigh.size == 0:
            return np.nan

        wsum = float(w.sum())
        if not np.isfinite(wsum) or wsum <= eps:
            return np.nan

        # Weighted batch mass among neighbors
        bc = batch_codes[neigh]
        # bincount with weights -> mass per batch
        mass = np.bincount(bc, weights=w, minlength=len(batch_uniques)).astype(np.float64, copy=False)

        p = mass / wsum
        denom = float(np.sum(p * p))
        if denom <= eps or not np.isfinite(denom):
            return np.nan

        return 1.0 / denom

    if n_cores is None or n_cores < 1:
        n_cores = 1

    scores = Parallel(n_jobs=n_cores, prefer="processes", batch_size="auto")(
        delayed(ilisi_one)(int(i)) for i in tqdm(idx, desc="Computing iLISI scores", total=len(idx))
    )

    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise RuntimeError("All iLISI scores were NaN/inf. Check your neighbor graph / batch labels.")

    return float(scores.mean())


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
print(f"Neighbors computed for {adata.n_obs} cells")

# Calculate batch integration metrics (ilisi, pcr_batch)
print("Calculating batch integration metrics...")

# Calculate ilisi (Integration Local Inverse Simpson's Index)
# Higher is better (more batch mixing)
print("  Calculating ilisi...")
# ilisi_score = ilisi_graph(adata, batch_key=args.batch_key, type_='embed', use_rep='cell_embeddings_zero_shot', n_cores=128)
ilisi_score = ilisi_graph_custom(
    adata,
    batch_key=args.batch_key,
    # n_cores=128,
    subsample=None,          # set e.g. 50000 for speed on huge datasets
    random_state=args.seed,
    exclude_self=True,
)
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

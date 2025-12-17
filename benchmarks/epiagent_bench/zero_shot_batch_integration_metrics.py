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
import sys

from scib.metrics import graph_connectivity

project_root = Path(__file__).parent.parent.parent
epiagent_path = project_root / 'EpiAgent'
sys.path.insert(0, str(epiagent_path))

from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF

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
    # use_rep: str | None = None,          # unused: kept for API compatibility
    # type_: str = "embed",                # unused: kept for API compatibility
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
                        default='Li2023b',
                        choices=['Li2023b', 'Buenrostro2018', 'Kanemaru2023'],
                        help='Dataset name')
    parser.add_argument('--batch_key', 
                        type=str, 
                        default='Batch (HSC)',
                        help='Batch key column name in obs')
    parser.add_argument('--root', 
                        type=str, 
                        default='/scratch/naimer/github/project-2-team-1/EpiAgent/data',
                        help='Root directory for datasets')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_cores', type=int, default=1, help='Number of cores for parallel processing')
    return parser


parser = get_args_parser()
args = parser.parse_args()

root = Path(args.root)

# Load the original dataset to get metadata
print("Loading the dataset...")
if args.dataset_name == 'Li2023b':
    input_path = root / 'Li2023b' / 'Li2023b-brain_tissue' / 'Li2023b-brain_tissue-cell_by_cCRE.h5ad'
elif args.dataset_name == 'Kanemaru2023':
    input_path = root / 'Kanemaru2023' / 'Kanemaru2023-cardiac_tissue' / 'Kanemaru2023-cardiac_tissue-cell_by_cCRE.h5ad'
else:
    raise ValueError(f"Dataset {args.dataset_name} not supported")

adata = sc.read_h5ad(input_path)

# Load cached tokenized data or create it
cache_dir = Path('./cache')
cache_dir.mkdir(exist_ok=True)
cached_tokenized_path = cache_dir / f'{args.dataset_name}_tokenized.h5ad'

if cached_tokenized_path.exists():
    print(f"Loading cached tokenized data from {cached_tokenized_path}...")
    adata_tfidf = sc.read_h5ad(cached_tokenized_path)
    print("Cached tokenized data loaded successfully.")
else:
    print("No cached tokenized data found. Computing from scratch...")
    cCRE_document_frequency = np.load(root / 'cCRE_document_frequency.npy')
    adata_tfidf = global_TFIDF(adata, cCRE_document_frequency)
    
    print("Performing tokenization... (this takes a while)")
    tokenization(adata_tfidf)
    
    print(f"Saving tokenized data to {cached_tokenized_path}...")
    adata_tfidf.write(cached_tokenized_path)
    print("Tokenized data saved successfully.")

# Load EpiAgent embeddings from saved file
embedding_save_path = f'./saved_embeddings/{args.dataset_name}/{args.dataset_name}-cell_embeddings.npy'

if not os.path.exists(embedding_save_path):
    raise FileNotFoundError(
        f"Embeddings file not found at {embedding_save_path}. "
        f"Please run zero_shot_feature_extraction_noshuffle.py first to generate the embeddings."
    )

print(f"Loading saved cell embeddings from {embedding_save_path}...")
cell_embeddings = np.load(embedding_save_path)
print(f"Cell embeddings loaded successfully with shape: {cell_embeddings.shape}")

# Assign embeddings to the AnnData object
adata_tfidf.obsm['cell_embeddings_zero_shot'] = cell_embeddings

# Verify batch key exists
if args.batch_key not in adata_tfidf.obs.columns:
    raise ValueError(
        f"Specified batch key '{args.batch_key}' not found in obs columns. "
        f"Available columns: {list(adata_tfidf.obs.columns)}"
    )

print(f"Using batch key: {args.batch_key}")

# Compute neighbors (required for ilisi_graph)
print("Computing neighbors for ilisi calculation...")
sc.pp.neighbors(adata_tfidf, use_rep='cell_embeddings_zero_shot')
print(f"Neighbors computed for {adata_tfidf.n_obs} cells")

# Calculate batch integration metrics (ilisi, pcr_batch)
print("Calculating batch integration metrics...")

# Calculate ilisi (Integration Local Inverse Simpson's Index)
# Higher is better (more batch mixing)
print("  Calculating ilisi...")
ilisi_score = ilisi_graph_custom(
    adata_tfidf,
    batch_key=args.batch_key,
    n_cores=args.n_cores,
    subsample=None,          # set e.g. 50000 for speed on huge datasets
    random_state=args.seed,
    exclude_self=True,
)
print(f"ilisi score: {ilisi_score:.4f} (higher is better)")

# Calculate PCR batch (Principal Component Regression)
# Lower is better (less batch effect), so we'll store 1-pcr for "higher is better" interpretation
print("  Calculating PCR batch...")
pcr_batch_score_raw = pcr(adata_tfidf, covariate=args.batch_key, embed='cell_embeddings_zero_shot', recompute_pca=True)
pcr_batch_score = 1 - pcr_batch_score_raw  # Invert so higher is better

print(f"  ilisi score: {ilisi_score:.4f} (higher is better)")
print(f"  PCR batch score (1-PCR): {pcr_batch_score:.4f} (higher is better, original PCR: {pcr_batch_score_raw:.4f})")

# Save results to CSV
output_dir = Path(f'./zero_shot_batch_integration_metrics/{args.dataset_name}')
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

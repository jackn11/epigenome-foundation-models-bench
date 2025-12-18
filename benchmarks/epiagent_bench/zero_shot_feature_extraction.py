import os
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy import sparse
import scanpy as sc
import numpy as np

from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF, global_TFIDF_with_shuffling, global_TFIDF_with_complete_shuffling
from epiagent.dataset import CellDataset, collate_fn
from epiagent.model import EpiAgent
from epiagent.inference import infer_cell_embeddings

import torch
from torch.utils.data import DataLoader
import io
from PIL import Image
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
from scib.metrics import silhouette, silhouette_batch, graph_connectivity, pcr
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to allow importing benchmarks module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.benchmark_utils import prepare_img, find_leiden_resolution_for_n_clusters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Li2023b')
    parser.add_argument('--batch_key', type=str, default='Batch (HSC)')
    parser.add_argument('--root', type=str, default='/scratch/naimer/github/project-2-team-1/EpiAgent/data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='/scratch/wkim/project-2-team-1/EpiAgent/model/pretrained_EpiAgent.pth')
    parser.add_argument('--token_cache_dir', type=str, default='./cache')
    return parser

def ilisi_graph_custom(
    adata,
    batch_key: str,
    connectivity_key: str = "connectivities",
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

    scores = [ilisi_one(int(i)) for i in tqdm(idx, desc="Computing iLISI scores", total=len(idx))]

    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise RuntimeError("All iLISI scores were NaN/inf. Check your neighbor graph / batch labels.")

    return float(scores.mean())


parser = get_args_parser()
args = parser.parse_args()

root = Path(args.root)

print("Loading the dataset...")
if args.dataset_name == 'Li2023b':
    input_path = root / 'Li2023b' / 'Li2023b-brain_tissue' / 'Li2023b-brain_tissue-cell_by_cCRE.h5ad'
elif args.dataset_name == 'Kanemaru2023':
    input_path = root / 'Kanemaru2023' / 'Kanemaru2023-cardiac_tissue' / 'Kanemaru2023-cardiac_tissue-cell_by_cCRE.h5ad'
else:
    raise ValueError(f"Dataset {args.dataset_name} not supported")

adata = sc.read_h5ad(input_path)

num_cell_types = len(adata.obs['cell_type'].unique())
print(f"Number of cell types in the dataset: {num_cell_types}")

cCRE_document_frequency = np.load(root / 'cCRE_document_frequency.npy')


cache_dir = Path(args.token_cache_dir)
cache_dir.mkdir(exist_ok=True)
cached_tokenized_path = cache_dir / f'{args.dataset_name}_tokenized.h5ad'

if cached_tokenized_path.exists():
    print(f"Loading cached tokenized data from {cached_tokenized_path}...")
    adata_tfidf = sc.read_h5ad(cached_tokenized_path)
    print("Cached tokenized data loaded successfully.")
else:
    print("No cached tokenized data found. Computing from scratch...")
    adata_tfidf = global_TFIDF(adata, cCRE_document_frequency)
    
    print("Performing tokenization... (this takes a while)")
    tokenization(adata_tfidf)
    
    print(f"Saving tokenized data to {cached_tokenized_path}...")
    adata_tfidf.write(cached_tokenized_path)
    print("Tokenized data saved successfully.")

print("Creating the dataset...")
cell_sentences = adata_tfidf.obs['cell_sentences'].tolist()
cell_dataset = CellDataset(cell_sentences=cell_sentences)

print("Creating the DataLoader...")
batch_size = 15
dataloader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

os.makedirs(f'./saved_embeddings/{args.dataset_name}', exist_ok=True)
embedding_save_path = f'./saved_embeddings/{args.dataset_name}/{args.dataset_name}-cell_embeddings.npy'

if os.path.exists(embedding_save_path):
    print(f"Loading saved cell embeddings from {embedding_save_path}...")
    cell_embeddings = np.load(embedding_save_path)
    print("Cell embeddings loaded successfully.")
else:
    print("No saved embeddings found. Computing from scratch...")
    print("Loading the pretrained model...")
    # model_path = '../model/pretrained_EpiAgent.pth'
    model_path = Path(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model = EpiAgent(vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8, max_rank_embeddings=8192, use_flash_attn=True, pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
    pretrained_model.load_state_dict(torch.load(model_path, map_location=device))

    print("Extracting cell embeddings...")
    cell_embeddings = infer_cell_embeddings(pretrained_model, device, dataloader)

    np.save(embedding_save_path, cell_embeddings)
    print(f"Cell embeddings saved to {embedding_save_path}")

adata_tfidf.obsm['cell_embeddings_zero_shot'] = cell_embeddings

# UMAP visualization
print("visualizing UMAP...")
sc.pp.neighbors(adata_tfidf, use_rep='cell_embeddings_zero_shot')
sc.tl.umap(adata_tfidf)
fig = sc.pl.umap(adata_tfidf, color='cell_type', return_fig=True, show=True, title='Cell embeddings (true labels)')
if fig is not None:
    axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('Cell type')
output_dir = Path(f'./zero_shot_feature_extraction/{args.dataset_name}')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'umap_cell_types_true_labels.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("UMAP visualization (cell type) saved")

# Plot UMAP with batch labels if batch key exists
if args.batch_key is not None and args.batch_key in adata_tfidf.obs.columns:
    fig = sc.pl.umap(adata_tfidf, color=args.batch_key, return_fig=True, show=True, title='Cell embeddings (batch labels)')
    if fig is not None:
        axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title('Batch')
    plt.savefig(output_dir / 'umap_batch_labels.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("UMAP visualization (batch labels) saved")



n_true_cell_types = len(adata_tfidf.obs['cell_type'].unique())
print(f"Number of cell types: {n_true_cell_types}")

# Perform leiden clustering with binary search to get exactly num_cell_types clusters
# Note: neighbors are already computed using the embeddings
optimal_resolution, n_clusters = find_leiden_resolution_for_n_clusters(
    adata_tfidf, 
    target_n_clusters=num_cell_types, 
    min_res=0.1, 
    max_res=2.0, 
    random_state=42
)

print(f"Optimal resolution: {optimal_resolution:.4f}")
print(f"Number of Leiden clusters: {n_clusters}")

print("visualizing UMAP with leiden clustering...")
fig = sc.pl.umap(adata_tfidf, color='leiden', legend_loc='on data', title='Leiden Clustering (no shuffling)', return_fig=True, show=True)
plt.savefig(output_dir / 'umap_leiden_clustering.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("UMAP visualization with leiden clustering saved")


print("calculating NMI and ARI scores...")
true_labels = adata_tfidf.obs['cell_type'].values
predicted_labels = adata_tfidf.obs['leiden'].values

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

print("calculating silhouette scores...")
silhouette_score = silhouette(adata_tfidf, label_key='cell_type', embed='cell_embeddings_zero_shot')
silhouette_batch_score = silhouette_batch(adata_tfidf, label_key='cell_type', batch_key=args.batch_key, embed='cell_embeddings_zero_shot')

# Calculate mean silhouette per group
cell_embeddings = adata_tfidf.obsm['cell_embeddings_zero_shot']
cell_types = adata_tfidf.obs['cell_type'].values
silhouette_samples_scores = silhouette_samples(cell_embeddings, cell_types)
mean_silhouette_per_group = pd.Series(silhouette_samples_scores, index=cell_types).groupby(cell_types).mean().mean()

print("Calculating graph connectivity...")
graph_connectivity_score = graph_connectivity(adata_tfidf, label_key='cell_type')
print(f"Graph connectivity score: {graph_connectivity_score:.4f}")

print("Training linear probe on embeddings...")
cell_types = adata_tfidf.obs['cell_type'].values

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
)
linear_probe.fit(X_train, y_train)

y_val_pred = linear_probe.predict(X_val)
linear_probe_accuracy = accuracy_score(y_val, y_val_pred)
linear_probe_f1_macro = f1_score(y_val, y_val_pred, average='macro')
linear_probe_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')

print(f"Linear probe accuracy: {linear_probe_accuracy:.4f}")
print(f"Linear probe F1 score (macro): {linear_probe_f1_macro:.4f}")
print(f"Linear probe F1 score (weighted): {linear_probe_f1_weighted:.4f}")


print("\nTraining linear probe on embeddings (batch labels)...")
batch_labels = adata_tfidf.obs[args.batch_key].values
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
)
linear_probe_batch.fit(X_train_batch, y_train_batch)

y_val_pred_batch = linear_probe_batch.predict(X_val_batch)
linear_probe_batch_accuracy = accuracy_score(y_val_batch, y_val_pred_batch)
linear_probe_batch_f1_macro = f1_score(y_val_batch, y_val_pred_batch, average='macro')
linear_probe_batch_f1_weighted = f1_score(y_val_batch, y_val_pred_batch, average='weighted')

print(f"Linear probe accuracy (batch): {linear_probe_batch_accuracy:.4f}")
print(f"Linear probe F1 score (macro, batch): {linear_probe_batch_f1_macro:.4f}")
print(f"Linear probe F1 score (weighted, batch): {linear_probe_batch_f1_weighted:.4f}")


print("Computing neighbors for ilisi calculation...")
sc.pp.neighbors(adata_tfidf, use_rep='cell_embeddings_zero_shot')
print(f"Neighbors computed for {adata_tfidf.n_obs} cells")

print("Calculating batch integration metrics...")
# iLISI (Integration Local Inverse Simpson's Index). Higher is better (more batch mixing)
print("  Calculating iLISI...")
ilisi_score = ilisi_graph_custom(
    adata_tfidf,
    batch_key=args.batch_key,
    subsample=None,          # set small e.g. 50000 for speed on huge datasets
    random_state=args.seed,
    exclude_self=True,
)
print(f"ilisi score: {ilisi_score:.4f} (higher is better)")

print("  Calculating PCR batch...")
# Lower is better (less batch effect), so we'll store 1-pcr for "higher is better" interpretation
pcr_batch_score_raw = pcr(adata_tfidf, covariate=args.batch_key, embed='cell_embeddings_zero_shot', recompute_pca=True)
pcr_batch_score = 1 - pcr_batch_score_raw  # Invert so higher is better

print(f"  PCR batch score (1-PCR): {pcr_batch_score:.4f} (higher is better, original PCR: {pcr_batch_score_raw:.4f})")



print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Silhouette score: {silhouette_score:.4f}")
print(f"Silhouette batch score: {silhouette_batch_score:.4f}")
print(f"Mean silhouette per group: {mean_silhouette_per_group:.4f}")
print(f"Graph connectivity score: {graph_connectivity_score:.4f}")
print(f"Linear probe accuracy: {linear_probe_accuracy:.4f}")
print(f"Linear probe F1 score (macro): {linear_probe_f1_macro:.4f}")
print(f"Linear probe F1 score (weighted): {linear_probe_f1_weighted:.4f}")
print(f"Linear probe batch accuracy: {linear_probe_batch_accuracy:.4f}")
print(f"Linear probe batch F1 score (macro): {linear_probe_batch_f1_macro:.4f}")
print(f"Linear probe batch F1 score (weighted): {linear_probe_batch_f1_weighted:.4f}")
print(f"\nNumber of true cell types: {len(adata_tfidf.obs['cell_type'].unique())}")
print(f"Number of predicted clusters: {len(adata_tfidf.obs['leiden'].unique())}")
print(f"ilisi score: {ilisi_score:.4f} (higher is better)")
print(f"PCR batch score (1-PCR): {pcr_batch_score:.4f} (higher is better)")
print(f"PCR batch score (raw): {pcr_batch_score_raw:.4f} (lower is better)")

results_df = pd.DataFrame({
    'Metric': [
        'Adjusted Rand Index', 
        'Normalized Mutual Information', 
        'Silhouette score', 
        'Silhouette batch score', 
        'Mean silhouette per group',
        'Cell type Linear probe accuracy',
        'Cell type Linear probe F1 score (macro)',
        'Cell type Linear probe F1 score (weighted)',
        'Batch label Linear probe accuracy',
        'Batch label Linear probe F1 score (macro)',
        'Batch label Linear probe F1 score (weighted)',
        'Graph connectivity score',
        'ilisi score',
        'PCR batch score (1-PCR)',
        'PCR batch score (raw)',
    ],
    'Value': [
        ari_score, 
        nmi_score, 
        silhouette_score, 
        silhouette_batch_score, 
        mean_silhouette_per_group,
        linear_probe_accuracy,
        linear_probe_f1_macro,
        linear_probe_f1_weighted,
        linear_probe_batch_accuracy,
        linear_probe_batch_f1_macro,
        linear_probe_batch_f1_weighted,
        graph_connectivity_score,
        ilisi_score,
        pcr_batch_score,
        pcr_batch_score_raw,
    ]
})
results_file = output_dir / 'results.csv'
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

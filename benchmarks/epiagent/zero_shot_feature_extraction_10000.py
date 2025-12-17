import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import scanpy as sc
import numpy as np
from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF, global_TFIDF_with_shuffling, global_TFIDF_with_complete_shuffling
from epiagent.dataset import CellDataset, collate_fn
from torch.utils.data import DataLoader
import os
from epiagent.model import EpiAgent
import torch
from epiagent.inference import infer_cell_embeddings
import io
from PIL import Image
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
from scib.metrics import silhouette, silhouette_batch
import pandas as pd

from benchmarks.benchmark_utils import prepare_img, find_leiden_resolution_for_n_clusters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Li2023b', choices=['Kanemaru2023', 'Li2023b'])
    parser.add_argument('--batch_key', type=str, default='Batch (HSC)')
    parser.add_argument('--root', type=str, default='/scratch/naimer/github/project-2-team-1/EpiAgent/data')
    parser.add_argument('--seed', type=int, default=42)
    return parser


parser = get_args_parser()
args = parser.parse_args()

root = Path(args.root)

print("Loading the dataset...")
if args.dataset_name == 'Li2023b':
    input_path = root / 'sample' / 'raw_h5ad' / 'Li2023b_downsampled_10000_cells.h5ad'
elif args.dataset_name == 'Kanemaru2023':
    input_path = root / 'sample' / 'raw_h5ad' / 'Kanemaru2023_downsampled_10000_cells.h5ad'
else:
    raise ValueError(f"Dataset {args.dataset_name} not supported")

adata = sc.read_h5ad(input_path)

# Load full dataset to transfer batch keys
print("Loading full dataset to transfer batch keys...")
if args.dataset_name == 'Li2023b':
    full_dataset_path = root / 'Li2023b' / 'Li2023b-brain_tissue' / 'Li2023b-brain_tissue-cell_by_cCRE.h5ad'
    batch_key_source = 'Batch (HSC)'
    # Create matching key using sample + cell_barcode
    adata_full = sc.read_h5ad(full_dataset_path)
    adata_full.obs['match_key'] = adata_full.obs['sample'].astype(str) + '_' + adata_full.obs['cell_barcode'].astype(str)
    adata.obs['match_key'] = adata.obs['sample'].astype(str) + '_' + adata.obs['cell_barcode'].astype(str)
elif args.dataset_name == 'Kanemaru2023':
    full_dataset_path = root / 'Kanemaru2023' / 'Kanemaru2023-cardiac_tissue' / 'Kanemaru2023-cardiac_tissue-cell_by_cCRE.h5ad'
    batch_key_source = 'Batch (HSC)'
    # Create matching key using sangerID
    adata_full = sc.read_h5ad(full_dataset_path)
    adata_full.obs['match_key'] = adata_full.obs['sangerID'].astype(str)
    adata.obs['match_key'] = adata.obs['sangerID'].astype(str)
else:
    raise ValueError(f"Dataset {args.dataset_name} not supported")

# Transfer batch key from full dataset
if batch_key_source in adata_full.obs.columns:
    # Create a mapping from match_key to batch
    batch_mapping = adata_full.obs.set_index('match_key')[batch_key_source].to_dict()
    
    # Map batch keys to downsampled dataset
    adata.obs[args.batch_key] = adata.obs['match_key'].map(batch_mapping)
    
    # Check how many cells got matched
    matched_count = adata.obs[args.batch_key].notna().sum()
    print(f"Transferred batch key '{args.batch_key}' to {matched_count} out of {len(adata)} cells")
    
    if matched_count < len(adata):
        print(f"Warning: {len(adata) - matched_count} cells could not be matched to full dataset")
else:
    print(f"Warning: Batch key '{batch_key_source}' not found in full dataset. Available columns: {list(adata_full.obs.columns)}")
    args.batch_key = None

# Clean up temporary match_key column
if 'match_key' in adata.obs.columns:
    adata.obs = adata.obs.drop(columns=['match_key'])

num_cell_types = len(adata.obs['cell_type'].unique())
print(f"Number of cell types in the dataset: {num_cell_types}")

cCRE_document_frequency = np.load(root / 'cCRE_document_frequency.npy')


cache_dir = Path('./cache')
cache_dir.mkdir(exist_ok=True)
cached_tokenized_path = cache_dir / f'{args.dataset_name}_10000_tokenized.h5ad'

if cached_tokenized_path.exists():
    print(f"Loading cached tokenized data from {cached_tokenized_path}...")
    adata_tfidf = sc.read_h5ad(cached_tokenized_path)
    print("Cached tokenized data loaded successfully.")
    # Transfer batch key if it's not already in the cached data
    if args.batch_key is not None and args.batch_key not in adata_tfidf.obs.columns and args.batch_key in adata.obs.columns:
        adata_tfidf.obs[args.batch_key] = adata.obs[args.batch_key].values
        print(f"Transferred batch key '{args.batch_key}' to cached tokenized data")
else:
    print("No cached tokenized data found. Computing from scratch...")
    adata_tfidf = global_TFIDF(adata, cCRE_document_frequency)
    
    # Ensure batch key is transferred to adata_tfidf
    if args.batch_key is not None and args.batch_key in adata.obs.columns:
        adata_tfidf.obs[args.batch_key] = adata.obs[args.batch_key].values
    
    print("Performing tokenization... (this takes a while)")
    tokenization(adata_tfidf)
    
    print(f"Saving tokenized data to {cached_tokenized_path}...")
    adata_tfidf.write(cached_tokenized_path)
    print("Tokenized data saved successfully.")

print("Creating the dataset...")
cell_sentences = adata_tfidf.obs['cell_sentences'].tolist()
cell_dataset = CellDataset(cell_sentences=cell_sentences)

print("Creating the DataLoader...")
# Create the DataLoader
batch_size = 15
dataloader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

os.makedirs(f'./saved_embeddings/{args.dataset_name}_10000', exist_ok=True)
embedding_save_path = f'./saved_embeddings/{args.dataset_name}_10000/{args.dataset_name}_10000-cell_embeddings.npy'

if os.path.exists(embedding_save_path):
    print(f"Loading saved cell embeddings from {embedding_save_path}...")
    cell_embeddings = np.load(embedding_save_path)
    print("Cell embeddings loaded successfully.")
else:
    print("No saved embeddings found. Computing from scratch...")
    print("Loading the pretrained model...")
    model_path = '../model/pretrained_EpiAgent.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model = EpiAgent(vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8, max_rank_embeddings=8192, use_flash_attn=True, pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
    pretrained_model.load_state_dict(torch.load(model_path, map_location=device))

    print("Extracting cell embeddings...")
    cell_embeddings = infer_cell_embeddings(pretrained_model, device, dataloader)

    np.save(embedding_save_path, cell_embeddings)
    print(f"Cell embeddings saved to {embedding_save_path}")

# Assign embeddings to the AnnData object
adata_tfidf.obsm['cell_embeddings_zero_shot'] = cell_embeddings

# UMAP visualization
print("visualizing UMAP...")
sc.pp.neighbors(adata_tfidf, use_rep='cell_embeddings_zero_shot')
sc.tl.umap(adata_tfidf)

# Plot UMAP with original cell types and capture the figure
fig = sc.pl.umap(adata_tfidf, color='cell_type', return_fig=True, show=True, title='Cell embeddings (true labels)')

# Customize legend to show "cell type" as description
if fig is not None:
    axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('Cell type')
output_dir = Path(f'./zero_shot_feature_extraction_{args.dataset_name}_10000')
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'umap_cell_types_true_labels.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("UMAP visualization saved")



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
fig = sc.pl.umap(adata_tfidf, color='leiden', legend_loc='on data', title='Leiden Clustering (10000 cells)', return_fig=True, show=True)
plt.savefig(output_dir / 'umap_leiden_clustering.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("UMAP visualization with leiden clustering saved")


# Calculate NMI and ARI scores
print("calculating NMI and ARI scores...")
# Get the true labels (cell types) and predicted labels (Leiden clusters)
true_labels = adata_tfidf.obs['cell_type'].values
predicted_labels = adata_tfidf.obs['leiden'].values

# Calculate ARI (Adjusted Rand Index)
ari_score = adjusted_rand_score(true_labels, predicted_labels)

# Calculate NMI (Normalized Mutual Information)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

# Calculate Silhouette scores
silhouette_score = silhouette(adata_tfidf, label_key='cell_type', embed='cell_embeddings_zero_shot')
silhouette_batch_score = silhouette_batch(adata_tfidf, label_key='cell_type', batch_key=args.batch_key, embed='cell_embeddings_zero_shot')

# Calculate mean silhouette per group
cell_embeddings = adata_tfidf.obsm['cell_embeddings_zero_shot']
cell_types = adata_tfidf.obs['cell_type'].values
silhouette_samples_scores = silhouette_samples(cell_embeddings, cell_types)
mean_silhouette_per_group = pd.Series(silhouette_samples_scores, index=cell_types).groupby(cell_types).mean().mean()

# Train linear probe on cell embeddings using cell type labels
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


# Train linear probe on embeddings using batch labels
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


print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Silhouette score: {silhouette_score:.4f}")
print(f"Silhouette batch score: {silhouette_batch_score:.4f}")
print(f"Mean silhouette per group: {mean_silhouette_per_group:.4f}")
print(f"Linear probe accuracy: {linear_probe_accuracy:.4f}")
print(f"Linear probe F1 score (macro): {linear_probe_f1_macro:.4f}")
print(f"Linear probe F1 score (weighted): {linear_probe_f1_weighted:.4f}")
print(f"Linear probe batch accuracy: {linear_probe_batch_accuracy:.4f}")
print(f"Linear probe batch F1 score (macro): {linear_probe_batch_f1_macro:.4f}")
print(f"Linear probe batch F1 score (weighted): {linear_probe_batch_f1_weighted:.4f}")
print(f"\nNumber of true cell types: {len(adata_tfidf.obs['cell_type'].unique())}")
print(f"Number of predicted clusters: {len(adata_tfidf.obs['leiden'].unique())}")

# Save results to CSV
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
        'Batch label Linear probe F1 score (weighted)'
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
    ]
})
results_file = output_dir / 'results.csv'
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

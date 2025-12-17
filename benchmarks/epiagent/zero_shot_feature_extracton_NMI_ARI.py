import random
import numpy as np
import torch
import wandb
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from scib.metrics import silhouette, silhouette_batch

from benchmarks.benchmark_utils import prepare_img, find_leiden_resolution_for_n_clusters



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="zero-shot-feature-extraction", name=f"zero_shot_embeddings_Li2023b_full_{timestamp}")

dataset_name = 'Li2023b'
batch_key = 'Batch (HSC)'

# Load the dataset
print("Loading the dataset...")
input_path = '../data/Li2023b/Li2023b-brain_tissue/Li2023b-brain_tissue-cell_by_cCRE.h5ad'
adata = sc.read_h5ad(input_path)

num_cell_types = len(adata.obs['cell_type'].unique())

print(f"Number of cell types in the dataset: {num_cell_types}")

# Load the cCRE document frequency data
cCRE_document_frequency = np.load('../data/cCRE_document_frequency.npy')

print("Applying TFIDF transformation...")


######################################################
# NO PERMUTATION
######################################################


# Apply TFIDF transformation
adata_tfidf = global_TFIDF(adata, cCRE_document_frequency)

print("Performing tokenization...")
# Perform tokenization
tokenization(adata_tfidf)

print("Creating the dataset...")
# Create the dataset
cell_sentences = adata_tfidf.obs['cell_sentences'].tolist()
cell_dataset = CellDataset(cell_sentences=cell_sentences)

print("Creating the DataLoader...")
# Create the DataLoader
batch_size = 15
dataloader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Load the pretrained model
print("Loading the pretrained model...")
model_path = '../model/pretrained_EpiAgent.pth'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

pretrained_model = EpiAgent(vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8, max_rank_embeddings=8192, use_flash_attn=True, pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
pretrained_model.load_state_dict(torch.load(model_path, map_location=device))

# Extract cell embeddings
print("Extracting cell embeddings...")
cell_embeddings = infer_cell_embeddings(pretrained_model, device, dataloader)

# Save cell embeddings to a file (e.g., numpy file)
os.makedirs(f'../data/saved_embeddings/{dataset_name}', exist_ok=True)
embedding_save_path = f'../data/saved_embeddings/{dataset_name}/{dataset_name}-cell_embeddings_{timestamp}.npy'
np.save(embedding_save_path, cell_embeddings)
print(f"Cell embeddings saved to {embedding_save_path}")

# Assign embeddings to the AnnData object
adata_tfidf.obsm['cell_embeddings_zero_shot'] = cell_embeddings

# UMAP visualization
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
plt.show()

# Convert figure to PIL Image for wandb using reusable function
img = prepare_img(fig)

wandb.log({"umap_cell_types_no_shuffling_true_labels": wandb.Image(img)})
plt.close(fig)

print("Pos weight for CCA: ", pretrained_model.criterion_CCA.pos_weight)

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

# Print the number of clusters found
print(f"Optimal resolution: {optimal_resolution:.4f}")
print(f"Number of Leiden clusters: {n_clusters}")

# Visualize the Leiden clusters on UMAP
fig = sc.pl.umap(adata_tfidf, color='leiden', legend_loc='on data', title='Leiden Clustering (no shuffling)', return_fig=True, show=True)
plt.show()
img = prepare_img(fig)
wandb.log({"umap_cell_types_no_shuffling_leiden_clustering": wandb.Image(img)})
plt.close(fig)

# Calculate NMI and ARI scores

# Get the true labels (cell types) and predicted labels (Leiden clusters)
true_labels = adata_tfidf.obs['cell_type'].values
predicted_labels = adata_tfidf.obs['leiden'].values

# Calculate ARI (Adjusted Rand Index)
ari_score = adjusted_rand_score(true_labels, predicted_labels)

# Calculate NMI (Normalized Mutual Information)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

# Calculate Silhouette scores
silhouette_score = silhouette(adata_tfidf, label_key='cell_type', embed='cell_embeddings_zero_shot')
silhouette_batch_score = silhouette_batch(adata_tfidf, label_key='cell_type', batch_key=batch_key, embed='cell_embeddings_zero_shot')

# Print the results
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Silhouette score: {silhouette_score:.4f}")
print(f"Silhouette batch score: {silhouette_batch_score:.4f}")
print(f"\nNumber of true cell types: {len(adata_tfidf.obs['cell_type'].unique())}")
print(f"Number of predicted clusters: {len(adata_tfidf.obs['leiden'].unique())}")

wandb.log({
    "ARI_no_shuffling": ari_score,
    "NMI_no_shuffling": nmi_score,
    "Silhouette_no_shuffling": silhouette_score,
    "Silhouette_batch_no_shuffling": silhouette_batch_score
})


######################################################
# PREMUTED LABELS
######################################################



# Apply TFIDF transformation
adata_tfidf_perm = global_TFIDF_with_shuffling(adata, cCRE_document_frequency)

# Perform tokenization
tokenization(adata_tfidf_perm)

# Create the dataset
cell_sentences_perm = adata_tfidf_perm.obs['cell_sentences'].tolist()
cell_dataset_perm = CellDataset(cell_sentences=cell_sentences_perm)

# Create the DataLoader
batch_size = 15
dataloader_perm = DataLoader(cell_dataset_perm, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Extract cell embeddings
cell_embeddings_perm = infer_cell_embeddings(pretrained_model, device, dataloader_perm)

# Assign embeddings to the AnnData object
adata_tfidf_perm.obsm['cell_embeddings_zero_shot'] = cell_embeddings_perm

# UMAP visualization
sc.pp.neighbors(adata_tfidf_perm, use_rep='cell_embeddings_zero_shot')
sc.tl.umap(adata_tfidf_perm)

# Plot UMAP with original cell types
fig = sc.pl.umap(adata_tfidf_perm, color='cell_type', return_fig=True, show=True)

if fig is not None:
    axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('Cell type')

plt.show()
img = prepare_img(fig)
wandb.log({"umap_cell_types_permuted_labels": wandb.Image(img)})
plt.close(fig)

# Assign embeddings to adata_perm and compute neighbors for Leiden clustering
adata_tfidf_perm.obsm['cell_embeddings_zero_shot'] = cell_embeddings_perm


# Perform Leiden clustering with binary search to get exactly num_cell_types clusters
optimal_resolution, n_clusters = find_leiden_resolution_for_n_clusters(
    adata_tfidf_perm, 
    target_n_clusters=num_cell_types, 
    min_res=0.1, 
    max_res=2.0, 
    random_state=42
)

# Print the number of clusters found
print(f"Optimal resolution: {optimal_resolution:.4f}")
print(f"Number of Leiden clusters: {n_clusters}")

# Visualize the Leiden clusters on UMAP
fig = sc.pl.umap(adata_tfidf_perm, color='leiden', legend_loc='on data', title='Leiden Clustering', return_fig=True, show=True)
plt.show()
img = prepare_img(fig)
wandb.log({"umap_cell_types_permuted_labels_leiden_clustering": wandb.Image(img)})
plt.close(fig)

# Calculate NMI and ARI scores
# Get the true labels (cell types) and predicted labels (Leiden clusters)
true_labels = adata_tfidf_perm.obs['cell_type'].values
predicted_labels = adata_tfidf_perm.obs['leiden'].values

# Calculate ARI (Adjusted Rand Index)
ari_score = adjusted_rand_score(true_labels, predicted_labels)

# Calculate NMI (Normalized Mutual Information)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

# Calculate Silhouette scores
silhouette_score = silhouette(adata_tfidf_perm, label_key='cell_type', embed='cell_embeddings_zero_shot')
silhouette_batch_score = silhouette_batch(adata_tfidf_perm, label_key='cell_type', batch_key=batch_key, embed='cell_embeddings_zero_shot')

# Print the results
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Silhouette score: {silhouette_score:.4f}")
print(f"Silhouette batch score: {silhouette_batch_score:.4f}")
print(f"\nNumber of true cell types: {len(adata_tfidf_perm.obs['cell_type'].unique())}")
print(f"Number of predicted clusters: {len(adata_tfidf_perm.obs['leiden'].unique())}")

wandb.log({
    "ARI_permuted_labels": ari_score,
    "NMI_permuted_labels": nmi_score,
    "Silhouette_permuted_labels": silhouette_score,
    "Silhouette_batch_permuted_labels": silhouette_batch_score
})



######################################################
# COMPLETELY PERMUTED LABELS
######################################################


# Apply TFIDF transformation
adata_tfidf_perm_complete = global_TFIDF_with_complete_shuffling(adata, cCRE_document_frequency)

# Perform tokenization
tokenization(adata_tfidf_perm_complete)

# Create the dataset
cell_sentences_perm_complete = adata_tfidf_perm_complete.obs['cell_sentences'].tolist()
cell_dataset_perm_complete = CellDataset(cell_sentences=cell_sentences_perm_complete)

# Create the DataLoader
batch_size = 15
dataloader_perm_complete = DataLoader(cell_dataset_perm_complete, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Extract cell embeddings
cell_embeddings_perm_complete = infer_cell_embeddings(pretrained_model, device, dataloader_perm_complete)

# Assign embeddings to the AnnData object
adata_tfidf_perm_complete.obsm['cell_embeddings_zero_shot'] = cell_embeddings_perm_complete

# UMAP visualization
sc.pp.neighbors(adata_tfidf_perm_complete, use_rep='cell_embeddings_zero_shot')
sc.tl.umap(adata_tfidf_perm_complete)

# Plot UMAP with original cell types
fig = sc.pl.umap(adata_tfidf_perm_complete, color='cell_type', return_fig=True, show=True)

if fig is not None:
    axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('Cell type')

plt.show()
img = prepare_img(fig)
wandb.log({"umap_cell_types_complete_shuffling": wandb.Image(img)})
plt.close(fig)


# Assign embeddings to adata_perm and compute neighbors for Leiden clustering
adata_tfidf_perm_complete.obsm['cell_embeddings_zero_shot'] = cell_embeddings_perm_complete

# Perform Leiden clustering with binary search to get exactly num_cell_types clusters
optimal_resolution, n_clusters = find_leiden_resolution_for_n_clusters(
    adata_tfidf_perm_complete, 
    target_n_clusters=num_cell_types, 
    min_res=0.1, 
    max_res=2.0, 
    random_state=42
)

# Print the number of clusters found
print(f"Optimal resolution: {optimal_resolution:.4f}")
print(f"Number of Leiden clusters: {n_clusters}")

# Visualize the Leiden clusters on UMAP
fig = sc.pl.umap(adata_tfidf_perm_complete, color='leiden', legend_loc='on data', title='Leiden Clustering', return_fig=True, show=True)
plt.show()
img = prepare_img(fig)
wandb.log({"umap_cell_types_complete_shuffling_leiden_clustering": wandb.Image(img)})
plt.close(fig)

# Calculate NMI and ARI scores
# Get the true labels (cell types) and predicted labels (Leiden clusters)
true_labels = adata_tfidf_perm_complete.obs['cell_type'].values
predicted_labels = adata_tfidf_perm_complete.obs['leiden'].values

# Calculate ARI (Adjusted Rand Index)
ari_score = adjusted_rand_score(true_labels, predicted_labels)

# Calculate NMI (Normalized Mutual Information)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

# Calculate Silhouette scores
silhouette_score = silhouette(adata_tfidf_perm_complete, label_key='cell_type', embed='cell_embeddings_zero_shot')
silhouette_batch_score = silhouette_batch(adata_tfidf_perm_complete, label_key='cell_type', batch_key=batch_key, embed='cell_embeddings_zero_shot')

# Print the results
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Silhouette score: {silhouette_score:.4f}")
print(f"Silhouette batch score: {silhouette_batch_score:.4f}")
print(f"\nNumber of true cell types: {len(adata_tfidf_perm_complete.obs['cell_type'].unique())}")
print(f"Number of predicted clusters: {len(adata_tfidf_perm_complete.obs['leiden'].unique())}")

wandb.log({
    "ARI_complete_shuffling": ari_score,
    "NMI_complete_shuffling": nmi_score,
    "Silhouette_complete_shuffling": silhouette_score,
    "Silhouette_batch_complete_shuffling": silhouette_batch_score
})

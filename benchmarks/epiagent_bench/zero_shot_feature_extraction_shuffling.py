import random
import numpy as np
import torch
import wandb
from datetime import datetime
import scanpy as sc
import argparse
from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF, global_TFIDF_with_shuffling, global_TFIDF_with_complete_shuffling
from epiagent.dataset import CellDataset, collate_fn
from torch.utils.data import DataLoader
import os
from epiagent.model import EpiAgent
from epiagent.inference import infer_cell_embeddings
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from scib.metrics import silhouette, silhouette_batch

from benchmarks.benchmark_utils import prepare_img, find_leiden_resolution_for_n_clusters


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Zero-shot feature extraction with shuffling experiments')
    
    parser.add_argument('--input_path', type=str, required=True, help='Path to input h5ad file')
    parser.add_argument('--cCRE_document_frequency_path', type=str, required=True, help='Path to cCRE document frequency numpy file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model file')
    parser.add_argument('--embeddings_output_dir', type=str, required=True, help='Directory to save embeddings')
    parser.add_argument('--device', type=str, required=True, help='Device to use (e.g., cuda:0, cuda:3, cpu)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--batch_key', type=str, required=True, help='Batch key for silhouette batch score calculation')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of worker processes for DataLoader')
    
    return parser.parse_args()


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


args = parse_args()

device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else "cpu")

dataset_name = args.dataset_name
batch_key = args.batch_key

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="zero-shot-feature-extraction", name=f"zero_shot_embeddings_{dataset_name}_full_{timestamp}")

print("Loading the dataset...")
adata = sc.read_h5ad(args.input_path)

num_cell_types = len(adata.obs['cell_type'].unique())

print(f"Number of cell types in the dataset: {num_cell_types}")

cCRE_document_frequency = np.load(args.cCRE_document_frequency_path)

print("Applying TFIDF transformation...")

print("Loading the pretrained model...")
pretrained_model = EpiAgent(vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8, max_rank_embeddings=8192, use_flash_attn=True, pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
pretrained_model.load_state_dict(torch.load(args.model_path, map_location=device))

print("Pos weight for CCA: ", pretrained_model.criterion_CCA.pos_weight)


def run_pipeline(tfidf_func, suffix, leiden_title_suffix="", save_embeddings=False, umap_title="Cell embeddings (true labels)"):
    """
    Run the complete pipeline for feature extraction and evaluation.
    
    Args:
        tfidf_func: Function to apply TFIDF transformation (global_TFIDF, global_TFIDF_with_shuffling, etc.)
        suffix: Suffix for wandb logging keys: "no_shuffling", "permuted_labels", "complete_shuffling"
        leiden_title_suffix: Additional suffix for Leiden clustering title
        save_embeddings: Whether to save embeddings to file (only for first run)
        umap_title: Title for the UMAP plot with cell types
    """
    print(f"Running pipeline: {suffix}")

    print("Applying TFIDF transformation...")
    adata_tfidf = tfidf_func(adata, cCRE_document_frequency)
    
    print("Performing tokenization...")
    tokenization(adata_tfidf)
    
    print("Creating the dataset...")
    cell_sentences = adata_tfidf.obs['cell_sentences'].tolist()
    cell_dataset = CellDataset(cell_sentences=cell_sentences)
    
    print("Creating the DataLoader...")
    dataloader = DataLoader(cell_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    print("Extracting cell embeddings...")
    cell_embeddings = infer_cell_embeddings(pretrained_model, device, dataloader)
    
    if save_embeddings:
        os.makedirs(args.embeddings_output_dir, exist_ok=True)
        embedding_save_path = os.path.join(args.embeddings_output_dir, f'{dataset_name}-cell_embeddings_{timestamp}.npy')
        np.save(embedding_save_path, cell_embeddings)
        print(f"Cell embeddings saved to {embedding_save_path}")
    
    adata_tfidf.obsm['cell_embeddings_zero_shot'] = cell_embeddings
    
    sc.pp.neighbors(adata_tfidf, use_rep='cell_embeddings_zero_shot')
    sc.tl.umap(adata_tfidf)
    
    fig = sc.pl.umap(adata_tfidf, color='cell_type', return_fig=True, show=True, title=umap_title)
    
    if fig is not None:
        axes = fig.axes if hasattr(fig, 'axes') else [ax for ax in fig.get_axes()]
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title('Cell type')
    plt.show()
    
    img = prepare_img(fig)
    wandb_key = f"umap_cell_types_{suffix}" if suffix != "no_shuffling" else "umap_cell_types_no_shuffling_true_labels"
    wandb.log({wandb_key: wandb.Image(img)})
    plt.close(fig)
    
    n_true_cell_types = len(adata_tfidf.obs['cell_type'].unique())
    print(f"Number of cell types: {n_true_cell_types}")
    
    optimal_resolution, n_clusters = find_leiden_resolution_for_n_clusters(
        adata_tfidf, 
        target_n_clusters=num_cell_types, 
        min_res=0.1, 
        max_res=2.0, 
        random_state=42
    )
    
    print(f"Optimal resolution: {optimal_resolution:.4f}")
    print(f"Number of Leiden clusters: {n_clusters}")
    
    leiden_title = f'Leiden Clustering{leiden_title_suffix}'
    fig = sc.pl.umap(adata_tfidf, color='leiden', legend_loc='on data', title=leiden_title, return_fig=True, show=True)
    plt.show()
    img = prepare_img(fig)
    wandb.log({f"umap_cell_types_{suffix}_leiden_clustering": wandb.Image(img)})
    plt.close(fig)
    
    true_labels = adata_tfidf.obs['cell_type'].values
    predicted_labels = adata_tfidf.obs['leiden'].values
    
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
    
    silhouette_score = silhouette(adata_tfidf, label_key='cell_type', embed='cell_embeddings_zero_shot')
    silhouette_batch_score = silhouette_batch(adata_tfidf, label_key='cell_type', batch_key=batch_key, embed='cell_embeddings_zero_shot')
    
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
    print(f"Silhouette score: {silhouette_score:.4f}")
    print(f"Silhouette batch score: {silhouette_batch_score:.4f}")
    print(f"\nNumber of true cell types: {len(adata_tfidf.obs['cell_type'].unique())}")
    print(f"Number of predicted clusters: {len(adata_tfidf.obs['leiden'].unique())}")
    
    wandb.log({
        f"ARI_{suffix}": ari_score,
        f"NMI_{suffix}": nmi_score,
        f"Silhouette_{suffix}": silhouette_score,
        f"Silhouette_batch_{suffix}": silhouette_batch_score
    })


######################################################
# NO PERMUTATION
######################################################
run_pipeline(
    tfidf_func=global_TFIDF,
    suffix="no_shuffling",
    leiden_title_suffix=" (no shuffling)",
    save_embeddings=True,
    umap_title="Cell embeddings (true labels)"
)

######################################################
# PERMUTED LABELS
######################################################
run_pipeline(
    tfidf_func=global_TFIDF_with_shuffling,
    suffix="permuted_labels",
    leiden_title_suffix="",
    save_embeddings=False
)

######################################################
# COMPLETELY PERMUTED LABELS
######################################################
run_pipeline(
    tfidf_func=global_TFIDF_with_complete_shuffling,
    suffix="complete_shuffling",
    leiden_title_suffix="",
    save_embeddings=False
)

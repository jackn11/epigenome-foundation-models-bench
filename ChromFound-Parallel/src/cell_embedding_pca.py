"""
Memory-efficient cell embedding with incremental PCA.
Processes embeddings in batches and saves only PCA-reduced output.

For large datasets (100K+ cells), this avoids loading all 1.3M-dim embeddings into RAM.
"""
import argparse
import logging
import os

import numpy as np
import scanpy as sc
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from src.data.dataset_ds import DatasetMultiPad
from src.models.chromfd_mixer import PretrainModelMambaLM
from src.utils.model_utils import ModelUtils

logging.basicConfig(level=logging.INFO)


def load_data(file_path):
    if file_path.endswith('.h5ad'):
        logging.info(f"Reading h5ad file from {file_path}")
        adata = sc.read_h5ad(file_path)
        if 'hg38_Start' in adata.var.columns:
            adata.var['hg38_Start'] = adata.var['hg38_Start'].astype(int)
        if 'hg38_End' in adata.var.columns:
            adata.var['hg38_End'] = adata.var['hg38_End'].astype(int)
        return adata
    else:
        raise ValueError("Unsupported file format. Please provide a .h5ad file.")


class EmbeddingModel(PretrainModelMambaLM):
    def __init__(self, **kwargs):
        super(EmbeddingModel, self).__init__(**kwargs)

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x = self.embedding(value, chromosome.long(), hg38_start.long(), hg38_end.long())
        x = self.backbone(x)
        return x


def generate_embeddings_with_incremental_pca(model, adata_dataloader, device, n_components=50):
    """
    Generate embeddings and apply incremental PCA in a memory-efficient way.
    
    Instead of storing all 1.3M-dim embeddings, we:
    1. First pass: Fit IncrementalPCA on batches
    2. Second pass: Transform batches and collect only PCA-reduced embeddings
    
    This uses O(n_cells * n_components) memory instead of O(n_cells * n_features).
    """
    model.eval()
    
    # First pass: Fit IncrementalPCA
    logging.info(f"Pass 1/2: Fitting IncrementalPCA with {n_components} components...")
    ipca = IncrementalPCA(n_components=n_components)
    
    # Accumulator for first partial_fit (needs at least n_components samples)
    first_fit_done = False
    accumulated_embeddings = []
    accumulated_count = 0
    
    with torch.no_grad():
        for batch_data in tqdm(adata_dataloader, desc="Fitting PCA"):
            value, chromosome, pos_start, pos_end, cell_type = batch_data
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)

            embedding = model(value, chromosome, pos_start, pos_end)
            embedding = embedding.mean(axis=-1)  # (batch, n_features)
            embedding = embedding.detach().cpu().numpy()
            
            if not first_fit_done:
                # Accumulate until we have enough samples for first partial_fit
                accumulated_embeddings.append(embedding)
                accumulated_count += embedding.shape[0]
                
                if accumulated_count >= n_components:
                    # Concatenate and do first partial_fit
                    first_batch = np.concatenate(accumulated_embeddings, axis=0)
                    ipca.partial_fit(first_batch)
                    first_fit_done = True
                    # Clear accumulator to free memory
                    accumulated_embeddings = []
                    accumulated_count = 0
            else:
                # Normal partial fit for subsequent batches
                ipca.partial_fit(embedding)
    
    logging.info(f"PCA fitted. Explained variance ratio: {ipca.explained_variance_ratio_.sum():.4f}")
    
    # Second pass: Transform and collect PCA-reduced embeddings
    logging.info("Pass 2/2: Transforming embeddings to PCA space...")
    pca_embeddings = []
    
    with torch.no_grad():
        for batch_data in tqdm(adata_dataloader, desc="Transforming"):
            value, chromosome, pos_start, pos_end, cell_type = batch_data
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)

            embedding = model(value, chromosome, pos_start, pos_end)
            embedding = embedding.mean(axis=-1)
            embedding = embedding.detach().cpu().numpy()
            
            # Transform to PCA space (1.3M -> 50 dims)
            pca_batch = ipca.transform(embedding)
            pca_embeddings.append(pca_batch)
    
    pca_embeddings = np.concatenate(pca_embeddings, axis=0)
    logging.info(f"Final PCA embeddings shape: {pca_embeddings.shape}")
    
    return pca_embeddings, ipca


def generate_embeddings_with_sampled_pca(model, adata_dataloader, device, n_components=50, n_samples_for_pca=1000):
    """
    MEMORY-EFFICIENT version: Two-pass approach that never stores all full embeddings.
    
    Strategy:
    1. Pass 1: Collect embeddings ONLY for ~1000 random cells → fit PCA
    2. Pass 2: Generate embeddings batch-by-batch, project immediately to 50-dim
    
    Memory usage: O(n_samples * n_features) + O(n_cells * n_components)
    Instead of:   O(n_cells * n_features) which would be ~680GB for full dataset
    """
    from sklearn.decomposition import PCA
    model.eval()
    
    # Get total cells from dataset
    total_cells = len(adata_dataloader.dataset)
    total_batches = len(adata_dataloader)
    
    logging.info(f"Total cells: {total_cells}, Total batches: {total_batches}")
    logging.info(f"Will sample {n_samples_for_pca} cells for PCA fitting")
    
    # Determine which cells to sample (by index)
    np.random.seed(42)
    n_samples = min(n_samples_for_pca, total_cells)
    sample_cell_indices = set(np.random.choice(total_cells, n_samples, replace=False))
    
    # ==================== PASS 1: Collect samples for PCA fitting ====================
    logging.info("Pass 1/2: Collecting samples for PCA fitting...")
    sample_embeddings = []
    current_cell_idx = 0
    
    with torch.no_grad():
        for batch_data in tqdm(adata_dataloader, desc="Pass 1 - Sampling"):
            value, chromosome, pos_start, pos_end, cell_type = batch_data
            batch_len = value.shape[0]
            
            # Check if any cells in this batch are in our sample
            batch_indices = set(range(current_cell_idx, current_cell_idx + batch_len))
            cells_to_sample = batch_indices & sample_cell_indices
            
            if cells_to_sample:
                # Generate embeddings for this batch
                value = value.to(device)
                chromosome = chromosome.to(device)
                pos_start = pos_start.to(device)
                pos_end = pos_end.to(device)

                embedding = model(value, chromosome, pos_start, pos_end)
                embedding = embedding.mean(axis=-1)
                embedding = embedding.detach().cpu().numpy()
                
                # Extract only the sampled cells from this batch
                for cell_idx in cells_to_sample:
                    local_idx = cell_idx - current_cell_idx
                    sample_embeddings.append(embedding[local_idx:local_idx+1])
            
            current_cell_idx += batch_len
    
    # Fit PCA on samples
    sample_embeddings = np.concatenate(sample_embeddings, axis=0)
    logging.info(f"Collected {sample_embeddings.shape[0]} samples, fitting PCA...")
    
    pca = PCA(n_components=n_components)
    pca.fit(sample_embeddings)
    logging.info(f"PCA fitted. Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Free sample embeddings
    del sample_embeddings
    
    # ==================== PASS 2: Generate all embeddings and project immediately ====================
    logging.info("Pass 2/2: Generating and projecting all embeddings...")
    pca_embeddings = []
    
    with torch.no_grad():
        for batch_data in tqdm(adata_dataloader, desc="Pass 2 - Projecting"):
            value, chromosome, pos_start, pos_end, cell_type = batch_data
            value = value.to(device)
            chromosome = chromosome.to(device)
            pos_start = pos_start.to(device)
            pos_end = pos_end.to(device)

            # Generate full embeddings
            embedding = model(value, chromosome, pos_start, pos_end)
            embedding = embedding.mean(axis=-1)
            embedding = embedding.detach().cpu().numpy()
            
            # Project to PCA space IMMEDIATELY (1.3M -> 50 dims)
            pca_batch = pca.transform(embedding)
            pca_embeddings.append(pca_batch)
            
            # Full embedding is discarded here, never accumulated!
    
    pca_embeddings = np.concatenate(pca_embeddings, axis=0)
    logging.info(f"Final PCA embeddings shape: {pca_embeddings.shape}")
    
    return pca_embeddings, pca


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help='GPU rank', default=0)
    parser.add_argument('--data_path', type=str, required=True, help='path of h5ad file')
    parser.add_argument('--pretrain_checkpoint_path', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--pretrain_model_file', type=str, required=True, help='file name of pre-trained model')
    parser.add_argument('--pretrain_config_file', type=str, required=True, help='file name of pre-trained config')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--cell_type_col', required=True, help='column name of cell type')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the updated h5ad file')
    parser.add_argument('--n_pca_components', type=int, default=50, help='Number of PCA components to keep')
    parser.add_argument('--n_samples_for_pca', type=int, default=1000, help='Number of cells to sample for PCA fitting (faster)')
    parser.add_argument('--use_incremental_pca', action='store_true', help='Use slower IncrementalPCA instead of sampled PCA')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    pretrain_path = args.pretrain_checkpoint_path
    pretrain_config_file = args.pretrain_config_file
    with open(os.path.join(pretrain_path, pretrain_config_file), 'r') as file:
        pretrain_config = yaml.safe_load(file)
    pretrain_model_args = pretrain_config['model_args']
    pretrain_data_args = pretrain_config["data_args"]
    chromosome_vocab = ModelUtils.get_chromosome_vocab(os.path.join(pretrain_path, "chromosome_vocab.yaml"))
    pretrain_data_args["chromosome_vocab"] = chromosome_vocab
    adata = load_data(args.data_path)
    
    max_length = adata.shape[1]
    cell_type = list(set(adata.obs['celltype'].unique().tolist()))
    cell_type_map = {cell_type: idx for idx, cell_type in enumerate(sorted(cell_type))}

    pretrain_data_args['cell_type_map'] = cell_type_map
    pretrain_model_args["cell_type_num"] = len(cell_type_map)
    pretrain_data_args['cell_type_col'] = args.cell_type_col
    pretrain_data_args["feature_num"] = adata.shape[1]
    pretrain_model_args["feature_num"] = adata.shape[1]
    pretrain_model_args["batch_size"] = args.batch_size
    pretrain_data_args["max_length"] = max_length
    pretrain_model_args["max_length"] = max_length
    pretrain_model_args["device"] = device
    pretrain_model_args["add_cls"] = pretrain_data_args["add_cls"]
    pretrain_model_args["mask_ratio"] = 0.0
    pretrain_data_args["return_batch_label"] = False

    model = EmbeddingModel(**pretrain_model_args)
    state_dict = torch.load(str(os.path.join(pretrain_path, args.pretrain_model_file)))
    model.load_state_dict(state_dict['module'])
    model = model.to(device)
    model.eval()
    
    adataset = DatasetMultiPad(*[adata], **pretrain_data_args)
    adata_dataloader = DataLoader(
        adataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=4, prefetch_factor=2
    )

    # Generate PCA-reduced embeddings
    if args.use_incremental_pca:
        logging.info("Using IncrementalPCA (slower but more accurate for very large datasets)")
        pca_embeddings, pca = generate_embeddings_with_incremental_pca(
            model, adata_dataloader, device, n_components=args.n_pca_components
        )
    else:
        logging.info(f"Using sampled PCA (FAST - fitting on {args.n_samples_for_pca} cells)")
        pca_embeddings, pca = generate_embeddings_with_sampled_pca(
            model, adata_dataloader, device, 
            n_components=args.n_pca_components, 
            n_samples_for_pca=args.n_samples_for_pca
        )
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logging.info(f"Created directory {args.output_path}")
    
    # Save PCA-reduced embeddings (much smaller!)
    adata.obsm['X_pca'] = pca_embeddings
    
    # Also save the explained variance for reference
    adata.uns['pca'] = {
        'variance_ratio': pca.explained_variance_ratio_,
        'variance': pca.explained_variance_,
        'n_components': args.n_pca_components
    }
    
    output_file = "embeddings_pca.h5ad"
    adata.write_h5ad(os.path.join(args.output_path, output_file))
    
    # Memory usage comparison
    full_size_gb = (adata.n_obs * max_length * 4) / (1024**3)
    pca_size_gb = (adata.n_obs * args.n_pca_components * 4) / (1024**3)
    logging.info(f"Memory saved: {full_size_gb:.2f} GB -> {pca_size_gb:.4f} GB ({full_size_gb/pca_size_gb:.0f}x reduction)")
    logging.info(f"PCA embeddings saved to {os.path.join(args.output_path, output_file)}")


if __name__ == '__main__':
    main()



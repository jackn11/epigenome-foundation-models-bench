#%%
import os
from pathlib import Path
import scanpy as sc
import numpy as np
import pandas as pd

path = Path("/scratch/wkim/project-2-team-1/benchmarks/results/chromfound/Pierce2021/merge10/embeddings_pca.h5ad")

# Load the h5ad file
print(f"Loading embeddings from: {path}")
adata = sc.read_h5ad(path)

# Basic information about the AnnData object
print("\n" + "="*80)
print("BASIC INFORMATION")
print("="*80)
print(f"Number of cells (observations): {adata.n_obs}")
print(f"Number of features (variables): {adata.n_vars}")
print(f"Shape: {adata.shape}")

# Inspect embeddings
print("\n" + "="*80)
print("EMBEDDINGS")
print("="*80)
if 'X_pca' in adata.obsm:
    embeddings = adata.obsm['X_pca']
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings min: {embeddings.min():.4f}, max: {embeddings.max():.4f}")
    print(f"Embeddings mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
    # print(f"\nFirst 5 rows of embeddings:")
    # print(embeddings[:5])
else:
    print("No 'X_pca' found in adata.obsm")
    print(f"Available keys in obsm: {list(adata.obsm.keys())}")

# Inspect labels/observations
print("\n" + "="*80)
print("LABELS / OBSERVATIONS")
print("="*80)
print(f"Observation columns: {list(adata.obs.columns)}")
# print(f"\nFirst few rows of observations:")
# print(adata.obs.head())

# Check for cell type column
if 'celltype' in adata.obs.columns:
    print('\ncelltype found')
    print(f"Cell types (unique): {adata.obs['celltype'].nunique()}")
    print(f"Cell type distribution:")
    print(adata.obs['celltype'].value_counts())
elif 'cell_type' in adata.obs.columns:
    print('\ncell_type found')
    print(f"Cell types (unique): {adata.obs['cell_type'].nunique()}")
    print(f"Cell type distribution:")
    print(adata.obs['cell_type'].value_counts())

# Check for PCA metadata
# print("\n" + "="*80)
# print("PCA METADATA")
# print("="*80)
# if 'pca' in adata.uns:
#     pca_info = adata.uns['pca']
    # print(f"PCA information: {pca_info}")
    # if 'variance_ratio' in pca_info:
    #     print(f"Explained variance ratio (sum): {pca_info['variance_ratio'].sum():.4f}")
    #     print(f"Number of components: {pca_info.get('n_components', 'N/A')}")
# else:
#     print("No PCA metadata found in adata.uns")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"File: {path}")
print(f"Cells: {adata.n_obs}, Features: {adata.n_vars}")
if 'X_pca' in adata.obsm:
    print(f"Embeddings: {adata.obsm['X_pca'].shape} (PCA-reduced)")
print(f"Observation columns: {len(adata.obs.columns)}")
# %%

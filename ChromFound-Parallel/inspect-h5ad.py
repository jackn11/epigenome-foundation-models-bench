#%%
import os
import scanpy as sc
import h5py
from pathlib import Path

#%%
root = Path('/scratch/wkim/project-2-team-1/ChromFound-Parallel/results-copy/chromfound')
# adata_full = sc.read_h5ad(root / 'Kanemaru2023_full' / 'merge10' / 'Kanemaru2023_full_preprocessed_merge10.h5ad')
adata_full = sc.read_h5ad(root / 'Kanemaru2023_full' / 'merge10' / 'embeddings_pca.h5ad')
adata_down = sc.read_h5ad(root / 'Kanemaru2023_downsampled' / 'merge10' / 'embeddings_pca.h5ad')
# %%
adata_full.obsm['X_pca']
#%%
adata_down.obsm.keys()
# %%

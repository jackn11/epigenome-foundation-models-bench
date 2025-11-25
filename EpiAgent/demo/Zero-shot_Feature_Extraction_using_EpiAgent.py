#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF

from epiagent.dataset import CellDataset, collate_fn
from torch.utils.data import DataLoader
from epiagent.model import EpiAgent
import torch
from epiagent.inference import infer_cell_embeddings
import scanpy as sc
print("imports complete")

#%%
# Load the dataset
print("Loading dataset")
input_path = '../data/sample/raw_h5ad/Kanemaru2023_downsampled_10000_cells.h5ad'
adata = sc.read_h5ad(input_path)
print("Dataset loaded")


# Load the cCRE document frequency data
print("Loading cCRE document frequency")
cCRE_document_frequency = np.load('../data/cCRE_document_frequency.npy')
print("cCRE document frequency loaded")
#%%
# Apply TFIDF transformation
print("Applying TFIDF transformation")
global_TFIDF(adata, cCRE_document_frequency)
print("TFIDF transformation applied")

# Perform tokenization
print("Performing tokenization")
tokenization(adata)
print("Tokenization complete")
#%%
# from epiagent.dataset import CellDataset, collate_fn
# from torch.utils.data import DataLoader

# Create the dataset
print("Creating dataset")
cell_sentences = adata.obs['cell_sentences'].tolist()
cell_dataset = CellDataset(cell_sentences=cell_sentences)

print("Dataset created")

# Create the DataLoader
batch_size = 8
dataloader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

#%%
# from epiagent.model import EpiAgent
# import torch

# Load the pretrained model
model_path = '../model/pretrained_EpiAgent.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model = EpiAgent(vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8, max_rank_embeddings=8192, use_flash_attn=True, pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
pretrained_model.load_state_dict(torch.load(model_path, map_location=device))

#%%
# from epiagent.inference import infer_cell_embeddings

# Extract cell embeddings
cell_embeddings = infer_cell_embeddings(pretrained_model, device, dataloader)

#%%
# import scanpy as sc

# Assign embeddings to the AnnData object
adata.obsm['cell_embeddings_zero_shot'] = cell_embeddings

# UMAP visualization
sc.pp.neighbors(adata, use_rep='cell_embeddings_zero_shot')
sc.tl.umap(adata)

# Plot UMAP with original cell types
sc.pl.umap(adata, color='cell_type')
plt.savefig('umap.png', dpi=300, bbox_inches='tight')
plt.close()
print("UMAP plot saved to umap.png")

# Save the processed AnnData
output_path = '../data/sample/processed_h5ad/Kanemaru2023_downsampled_10000_cells_EpiAgent_zero_shot_outputs.h5ad'
adata.write(output_path)
print(f"Processed AnnData saved at {output_path}")
# %%

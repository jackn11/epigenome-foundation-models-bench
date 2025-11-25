#%%
import scanpy as sc
import numpy as np
from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF

# Load the AnnData object
print("Loading dataset")
input_path = '../data/sample/raw_h5ad/Li2023b_downsampled_10000_cells.h5ad'
adata = sc.read_h5ad(input_path)
print("Dataset loaded")

# Load the cCRE document frequency and compute TF-IDF
print("Loading cCRE document frequency")
cCRE_document_frequency = np.load('../data/cCRE_document_frequency.npy')
print("cCRE document frequency loaded")

print("Applying TFIDF")
adata = global_TFIDF(adata, cCRE_document_frequency)
print("TFIDF applied")

# Tokenize the data using the tokenization function
tokenization(adata)

print("Data preprocessing completed: TFIDF applied, and cell sentences generated.")
#%%
from epiagent.dataset import CellDataset, collate_fn
from torch.utils.data import DataLoader

# Create dataset
cell_sentences = adata.obs['cell_sentences'].tolist()
dataset = CellDataset(cell_sentences=cell_sentences, max_length=8192)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=4, collate_fn=collate_fn)

print("Dataset and DataLoader successfully created.")
#%%
import torch
import json
from sklearn.preprocessing import LabelEncoder
from epiagent.model import EpiAgent_supervised

# Load cell types for EpiAgent-NT
cell_type_file = '../model/EpiAgent-B/EpiAgent-B_cell_types.json'
with open(cell_type_file, 'r') as f:
    cell_types_for_epiant_b = json.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(cell_types_for_epiant_b)

# Load pretrained model
model_path = '../model/EpiAgent-B/EpiAgent-B.pth'
epiagent_b = EpiAgent_supervised(
    vocab_size=1355449, num_layers=18, embedding_dim=512, num_attention_heads=8, 
    max_rank_embeddings=8192, num_classes=len(cell_types_for_epiant_b), use_flash_attn=True
)
epiagent_b.load_state_dict(torch.load(model_path))

print("Pretrained EpiAgent-NT model loaded successfully.")
#%%
from epiagent.inference import infer_cell_types, filter_rare_cell_types

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epiagent_b.to(device)

# Perform inference
results = infer_cell_types(model=epiagent_b, device=device, dataloader=dataloader, need_cell_embeddings=True)

# Filter rare predicted cell types
predicted_labels_filtered = filter_rare_cell_types(
    results['predicted_labels'], np.array(results['predicted_probabilities']), threshold=0.005
)

# Map predicted labels to real cell types
predicted_cell_types = label_encoder.inverse_transform(predicted_labels_filtered)

print("Inference completed: Predicted cell types and embeddings generated.")
#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Assign predictions to AnnData
adata.obs['predicted_cell_types'] = predicted_cell_types
adata.obsm['cell_embeddings_from_EpiAgent-B'] = results['cell_embeddings']

# UMAP visualization
sc.pp.neighbors(adata, use_rep='cell_embeddings_from_EpiAgent-B')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['cell_type'], save="umap_true_cell_types.png")
sc.pl.umap(adata, color=['predicted_cell_types'], save="umap_predicted_cell_types.png")

# Prepare labels for confusion matrix visualization
true_labels = adata.obs['cell_type']
predicted_labels = adata.obs['predicted_cell_types']
true_class_list = sorted(true_labels.unique())
pred_class_list = sorted(predicted_labels.unique())

# Create an empty confusion matrix
conf_matrix = np.zeros((len(true_class_list), len(pred_class_list)), dtype=int)

# Map class names to indices
true_class_to_index = {cls: idx for idx, cls in enumerate(true_class_list)}
pred_class_to_index = {cls: idx for idx, cls in enumerate(pred_class_list)}

# Populate the confusion matrix
for pred, true in zip(predicted_labels, true_labels):
    pred_idx = pred_class_to_index[pred]
    true_idx = true_class_to_index[true]
    conf_matrix[true_idx, pred_idx] += 1

# Convert to DataFrame for better readability
conf_matrix_df = pd.DataFrame(conf_matrix, index=true_class_list, columns=pred_class_list)

# Normalize the confusion matrix
normalized_conf_matrix = conf_matrix_df.div(conf_matrix_df.sum(axis=1), axis=0)

# Visualize the confusion matrix
plt.figure(figsize=(0.3 * len(pred_class_list), 0.5 * len(true_class_list)))
sns.heatmap(normalized_conf_matrix, annot=False, fmt=".2f", xticklabels=pred_class_list, yticklabels=true_class_list, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.show()

# # Save processed AnnData
# output_path = '../data/sample/processed_h5ad/Li2023_downsampled_10000_cells_EpiAgent-B_outputs.h5ad'
# adata.write(output_path)
# print(f"Processed AnnData saved at {output_path}")
#%%

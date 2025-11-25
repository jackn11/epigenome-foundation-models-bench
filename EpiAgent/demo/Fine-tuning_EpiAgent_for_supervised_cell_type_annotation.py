#%%
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
input_path = '../data/sample/raw_h5ad/Kanemaru2023_downsampled_10000_cells.h5ad'
adata = sc.read_h5ad(input_path)

# Split the dataset into training and test sets, stratified by cell type
train_adata, test_adata = train_test_split(
    adata,
    test_size=0.33,
    stratify=adata.obs['cell_type'],
    random_state=42
)

print(f"Training set size: {train_adata.n_obs} cells")
print(f"Test set size: {test_adata.n_obs} cells")
#%%
import numpy as np
from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF

# Load the cCRE document frequency data
cCRE_document_frequency = np.load('../data/cCRE_document_frequency.npy')

# Apply TF-IDF transformation to the training set
import ipdb; ipdb.set_trace()
global_TFIDF(train_adata, cCRE_document_frequency)
# Perform tokenization on the training set to create cell sentences
tokenization(train_adata)

# Repeat the TF-IDF transformation and tokenization for the test set
global_TFIDF(test_adata, cCRE_document_frequency)
tokenization(test_adata)

#%%
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder on the training cell types and transform
train_adata.obs['cell_type_encoded'] = label_encoder.fit_transform(train_adata.obs['cell_type'])

# Transform the test cell types using the same encoder
test_adata.obs['cell_type_encoded'] = label_encoder.transform(test_adata.obs['cell_type'])

# Get the number of unique cell types (classes)
num_classes = len(label_encoder.classes_)
print(f"Number of cell type classes: {num_classes}")

#%%
from epiagent.dataset import CellDatasetForSCA, collate_fn_for_SCA
from torch.utils.data import DataLoader

# Extract cell sentences and labels from the training AnnData
train_cell_sentences = train_adata.obs['cell_sentences'].tolist()
train_cell_types = train_adata.obs['cell_type_encoded'].tolist()

# Create the training dataset
train_dataset = CellDatasetForSCA(
    cell_sentences=train_cell_sentences,
    cell_types=train_cell_types,
    max_length=8192,
    is_random=False
)

# Create the training DataLoader
train_batch_size = 1
train_dataloader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=16,
    collate_fn=collate_fn_for_SCA
)

#%%
from epiagent.dataset import CellDataset, collate_fn

# Extract cell sentences from the test AnnData
test_cell_sentences = test_adata.obs['cell_sentences'].tolist()

# Create the test dataset
test_dataset = CellDataset(
    cell_sentences=test_cell_sentences,
    max_length=8192,
    is_random=False
)

# Create the test DataLoader
test_batch_size = 8
test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

#%%
import torch
import torch.nn as nn
from epiagent.model import EpiAgent, EpiAgent_supervised

#%%
# Specify the path to the pre-trained unsupervised model
pretrained_model_path = '../model/pretrained_EpiAgent.pth'

# Set the device (GPU if available)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the pre-trained EpiAgent model
pretrained_model = EpiAgent(
    vocab_size=1355449,
    num_layers=18,
    embedding_dim=512,
    num_attention_heads=8,
    max_rank_embeddings=8192,
    use_flash_attn=True,
    pos_weight_for_RLM=torch.tensor(1.0),
    pos_weight_for_CCA=torch.tensor(1.0)
)

# Load the pre-trained weights into the model
pretrained_model.load_state_dict(torch.load(pretrained_model_path))

#%%
# Initialize the supervised EpiAgent model with the appropriate number of classes
from epiagent.model import EpiAgent_supervised

num_classes = len(label_encoder.classes_)

supervised_model = EpiAgent_supervised(
    vocab_size=1355449,
    num_layers=18,
    embedding_dim=512,
    num_attention_heads=8,
    max_rank_embeddings=8192,
    num_classes=num_classes,
    use_flash_attn=True
)

# Transfer matching weights from the pre-trained model to the supervised model
# This involves copying parameters with the same names
pretrained_state_dict = pretrained_model.state_dict()
supervised_state_dict = supervised_model.state_dict()

# Identify parameters that are common between the two models
common_params = {k: v for k, v in pretrained_state_dict.items() if k in supervised_state_dict and v.size() == supervised_state_dict[k].size()}

# Update the supervised model's state dict with pre-trained parameters
supervised_state_dict.update(common_params)

# Load the updated state dict into the supervised model
supervised_model.load_state_dict(supervised_state_dict)

# Set the device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the supervised model to the specified device
supervised_model.to(device)

print("Supervised model initialized and pre-trained weights loaded.")

#%%
from epiagent.train import fine_tune_epiagent_for_SCA

# Fine-tune the model
fine_tuned_model = fine_tune_epiagent_for_SCA(
    model=supervised_model,
    train_dataloader=train_dataloader,
    num_steps=100000,  # Total training steps
    save_dir='../model/fine_tune/SCA/demo_dataset/',
    device=device,
    learning_rate=1e-4,
    save_steps=20000,
    log_steps=500,
    warmup_steps=10000,
    is_logging=True
)

#%%
from epiagent.inference import infer_cell_types

# Ensure the model is in evaluation mode
fine_tuned_model.eval()

# Perform inference on the test set
results = infer_cell_types(
    model=fine_tuned_model, 
    device=device, 
    dataloader=test_dataloader, 
    need_cell_embeddings=True
)

# Extract predicted labels and cell embeddings from the results
predicted_labels = results['predicted_labels']
cell_embeddings = results['cell_embeddings']

# Map predicted labels to real cell types using the label encoder
predicted_cell_types = label_encoder.inverse_transform(predicted_labels)

print("Inference completed: Predicted cell types and embeddings generated.")

#%%
import numpy as np

# Assign predicted cell types to the test AnnData object
test_adata.obs['predicted_cell_types'] = predicted_cell_types

# Assign cell embeddings to .obsm
test_adata.obsm['cell_embeddings_from_EpiAgent'] = cell_embeddings

#%%
import scanpy as sc

# Use the cell embeddings for UMAP visualization
sc.pp.neighbors(test_adata, use_rep='cell_embeddings_from_EpiAgent')

# Compute UMAP embedding
sc.tl.umap(test_adata)

# Plot UMAP colored by true cell types
sc.pl.umap(test_adata, color=['cell_type'], title='UMAP of True Cell Types (Test Set)')

# Plot UMAP colored by predicted cell types
sc.pl.umap(test_adata, color=['predicted_cell_types'], title='UMAP of Predicted Cell Types (Test Set)')

#%%
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# True labels and predicted labels for the test set
true_labels = test_adata.obs['cell_type_encoded'].values
predicted_labels = predicted_labels  # From the inference step

# Calculate performance metrics
accuracy = accuracy_score(true_labels, predicted_labels)
kappa = cohen_kappa_score(true_labels, predicted_labels)
macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Map encoded labels back to original cell type names
true_cell_types = test_adata.obs['cell_type'].values  # True cell type names
predicted_cell_types = test_adata.obs['predicted_cell_types'].values  # Predicted cell type names

# Create a DataFrame for the confusion matrix
confusion_df = pd.crosstab(
    pd.Series(true_cell_types, name='Actual'),
    pd.Series(predicted_cell_types, name='Predicted'),
    normalize='index'
)

# Visualize the normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_df, annot=True, fmt=".2f", cmap="Blues")
plt.title("Normalized Confusion Matrix (Test Set)")
plt.show()

#%%
# Save the test AnnData with predictions and embeddings
output_path = '../data/sample/processed_h5ad/Kanemaru2023_downsampled_10000_cells_EpiAgent_SCA_outputs.h5ad'
test_adata.write(output_path)
print(f"Processed AnnData saved at {output_path}")
# %%

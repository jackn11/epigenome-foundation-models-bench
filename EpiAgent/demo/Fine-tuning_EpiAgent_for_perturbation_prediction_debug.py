#%%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from epiagent.model import EpiAgent
from epiagent.dataset import CellDataset, collate_fn

#%%
# Load the dataset
input_path = '/scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/All_data_K562_1.csv'
df = pd.read_csv(input_path,usecols=['cell_indices','perturbation','cell_line','ncounts','ngenes'])

# Remove cells with unknown perturbation type
df = df[~df['perturbation'].isna()]
print(df['perturbation'].value_counts())
#%%
all_peat_list = list(np.unique(df['perturbation']))
all_peat_list.remove('control')
pert_to_index = {element: index for index, element in enumerate(all_peat_list)}

Test_pert_list = ['PRDM9','SETD2','SMARCA4','SMARCB1','TET2']
metadata = df.copy()
Train_df = metadata[~metadata['perturbation'].isin(Test_pert_list)]
Test_df = metadata[metadata['perturbation'].isin(Test_pert_list)]

cell_indeces = [json.loads(instance) for instance in Train_df['cell_indices'].tolist()]
cell_dataset = CellDataset(cell_indeces,8192,True)
dataloader = torch.utils.data.DataLoader(cell_dataset, batch_size=4, shuffle=False,num_workers=4, collate_fn=collate_fn)


#%%
# Specify the path to the pre-trained model
# model_path = '/home/chenxiaoyang/program/scCASdata/model/pretrained_EpiAgent.pth'
model_path = '/scratch/wkim/project-2-team-1/EpiAgent/model/pretrained_EpiAgent.pth'

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the EpiAgent model with appropriate configurations
pretrained_model = EpiAgent(
    vocab_size=1355449,
    num_layers=18,
    embedding_dim=512,
    num_attention_heads=8,
    max_rank_embeddings=8192,
    use_flash_attn=True,
    pos_weight_for_RLM=torch.tensor(1.),
    pos_weight_for_CCA=torch.tensor(1.)
)

# Load the pre-trained weights into the model
pretrained_model.load_state_dict(torch.load(model_path))

# Move the model to the specified device
pretrained_model.to(device)

#%%
from torch.cuda.amp import autocast
import umap

pretrained_model.eval()

cell_embeddings = []
for _, batch in enumerate(dataloader):
    torch.cuda.empty_cache()
    input_ids = batch.to(device)
    with autocast():
        output_embeddings = pretrained_model(input_ids)['transformer_outputs'][:,0,:].cpu().detach().numpy()
    input_ids = input_ids.cpu().detach().numpy()
    for output_embedding in output_embeddings:
        cell_embeddings.append(output_embedding)
cell_embeddings = np.array(cell_embeddings)
umap_embeddings = umap.UMAP(n_components=2, random_state=32).fit_transform(np.array(cell_embeddings))

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization perturbation of 'Control' vs 'Not-control'
cell_label = Train_df["perturbation"].tolist()
new_cell_label = ['Control' if item == 'control' else 'Not-control' for item in cell_label]
data = pd.DataFrame({'x': umap_embeddings[:, 0], 'y': umap_embeddings[:, 1], 'label': new_cell_label})

plt.figure(figsize=(8,8))
sns.scatterplot(x='x', y='y', hue='label',palette='tab20',s=15, data=data)
plt.legend(bbox_to_anchor=(0., -0.2, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

#%%
import ot

torch.cuda.empty_cache()

# Extract the indices of ‘Stimulated’ and ‘Resting’ from the ‘condition’ column
match_types = []
match_idxs = []
for pert in np.unique(Train_df['perturbation']):
    if pert == 'control':
        continue
    
    stim_indices = np.array(Train_df['perturbation'] == pert)
    control_indices = np.array(Train_df['perturbation'] == 'control')

    # Split cell_embeddings into stim and control based on the indices
    stim = cell_embeddings[stim_indices]
    ctrl = cell_embeddings[control_indices]
    M = ot.dist(stim, ctrl, metric='cosine')
    G = ot.emd(torch.ones(stim.shape[0]) / stim.shape[0],
               torch.ones(ctrl.shape[0]) / ctrl.shape[0],
               torch.tensor(M), numItermax=100000)
    match_idx = torch.max(G, 0)[1].numpy()

    match_types.append(pert + '-' + 'control')
    match_idxs.append(match_idx)
    
match_idx = dict(zip(match_types, match_idxs))

#%%
match_idx.keys()

#%%
import scanpy as sc

# Load the anndata object
# adata_path = '/data/user/chenxiaoyang/data/scATAC/Liscovitch-BrauerSanjana2021/K562_1/tfidf_data.h5ad'
adata_path = '/scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/tfidf_data.h5ad'
adata = sc.read_h5ad(adata_path)

# Select the top 50,000 peaks ranked by chromatin accessibility
peak_sum = np.sum(adata.X, axis=0)
peak_sum = np.array(peak_sum).reshape(-1)
peak_sum_sortidx = np.argsort(peak_sum)[::-1]
peak_sum_sortidx = peak_sum_sortidx[:50000]
adata = adata[:, peak_sum_sortidx]
adata = adata[~adata.obs['perturbation'].isna()]

# Split to 'train_adata' and 'test_adata'
train_adata = adata[~adata.obs['perturbation'].isin(Test_pert_list)]
test_adata = adata[adata.obs['perturbation'].isin(Test_pert_list)]

#%%
New_train_df = pd.DataFrame()
Train_control_df = Train_df[Train_df['perturbation'] == 'control'].copy()
Train_control_df.reset_index(drop=True, inplace=True)
Train_control_df['predicted_cell_indices'] = Train_control_df['cell_indices']

train_resting = train_adata[train_adata.obs['perturbation'] == 'control']
New_train_df = pd.concat([New_train_df, Train_control_df], ignore_index=True)
New_train_adata = train_resting

for pert in np.unique(Train_df['perturbation']): 
    if pert == 'control':
        continue
    Train_stimulation_df = Train_df[Train_df['perturbation'] == pert].copy()
    Train_stimulation_df.reset_index(drop=True, inplace=True)

    train_stimulated = train_adata[train_adata.obs['perturbation'] == pert]
    
    # Add a 'predicted_cell_indices' column with the same values as the 'cell_indices column'
    Train_stimulation_df['predicted_cell_indices'] = Train_stimulation_df['cell_indices']

    Train_control_mapping_df = Train_control_df.copy()
    # Update the 'predicted_cell_indices' column in Train_control_mapping_df
    Train_control_mapping_df['predicted_cell_indices'] = Train_stimulation_df.loc[match_idx[pert + '-' + 'control'], 'cell_indices'].values
    Train_control_mapping_df['perturbation'] = "control_mapping_to_" + pert
    # Merge the three DataFrames
    New_train_df = pd.concat([New_train_df, Train_stimulation_df, Train_control_mapping_df], ignore_index=True)

    New_train_adata = sc.concat([New_train_adata, train_stimulated, train_stimulated[match_idx[pert + '-' + 'control']]], join='outer')

New_train_df.reset_index(drop=True, inplace=True)
New_train_df.to_csv("./New_train_df.csv")

#%%
New_train_adata.obs.index = [str(i) for i in range(New_train_adata.shape[0])]
New_train_df = pd.read_csv("./New_train_df.csv",index_col=0)
assert New_train_adata.shape[0] == New_train_df.shape[0], "The number of rows in New_train_df must match the number of cells in the New_train_adata."
New_train_adata.obs = New_train_df
New_train_adata.obs.index = [str(i) for i in range(New_train_adata.shape[0])]
New_train_adata.write_h5ad('./New_train_adata.h5ad')
#%%
New_test_df = pd.DataFrame()
Test_control_df = Train_df[Train_df['perturbation'] == 'control'].copy()
Test_control_df.reset_index(drop=True, inplace=True)

test_resting = train_adata[train_adata.obs['perturbation'] == 'control']
New_test_df = pd.concat([New_test_df, Test_control_df], ignore_index=True)
New_test_adata = test_resting

for pert in np.unique(Test_df['perturbation']): 
    if pert == 'control':
        continue
    Test_stimulation_df = Test_df[Test_df['perturbation'] == pert].copy()

    # 重新标号索引，从0开始
    Test_stimulation_df.reset_index(drop=True, inplace=True)

    test_stimulated = test_adata[test_adata.obs['perturbation'] == pert]

    Test_control_mapping_df = Test_control_df.copy()
    Test_control_mapping_df['perturbation'] = "control_mapping_to_" + pert
    # 合并这三个DataFrame
    New_test_df = pd.concat([New_test_df, Test_stimulation_df, Test_control_mapping_df], ignore_index=True)

    New_test_adata = sc.concat([New_test_adata, test_stimulated, test_resting], join='outer')

New_test_df.reset_index(drop=True, inplace=True)
New_test_df.to_csv("./New_test_df.csv")

#%%
New_test_adata.obs.index = [str(i) for i in range(New_test_adata.shape[0])]
New_test_df = pd.read_csv("./New_test_df.csv",index_col=0)
assert New_test_adata.shape[0] == New_test_df.shape[0], "The number of rows in New_test_df must match the number of cells in the New_test_adata."
New_test_adata.obs = New_test_df
New_test_adata.write_h5ad('./New_test_adata.h5ad')

#%%
from multiprocessing import Pool
import networkx as nx
import os
from tqdm import tqdm

class GeneSimNetwork():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """
    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """

        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        #self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

def make_GO(data_path, pert_list, data_name, num_workers=25, save=False):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    # fname = './data/go_essential_' + data_name + '.csv'
    fname = '/scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                      total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})
    
    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list

def get_similarity_network(data_path, data_name, pert_list=None, k=20):

    df_jaccard = make_GO(data_path, pert_list, data_name)

    df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,['importance'])).reset_index(drop = True)

    return df_out

def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list

#%%
from pathlib import Path
import pickle

gene2go_path = Path('/scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/gene2go_all.pkl')
with open(gene2go_path, "rb") as f:
    data = pickle.load(f)

# Read or Create GO
data_name='brauersanjana2021_k562_1'
edge_list = get_similarity_network(data_path=gene2go_path.parent,data_name=data_name,pert_list=all_peat_list)

sim_network = GeneSimNetwork(edge_list, all_peat_list, node_map = pert_to_index)
G_go = sim_network.edge_index
G_go_weight = sim_network.edge_weight

np.save(data_name + '_G_go.npy', G_go.cpu().numpy())
np.save(data_name + '_G_go_weight.npy', G_go_weight.cpu().numpy())

#%%
print(G_go.shape, G_go_weight.shape)

#%%
from epiagent.model import EpiAgent_PT

# Free up GPU memory by deleting the first model and clearing cache
# The first pretrained_model (EpiAgent) is no longer needed after generating embeddings
del pretrained_model
torch.cuda.empty_cache()
import gc
gc.collect()

# Initialize the EpiAgent model with appropriate configurations
pretrained_model = EpiAgent_PT(
    vocab_size=1355449,
    num_layers=18,
    embedding_dim=512,
    num_attention_heads=8,
    max_rank_embeddings=8192,
    use_flash_attn=True,
    pos_weight_for_RLM=torch.tensor(1.),
    pos_weight_for_CCA=torch.tensor(1.),
    GO=G_go,
    GO_weight=G_go_weight,
    all_pert_list=all_peat_list
)

# Specify the path to the pre-trained model
# model_path = '/home/chenxiaoyang/program/scCASdata/model/pretrained_EpiAgent.pth'
model_path = '/scratch/wkim/project-2-team-1/EpiAgent/model/pretrained_EpiAgent.pth'

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained weights into the model
pretrained_model.load_state_dict(torch.load(model_path),strict=False)

# Move the model to the specified device
pretrained_model.to(device)

#%%
# MEMORY OPTIMIZATION: Enable gradient checkpointing to reduce memory usage
# This trades compute for memory by not storing intermediate activations
# if hasattr(pretrained_model.EpiAgent_transformer, 'gradient_checkpointing_enable'):
#     pretrained_model.EpiAgent_transformer.gradient_checkpointing_enable()
#     print("Gradient checkpointing enabled on transformer")
# else:
#     # Try to enable gradient checkpointing on individual layers
#     try:
#         for layer in pretrained_model.EpiAgent_transformer.layer:
#             if hasattr(layer, 'gradient_checkpointing'):
#                 layer.gradient_checkpointing = True
#         print("Gradient checkpointing enabled on transformer layers")
#     except:
#         print("Could not enable gradient checkpointing (may not be supported)")

pretrained_model.signal_decoder = nn.Linear(512, 50000)
pretrained_model.criterion_SR = nn.MSELoss()
original_vocab_size = pretrained_model.vocab_size
pretrained_model.vocab_size = original_vocab_size + len(all_peat_list)

original_weights = pretrained_model.cCRE_embedding.weight.data
new_embedding = nn.Embedding(pretrained_model.vocab_size, 512)
new_embedding.weight.data[:original_vocab_size] = original_weights[:original_vocab_size]
pretrained_model.cCRE_embedding = new_embedding

# Ensure the CCA loss uses a positive weight of 1
pretrained_model.criterion_CCA.pos_weight = torch.tensor(1.)

#%%
# MEMORY OPTIMIZATION: Freeze early transformer layers to reduce memory usage
# This keeps gradients from being computed for frozen layers, saving memory
# Uncomment and adjust the number of layers to freeze based on your GPU memory
# Freezing more layers = less memory but potentially less fine-tuning flexibility

# Option 1: Freeze first 12 out of 18 layers (keeps last 6 trainable)
# num_layers_to_freeze = 12
# for i in range(num_layers_to_freeze):
#     if hasattr(pretrained_model.EpiAgent_transformer, 'layer') and i < len(pretrained_model.EpiAgent_transformer.layer):
#         for param in pretrained_model.EpiAgent_transformer.layer[i].parameters():
#             param.requires_grad = False
#     print(f"Froze transformer layer {i}")

# Option 2: Freeze embeddings (saves significant memory)
for param in pretrained_model.cCRE_embedding.parameters():
    param.requires_grad = False
for param in pretrained_model.rank_embedding.parameters():
    param.requires_grad = False
print("Froze embedding layers")

print("Note: To reduce memory further, uncomment the freezing code above")
#%%
from epiagent.dataset import TrainCellDatasetForPT, collate_fn_for_PT_train

# Create the training dataset
train_dataset = TrainCellDatasetForPT(
    adata=New_train_adata,
    pert_to_index=pert_to_index)

# Create the training DataLoader
# MEMORY OPTIMIZATION: Set num_workers=0 to reduce memory overhead from data loading
train_batch_size = 1
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=train_batch_size, 
    shuffle=True, 
    num_workers=0,  # Reduced from 2 to 0 to save GPU memory
    collate_fn=collate_fn_for_PT_train,
    pin_memory=False)  # Disable pin_memory to save memory

#%%
from epiagent.dataset import TestCellDatasetForPT, collate_fn_for_PT_test

# Create the inference dataset
test_dataset = TestCellDatasetForPT(adata=New_test_adata)

# Create the inference DataLoader
# MEMORY OPTIMIZATION: Set num_workers=0 to reduce memory overhead
inference_batch_size = 1
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=inference_batch_size, 
    shuffle=False, 
    num_workers=0,  # Reduced from 2 to 0 to save GPU memory
    collate_fn=collate_fn_for_PT_test,
    pin_memory=False)  # Disable pin_memory to save memory

cell_label = New_test_adata.obs['perturbation'].tolist()

#%%
from epiagent.train import fine_tune_epiagent_for_UFE

# MEMORY OPTIMIZATION: Configure PyTorch for memory efficiency
# Set memory allocation strategy to reduce fragmentation
if torch.cuda.is_available():
    # Use memory pool to reduce fragmentation
    torch.cuda.empty_cache()
    # Set memory fraction if needed (uncomment and adjust if you have multiple processes)
    # torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Enable memory-efficient attention if available (PyTorch 2.0+)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("Flash attention enabled via PyTorch")
    except:
        pass

# Clear GPU cache one more time before fine-tuning to ensure maximum free memory
torch.cuda.empty_cache()
import gc
gc.collect()

# Fine-tune the model
fine_tuned_model = fine_tune_epiagent_for_UFE(
    model=pretrained_model,
    train_dataloader=train_dataloader,
    num_steps=500000, 
    # num_steps=500, 
    save_dir='./model/fine_tune/PT/demo_dataset/',
    device=device,
    learning_rate=5e-5,
    save_steps=20000,
    log_steps=500,
    warmup_steps=10000,
    is_logging=True
)

#%%
import torch
from epiagent.model import EpiAgent_PT
import scanpy as sc
from epiagent.dataset import TestCellDatasetForPT, collate_fn_for_PT_test
import numpy as np
import pandas as pd
import torch.nn as nn

# Load supporting data
input_path = '/scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/All_data_K562_1.csv'
df = pd.read_csv(input_path,usecols=['cell_indices','perturbation','cell_line','ncounts','ngenes'])
df = df[~df['perturbation'].isna()]
all_peat_list = list(np.unique(df['perturbation']))
all_peat_list.remove('control')
pert_to_index = {element: index for index, element in enumerate(all_peat_list)}

G_go = np.load("./brauersanjana2021_k562_1_G_go.npy")
G_go_weight = np.load("./brauersanjana2021_k562_1_G_go_weight.npy")

# Load the Fine-tuned model
# model_path = "/home/likeyi/program/EpiAgent/20250512_tutorail/model/fine_tune/PT/demo_dataset/checkpoint_step_480000.pth"
model_path = "/scratch/wkim/project-2-team-1/EpiAgent/demo/model/fine_tune/PT/demo_dataset/checkpoint_step_480000.pth"
model = EpiAgent_PT(
    vocab_size=1355449 + len(all_peat_list),
    num_layers=18,
    embedding_dim=512,
    num_attention_heads=8,
    max_rank_embeddings=8192,
    use_flash_attn=True,
    pos_weight_for_RLM=torch.tensor(1.),
    pos_weight_for_CCA=torch.tensor(1.),
    GO=G_go,
    GO_weight=G_go_weight,
    all_pert_list=all_peat_list
)
model.signal_decoder = nn.Linear(512, 50000)
model.criterion_SR = nn.MSELoss()

model.load_state_dict(torch.load(model_path, map_location='cpu'))

# Load the processed Test Anndata
adata = sc.read_h5ad("./New_test_adata.h5ad")

# Create Dataset and DataLoader
test_dataset = TestCellDatasetForPT(adata=adata,pert_to_index=pert_to_index)

inference_batch_size = 2
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=inference_batch_size, 
    shuffle=False, 
    num_workers=2, 
    collate_fn=collate_fn_for_PT_test)

#%%
from torch.cuda.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

cell_embeddings = []
predicted_signals = []
for _, batch in enumerate(test_dataloader):
    torch.cuda.empty_cache()
    input_ids = batch.to(device)
    with autocast():
        output_embeddings = model(input_ids)['transformer_outputs'][:,0,:]
        predicted_signal = model.signal_decoder(output_embeddings).cpu().detach().numpy()
        output_embeddings = output_embeddings.cpu().detach().numpy()

    for output_embedding in output_embeddings:
        cell_embeddings.append(output_embedding)
    for signal in predicted_signal:
        predicted_signals.append(signal)
        
cell_embeddings = np.array(cell_embeddings)
predicted_signals = np.array(predicted_signals)

adata.layers['predicted_signals'] = predicted_signals
adata.obsm['cell_embeddings'] = cell_embeddings

#%%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_distances
import ot

def cal_DE_peaks(adata, pert):
    # MEMORY OPTIMIZATION: Work directly on adata and clean up intermediate results
    # instead of creating expensive full copies
    
    # FIX: Convert data to float32 to avoid numba float16 error
    # Numba (used by scanpy's rank_genes_groups) doesn't support float16
    from scipy.sparse import issparse
    # Check and convert float16 to float32
    x_dtype_str = str(adata.X.dtype)
    if 'float16' in x_dtype_str or adata.X.dtype == np.float16:
        adata.X = adata.X.astype(np.float32)
    
    # Compute differential peaks between Stimulated and Resting conditions
    sc.tl.rank_genes_groups(adata, groupby='perturbation', groups=[pert], reference='control', method='wilcoxon')
    
    # Extract results immediately before they get overwritten
    differential_peaks_stim_vs_rest = pd.DataFrame({
        'names': adata.uns['rank_genes_groups']['names'][pert],
        'pvals_adj': adata.uns['rank_genes_groups']['pvals_adj'][pert],
        'logfoldchanges': adata.uns['rank_genes_groups']['logfoldchanges'][pert]
    })
    
    # Sort the adjusted p-values
    differential_peaks_stim_vs_rest = differential_peaks_stim_vs_rest.sort_values(by='pvals_adj')
    
    # Clean up the first result from uns to free memory
    # Note: rank_genes_groups will overwrite this anyway, but cleaning explicitly helps
    if 'rank_genes_groups' in adata.uns:
        # Store only what we need, then let it be overwritten
        pass

    # Detect DEGs (Differentially Expressed Genes) using predicted_signals layer
    # FIX: Convert predicted_signals layer to float32 to avoid numba float16 error
    if 'predicted_signals' in adata.layers:
        pred_sig = adata.layers['predicted_signals']
        pred_dtype_str = str(pred_sig.dtype)
        if 'float16' in pred_dtype_str or pred_sig.dtype == np.float16:
            adata.layers['predicted_signals'] = pred_sig.astype(np.float32)
    
    # This will overwrite the previous rank_genes_groups results
    sc.tl.rank_genes_groups(adata, groupby='perturbation', groups=["control_mapping_to_" + pert], reference='control', method='wilcoxon', layer='predicted_signals')

    # Extract results immediately
    differential_genes_rest_map_to_stim_vs_rest = pd.DataFrame({
        'names': adata.uns['rank_genes_groups']['names']["control_mapping_to_" + pert],
        'pvals_adj': adata.uns['rank_genes_groups']['pvals_adj']["control_mapping_to_" + pert],
        'logfoldchanges': adata.uns['rank_genes_groups']['logfoldchanges']["control_mapping_to_" + pert]
    })
    
    # Sort the adjusted p-values
    differential_genes_rest_map_to_stim_vs_rest = differential_genes_rest_map_to_stim_vs_rest.sort_values(by='pvals_adj')
    
    # Clean up the uns dictionary to free memory (optional but helpful)
    # We can delete the rank_genes_groups if it's no longer needed
    # Note: This might affect other code that expects it, so we'll be conservative
    # and only clean if we're sure it's safe

    return differential_peaks_stim_vs_rest, differential_genes_rest_map_to_stim_vs_rest

def cal_DE_peak_Pearson_and_Jaccard(adata,pert,differential_peaks_stim_vs_rest,n=1000):
    New_test_adata = adata.copy()
    
    # Select the top n differential peaks
    top_n_peaks = differential_peaks_stim_vs_rest.head(n)['names']
    
    # Create a new AnnData object retaining only the top n differential peaks
    top_n_adata = New_test_adata[:, New_test_adata.var.index.isin(top_n_peaks)].copy()
    
    # Extract data with condition ‘Stimulated’ from the raw dataset
    stimulated_raw = top_n_adata[top_n_adata.obs['perturbation'] == pert]
    stimulated_raw_matrix = stimulated_raw.X.toarray()
    stimulated_raw_mean = np.mean(stimulated_raw_matrix, axis=0)
    
    # Extract data with condition ‘Resting_mapping_to_stimulated’ from predicted_signals
    predicted_matrix = top_n_adata.layers['predicted_signals']#.toarray()
    predicted_obs = top_n_adata.obs['perturbation'] == "control_mapping_to_" + pert
    predicted_signals_matrix = predicted_matrix[predicted_obs, :]
    predicted_signals_mean = np.mean(predicted_signals_matrix, axis=0)
    
    pearson_corr = np.corrcoef(stimulated_raw_mean, predicted_signals_mean)[0, 1]
    return pearson_corr

#%%
import gc

differential_peaks_stim_vs_rests = []
differential_genes_rest_map_to_stim_vs_rests = []
indexs = []
for pert in tqdm(np.unique(adata.obs['perturbation'])):
    # try:
    if 'control' in  pert:
        continue
    if sum(adata.obs['perturbation'] == pert) <= 1:
        continue
    print(pert)

    # differential_peaks_stim_vs_rests.append(cal_DE_peaks(adata, pert)[0])
    # differential_genes_rest_map_to_stim_vs_rests.append(cal_DE_peaks(adata, pert)[1])
    # Call cal_DE_peaks once and store both results to avoid duplicate computation
    de_peaks, de_genes = cal_DE_peaks(adata, pert)
    differential_peaks_stim_vs_rests.append(de_peaks)
    differential_genes_rest_map_to_stim_vs_rests.append(de_genes)
    
    # MEMORY OPTIMIZATION: Clean up intermediate results stored in adata.uns
    # This prevents memory accumulation across iterations
    if 'rank_genes_groups' in adata.uns:
        # Clear the rank_genes_groups to free memory (will be regenerated next iteration)
        del adata.uns['rank_genes_groups']
    
    # Force garbage collection periodically to free memory
    gc.collect()

    # except:
    #     print(f'Error for {pert}')
    #     # Clean up even on error
    #     if 'rank_genes_groups' in adata.uns:
    #         del adata.uns['rank_genes_groups']
    #     gc.collect()

#%%
for n in [100,200,500,1000,2000]:
    print(f'n = {n}')
    pearson_corrs = []
    jaccard_sims = []
    idx = 0
    for pert in np.unique(adata.obs['perturbation']):
        if 'control' in  pert:
            continue
        if sum(adata.obs['perturbation'] == pert) <= 1:
            continue
        pearson_corr = cal_DE_peak_Pearson_and_Jaccard(adata, pert, differential_peaks_stim_vs_rests[idx], n=n)
        pearson_corrs.append(pearson_corr)

    pearson_corrs = np.array(pearson_corrs)
    print(f'Mean value of Pearson correlation: {np.mean(pearson_corrs):.4f}')
    print(f'Std value of Pearson correlation: {np.std(pearson_corrs):.4f}')

#%%
def cal_regulated_rate(adata,pert,differential_peaks_stim_vs_rest,m=1000):
    New_test_adata = adata.copy()
    
    # Extract top m upregulated and top m downregulated peaks in Stimulated based on smallest p-values
    upregulated_peaks = differential_peaks_stim_vs_rest[differential_peaks_stim_vs_rest['logfoldchanges'] > 0].sort_values(by='pvals_adj').head(m)['names']
    downregulated_peaks = differential_peaks_stim_vs_rest[differential_peaks_stim_vs_rest['logfoldchanges'] < 0].sort_values(by='pvals_adj').head(m)['names']
    
    # Extract the matrix of the selected top m upregulated and m downregulated peaks from New_test_adata
    upregulated_data = New_test_adata[:, New_test_adata.var.index.isin(upregulated_peaks)].copy()
    downregulated_data = New_test_adata[:, New_test_adata.var.index.isin(downregulated_peaks)].copy()
    
    # Extract the subset of data where the condition is either 'Resting' or 'Resting_mapping_to_stimulated'
    resting_indices = np.where(New_test_adata.obs['perturbation'] == 'control')[0]
    resting_mapping_to_stimulated_indices = np.where(New_test_adata.obs['perturbation'] == "control_mapping_to_" + pert)[0]
    
    # Calculate the mean signal change for upregulated peaks
    upregulated_resting_mean = np.mean(upregulated_data[resting_indices, :].X.toarray(), axis=0)
    upregulated_mapping_to_stimulated_mean = np.mean(upregulated_data[resting_mapping_to_stimulated_indices, :].layers['predicted_signals'], axis=0)
    
    # Calculate the mean signal change for downregulated peaks
    downregulated_resting_mean = np.mean(downregulated_data[resting_indices, :].X.toarray(), axis=0)
    downregulated_mapping_to_stimulated_mean = np.mean(downregulated_data[resting_mapping_to_stimulated_indices, :].layers['predicted_signals'], axis=0)
    
    # Check whether the mean signal change indicates an increase or decrease
    upregulated_change = upregulated_mapping_to_stimulated_mean - upregulated_resting_mean
    downregulated_change = downregulated_mapping_to_stimulated_mean - downregulated_resting_mean
    
    # Compute increase ratio for upregulated peaks and decrease ratio for downregulated peaks
    upregulated_increase_ratio = np.sum(upregulated_change > 0) / m
    downregulated_decrease_ratio = np.sum(downregulated_change < 0) / m
    
    return upregulated_increase_ratio, downregulated_decrease_ratio

#%%
for n in [100,200,500,1000,2000]:
    print(f'n = {n}')
    upregulated_increase_ratios = []
    downregulated_decrease_ratios = []

    for pert in np.unique(adata.obs['perturbation']):
        if 'control' in  pert:
            continue
        if sum(adata.obs['perturbation'] == pert) <= 1:
            continue
        upregulated_increase_ratio, downregulated_decrease_ratio = cal_regulated_rate(adata,pert,differential_peaks_stim_vs_rests[idx],m=n)
        upregulated_increase_ratios.append(upregulated_increase_ratio)
        downregulated_decrease_ratios.append(downregulated_decrease_ratio)

    upregulated_increase_ratios = np.array(upregulated_increase_ratios)
    downregulated_decrease_ratios = np.array(downregulated_decrease_ratios)
    print(f'Mean value of Upregulated increase ratios: {np.mean(upregulated_increase_ratios):.4f}')
    print(f'Std value of Upregulated increase ratios: {np.std(upregulated_increase_ratios):.4f}')
    print(f'Mean value of Downregulated decrease ratios: {np.mean(downregulated_decrease_ratios):.4f}')
    print(f'Std value of Downregulated decrease ratios: {np.std(downregulated_decrease_ratios):.4f}')
# %%

import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
import torch
import umap
import matplotlib.pyplot as plt
import hdbscan
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ---------------------------
# 1. Configuration
# ---------------------------

# CSV_PATH = "/scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/All_data_K562_1.csv"
# GENES_OF_INTEREST = ["CHD5", "KDM6A", "DNMT3A", "HDAC9", "PBRM1", "MBD1", "PRDM9", "ING1", "EZH2", "TET2", "ARID1A", "SETD2", "HIST1H3B", "PHF6", "ATRX", "H3F3B", "SMARCB1", "SMARCA4", "CHD8", "H3F3A", "CHD4"]

CSV_PATH = "/scratch/wkim/project-2-team-1/EpiAgent/external_data/GSE168851_crispr_perturb/All_data_SpearATAC_K562_LargeScreen.csv"
# GENES_OF_INTEREST = ['UNK', 'sgARID2', 'sgARID3A', 'sgATF1', 'sgATF3', 'sgBCLAF1', 'sgBRF2', 'sgCAD',
#  'sgCDC5L', 'sgCEBPB', 'sgCEBPZ', 'sgCTCF', 'sgCUX1', 'sgELF1', 'sgFOSL1',
#  'sgGABPA', 'sgGATA1', 'sgGTF2B', 'sgHINFP', 'sgHSPA5', 'sgKLF1', 'sgKLF16',
#  'sgMAX', 'sgMYC', 'sgNFE2', 'sgNFYB', 'sgNRF1', 'sgPBX2', 'sgPOLR1D', 'sgREST',
#  'sgRPL9', 'sgSETDB1', 'sgTBP', 'sgTFDP1', 'sgTHAP1', 'sgTRIM28', 'sgYY1',
#  'sgZBTB11', 'sgZNF280A', 'sgZNF407', 'sgZZZ3', 'sgsgNT']
GENES_OF_INTEREST = ['sgGATA1', 'sgMAX', 'sgYY1']

PRETRAINED_MODEL_PATH = "/scratch/wkim/project-2-team-1/EpiAgent/model/pretrained_EpiAgent.pth"


# Max number of cells per group to include in UMAP (for plotting sanity)
MAX_CELLS_PER_GROUP = 500

# EpiAgent model hyperparameters (must match the checkpoint)
VOCAB_SIZE = 1355449
NUM_LAYERS = 18
EMBED_DIM = 512
NUM_HEADS = 8
MAX_SEQ_LEN = 8192

# Random seed for reproducibility
SEED = 42

# Output directory for plots
# OUTPUT_DIR = "./zeroshot_perturbation_effect_prediction_umaps_GSE168851"
# OUTPUT_DIR = "./zeroshot_perturbation_effect_prediction_PCA_on_controls_GSE168851"
OUTPUT_DIR = "./zeroshot_perturbation_effect_prediction_tSNE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_DIR = "./epiagent_embedding_cache_GSE168851"
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "cell_embeddings.npy")
PERTURBATION_CACHE_PATH = os.path.join(CACHE_DIR, "perturbations.npy")

# ---------------------------
# 2. Reproducibility
# ---------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------
# 3. Import EpiAgent + datasets from epiagent
# ---------------------------

from epiagent.model import EpiAgent
from epiagent.dataset import CellDataset, collate_fn


# ---------------------------
# 4. Load metadata and build cell_indices
# ---------------------------

def load_metadata_and_cell_indices(csv_path):
    """
    Load All_data_K562_1.csv and parse cell_indices and perturbation labels.
    """
    print(f"[INFO] Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, usecols=["cell_indices", "perturbation"])

    # Drop rows with missing perturbation labels
    df = df[~df["perturbation"].isna()].reset_index(drop=True)

    # map sgsgNT to "control"
    df.loc[df["perturbation"] == "sgsgNT", "perturbation"] = "control"
    # OPTIONAL: drop unknown assignments
    df = df[df["perturbation"] != "UNK"].reset_index(drop=True)

    # Parse JSON-encoded cell_indices into Python lists
    print("[INFO] Parsing cell_indices...")
    cell_indices_list = [json.loads(s) for s in tqdm(df["cell_indices"].tolist())]

    perturbations = df["perturbation"].values  # numpy array of labels

    print(f"[INFO] Loaded {len(df)} cells in total.")
    print(f"[INFO] Unique perturbations: {np.unique(perturbations)}")

    return cell_indices_list, perturbations


# ---------------------------
# 5. Build EpiAgent and compute embeddings
# ---------------------------

def build_epigagent_and_embed(cell_indices_list, device="cuda"):
    """
    Build base EpiAgent, load pretrained weights, and compute embeddings for all cells.
    Returns:
        cell_embeddings: np.ndarray of shape (N_cells, EMBED_DIM)
    """
    print("[INFO] Initializing EpiAgent...")
    model = EpiAgent(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBED_DIM,
        num_attention_heads=NUM_HEADS,
        max_rank_embeddings=MAX_SEQ_LEN,
        use_flash_attn=True,
        pos_weight_for_RLM=torch.tensor(1.0),
        pos_weight_for_CCA=torch.tensor(1.0),
    )

    print(f"[INFO] Loading pretrained weights from {PRETRAINED_MODEL_PATH}")
    state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Dataset & DataLoader
    print("[INFO] Building CellDataset and DataLoader...")
    dataset = CellDataset(cell_indices_list, MAX_SEQ_LEN, True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    all_embeddings = []

    print("[INFO] Computing embeddings with EpiAgent...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch.to(device)
            # ensure we run in half precision for flash_attn
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(input_ids)
                emb = outputs["transformer_outputs"][:, 0, :]  # (B, EMBED_DIM)
            # cast back to float32 for numpy / downstream use
            all_embeddings.append(emb.float().cpu().numpy())

    print(f"[INFO] concattenating all_embeddings shape...")
    cell_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[INFO] cell_embeddings shape: {cell_embeddings.shape}")
    return cell_embeddings

def get_or_compute_embeddings(cell_indices_list, perturbations, device="cuda"):
    """
    If cached embeddings exist, load and return them.
    Otherwise, compute embeddings with EpiAgent, save to disk, and return.

    We cache embeddings BEFORE HDBSCAN filtering. HDBSCAN is cheap
    compared to the forward pass, so we rerun it each time.
    """
    # If cache exists, load and sanity-check shapes
    if os.path.exists(EMBEDDING_CACHE_PATH) and os.path.exists(PERTURBATION_CACHE_PATH):
        print(f"[INFO] Loading cached embeddings from {EMBEDDING_CACHE_PATH}")
        cached_embeddings = np.load(EMBEDDING_CACHE_PATH)
        cached_perts = np.load(PERTURBATION_CACHE_PATH, allow_pickle=True)

        if cached_embeddings.shape[0] != len(perturbations):
            print("[WARN] Cached embeddings shape does not match current "
                  "number of cells; recomputing embeddings.")
        elif not np.array_equal(cached_perts, perturbations):
            print("[WARN] Cached perturbation labels do not match current "
                  "labels; recomputing embeddings.")
        else:
            print("[INFO] Loaded cached embeddings successfully.")
            return cached_embeddings, cached_perts

    # Otherwise compute from scratch
    print("[INFO] No valid cache found; computing embeddings with EpiAgent...")
    embeddings = build_epigagent_and_embed(cell_indices_list, device=device)

    # Save cache
    np.save(EMBEDDING_CACHE_PATH, embeddings)
    np.save(PERTURBATION_CACHE_PATH, np.asarray(perturbations, dtype=str))
    print(f"[INFO] Saved embeddings cache to {EMBEDDING_CACHE_PATH}")
    print(f"[INFO] Saved perturbation labels cache to {PERTURBATION_CACHE_PATH}")

    return embeddings, perturbations



# ---------------------------
# 6. Utility: subsampling indices
# ---------------------------

def subsample_indices(idx_array, max_n, seed=SEED):
    """
    Randomly subsample indices to at most max_n.
    """
    idx_array = np.array(idx_array)
    if len(idx_array) <= max_n:
        return idx_array
    rng = np.random.default_rng(seed)
    return rng.choice(idx_array, size=max_n, replace=False)

# ---------------------------
# 6b. Filter out "ribbon" cells via HDBSCAN on embeddings
# ---------------------------

def filter_major_cluster(embeddings, perturbations,
                         min_cluster_size=300,
                         min_samples=20,
                         mode="most_compact"):
    """
    Run HDBSCAN on the high-dimensional embeddings and keep only one
    non-noise cluster.

    mode = "largest": keep cluster with max size
    mode = "most_compact": keep cluster with smallest mean squared
                           distance to its centroid (better when you
                           expect a ball + ribbon structure).

    Returns:
        filtered_embeddings, filtered_perturbations
    """
    print("[INFO] Running HDBSCAN to identify major cluster...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=4,
    )
    labels = clusterer.fit_predict(embeddings)

    # HDBSCAN uses -1 for noise / unassigned points
    unique, counts = np.unique(labels, return_counts=True)
    print("[INFO] HDBSCAN cluster labels and counts (label: count):")
    for u, c in zip(unique, counts):
        print(f"    {u}: {c}")

    non_noise = unique[unique >= 0]
    if len(non_noise) == 0:
        print("[WARN] HDBSCAN found only noise (label = -1). "
              "Skipping filtering and returning original embeddings.")
        return embeddings, perturbations

    if mode == "largest":
        # Old behavior: keep largest cluster by size
        largest_label = non_noise[np.argmax(counts[unique >= 0])]
        chosen_label = largest_label
        print(f"[INFO] [mode=largest] Keeping cluster {chosen_label}.")
    else:
        # New behavior: keep most compact cluster
        print("[INFO] [mode=most_compact] Selecting cluster with smallest "
              "mean squared distance to its centroid.")
        best_label = None
        best_compactness = None
        for lab in non_noise:
            mask = (labels == lab)
            emb_lab = embeddings[mask]
            centroid = emb_lab.mean(axis=0)
            # mean squared distance to centroid
            mse = ((emb_lab - centroid) ** 2).sum(axis=1).mean()
            print(f"    cluster {lab}: size={mask.sum()}, compactness={mse:.4f}")
            if (best_compactness is None) or (mse < best_compactness):
                best_compactness = mse
                best_label = lab
        chosen_label = best_label
        print(f"[INFO] Chosen cluster {chosen_label} with compactness={best_compactness:.4f}")

    keep_mask = (labels == chosen_label)
    print(f"[INFO] Keeping {keep_mask.sum()} / {len(keep_mask)} cells "
          f"({keep_mask.sum() / len(keep_mask) * 100:.1f}%).")

    filtered_embeddings = embeddings[keep_mask]
    filtered_perturbations = perturbations[keep_mask]

    print("[INFO] Unique perturbations after filtering:",
          np.unique(filtered_perturbations))

    return filtered_embeddings, filtered_perturbations




# ---------------------------
# 7. Zero-shot perturbation direction test for one gene
# ---------------------------

def zero_shot_test_for_gene(gene, embeddings, perturbations, max_cells_per_group=500):
    """
    For a given gene, compute:
        v_gene = mean_emb(perturbed_gene) - mean_emb(control)
    Then apply v_gene to control embeddings to get pseudo-perturbed embeddings,
    and plot UMAP of:
        - control
        - real perturbed cells
        - pseudo-perturbed control cells
    """

    print(f"\n[INFO] === Testing zero-shot direction for gene: {gene} ===")

    # Indices
    ctrl_idx = np.where(perturbations == "control")[0]
    gene_idx = np.where(perturbations == gene)[0]

    if len(ctrl_idx) == 0:
        print("[WARN] No control cells found after filtering; skipping this gene.")
        return

    if len(gene_idx) == 0:
        print(f"[WARN] No cells found for gene {gene}, skipping.")
        return

    print(f"[INFO] #control cells: {len(ctrl_idx)}, # {gene}-perturbed cells: {len(gene_idx)}")

    # Compute mean embeddings
    mu_ctrl = embeddings[ctrl_idx].mean(axis=0)
    mu_gene = embeddings[gene_idx].mean(axis=0)
    v_gene = mu_gene - mu_ctrl

    # Subsample for plotting
    ctrl_idx_sub = subsample_indices(ctrl_idx, max_cells_per_group)
    gene_idx_sub = subsample_indices(gene_idx, max_cells_per_group)

    # Embeddings for plotting
    emb_ctrl = embeddings[ctrl_idx_sub]                # (Nc, d)
    emb_gene_real = embeddings[gene_idx_sub]           # (Ng, d)
    emb_gene_pseudo = emb_ctrl + v_gene[np.newaxis, :] # (Nc, d)

    # # -----------------------------
    # # Balanced subsampling for plot
    # # -----------------------------
    # # We want similar point counts in each class to make the t-SNE readable.
    # # Choose n_plot as the min of:
    # #   - max_cells_per_group (e.g. 500)
    # #   - # available controls
    # #   - # available perturbed cells
    # n_plot = min(max_cells_per_group, len(ctrl_idx), len(gene_idx))

    # if n_plot < 10:
    #     print(f"[WARN] Too few cells for {gene} after filtering (n_plot={n_plot}); skipping.")
    #     return

    # # Subsample controls and real perturbed cells to n_plot
    # ctrl_idx_sub = subsample_indices(ctrl_idx, n_plot)
    # gene_idx_sub = subsample_indices(gene_idx, n_plot)

    # # Embeddings for plotting
    # emb_ctrl = embeddings[ctrl_idx_sub]              # (n_plot, d)
    # emb_gene_real = embeddings[gene_idx_sub]         # (n_plot, d)
    # emb_gene_pseudo = emb_ctrl + v_gene[np.newaxis, :]  # (n_plot, d)

    # Prepare data for t-SNE / UMAP / PCA
    X = np.concatenate([emb_ctrl, emb_gene_real, emb_gene_pseudo], axis=0)
    labels = (
        ["control"] * len(emb_ctrl)
        + [f"{gene}_real"] * len(emb_gene_real)
        + [f"{gene}_pseudo"] * len(emb_gene_pseudo)
    )

    # Prepare data for UMAP
    X = np.concatenate([emb_ctrl, emb_gene_real, emb_gene_pseudo], axis=0)
    labels = (
        ["control"] * len(emb_ctrl)
        + [f"{gene}_real"] * len(emb_gene_real)
        + [f"{gene}_pseudo"] * len(emb_gene_pseudo)
    )

    print("Label counts for this gene:", Counter(labels))

    ##########################################################################
    # UMAP!! #################################################################
    # print("[INFO] Running UMAP...")
    # reducer = umap.UMAP(
    #     n_components=2,
    #     random_state=SEED,
    #     n_neighbors=30,
    #     min_dist=0.3,
    #     metric="cosine",
    # )
    # X_umap = reducer.fit_transform(X)

    # # Plot
    # print("[INFO] Plotting UMAP...")
    # fig, ax = plt.subplots(figsize=(8, 8))

    # label_to_color = {
    #     "control": "tab:blue",
    #     f"{gene}_real": "tab:red",
    #     f"{gene}_pseudo": "tab:green",
    # }

    # for lab in np.unique(labels):
    #     mask = np.array(labels) == lab
    #     ax.scatter(
    #         X_umap[mask, 0],
    #         X_umap[mask, 1],
    #         s=10,
    #         alpha=0.7,
    #         label=lab,
    #         c=label_to_color.get(lab, "gray"),
    #     )

    # ax.set_title(f"Zero-shot perturbation direction: {gene}")
    # ax.set_xlabel("UMAP-1")
    # ax.set_ylabel("UMAP-2")
    # ax.legend(markerscale=2, fontsize=8, loc="best")

    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # out_path = os.path.join(OUTPUT_DIR, f"umap_zero_shot_{gene}.png")
    # plt.tight_layout()
    # plt.savefig(out_path, dpi=200)
    # plt.close(fig)

    # print(f"[INFO] Saved UMAP plot for {gene} to: {out_path}")


    ##########################################################################
    # tSNE
        # -----------------------------
    # t-SNE on concatenated embeddings
    # -----------------------------
    # ---- PCA before t-SNE ----
    print("[INFO] Running PCA for t-SNE preprocessing...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    # ---- t-SNE ----
    print("[INFO] Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=SEED
    )
    X_tsne = tsne.fit_transform(X_pca)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 8))
    for lab in np.unique(labels):
        mask = np.array(labels) == lab
        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=10,
            alpha=0.7,
            label=lab
        )
    ax.set_title(f"Zero-shot perturbation direction (t-SNE): {gene}")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"tSNE_zero_shot_{gene}.png"))



    ##########################################################################

    # PCA!! #################################################################
    # emb_ctrl = embeddings[ctrl_idx_sub]                # (Nc, d)
    # emb_gene_real = embeddings[gene_idx_sub]           # (Ng, d)
    # emb_gene_pseudo = emb_ctrl + v_gene[np.newaxis, :] # (Nc, d)

    # # Stack all embeddings (control, real, pseudo)
    # X = np.concatenate([emb_ctrl, emb_gene_real, emb_gene_pseudo], axis=0)
    # labels = (
    #     ["control"] * len(emb_ctrl)
    #     + [f"{gene}_real"] * len(emb_gene_real)
    #     + [f"{gene}_pseudo"] * len(emb_gene_pseudo)
    # )
    # labels = np.array(labels)

    # # ---------------------------
    # # PCA on EpiAgent embeddings
    # # ---------------------------
    # print("[INFO] Running PCA...")
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)   # shape (N, 2)

    # # ---------------------------
    # # Projection onto v_gene
    # # ---------------------------
    # # Normalize perturbation direction to unit length
    # v_norm = np.linalg.norm(v_gene)
    # if v_norm < 1e-8:
    #     print("[WARN] v_gene has near-zero norm; skipping projection plot.")
    #     proj_all = np.zeros(X.shape[0])
    # else:
    #     v_unit = v_gene / v_norm
    #     # projection = dot(X, v_unit)
    #     proj_all = X.dot(v_unit)   # shape (N,)

    # # ---------------------------
    # # Plot: PCA (left) and PCA+projection (right)
    # # ---------------------------
    # print("[INFO] Plotting PCA and PCA+projection...")
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # ax_pca, ax_proj = axes

    # label_to_color = {
    #     "control": "tab:blue",
    #     f"{gene}_real": "tab:red",
    #     f"{gene}_pseudo": "tab:green",
    # }

    # unique_labels = np.unique(labels)

    # # Left: PC1 vs PC2
    # for lab in unique_labels:
    #     mask = (labels == lab)
    #     ax_pca.scatter(
    #         X_pca[mask, 0],
    #         X_pca[mask, 1],
    #         s=10,
    #         alpha=0.7,
    #         label=lab,
    #         c=label_to_color.get(lab, "gray"),
    #     )
    # ax_pca.set_title(f"{gene}: PCA (PC1 vs PC2)")
    # ax_pca.set_xlabel("PC1")
    # ax_pca.set_ylabel("PC2")

    # # Right: PC1 vs projection onto v_gene
    # for lab in unique_labels:
    #     mask = (labels == lab)
    #     ax_proj.scatter(
    #         X_pca[mask, 0],       # same x-axis (PC1)
    #         proj_all[mask],       # y-axis = projection on v_gene
    #         s=10,
    #         alpha=0.7,
    #         label=lab,
    #         c=label_to_color.get(lab, "gray"),
    #     )
    # ax_proj.set_title(f"{gene}: PC1 vs ⟨emb, v_gene⟩")
    # ax_proj.set_xlabel("PC1")
    # ax_proj.set_ylabel("Projection on v_gene")

    # # One shared legend
    # handles, legend_labels = ax_pca.get_legend_handles_labels()
    # fig.legend(handles, legend_labels, loc="upper center", ncol=3)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # out_path = os.path.join(OUTPUT_DIR, f"pca_zero_shot_{gene}.png")
    # plt.savefig(out_path, dpi=200)
    # plt.close(fig)

    # print(f"[INFO] Saved PCA figure for {gene} to: {out_path}")

    # print(f"emb_ctrl shape: {emb_ctrl.shape}")
    # print(f"emb_gene_real shape: {emb_gene_real.shape}")
    # print(f"emb_gene_pseudo shape: {emb_gene_pseudo.shape}")
    # print(f"X shape: {X.shape}")
    # print("Unique labels and counts:",
    #     {lab: np.sum(labels == lab) for lab in np.unique(labels)})
    # print("Explained variance ratio (first 5 PCs):", pca.explained_variance_ratio_[:5])

    # Embeddings for plotting / PCA on controls!!
    # emb_ctrl = embeddings[ctrl_idx_sub]                # (Nc, d)
    # emb_gene_real = embeddings[gene_idx_sub]           # (Ng, d)
    # emb_gene_pseudo = emb_ctrl + v_gene[np.newaxis, :] # (Nc, d)

    # # Stack all embeddings (control, real, pseudo) for later use
    # X = np.concatenate([emb_ctrl, emb_gene_real, emb_gene_pseudo], axis=0)
    # labels = (
    #     ["control"] * len(emb_ctrl)
    #     + [f"{gene}_real"] * len(emb_gene_real)
    #     + [f"{gene}_pseudo"] * len(emb_gene_pseudo)
    # )
    # labels = np.array(labels)


    ##########################################################################
    # # ---------- NEW: scale + PCA fitted on control only ----------
    # print("[INFO] Running PCA (fit on control only)...")

    # # 1) Fit scaler on control embeddings
    # scaler = StandardScaler()
    # emb_ctrl_scaled = scaler.fit_transform(emb_ctrl)
    # emb_gene_real_scaled = scaler.transform(emb_gene_real)
    # emb_gene_pseudo_scaled = scaler.transform(emb_gene_pseudo)

    # # 2) Fit PCA on *scaled control* only
    # pca = PCA(n_components=2, random_state=SEED)   # <--- no whiten
    # pca.fit(emb_ctrl_scaled)

    # # 3) Transform all groups with that PCA
    # X_scaled = np.concatenate(
    #     [emb_ctrl_scaled, emb_gene_real_scaled, emb_gene_pseudo_scaled],
    #     axis=0,
    # )
    # X_pca = pca.transform(X_scaled)   # shape (N, 2)

    # print("Explained variance ratio (first 2 PCs):", pca.explained_variance_ratio_)

    # # ---------- Projection onto v_gene (in original space) ----------
    # v_norm = np.linalg.norm(v_gene)
    # if v_norm < 1e-8:
    #     print("[WARN] v_gene has near-zero norm; skipping projection plot.")
    #     proj_all = np.zeros(X.shape[0])
    # else:
    #     v_unit = v_gene / v_norm
    #     proj_all = X.dot(v_unit)   # use unscaled embeddings for this

    # # ---------- Plot: PCA (left) and PC1 vs projection (right) ----------
    # print("[INFO] Plotting PCA and PCA+projection...")
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # ax_pca, ax_proj = axes

    # label_to_color = {
    #     "control": "tab:blue",
    #     f"{gene}_real": "tab:red",
    #     f"{gene}_pseudo": "tab:green",
    # }
    # unique_labels = np.unique(labels)

    # # Left: PC1 vs PC2
    # for lab in unique_labels:
    #     m = (labels == lab)
    #     ax_pca.scatter(
    #         X_pca[m, 0],
    #         X_pca[m, 1],
    #         s=10,
    #         alpha=0.7,
    #         label=lab,
    #         c=label_to_color.get(lab, "gray"),
    #     )
    # ax_pca.set_title(f"{gene}: PCA (PC1 vs PC2)")
    # ax_pca.set_xlabel("PC1")
    # ax_pca.set_ylabel("PC2")

    # # Right: PC1 vs projection on v_gene
    # for lab in unique_labels:
    #     m = (labels == lab)
    #     ax_proj.scatter(
    #         X_pca[m, 0],
    #         proj_all[m],
    #         s=10,
    #         alpha=0.7,
    #         label=lab,
    #         c=label_to_color.get(lab, "gray"),
    #     )
    # ax_proj.set_title(f"{gene}: PC1 vs ⟨emb, v_gene⟩")
    # ax_proj.set_xlabel("PC1")
    # ax_proj.set_ylabel("Projection on v_gene")

    # handles, legend_labels = ax_pca.get_legend_handles_labels()
    # fig.legend(handles, legend_labels, loc="upper center", ncol=3)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # out_path = os.path.join(OUTPUT_DIR, f"pca_zero_shot_{gene}.png")
    # plt.savefig(out_path, dpi=200)
    # plt.close(fig)

    # print(f"[INFO] Saved PCA figure for {gene} to: {out_path}")
    # print(f"emb_ctrl shape: {emb_ctrl.shape}")
    # print(f"emb_gene_real shape: {emb_gene_real.shape}")
    # print(f"emb_gene_pseudo shape: {emb_gene_pseudo.shape}")
    # print(f"X shape: {X.shape}")
    # print("Unique labels and counts:",
    #     {lab: np.sum(labels == lab) for lab in np.unique(labels)})

    # fig, ax = plt.subplots(figsize=(6, 4))
    # for lab, color in [
    #     ("control", "tab:blue"),
    #     (f"{gene}_real", "tab:red"),
    #     (f"{gene}_pseudo", "tab:green"),
    # ]:
    #     m = (labels == lab)
    #     ax.hist(
    #         proj_all[m],
    #         bins=30,
    #         alpha=0.4,
    #         label=lab,
    #         color=label_to_color.get(lab, color),
    #         density=True,
    #     )
    # ax.set_title(f"{gene}: distribution of ⟨emb, v_gene⟩")
    # ax.set_xlabel("Projection on v_gene")
    # ax.set_ylabel("Density")
    # ax.legend()
    # hist_path = os.path.join(OUTPUT_DIR, f"proj_hist_{gene}.png")
    # plt.tight_layout()
    # plt.savefig(hist_path, dpi=200)
    # plt.close(fig)
    # print(f"[INFO] Saved projection histogram for {gene} to: {hist_path}")

# ---------------------------
# 8. Main
# ---------------------------

def main():
    # 1) Load metadata and cell_indices
    cell_indices_list, perturbations = load_metadata_and_cell_indices(CSV_PATH)

    # 2) Build EpiAgent and compute embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, perturbations = get_or_compute_embeddings(
        cell_indices_list,
        perturbations,
        device=device,
    )

    # 2b) Filter out "ribbon" cells by keeping the largest HDBSCAN cluster
    embeddings, perturbations = filter_major_cluster(
        embeddings,
        perturbations,
        min_cluster_size=300,   # you can tune these
        min_samples=20,
        mode="most_compact",
    )

    # 3) Run zero-shot test for each gene on the filtered set
    unique_perts = np.unique(perturbations)
    print(f"[INFO] Unique perturbations in filtered data: {unique_perts}")
    
    for gene in GENES_OF_INTEREST:
        if gene not in unique_perts:
            print(f"[WARN] Gene {gene} not present in perturbation labels; skipping.")
            continue
        print(f"[INFO] Scheduling zero-shot test for gene {gene}")
        zero_shot_test_for_gene(
            gene,
            embeddings,
            perturbations,
            max_cells_per_group=MAX_CELLS_PER_GROUP,
        )

if __name__ == "__main__":
    main()
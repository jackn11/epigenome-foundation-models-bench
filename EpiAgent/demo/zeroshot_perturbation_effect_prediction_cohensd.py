import os
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import hdbscan

# ============================================================
# 1. Configuration
# ============================================================

PRETRAINED_MODEL_PATH = (
    "/scratch/wkim/project-2-team-1/EpiAgent/model/pretrained_EpiAgent.pth"
)

CSV_PATH = "/scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/All_data_K562_1.csv"
GENES_OF_INTEREST = ["CHD5", "KDM6A", "DNMT3A", "HDAC9", "PBRM1", "MBD1", "PRDM9", "ING1", "EZH2", "TET2", "ARID1A", "SETD2", "HIST1H3B", "PHF6", "ATRX", "H3F3B", "SMARCB1", "SMARCA4", "CHD8", "H3F3A", "CHD4"]

# CSV_PATH = (
#     "/scratch/wkim/project-2-team-1/EpiAgent/external_data/"
#     "GSE168851_crispr_perturb/All_data_SpearATAC_K562_LargeScreen.csv"
# )
# GENES_OF_INTEREST = ["sgGATA1", "sgMAX", "sgYY1"]
# GENES_OF_INTEREST = ['UNK', 'sgARID2', 'sgARID3A', 'sgATF1', 'sgATF3', 'sgBCLAF1', 'sgBRF2', 'sgCAD',
#  'sgCDC5L', 'sgCEBPB', 'sgCEBPZ', 'sgCTCF', 'sgCUX1', 'sgELF1', 'sgFOSL1',
#  'sgGABPA', 'sgGATA1', 'sgGTF2B', 'sgHINFP', 'sgHSPA5', 'sgKLF1', 'sgKLF16',
#  'sgMAX', 'sgMYC', 'sgNFE2', 'sgNFYB', 'sgNRF1', 'sgPBX2', 'sgPOLR1D', 'sgREST',
#  'sgRPL9', 'sgSETDB1', 'sgTBP', 'sgTFDP1', 'sgTHAP1', 'sgTRIM28', 'sgYY1',
#  'sgZBTB11', 'sgZNF280A', 'sgZNF407', 'sgZZZ3', 'sgsgNT']

CONTROL_LABEL = "control"        # we already remapped sgsgNT -> control
UNK_LABEL = "UNK"                # will be dropped

MAX_SEQ_LEN = 8192
VOCAB_SIZE = 1355449
NUM_LAYERS = 18
EMBED_DIM = 512
NUM_HEADS = 8

K_NEIGHBORS = 20                # k for local neighborhood
MIN_GENE_CELLS = 30             # skip genes with fewer cells
SEED = 42

OUTPUT_DIR = "./local_neighborhood_deviation_K562"
# OUTPUT_DIR = "./local_neighborhood_deviation_GSE168851"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache paths for embeddings
CACHE_DIR = "./epiagent_embedding_cache_K562"
# CACHE_DIR = "./epiagent_embedding_cache_GSE168851"
os.makedirs(CACHE_DIR, exist_ok=True)
EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "cell_embeddings.npy")
PERTURBATION_CACHE_PATH = os.path.join(CACHE_DIR, "perturbations.npy")

# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================================================
# 3. Import EpiAgent / dataset
# ============================================================

from epiagent.model import EpiAgent
from epiagent.dataset import CellDataset, collate_fn

# ============================================================
# 4. Load metadata + cell_indices
# ============================================================

def load_metadata_and_cell_indices(csv_path):
    print(f"[INFO] Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, usecols=["cell_indices", "perturbation"])

    # map sgsgNT -> control, drop UNK and NaNs
    df.loc[df["perturbation"] == "sgsgNT", "perturbation"] = CONTROL_LABEL
    df = df[df["perturbation"] != UNK_LABEL].reset_index(drop=True)
    df = df.dropna(subset=["perturbation"]).reset_index(drop=True)

    print("[INFO] Parsing cell_indices...")
    cell_indices_list = [json.loads(s) for s in tqdm(df["cell_indices"].tolist())]
    perturbations = df["perturbation"].values

    print(f"[INFO] Loaded {len(df)} cells.")
    print(f"[INFO] Unique perturbations: {np.unique(perturbations)}")
    return cell_indices_list, perturbations

# ============================================================
# 5. Build EpiAgent and compute embeddings
# ============================================================

def build_epigagent_and_embed(cell_indices_list, device="cuda"):
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
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(input_ids)
                emb = outputs["transformer_outputs"][:, 0, :]  # (B, 512)
            all_embeddings.append(emb.float().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    return embeddings

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

# ============================================================
# 6. HDBSCAN filtering to main "ball"
# ============================================================

def hdbscan_filter_main_cluster(embeddings, perturbations):
    """
    Run HDBSCAN and keep only the largest non-noise cluster.
    """
    print("[INFO] Running HDBSCAN to identify major cluster...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=200,
        min_samples=10,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(embeddings)

    unique, counts = np.unique(labels, return_counts=True)
    print("[INFO] HDBSCAN cluster labels and counts (label: count):")
    for lab, cnt in zip(unique, counts):
        print(f"    {lab}: {cnt}")

    # ignore noise label -1 when picking largest cluster
    valid = unique[unique != -1]
    if len(valid) == 0:
        print("[WARN] HDBSCAN found no clusters; returning original data.")
        return embeddings, perturbations

    # pick label with max count among non-noise
    label_counts = {lab: cnt for lab, cnt in zip(unique, counts) if lab != -1}
    main_label = max(label_counts, key=label_counts.get)
    keep_mask = labels == main_label

    emb_filt = embeddings[keep_mask]
    pert_filt = perturbations[keep_mask]

    print(f"[INFO] Keeping cluster {main_label}: {emb_filt.shape[0]} / {embeddings.shape[0]} cells")
    print(f"[INFO] Unique perturbations after filtering: {np.unique(pert_filt)}")
    return emb_filt, pert_filt

# ============================================================
# 7. Local neighborhood deviation metric (Option A)
# ============================================================

def compute_local_knn_dists(embeddings, perturbations, gene, k_neighbors=20):
    """
    For a given gene, compute:
      - baseline: each control cell -> mean distance to its k nearest control neighbors
      - gene:     each gene-perturbed cell -> mean distance to its k nearest control neighbors
    """
    ctrl_mask = perturbations == CONTROL_LABEL
    gene_mask = perturbations == gene

    ctrl_idx = np.where(ctrl_mask)[0]
    gene_idx = np.where(gene_mask)[0]

    if len(gene_idx) < MIN_GENE_CELLS:
        print(f"[WARN] {gene}: only {len(gene_idx)} cells (< {MIN_GENE_CELLS}), skipping.")
        return None, None

    emb_ctrl = embeddings[ctrl_idx]
    emb_gene = embeddings[gene_idx]

    # fit kNN on control cells
    k = min(k_neighbors + 1, emb_ctrl.shape[0])  # +1 to account for self in baseline
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(emb_ctrl)

    # baseline: control -> control
    d_ctrl, idx_ctrl = nn.kneighbors(emb_ctrl, return_distance=True)
    # drop self (distance 0)
    baseline_dists = d_ctrl[:, 1:].mean(axis=1)  # shape (n_ctrl,)

    # gene -> control
    d_gene, idx_gene = nn.kneighbors(emb_gene, return_distance=True)
    gene_dists = d_gene.mean(axis=1)             # shape (n_gene,)

    return baseline_dists, gene_dists, len(ctrl_idx), len(gene_idx)

def summarize_and_plot_local_deviation(
    gene, baseline_dists, gene_dists, n_ctrl, n_gene, out_dir
):
    """
    Save metrics + histogram visualization for one gene.
    """
    mean_ctrl = float(baseline_dists.mean())
    std_ctrl = float(baseline_dists.std(ddof=1))
    mean_gene = float(gene_dists.mean())
    std_gene = float(gene_dists.std(ddof=1))

    # pooled std for Cohen's d
    pooled_var = (
        ((n_ctrl - 1) * std_ctrl**2 + (n_gene - 1) * std_gene**2)
        / (n_ctrl + n_gene - 2)
    )
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
    cohens_d = (
        (mean_gene - mean_ctrl) / pooled_std
        if pooled_std > 0
        else 0.0
    )

    print(
        f"[RESULT] {gene}: "
        f"mean_ctrl={mean_ctrl:.4f}, mean_gene={mean_gene:.4f}, "
        f"Cohen's d={cohens_d:.3f}"
    )

    # ---- visualization ----
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        baseline_dists,
        bins=40,
        alpha=0.6,
        density=True,
        label=f"control↔control (n={n_ctrl})",
    )
    ax.hist(
        gene_dists,
        bins=40,
        alpha=0.6,
        density=True,
        label=f"{gene}↔control (n={n_gene})",
    )

    ax.axvline(mean_ctrl, color="blue", linestyle="--", label=f"ctrl mean={mean_ctrl:.4f}")
    ax.axvline(mean_gene, color="red", linestyle="--", label=f"{gene} mean={mean_gene:.4f}")

    ax.set_title(f"Local k-NN distance to control manifold in 512D: {gene}")
    ax.set_xlabel(f"Mean distance to {K_NEIGHBORS} nearest controls (L2)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    text = f"Cohen's d = {cohens_d:.3f}"
    ax.text(
        0.98, 0.95, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"local_knn_distance_{gene}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # ---- per-gene violin plot: control vs gene ----
    fig_v, ax_v = plt.subplots(figsize=(5, 4))

    # two groups: baseline (control↔control) and gene↔control
    data_violin = [baseline_dists, gene_dists]

    parts = ax_v.violinplot(
        data_violin,
        positions=[1, 2],
        showmeans=True,
        showextrema=True,
        showmedians=False,
    )

    # Optional: slightly different facecolors
    for i, pc in enumerate(parts['bodies']):
        if i == 0:
            pc.set_facecolor("lightblue")
        else:
            pc.set_facecolor("lightcoral")
        pc.set_alpha(0.7)

    ax_v.set_xticks([1, 2])
    ax_v.set_xticklabels(
        [f"control (n={n_ctrl})", f"{gene} (n={n_gene})"],
        rotation=20,
        ha="right",
        fontsize=9,
    )
    ax_v.set_ylabel(f"Mean distance to {K_NEIGHBORS} nearest controls (L2)")
    ax_v.set_title(f"Local k-NN distances: control vs {gene}")

    fig_v.tight_layout()
    violin_path = os.path.join(out_dir, f"local_knn_violin_{gene}.png")
    fig_v.savefig(violin_path, dpi=200)
    plt.close(fig_v)


    metrics = {
        "gene": gene,
        "n_ctrl": n_ctrl,
        "n_gene": n_gene,
        "mean_ctrl": mean_ctrl,
        "std_ctrl": std_ctrl,
        "mean_gene": mean_gene,
        "std_gene": std_gene,
        "cohens_d": cohens_d,
    }
    return metrics

def compute_perturbation_directions(embeddings, perturbations, genes, control_label=CONTROL_LABEL):
    """
    Compute normalized perturbation direction vectors v_g = (mu_gene - mu_ctrl) / ||...||
    for each gene in `genes`.

    Returns:
        genes_used: list of genes that had valid vectors
        dir_matrix: np.ndarray of shape (len(genes_used), EMBED_DIM)
    """
    ctrl_mask = (perturbations == control_label)
    if ctrl_mask.sum() == 0:
        raise ValueError("No control cells found for computing perturbation directions.")

    emb_ctrl = embeddings[ctrl_mask]
    mu_ctrl = emb_ctrl.mean(axis=0)

    dir_vectors = []
    genes_used = []

    for gene in genes:
        gene_mask = (perturbations == gene)
        if gene_mask.sum() == 0:
            continue

        emb_gene = embeddings[gene_mask]
        mu_gene = emb_gene.mean(axis=0)
        v = mu_gene - mu_ctrl
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            print(f"[WARN] Perturbation direction nearly zero for {gene}, skipping in correlation heatmap.")
            continue

        dir_vectors.append(v / norm)
        genes_used.append(gene)

    if len(dir_vectors) == 0:
        return [], None

    dir_matrix = np.vstack(dir_vectors)  # shape (G, 512)
    return genes_used, dir_matrix


# ============================================================
# 8. Main
# ============================================================

def main():
    # 1) Load metadata
    cell_indices_list, perturbations = load_metadata_and_cell_indices(CSV_PATH)

    # 2) Get or compute EpiAgent embeddings (with caching)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, perturbations = get_or_compute_embeddings(cell_indices_list, perturbations, device=device)

    # 3) HDBSCAN filter to main cluster
    # embeddings, perturbations = hdbscan_filter_main_cluster(embeddings, perturbations)

    # 4) Compute metrics per gene
    all_metrics = []

    unique_perts = np.unique(perturbations)
    print(f"[INFO] Unique perturbations after filter: {unique_perts}")

    for gene in GENES_OF_INTEREST:
        if gene not in unique_perts:
            print(f"[WARN] {gene} not found after filtering; skipping.")
            continue

        baseline_dists, gene_dists, n_ctrl, n_gene = compute_local_knn_dists(
            embeddings, perturbations, gene, k_neighbors=K_NEIGHBORS
        )
        if baseline_dists is None:
            continue

        metrics = summarize_and_plot_local_deviation(
            gene, baseline_dists, gene_dists, n_ctrl, n_gene, OUTPUT_DIR
        )
        all_metrics.append(metrics)

    # 5) Save metrics table
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        csv_path = os.path.join(OUTPUT_DIR, "local_neighborhood_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"[INFO] Saved metrics table to {csv_path}")
        
        # 6) Plot bar plot of Cohen's d for all genes
        print("[INFO] Creating bar plot of Cohen's d for all genes...")
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sort by Cohen's d for better visualization
        df_sorted = df_metrics.sort_values('cohens_d', ascending=False)
        
        genes = df_sorted['gene'].values
        cohens_d_values = df_sorted['cohens_d'].values
        
        bars = ax.bar(range(len(genes)), cohens_d_values, alpha=0.7)
        
        # Color bars based on positive/negative values
        for i, (bar, val) in enumerate(zip(bars, cohens_d_values)):
            if val >= 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        ax.set_xlabel('Gene', fontsize=12)
        ax.set_ylabel("Cohen's d", fontsize=12)
        ax.set_title("Local Neighborhood Deviation Metric (Cohen's d) for All Genes", fontsize=14)
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        bar_plot_path = os.path.join(OUTPUT_DIR, "cohens_d_barplot_all_genes.png")
        fig.savefig(bar_plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved bar plot to {bar_plot_path}")
    else:
        print("[WARN] No metrics computed (check GENES_OF_INTEREST).")


    # 7) Heatmap of gene–gene perturbation direction correlations
    # print("[INFO] Computing gene–gene perturbation direction correlations...")

    # # Use the same gene ordering as the bar plot (df_sorted)
    # genes_for_dirs = df_sorted['gene'].values

    # genes_used, dir_matrix = compute_perturbation_directions(
    #     embeddings=embeddings,
    #     perturbations=perturbations,
    #     genes=genes_for_dirs,
    #     control_label=CONTROL_LABEL,
    # )

    # if dir_matrix is not None and len(genes_used) > 1:
    #     # cosine similarity matrix (G x G)
    #     # because dir_matrix rows are unit vectors, dot products = cosines
    #     corr_mat = np.matmul(dir_matrix, dir_matrix.T)

    #     fig_h, ax_h = plt.subplots(figsize=(10, 8))
    #     im = ax_h.imshow(
    #         corr_mat,
    #         vmin=-1.0, vmax=1.0,
    #         cmap="coolwarm",
    #         aspect="auto",
    #     )

    #     ax_h.set_xticks(range(len(genes_used)))
    #     ax_h.set_yticks(range(len(genes_used)))
    #     ax_h.set_xticklabels(genes_used, rotation=90, fontsize=7)
    #     ax_h.set_yticklabels(genes_used, fontsize=7)

    #     ax_h.set_title("Cosine similarity of perturbation directions in EpiAgent embedding space")

    #     cbar = fig_h.colorbar(im, ax=ax_h)
    #     cbar.set_label("Cosine similarity", rotation=90)

    #     fig_h.tight_layout()
    #     heatmap_path = os.path.join(OUTPUT_DIR, "perturbation_direction_correlation_heatmap.png")
    #     fig_h.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    #     plt.close(fig_h)

    #     print(f"[INFO] Saved perturbation direction correlation heatmap to {heatmap_path}")
    # else:
    #     print("[WARN] Not enough valid perturbation directions to build a heatmap.")



if __name__ == "__main__":
    main()

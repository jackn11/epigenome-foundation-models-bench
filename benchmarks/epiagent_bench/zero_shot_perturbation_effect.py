import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import hdbscan

from epiagent.model import EpiAgent
from epiagent.dataset import CellDataset, collate_fn


CONTROL_LABEL = "control"
UNK_LABEL = "UNK"

MAX_SEQ_LEN = 8192
VOCAB_SIZE = 1355449
NUM_LAYERS = 18
EMBED_DIM = 512
NUM_HEADS = 8

K_NEIGHBORS = 20
MIN_GENE_CELLS = 30

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def get_args_parser():
    parser = argparse.ArgumentParser('Zero-shot perturbation effect prediction', add_help=False)
    parser.add_argument('--csv_path',
                        default='/scratch/wkim/project-2-team-1/EpiAgent/external_data/GSE168851_crispr_perturb/All_data_SpearATAC_K562_LargeScreen.csv', type=str)
    parser.add_argument('--genes_of_interest',
                        default=['sgGATA1', 'sgMAX', 'sgYY1'],
                        nargs='+',
                        type=str,)
    parser.add_argument('--output_dir', default='./zeroshot_perturbation_effect_prediction_CohensD_outputs', type=str)
    parser.add_argument('--cache_dir', default='/path/to/embeddings/cache', type=str)
    parser.add_argument('--pretrained_model_path', default='/scratch/wkim/project-2-team-1/EpiAgent/model/pretrained_EpiAgent.pth', type=str)
    # parser.add_argument('--chromVAR_rank_path',
    #                     default='./Pierce2021_chromVAR/Pierce2021_gene_ranking_chromvar.csv',
    #                     type=str,
    #                     help='Path to chromVAR gene ranking file')
    return parser


def load_metadata_and_cell_indices(csv_path):
    print(f"[INFO] Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, usecols=["cell_indices", "perturbation"])

    df.loc[df["perturbation"] == "sgsgNT", "perturbation"] = CONTROL_LABEL
    df = df[df["perturbation"] != UNK_LABEL].reset_index(drop=True)
    df = df.dropna(subset=["perturbation"]).reset_index(drop=True)

    print("[INFO] Parsing cell_indices...")
    cell_indices_list = [json.loads(s) for s in tqdm(df["cell_indices"].tolist())]
    perturbations = df["perturbation"].values

    print(f"[INFO] Loaded {len(df)} cells.")
    print(f"[INFO] Unique perturbations: {np.unique(perturbations)}")
    return cell_indices_list, perturbations

def build_epigagent_and_embed(args, cell_indices_list, device="cuda"):
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

    print(f"[INFO] Loading pretrained weights from {args.pretrained_model_path}")
    state_dict = torch.load(args.pretrained_model_path, map_location="cpu")
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
                emb = outputs["transformer_outputs"][:, 0, :]   
            all_embeddings.append(emb.float().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    return embeddings

def get_or_compute_embeddings(args, cell_indices_list, perturbations, device="cuda"):
    """
    If cached embeddings exist, load and return them.
    Otherwise, compute embeddings with EpiAgent, save to disk, and return.

    We cache embeddings BEFORE HDBSCAN filtering. HDBSCAN is cheap
    compared to the forward pass, so we rerun it each time.
    """
    embedding_cache_path = os.path.join(args.cache_dir, "cell_embeddings.npy")
    perturbation_cache_path = os.path.join(args.cache_dir, "perturbations.npy")
    if os.path.exists(embedding_cache_path) and os.path.exists(perturbation_cache_path):
        print(f"[INFO] Loading cached embeddings from {embedding_cache_path}")
        cached_embeddings = np.load(embedding_cache_path)
        cached_perts = np.load(perturbation_cache_path, allow_pickle=True)

        if cached_embeddings.shape[0] != len(perturbations):
            print("[WARN] Cached embeddings shape does not match current "
                  "number of cells; recomputing embeddings.")
        elif not np.array_equal(cached_perts, perturbations):
            print("[WARN] Cached perturbation labels do not match current "
                  "labels; recomputing embeddings.")
        else:
            print("[INFO] Loaded cached embeddings successfully.")
            return cached_embeddings, cached_perts

    print("[INFO] No valid cache found; computing embeddings with EpiAgent...")
    embeddings = build_epigagent_and_embed(args, cell_indices_list, device=device)

    np.save(embedding_cache_path, embeddings)
    np.save(perturbation_cache_path, np.asarray(perturbations, dtype=str))
    print(f"[INFO] Saved embeddings cache to {embedding_cache_path}")
    print(f"[INFO] Saved perturbation labels cache to {perturbation_cache_path}")

    return embeddings, perturbations

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

    valid = unique[unique != -1]
    if len(valid) == 0:
        print("[WARN] HDBSCAN found no clusters; returning original data.")
        return embeddings, perturbations

    label_counts = {lab: cnt for lab, cnt in zip(unique, counts) if lab != -1}
    main_label = max(label_counts, key=label_counts.get)
    keep_mask = labels == main_label

    emb_filt = embeddings[keep_mask]
    pert_filt = perturbations[keep_mask]

    print(f"[INFO] Keeping cluster {main_label}: {emb_filt.shape[0]} / {embeddings.shape[0]} cells")
    print(f"[INFO] Unique perturbations after filtering: {np.unique(pert_filt)}")
    return emb_filt, pert_filt

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

    k = min(k_neighbors + 1, emb_ctrl.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(emb_ctrl)

    d_ctrl, idx_ctrl = nn.kneighbors(emb_ctrl, return_distance=True)
    baseline_dists = d_ctrl[:, 1:].mean(axis=1)

    d_gene, idx_gene = nn.kneighbors(emb_gene, return_distance=True)
    gene_dists = d_gene.mean(axis=1)

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


def main(args):
    cell_indices_list, perturbations = load_metadata_and_cell_indices(args.csv_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, perturbations = get_or_compute_embeddings(args, cell_indices_list, perturbations, device=device)

    all_metrics = []

    unique_perts = np.unique(perturbations)
    print(f"[INFO] Unique perturbations after filter: {unique_perts}")

    for gene in args.genes_of_interest:
        if gene not in unique_perts:
            print(f"[WARN] {gene} not found after filtering; skipping.")
            continue

        baseline_dists, gene_dists, n_ctrl, n_gene = compute_local_knn_dists(
            embeddings, perturbations, gene, k_neighbors=K_NEIGHBORS
        )
        if baseline_dists is None:
            continue

        metrics = summarize_and_plot_local_deviation(
            gene, baseline_dists, gene_dists, n_ctrl, n_gene, args.output_dir
        )
        all_metrics.append(metrics)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        csv_path = os.path.join(args.output_dir, "local_neighborhood_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"[INFO] Saved metrics table to {csv_path}")

        df_metrics["weight"] = df_metrics["n_gene"]
        weighted_mean_d = np.average(df_metrics["cohens_d"], weights=df_metrics["weight"])
        
        print(f"[MODEL SCORE] Weighted mean Cohen's d across genes = {weighted_mean_d:.4f}")
        
        with open(os.path.join(args.output_dir, "model_score.txt"), "w") as f:
            f.write(f"Weighted mean Cohen's d = {weighted_mean_d:.6f}\n")

        if "Pierce2021" in args.csv_path:
            chromVAR_rank = ['sgGATA1', 'sgCAD', 'sgRPL9', 'sgCDC5L', 'sgKLF1', 'sgNFE2', 'sgNRF1', 'sgARID2', 'sgGABPA', 'sgARID3A', 'sgZNF407', 'sgFOSL1', 'sgMAX', 'sgATF3', 'sgHSPA5', 'sgCTCF', 'sgGTF2B', 'sgMYC', 'sgHINFP', 'sgKLF16', 'sgNFYB', 'sgCUX1', 'sgTHAP1', 'sgZBTB11', 'sgPBX2', 'sgATF1', 'sgYY1', 'sgPOLR1D', 'sgBCLAF1', 'sgREST', 'sgZZZ3', 'sgCEBPB', 'sgTFDP1', 'sgTBP', 'sgBRF2', 'sgCEBPZ', 'sgSETDB1', 'sgZNF280A', 'sgTRIM28', 'sgELF1']
            # chromVAR_rank_df = pd.read_csv(args.chromVAR_rank_path)
            # chromVAR_rank = chromVAR_rank_df['gene'].tolist() 

            chromvar_rank_dict = {gene: rank + 1 for rank, gene in enumerate(chromVAR_rank)}
            
            ranks = []
            d_values_rank = []

            for _, row in df_metrics.iterrows():
                gene = row["gene"]
                d_val = row["cohens_d"]

                if gene in chromvar_rank_dict:
                    ranks.append(chromvar_rank_dict[gene])
                    d_values_rank.append(d_val)
                else:
                    continue

            if len(ranks) > 0:
                rho_rank, pval_rank = spearmanr(ranks, d_values_rank)
                print(f"[BIO SCORE (RANK)] Spearman correlation between chromVAR order and Cohen's d = {rho_rank:.4f} (p={pval_rank:.3g})")
                with open(os.path.join(args.output_dir, "chromVAR_spearman_correlation_rank.txt"), "w") as f:
                    f.write(f"Spearman correlation = {rho_rank:.6f}, p={pval_rank:.6g}\n")
            else:
                print("[WARN] No genes found in chromVAR_rank for rank-based metric calculation.")


        print("[INFO] Creating bar plot of Cohen's d for all genes...")
        fig, ax = plt.subplots(figsize=(14, 6))
        
        df_sorted = df_metrics.sort_values('cohens_d', ascending=False)
        
        genes = df_sorted['gene'].values
        cohens_d_values = df_sorted['cohens_d'].values
        
        bars = ax.bar(range(len(genes)), cohens_d_values, alpha=0.7)
        
        for i, (bar, val) in enumerate(zip(bars, cohens_d_values)):
            if val >= 0:
                bar.set_color('salmon')
            else:
                bar.set_color('steelblue')
        
        ax.set_xlabel('Gene', fontsize=12)
        ax.set_ylabel("Cohen's d", fontsize=12)
        ax.set_ylim([-0.5, 3.5])
        ax.set_title("Local Neighborhood Deviation Metric (Cohen's d) for All Genes", fontsize=14)
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        bar_plot_path = os.path.join(args.output_dir, "cohens_d_barplot_all_genes.png")
        fig.savefig(bar_plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved bar plot to {bar_plot_path}")
    else:
        print("[WARN] No metrics computed (check GENES_OF_INTEREST).")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    main(args)

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.neighbors import NearestNeighbors

CONTROL_LABEL = "control"        # we will remap sgsgNT -> control
UNK_LABEL = "UNK"                # will be dropped

K_NEIGHBORS = 20                # k for local neighborhood
MIN_GENE_CELLS = 30             # skip genes with fewer cells

def get_args_parser():
    parser = argparse.ArgumentParser('Zero-shot perturbation effect prediction with ChromFound embeddings', add_help=False)
    parser.add_argument('--h5ad_path',
                        default='/scratch/wkim/project-2-team-1/benchmarks/results/chromfound/Pierce2021/merge10/embeddings_pca.h5ad', 
                        type=str,
                        help='Path to h5ad file with embeddings')
    parser.add_argument('--genes_of_interest',
                        default=['sgGATA1', 'sgMAX', 'sgYY1'],
                        nargs='+',
                        type=str,
                        help='List of genes to analyze (with sg prefix)')
    parser.add_argument('--output_dir', 
                        default='./chromfound_zeroshot_perturbation_effect_prediction_CohensD_outputs', 
                        type=str,
                        help='Directory to save outputs')
    parser.add_argument('--use_hdbscan_filter',
                        action='store_true',
                        help='Apply HDBSCAN filtering to keep main cluster')
    # parser.add_argument('--chromVAR_rank_path',
    #                     default='./Pierce2021_chromVAR/Pierce2021_gene_ranking_chromvar.csv',
    #                     type=str,
    #                     help='Path to chromVAR gene ranking file')
    return parser


def load_embeddings_and_labels(h5ad_path):
    """
    Load ChromFound embeddings and perturbation labels from h5ad file.
    """
    print(f"[INFO] Loading h5ad file from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    # Extract embeddings from obsm
    if 'X_pca' not in adata.obsm:
        raise ValueError("No 'X_pca' found in adata.obsm. Check your h5ad file.")
    
    embeddings = adata.obsm['X_pca']
    print(f"[INFO] Loaded embeddings with shape: {embeddings.shape}")
    
    # Extract perturbations from obs
    if 'celltype' not in adata.obs.columns:
        raise ValueError("No 'celltype' column found in adata.obs. Check your h5ad file.")
    
    perturbations = adata.obs['celltype'].values
    print(f"[INFO] Loaded {len(perturbations)} cell perturbation labels")
    
    # Map sgsgNT -> control, drop UNK and NaNs
    print(f"[INFO] Original unique perturbations: {np.unique(perturbations)}")
    
    # Create a mask for filtering
    keep_mask = (perturbations != UNK_LABEL)
    embeddings = embeddings[keep_mask]
    perturbations = perturbations[keep_mask]
    
    # Remap sgsgNT to control
    perturbations = np.array([CONTROL_LABEL if p == "sgsgNT" else p for p in perturbations])
    
    print(f"[INFO] After filtering: {len(perturbations)} cells")
    print(f"[INFO] Unique perturbations after mapping: {np.unique(perturbations)}")
    
    return embeddings, perturbations


def hdbscan_filter_main_cluster(embeddings, perturbations):
    """
    Run HDBSCAN and keep only the largest non-noise cluster.
    """
    import hdbscan
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
        return None, None, None, None

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
    gene, baseline_dists, gene_dists, n_ctrl, n_gene, out_dir, embedding_dim=50
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

    ax.set_title(f"Local k-NN distance to control manifold (ChromFound {embedding_dim}D): {gene}")
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
    # 1) Load embeddings and labels from h5ad
    embeddings, perturbations = load_embeddings_and_labels(args.h5ad_path)

    # 2) Optional HDBSCAN filter to main cluster
    if args.use_hdbscan_filter:
        embeddings, perturbations = hdbscan_filter_main_cluster(embeddings, perturbations)

    # 3) Compute metrics per gene
    all_metrics = []

    unique_perts = np.unique(perturbations)
    print(f"[INFO] Unique perturbations available: {unique_perts}")

    embedding_dim = embeddings.shape[1]
    
    for gene in args.genes_of_interest:
        if gene not in unique_perts:
            print(f"[WARN] {gene} not found in data; skipping.")
            continue

        baseline_dists, gene_dists, n_ctrl, n_gene = compute_local_knn_dists(
            embeddings, perturbations, gene, k_neighbors=K_NEIGHBORS
        )
        if baseline_dists is None:
            continue

        metrics = summarize_and_plot_local_deviation(
            gene, baseline_dists, gene_dists, n_ctrl, n_gene, args.output_dir, embedding_dim
        )
        all_metrics.append(metrics)

    # 4) Save metrics table
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        csv_path = os.path.join(args.output_dir, "local_neighborhood_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"[INFO] Saved metrics table to {csv_path}")

        # Compute one scalar "model performance score" across all genes based on Cohen's d
        df_metrics["weight"] = df_metrics["n_gene"]
        weighted_mean_d = np.average(df_metrics["cohens_d"], weights=df_metrics["weight"])
        
        print(f"[MODEL SCORE] Weighted mean Cohen's d across genes = {weighted_mean_d:.4f}")
        
        with open(os.path.join(args.output_dir, "model_score.txt"), "w") as f:
            f.write(f"Weighted mean Cohen's d = {weighted_mean_d:.6f}\n")

        if "Pierce2021" in args.h5ad_path:
            chromVAR_rank = ['sgGATA1', 'sgCAD', 'sgRPL9', 'sgCDC5L', 'sgKLF1', 'sgNFE2', 'sgNRF1', 'sgARID2', 'sgGABPA', 'sgARID3A', 'sgZNF407', 'sgFOSL1', 'sgMAX', 'sgATF3', 'sgHSPA5', 'sgCTCF', 'sgGTF2B', 'sgMYC', 'sgHINFP', 'sgKLF16', 'sgNFYB', 'sgCUX1', 'sgTHAP1', 'sgZBTB11', 'sgPBX2', 'sgATF1', 'sgYY1', 'sgPOLR1D', 'sgBCLAF1', 'sgREST', 'sgZZZ3', 'sgCEBPB', 'sgTFDP1', 'sgTBP', 'sgBRF2', 'sgCEBPZ', 'sgSETDB1', 'sgZNF280A', 'sgTRIM28', 'sgELF1']
            # chromVAR_rank_df = pd.read_csv(args.chromVAR_rank_path)
            # chromVAR_rank = chromVAR_rank_df['gene'].tolist() 

            # Create mapping from gene name to rank (1-indexed: first gene = rank 1)
            chromvar_rank_dict = {gene: rank + 1 for rank, gene in enumerate(chromVAR_rank)}
            
            ranks = []
            d_values_rank = []

            for _, row in df_metrics.iterrows():
                gene = row["gene"]  # keep "sg" prefix for chromVAR_rank matching
                d_val = row["cohens_d"]

                if gene in chromvar_rank_dict:
                    ranks.append(chromvar_rank_dict[gene])
                    d_values_rank.append(d_val)
                else:
                    # Skip genes not in chromVAR_rank for this metric
                    continue

            if len(ranks) > 0:
                rho_rank, pval_rank = spearmanr(ranks, d_values_rank)
                print(f"[BIO SCORE (RANK)] Spearman correlation between chromVAR order and Cohen's d = {rho_rank:.4f} (p={pval_rank:.3g})")
                with open(os.path.join(args.output_dir, "chromVAR_spearman_correlation_rank.txt"), "w") as f:
                    f.write(f"Spearman correlation = {rho_rank:.6f}, p={pval_rank:.6g}\n")
            else:
                print("[WARN] No genes found in chromVAR_rank for rank-based metric calculation.")

        # 5) Plot bar plot of Cohen's d for all genes
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
                bar.set_color('salmon')
            else:
                bar.set_color('steelblue')
        
        ax.set_xlabel('Gene', fontsize=12)
        ax.set_ylabel("Cohen's d", fontsize=12)
        ax.set_ylim([-0.5, 3.5])
        ax.set_title("ChromFound: Local Neighborhood Deviation Metric (Cohen's d) for All Genes", fontsize=14)
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
        print("[WARN] No metrics computed (check genes_of_interest).")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)


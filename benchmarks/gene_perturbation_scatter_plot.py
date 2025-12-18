#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

chromvar_df = pd.read_csv("Pierce2021_gene_ranking_chromvar.csv")
epiagent_metrics_df = pd.read_csv("epiagent_bench/zero_shot_perturbation_effect/Pierce2021/local_neighborhood_metrics.csv")
chromfound_metrics_df = pd.read_csv("chromfound_bench/zero_shot_perturbation_effect/Pierce2021/local_neighborhood_metrics.csv")

epiagent_merged_df = chromvar_df.merge(
    epiagent_metrics_df[["gene", "cohens_d"]],
    on="gene",
    how="inner",
)
chromfound_merged_df = chromvar_df.merge(
    chromfound_metrics_df[["gene", "cohens_d"]],
    on="gene",
    how="inner",
)

epiagent_merged_df = epiagent_merged_df.dropna(subset=["chromvar", "cohens_d"]).copy()
chromfound_merged_df = chromfound_merged_df.dropna(subset=["chromvar", "cohens_d"]).copy()

# Create copies for "all genes" plot
epiagent_all = epiagent_merged_df.copy()
chromfound_all = chromfound_merged_df.copy()

# Remove specific genes (e.g., sgGATA1) for filtered plot
genes_to_exclude = ['sgGATA1']
epiagent_filtered = epiagent_merged_df[~epiagent_merged_df['gene'].isin(genes_to_exclude)].copy()
chromfound_filtered = chromfound_merged_df[~chromfound_merged_df['gene'].isin(genes_to_exclude)].copy()

def annotate_outliers(ax, df, color, dx=5, dy=5):
    """Annotate genes; dx/dy are pixel offsets to reduce overlap with the marker."""
    for _, row in df.iterrows():
        ax.annotate(
            row["gene"],
            (row["chromvar"], row["cohens_d"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            color=color,
            alpha=0.9,
        )

def create_plot(epiagent_df, chromfound_df, filename, title_suffix=""):
    """Create a scatter plot with best-fit lines."""
    x_e = epiagent_df["chromvar"].to_numpy()
    y_e = epiagent_df["cohens_d"].to_numpy()
    x_c = chromfound_df["chromvar"].to_numpy()
    y_c = chromfound_df["cohens_d"].to_numpy()
    
    # Best-fit (linear) lines
    epiagent_slope, epiagent_intercept = np.polyfit(x_e, y_e, 1)
    chromfound_slope, chromfound_intercept = np.polyfit(x_c, y_c, 1)
    
    # Calculate R^2
    y_e_pred = epiagent_slope * x_e + epiagent_intercept
    y_c_pred = chromfound_slope * x_c + chromfound_intercept
    ss_res_e = np.sum((y_e - y_e_pred) ** 2)
    ss_tot_e = np.sum((y_e - np.mean(y_e)) ** 2)
    r2_epiagent = 1 - (ss_res_e / ss_tot_e) if ss_tot_e > 0 else 0
    
    ss_res_c = np.sum((y_c - y_c_pred) ** 2)
    ss_tot_c = np.sum((y_c - np.mean(y_c)) ** 2)
    r2_chromfound = 1 - (ss_res_c / ss_tot_c) if ss_tot_c > 0 else 0
    
    x_line = np.linspace(min(x_e.min(), x_c.min()), max(x_e.max(), x_c.max()), 200)
    
    # Outlier labeling (top-k by |cohens_d|)
    topk = 5
    epiagent_outliers = epiagent_df.iloc[
        np.argsort(np.abs(epiagent_df["cohens_d"].to_numpy()))[-topk:]
    ]
    chromfound_outliers = chromfound_df.iloc[
        np.argsort(np.abs(chromfound_df["cohens_d"].to_numpy()))[-topk:]
    ]
    
    # Create plot
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_e, y_e, alpha=0.6, s=50, color="steelblue", label="EpiAgent")
    ax.scatter(x_c, y_c, alpha=0.6, s=50, color="salmon", label="ChromFound")
    
    ax.plot(x_line, epiagent_slope * x_line + epiagent_intercept, "b--", alpha=0.3, linewidth=2, 
            label=f"EpiAgent fit (R²={r2_epiagent:.3f})")
    ax.plot(x_line, chromfound_slope * x_line + chromfound_intercept, "r--", alpha=0.3, linewidth=2, 
            label=f"ChromFound fit (R²={r2_chromfound:.3f})")
    
    # Add R^2 text annotations on the plot
    x_text_pos = x_line[-1] * 0.85  # Position near the right edge (moved left)
    y_range = max(y_e.max(), y_c.max()) - min(y_e.min(), y_c.min())
    y_offset = y_range * 0.1  # Move upward by 5% of y range
    y_text_e = epiagent_slope * x_text_pos + epiagent_intercept + y_offset
    y_text_c = chromfound_slope * x_text_pos + chromfound_intercept + y_offset
    
    ax.text(x_text_pos, y_text_e, f"R²={r2_epiagent:.3f}", 
            fontsize=10, color="steelblue", fontweight="bold",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='steelblue'))
    ax.text(x_text_pos, y_text_c, f"R²={r2_chromfound:.3f}", 
            fontsize=10, color="salmon", fontweight="bold",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='salmon'))
    
    # annotate_outliers(ax, epiagent_outliers, color="blue")
    # annotate_outliers(ax, chromfound_outliers, color="red")
    
    ax.set_xlabel("chromVAR (DiffDevScore)", fontsize=12)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title(f"Gene Perturbation Effect: chromVAR vs Cohen's d{title_suffix}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {filename}")
    print(f"Number of genes plotted (EpiAgent): {len(epiagent_df)}")
    print(f"Number of genes plotted (ChromFound): {len(chromfound_df)}")
    print(f"R² (EpiAgent): {r2_epiagent:.4f}")
    print(f"R² (ChromFound): {r2_chromfound:.4f}")
    print(f"Labeled outliers per model (topk): {topk}")
    print()

# Plot 1: All genes
create_plot(epiagent_all, chromfound_all, "gene_perturbation_scatter_plot.png", "")

# Plot 2: With sgGATA1 (outlier) removed
create_plot(epiagent_filtered, chromfound_filtered, "gene_perturbation_scatter_plot_no_sgGATA1.png", " (sgGATA1 excluded)")

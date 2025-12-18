#%%
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#%%
def load_epiagent_results():
    """Load EpiAgent benchmark results"""
    feature_extraction_datasets = ["Kanemaru2023", "Li2023b"]
    # epiagent_root = Path("../EpiAgent/benchmarks")
    epiagent_root = Path("epiagent_bench")
    feature_extraction_result_csvs = {dataset: epiagent_root / f"zero_shot_feature_extraction/{dataset}/results.csv" for dataset in feature_extraction_datasets}

    feature_extraction_results = {}
    for dataset, csv_path in feature_extraction_result_csvs.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            metrics_dict = dict(zip(df['Metric'], df['Value']))
            feature_extraction_results[dataset] = metrics_dict
        else:
            print(f"Warning: EpiAgent results file not found for {dataset}: {csv_path}")

    # Load perturbation benchmark results
    perturbation_datasets = ["Liscovitch_Brauer2021", "Pierce2021"]
    perturbation_cohens_d_paths = {dataset: epiagent_root / f"zero_shot_perturbation_effect/{dataset}/model_score.txt" for dataset in perturbation_datasets}
    perturbation_spearmanr_paths = {dataset: epiagent_root / f"zero_shot_perturbation_effect/{dataset}/chromVAR_spearman_correlation_rank.txt" for dataset in perturbation_datasets}
    perturbation_results = {}
    for dataset in perturbation_datasets:
        cohens_d_path = perturbation_cohens_d_paths[dataset]
        spearmanr_path = perturbation_spearmanr_paths[dataset]
        if cohens_d_path.exists():
            with open(cohens_d_path, 'r') as file:
                cohens_d = file.read().strip()
                match = re.search(r"Weighted mean Cohen's d[^\d\-]*([-+]?\d*\.\d+|\d+)", cohens_d)
                cohens_d = float(match.group(1)) 
        if spearmanr_path.exists():
            with open(spearmanr_path, 'r') as file:
                chromVAR_spearmanr = file.read().strip()
                match = re.search(r"Spearman correlation[^\d\-]*([-+]?\d*\.\d+|\d+)", chromVAR_spearmanr)
                chromVAR_spearmanr = -float(match.group(1))
            perturbation_results[dataset] = {
                "cohens_d": cohens_d,
                "chromVAR_spearmanr": chromVAR_spearmanr
            }
        else:
            print(f"Warning: EpiAgent perturbation results not found for {dataset}")

    return feature_extraction_results, perturbation_results, feature_extraction_datasets, perturbation_datasets


def load_chromfound_results():
    """Load ChromFound benchmark results"""
    feature_extraction_datasets = ["Kanemaru2023_full", "Li2023b_full"]
    # chromfound_root = Path("../ChromFound-Parallel")
    chromfound_root = Path("chromfound_bench")
    feature_extraction_result_csvs = {dataset: chromfound_root / f"zero_shot_feature_extraction/{dataset}/results.csv" for dataset in feature_extraction_datasets}

    feature_extraction_results = {}
    for dataset, csv_path in feature_extraction_result_csvs.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            metrics_dict = dict(zip(df['Metric'], df['Value']))
            feature_extraction_results[dataset] = metrics_dict
        else:
            print(f"Warning: ChromFound results file not found for {dataset}: {csv_path}")

    perturbation_datasets = ["Liscovitch_Brauer2021", "Pierce2021"]
    perturbation_cohens_d_paths = {dataset: chromfound_root / f"zero_shot_perturbation_effect/{dataset}/model_score.txt" for dataset in perturbation_datasets}
    perturbation_spearmanr_paths = {dataset: chromfound_root / f"zero_shot_perturbation_effect/{dataset}/chromVAR_spearman_correlation_rank.txt" for dataset in perturbation_datasets}
    perturbation_results = {}
    for dataset in perturbation_datasets:
        cohens_d_path = perturbation_cohens_d_paths[dataset]
        spearmanr_path = perturbation_spearmanr_paths[dataset]
        if cohens_d_path.exists():
            with open(cohens_d_path, 'r') as file:
                cohens_d = file.read().strip()
                match = re.search(r"Weighted mean Cohen's d[^\d\-]*([-+]?\d*\.\d+|\d+)", cohens_d)
                cohens_d = float(match.group(1)) 
        if spearmanr_path.exists():
            with open(spearmanr_path, 'r') as file:
                chromVAR_spearmanr = file.read().strip()
                match = re.search(r"Spearman correlation[^\d\-]*([-+]?\d*\.\d+|\d+)", chromVAR_spearmanr)
                chromVAR_spearmanr = -float(match.group(1))
            perturbation_results[dataset] = {
                "cohens_d": cohens_d,
                "chromVAR_spearmanr": chromVAR_spearmanr
            }
        else:
            print(f"Warning: ChromFound perturbation results not found for {dataset}")

    return feature_extraction_results, perturbation_results, feature_extraction_datasets, perturbation_datasets


def aggregate_metrics(feature_extraction_results, perturbation_results, feature_extraction_datasets, perturbation_datasets):
    """Aggregate metrics from results"""
    # Aggregate feature extraction metrics
    nmi_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            nmi_values.append(feature_extraction_results[dataset]['Normalized Mutual Information'])
    nmi = np.array(nmi_values).mean() if len(nmi_values) > 0 else 0

    ari_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            ari_values.append(feature_extraction_results[dataset]['Adjusted Rand Index'])
    ari = np.array(ari_values).mean() if len(ari_values) > 0 else 0

    silhouette_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            silhouette_values.append(feature_extraction_results[dataset]['Silhouette score'])
    silhouette = np.array(silhouette_values).mean() if len(silhouette_values) > 0 else 0

    silhouette_batch_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            if 'Silhouette batch score' in feature_extraction_results[dataset]:
                silhouette_batch_values.append(feature_extraction_results[dataset]['Silhouette batch score'])
    silhouette_batch = np.array(silhouette_batch_values).mean() if len(silhouette_batch_values) > 0 else 0

    cell_type_linear_probe_f1_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            cell_type_linear_probe_f1_values.append(feature_extraction_results[dataset]['Cell type Linear probe F1 score (macro)'])
    cell_type_linear_probe_f1 = np.array(cell_type_linear_probe_f1_values).mean() if len(cell_type_linear_probe_f1_values) > 0 else 0

    batch_linear_probe_f1_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            if 'Batch label Linear probe F1 score (macro)' in feature_extraction_results[dataset]:
                batch_linear_probe_f1_values.append(feature_extraction_results[dataset]['Batch label Linear probe F1 score (macro)'])
    batch_linear_probe_f1 = np.array(batch_linear_probe_f1_values).mean() if len(batch_linear_probe_f1_values) > 0 else 0

    # Graph connectivity
    graph_connectivity_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            if 'Graph connectivity score' in feature_extraction_results[dataset]:
                graph_connectivity_values.append(feature_extraction_results[dataset]['Graph connectivity score'])
    graph_connectivity = np.array(graph_connectivity_values).mean() if len(graph_connectivity_values) > 0 else 0

    # ilisi
    ilisi_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            if 'ilisi score' in feature_extraction_results[dataset]:
                ilisi_values.append(feature_extraction_results[dataset]['ilisi score'])
    ilisi = np.array(ilisi_values).mean() if len(ilisi_values) > 0 else 0

    # PCR batch (already inverted to 1-PCR in the feature extraction script)
    pcr_batch_values = []
    for dataset in feature_extraction_datasets:
        if dataset in feature_extraction_results:
            if 'PCR batch score (1-PCR)' in feature_extraction_results[dataset]:
                pcr_batch_values.append(feature_extraction_results[dataset]['PCR batch score (1-PCR)'])
    pcr_batch = np.array(pcr_batch_values).mean() if len(pcr_batch_values) > 0 else 0

    # Aggregate perturbation metrics
    cohens_d_values = []
    for dataset in perturbation_datasets:
        if dataset in perturbation_results:
            cohens_d_values.append(perturbation_results[dataset]['cohens_d'])
    cohens_d = np.array(cohens_d_values).mean() if len(cohens_d_values) > 0 else 0

    chromVAR_spearmanr_values = []
    for dataset in perturbation_datasets:
        if dataset in perturbation_results:
            chromVAR_spearmanr_values.append(perturbation_results[dataset]['chromVAR_spearmanr'])
    chromVAR_spearmanr = np.array(chromVAR_spearmanr_values).mean() if len(chromVAR_spearmanr_values) > 0 else 0

    return {
        'nmi': nmi,
        'ari': ari,
        'silhouette': silhouette,
        'silhouette_batch': silhouette_batch,
        'cell_type_linear_probe_f1': cell_type_linear_probe_f1,
        'batch_linear_probe_f1': batch_linear_probe_f1,
        'graph_connectivity': graph_connectivity,
        'ilisi': ilisi,
        'pcr_batch': pcr_batch,
        'cohens_d': cohens_d,
        'chromVAR_spearmanr': chromVAR_spearmanr
    }


#%%
# Load results from both methods
print("Loading EpiAgent results...")
epiagent_feat_results, epiagent_pert_results, epiagent_feat_datasets, epiagent_pert_datasets = load_epiagent_results()

print("\nLoading ChromFound results...")
chromfound_feat_results, chromfound_pert_results, chromfound_feat_datasets, chromfound_pert_datasets = load_chromfound_results()

# Aggregate metrics
print("\nAggregating EpiAgent metrics...")
epiagent_metrics = aggregate_metrics(epiagent_feat_results, epiagent_pert_results, epiagent_feat_datasets, epiagent_pert_datasets)
print(f"EpiAgent metrics: {epiagent_metrics}")

print("\nAggregating ChromFound metrics...")
chromfound_metrics = aggregate_metrics(chromfound_feat_results, chromfound_pert_results, chromfound_feat_datasets, chromfound_pert_datasets)
print(f"ChromFound metrics: {chromfound_metrics}")

#%%
# Prepare data for plotting
labels = [
    "NMI",
    "ARI",
    "Silhouette",
    "Graph\nConnectivity",
    "Cell type\nLinear Probe (F1)",
    "Silhouette\nBatch",
    "Batch Label\nLinear Probe (1-F1)",
    "iLISI",
    "PCR Batch\n(1-PCR)",
    "Perturbation Effect\nCohen's d",
    "Perturbation Effect\nChromVAR Spearman-R",
]

mins = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
maxs = np.array([1, 1, 1, 1.25, 1, 1, 1, 5.5, 1, 0.25, 0.75])

epiagent_values = np.array([
    epiagent_metrics['nmi'],
    epiagent_metrics['ari'],
    epiagent_metrics['silhouette'],
    epiagent_metrics['graph_connectivity'],
    epiagent_metrics['cell_type_linear_probe_f1'],
    epiagent_metrics['silhouette_batch'],
    1 - epiagent_metrics['batch_linear_probe_f1'],
    epiagent_metrics['ilisi'],
    epiagent_metrics['pcr_batch'],
    epiagent_metrics['cohens_d'],
    epiagent_metrics['chromVAR_spearmanr']
])

chromfound_values = np.array([
    chromfound_metrics['nmi'],
    chromfound_metrics['ari'],
    chromfound_metrics['silhouette'],
    chromfound_metrics['graph_connectivity'],
    chromfound_metrics['cell_type_linear_probe_f1'],
    chromfound_metrics['silhouette_batch'],
    1 - chromfound_metrics['batch_linear_probe_f1'],
    chromfound_metrics['ilisi'],
    chromfound_metrics['pcr_batch'],
    chromfound_metrics['cohens_d'],
    chromfound_metrics['chromVAR_spearmanr']
])

epiagent_norm = (epiagent_values - mins) / (maxs - mins)
chromfound_norm = (chromfound_values - mins) / (maxs - mins)

labels_closed = labels + [labels[0]]
epiagent_norm_closed = np.append(epiagent_norm, epiagent_norm[0])
chromfound_norm_closed = np.append(chromfound_norm, chromfound_norm[0])

angles = np.linspace(0, 2 * np.pi, len(labels_closed))

#%%
# Create figure with single subplot for overlay comparison
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

ax.set_ylim(0, 1)
ax.set_rlabel_position(0)
ax.set_yticks(list(np.linspace(0, 1, len(labels_closed))))
ax.set_yticklabels([""] * len(labels_closed))
ax.yaxis.set_tick_params(labelleft=False, labelright=False)

# Plot EpiAgent
ax.plot(angles, epiagent_norm_closed, marker='o', label='EpiAgent', color='#1f77b4', linewidth=2.5, markersize=8)
ax.fill(angles, epiagent_norm_closed, alpha=0.2, color='#1f77b4')

# Plot ChromFound
ax.plot(angles, chromfound_norm_closed, marker='s', label='ChromFound', color='#ff7f0e', linewidth=2.5, markersize=8)
ax.fill(angles, chromfound_norm_closed, alpha=0.2, color='#ff7f0e')

# Add value annotations showing both values
epiagent_values_display = epiagent_values
chromfound_values_display = chromfound_values
angles_display = angles[:-1]
epiagent_norm_display = epiagent_norm
chromfound_norm_display = chromfound_norm

for angle, epi_norm, epi_val, chrom_norm, chrom_val in zip(angles_display, epiagent_norm_display, epiagent_values_display, chromfound_norm_display, chromfound_values_display):
    # Position annotations dynamically based on which value is larger
    # The method with the larger value gets the outer position (further from center)
    if chrom_val > epi_val:
        # ChromFound is larger, so it goes outer
        chrom_text_radius = max(chrom_norm, epi_norm) + 0.12
        epi_text_radius = min(chrom_norm, epi_norm) - 0.12
    else:
        # EpiAgent is larger or equal, so it goes outer
        epi_text_radius = max(chrom_norm, epi_norm) + 0.12
        chrom_text_radius = min(chrom_norm, epi_norm) - 0.12
    
    # Adjust if they would go out of bounds
    if epi_text_radius > 1.0:
        epi_text_radius = max(chrom_norm, epi_norm) - 0.15
    if chrom_text_radius < 0:
        chrom_text_radius = min(chrom_norm, epi_norm) + 0.15
    if epi_text_radius < 0:
        epi_text_radius = max(chrom_norm, epi_norm) - 0.15
    if chrom_text_radius > 1.0:
        chrom_text_radius = min(chrom_norm, epi_norm) + 0.15
    
    # EpiAgent value annotation
    ax.text(angle, epi_text_radius, f'E:{epi_val:.3f}', 
            ha='center', va='center', fontsize=16, 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1f77b4', alpha=0.7, edgecolor='none'),
            color='white', fontweight='bold')
    
    # ChromFound value annotation
    ax.text(angle, chrom_text_radius, f'C:{chrom_val:.3f}', 
            ha='center', va='center', fontsize=16, 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ff7f0e', alpha=0.7, edgecolor='none'),
            color='white', fontweight='bold')


ax.set_xticklabels([]) # Remove angle degree labels (keep grid lines but hide labels)
ax.set_xticks(angles)
# ax.set_xticklabels(labels_closed)
# ax.tick_params(pad=30)  # Move xtick labels outward
# ax.set_title('EpiAgent vs ChromFound Comparison', pad=20, fontsize=14, fontweight='bold')
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig("radar_plot_comparison.png", dpi=300, bbox_inches='tight')
print("\nComparison radar plot saved to radar_plot_comparison.png")
plt.show()

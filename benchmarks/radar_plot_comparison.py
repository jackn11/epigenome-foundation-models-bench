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
    # Load feature extraction benchmark results
    feature_extraction_datasets = ["Kanemaru2023_10000", "Li2023b_10000"]
    epiagent_root = Path("../EpiAgent/benchmarks")
    feature_extraction_result_csvs = {dataset: epiagent_root / f"zero_shot_feature_extraction_{dataset}/results.csv" for dataset in feature_extraction_datasets}

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
    perturbation_cohens_d_paths = {dataset: epiagent_root / f"zero_shot_perturbation_effect_prediction_{dataset}/model_score.txt" for dataset in perturbation_datasets}
    perturbation_plausibility_paths = {dataset: epiagent_root / f"zero_shot_perturbation_effect_prediction_{dataset}/biological_plausibility_score.txt" for dataset in perturbation_datasets}
    perturbation_results = {}
    for dataset, cohens_d_path in perturbation_cohens_d_paths.items():
        plausibility_path = perturbation_plausibility_paths[dataset]
        if cohens_d_path.exists() and plausibility_path.exists():
            with open(cohens_d_path, 'r') as file:
                cohens_d = file.read().strip()
                match = re.search(r"Weighted mean Cohen's d[^\d\-]*([-+]?\d*\.\d+|\d+)", cohens_d)
                cohens_d = float(match.group(1)) 
            with open(plausibility_path, 'r') as file:
                biological_plausibility = file.read().strip()
                match = re.search(r"Spearman correlation[^\d\-]*([-+]?\d*\.\d+|\d+)", biological_plausibility)
                biological_plausibility = -float(match.group(1))
            perturbation_results[dataset] = {
                "cohens_d": cohens_d,
                "biological_plausibility": biological_plausibility
            }
        else:
            print(f"Warning: EpiAgent perturbation results not found for {dataset}")

    return feature_extraction_results, perturbation_results, feature_extraction_datasets, perturbation_datasets


def load_chromfound_results():
    """Load ChromFound benchmark results"""
    # Load feature extraction benchmark results
    feature_extraction_datasets = ["Kanemaru2023_downsampled", "Li2023b_downsampled"]
    chromfound_root = Path("../ChromFound-Parallel")
    feature_extraction_result_csvs = {dataset: chromfound_root / f"zero_shot_feature_extraction_chromfound_{dataset}/results.csv" for dataset in feature_extraction_datasets}

    feature_extraction_results = {}
    for dataset, csv_path in feature_extraction_result_csvs.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            metrics_dict = dict(zip(df['Metric'], df['Value']))
            feature_extraction_results[dataset] = metrics_dict
        else:
            print(f"Warning: ChromFound results file not found for {dataset}: {csv_path}")

    # Load perturbation benchmark results
    perturbation_datasets = ["Liscovitch_Brauer2021", "Pierce2021"]
    perturbation_cohens_d_paths = {dataset: chromfound_root / f"zero_shot_perturbation_effect_prediction_chromfound_outputs/{dataset}/model_score.txt" for dataset in perturbation_datasets}
    perturbation_plausibility_paths = {dataset: chromfound_root / f"zero_shot_perturbation_effect_prediction_chromfound_outputs/{dataset}/biological_plausibility_score.txt" for dataset in perturbation_datasets}
    perturbation_results = {}
    for dataset, cohens_d_path in perturbation_cohens_d_paths.items():
        plausibility_path = perturbation_plausibility_paths[dataset]
        if cohens_d_path.exists() and plausibility_path.exists():
            with open(cohens_d_path, 'r') as file:
                cohens_d = file.read().strip()
                match = re.search(r"Weighted mean Cohen's d[^\d\-]*([-+]?\d*\.\d+|\d+)", cohens_d)
                cohens_d = float(match.group(1)) 
            with open(plausibility_path, 'r') as file:
                biological_plausibility = file.read().strip()
                match = re.search(r"Spearman correlation[^\d\-]*([-+]?\d*\.\d+|\d+)", biological_plausibility)
                biological_plausibility = -float(match.group(1))
            perturbation_results[dataset] = {
                "cohens_d": cohens_d,
                "biological_plausibility": biological_plausibility
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

    # Aggregate perturbation metrics
    cohens_d_values = []
    for dataset in perturbation_datasets:
        if dataset in perturbation_results:
            cohens_d_values.append(perturbation_results[dataset]['cohens_d'])
    cohens_d = np.array(cohens_d_values).mean() if len(cohens_d_values) > 0 else 0

    biological_plausibility_values = []
    for dataset in perturbation_datasets:
        if dataset in perturbation_results:
            biological_plausibility_values.append(perturbation_results[dataset]['biological_plausibility'])
    biological_plausibility = np.array(biological_plausibility_values).mean() if len(biological_plausibility_values) > 0 else 0

    return {
        'nmi': nmi,
        'ari': ari,
        'silhouette': silhouette,
        'silhouette_batch': silhouette_batch,
        'cell_type_linear_probe_f1': cell_type_linear_probe_f1,
        'batch_linear_probe_f1': batch_linear_probe_f1,
        'cohens_d': cohens_d,
        'biological_plausibility': biological_plausibility
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
    "Silhouette_batch",
    "Cell type\n linear probe (F1)",
    "Batch label\n linear probe (1-F1)",
    "Perturbation Effect Captured",
    "Perturbation Effect\n Biological Plausibility",
]

mins = np.array([0, 0, 0, 0, 0, 0, 0, 0])
maxs = np.array([1, 1, 1, 1, 1, 1, 0.2, 0.4])

# EpiAgent values
epiagent_values = np.array([
    epiagent_metrics['nmi'],
    epiagent_metrics['ari'],
    epiagent_metrics['silhouette'],
    epiagent_metrics['silhouette_batch'],
    epiagent_metrics['cell_type_linear_probe_f1'],
    1 - epiagent_metrics['batch_linear_probe_f1'],
    epiagent_metrics['cohens_d'],
    epiagent_metrics['biological_plausibility']
])

# ChromFound values
chromfound_values = np.array([
    chromfound_metrics['nmi'],
    chromfound_metrics['ari'],
    chromfound_metrics['silhouette'],
    chromfound_metrics['silhouette_batch'],
    chromfound_metrics['cell_type_linear_probe_f1'],
    1 - chromfound_metrics['batch_linear_probe_f1'],
    chromfound_metrics['cohens_d'],
    chromfound_metrics['biological_plausibility']
])

# Normalize values
epiagent_norm = (epiagent_values - mins) / (maxs - mins)
chromfound_norm = (chromfound_values - mins) / (maxs - mins)

# Close the loop
labels_closed = labels + [labels[0]]
epiagent_norm_closed = np.append(epiagent_norm, epiagent_norm[0])
chromfound_norm_closed = np.append(chromfound_norm, chromfound_norm[0])

angles = np.linspace(0, 2 * np.pi, len(labels_closed))

#%%
# Create figure with single subplot for overlay comparison
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

ax.set_ylim(0, 1)
ax.set_rlabel_position(0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([""] * 5)
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
    # Position annotations to avoid overlap
    # EpiAgent annotation above, ChromFound below
    epi_text_radius = epi_norm + 0.12
    chrom_text_radius = chrom_norm - 0.12
    
    # Adjust if they would go out of bounds
    if epi_text_radius > 1.0:
        epi_text_radius = epi_norm - 0.15
    if chrom_text_radius < 0:
        chrom_text_radius = chrom_norm + 0.15
    
    # EpiAgent value annotation
    ax.text(angle, epi_text_radius, f'E:{epi_val:.3f}', 
            ha='center', va='center', fontsize=7, 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1f77b4', alpha=0.7, edgecolor='none'),
            color='white', fontweight='bold')
    
    # ChromFound value annotation
    ax.text(angle, chrom_text_radius, f'C:{chrom_val:.3f}', 
            ha='center', va='center', fontsize=7, 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ff7f0e', alpha=0.7, edgecolor='none'),
            color='white', fontweight='bold')

ax.set_xticks(angles)
ax.set_xticklabels(labels_closed)
ax.set_title('EpiAgent vs ChromFound Comparison', pad=20, fontsize=14, fontweight='bold')

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig("radar_plot_comparison.png", dpi=300, bbox_inches='tight')
print("\nComparison radar plot saved to radar_plot_comparison.png")
plt.show()

#%%
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

#%%
# Load metrics
feature_extraction_datasets = ["Buenrostro2018", "Kanemaru2023", "Li2023b"]
feature_extraction_result_csvs = {dataset:f"zero_shot_feature_extraction_{dataset}/results.csv" for dataset in feature_extraction_datasets}

feature_extraction_results = {}
for dataset, csv_path in feature_extraction_result_csvs.items():
    df = pd.read_csv(csv_path)
    metrics_dict = dict(zip(df['Metric'], df['Value']))
    feature_extraction_results[dataset] = metrics_dict

perturbation_datasets = ["Liscovitch_Brauer2021", "Pierce2021"]
perturbation_cohens_d_paths = {dataset:f"zero_shot_perturbation_effect_prediction_{dataset}/model_score.txt" for dataset in perturbation_datasets}
perturbation_plausibility_paths = {dataset:f"zero_shot_perturbation_effect_prediction_{dataset}/biological_plausibility_score.txt" for dataset in perturbation_datasets}
perturbation_results = {}
for dataset, cohens_d_path in perturbation_cohens_d_paths.items():
    with open(cohens_d_path, 'r') as file:
        cohens_d = file.read().strip()
        match = re.search(r"Weighted mean Cohen's d[^\d\-]*([-+]?\d*\.\d+|\d+)", cohens_d) # first floating point number after "Weighted mean Cohen's d"
        cohens_d = float(match.group(1)) 
    with open(perturbation_plausibility_paths[dataset], 'r') as file:
        biological_plausibility = file.read().strip()
        match = re.search(r"Spearman correlation[^\d\-]*([-+]?\d*\.\d+|\d+)", biological_plausibility)
        biological_plausibility = -float(match.group(1)) 
    perturbation_results[dataset] = {
        "cohens_d": cohens_d,
        "biological_plausibility": biological_plausibility
    }

print(feature_extraction_results)
print(perturbation_results)

#%%
# aggregate feature extraction metrics (NMI, ARI, Silhouette, Silhouette batch)
nmi_values = []
for dataset in feature_extraction_datasets:
    nmi_values.append(feature_extraction_results[dataset]['Normalized Mutual Information'])
nmi_values = np.array(nmi_values)
nmi = nmi_values.mean()

ari_values = []
for dataset in feature_extraction_datasets:
    ari_values.append(feature_extraction_results[dataset]['Adjusted Rand Index'])
ari_values = np.array(ari_values)
ari = ari_values.mean()

silhouette_values = []
for dataset in feature_extraction_datasets:
    silhouette_values.append(feature_extraction_results[dataset]['Silhouette score'])
silhouette_values = np.array(silhouette_values)
silhouette = silhouette_values.mean()

silhouette_batch_values = []
for dataset in feature_extraction_datasets:
    silhouette_batch_values.append(feature_extraction_results[dataset]['Silhouette batch score'])
silhouette_batch_values = np.array(silhouette_batch_values)
silhouette_batch = silhouette_batch_values.mean()

print(nmi, ari, silhouette, silhouette_batch)

#aggregate perturbation metrics (Cohen's D, Biological Plausibility)
cohens_d_values = []
for dataset in perturbation_datasets:
    cohens_d_values.append(perturbation_results[dataset]['cohens_d'])
cohens_d_values = np.array(cohens_d_values)
cohens_d = cohens_d_values.mean()

biological_plausibility_values = []
for dataset in perturbation_datasets:
    biological_plausibility_values.append(perturbation_results[dataset]['biological_plausibility'])
biological_plausibility_values = np.array(biological_plausibility_values)
biological_plausibility = biological_plausibility_values.mean()

print(cohens_d, biological_plausibility)

#%%
labels = [
    "NMI",
    "ARI",
    "Silhouette",
    "Silhouette_batch",
    "Perturbation Effect Captured",
    "Perturbation Effect\n Biological Plausibility",
]

values = np.array([nmi, ari, silhouette, silhouette_batch, cohens_d, biological_plausibility])
print("values: ", values)
mins   = np.array([0, 0, 0, 0, 0, 0])
maxs   = np.array([1, 1, 1, 1, 0.4, 0.4])

# 2. Normalize into [0, 1]
norm = (values - mins) / (maxs - mins)
print("norm: ", norm)

# Close the loop
labels = labels + [labels[0]]
norm   = np.append(norm, norm[0])

angles = np.linspace(0, 2 * np.pi, len(labels))

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

ax.set_ylim(0, 1)

# Remove radial number labels but keep rings
ax.set_rlabel_position(0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([""] * 5)
ax.yaxis.set_tick_params(labelleft=False, labelright=False)

ax.plot(angles, norm, marker='o')
ax.fill(angles, norm, alpha=0.25)

# Use your text labels instead of degree numbers
ax.set_xticks(angles)
ax.set_xticklabels(labels)
# ax.set_xticklabels([])

# --- REMOVE this whole block:
# for angle, vmin, vmax in zip(angles, mins, maxs):
#     ax.text(angle, 0.02, f"{vmin:g}", ha='center', va='center', fontsize=9)
#     ax.text(angle, 1.02, f"{vmax:g}", ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("radar_plot.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# data from wandb
data = {
    "Kanemaru2023": {
        "no-permutation": {
            "NMI": 0.7440606717895658,
            "ARI": 0.6287278266091394,
            "ASW": 0.5720129758119583,
            "ASW_batch": 0.8848167794239891
        },
        "permuted-labels": {
            "NMI": 0.7473509503305112,
            "ARI": 0.622594077606266,
            "ASW": 0.5747807323932648,
            "ASW_batch": 0.8825808285029856
        },
        "complete-shuffling": {
            "NMI": 0.045462252551391635,
            "ARI": 0.013802392049131385,
            "ASW": 0.46779487654566765,
            "ASW_batch": 0.8777853068956452
        }
    },

    "Li2023b": {
        "no-permutation": {
            "NMI": 0.61852051990074,
            "ARI": 0.30711020759537333,
            "ASW": 0.589332640171051,
            "ASW_batch": 0.8698944464044077 
        },
        "permuted-labels": {
            "NMI": 0.6220005627584155,
            "ARI": 0.3233572426701724,
            "ASW": 0.5885151624679565,
            "ASW_batch": 0.8677245744356454
        },
        "complete-shuffling": {
            "NMI": 0.07468240308888015,
            "ARI": 0.04212687088502028,
            "ASW": 0.4761756993830204,
            "ASW_batch": 0.8535288950848378
        }
    },

    "Buenrostro2018": {
        "no-permutation": {
            "NMI": 0.6010202424725555,
            "ARI": 0.40715040641633826,
            "ASW": 0.5222255270928144,
            "ASW_batch": 0.84766119786269
        },
        "permuted-labels": {
            "NMI": 0.6046483696255189,
            "ARI": 0.44075564068333684,
            "ASW": 0.5239878073334694,
            "ASW_batch": 0.8517856527976342
        },
        "complete-shuffling": {
            "NMI": 0.05061882950389105,
            "ARI": 0.023302843662206956,
            "ASW": 0.42081795632839203,
            "ASW_batch": 0.8835116034631877
        }
    }
}

# plot the data

datasets = list(data.keys())
conditions = ["no-permutation", "permuted-labels", "complete-shuffling"]
x = np.arange(len(datasets))
width = 0.25  # Width of bars

# Colors for conditions
colors = {
    "no-permutation": "#2E86AB",      # Blue
    "permuted-labels": "#A23B72",     # Purple
    "complete-shuffling": "#F18F01"   # Orange
}

# Plot NMI - separate figure
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i, condition in enumerate(conditions):
    nmi_values = [data[dataset][condition]["NMI"] for dataset in datasets]
    ax1.bar(x + i*width, nmi_values, width, label=condition.replace("-", " ").title(), 
            color=colors[condition], alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Dataset', fontsize=11, fontweight='bold')
ax1.set_ylabel('Normalized Mutual Information (NMI)', fontsize=11, fontweight='bold')
ax1.set_title('NMI Comparison Across Datasets and Conditions', fontsize=12, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(datasets, rotation=0)
ax1.set_ylim([0, 1.0])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('nmi_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot ARI - separate figure
fig2, ax2 = plt.subplots(figsize=(8, 6))
for i, condition in enumerate(conditions):
    ari_values = [data[dataset][condition]["ARI"] for dataset in datasets]
    ax2.bar(x + i*width, ari_values, width, label=condition.replace("-", " ").title(), 
            color=colors[condition], alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Dataset', fontsize=11, fontweight='bold')
ax2.set_ylabel('Adjusted Rand Index (ARI)', fontsize=11, fontweight='bold')
ax2.set_title('ARI Comparison Across Datasets and Conditions', fontsize=12, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(datasets, rotation=0)
ax2.set_ylim([0, 1.0])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('ari_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional: Create a summary table visualization
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Dataset':<20} {'Condition':<20} {'NMI':<10} {'ARI':<10}")
print("-"*80)
for dataset in datasets:
    for condition in conditions:
        nmi = data[dataset][condition]["NMI"]
        ari = data[dataset][condition]["ARI"]
        print(f"{dataset:<20} {condition.replace('-', ' ').title():<20} {nmi:<10.4f} {ari:<10.4f}")
print("="*80)
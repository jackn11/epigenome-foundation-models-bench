"""
Helper script to run ChromFound perturbation analysis on ALL genes found in the h5ad file.
"""
import scanpy as sc
import numpy as np
import subprocess
import sys

# Path to the h5ad file
h5ad_path = "/scratch/wkim/project-2-team-1/benchmarks/results/chromfound/Pierce2021/merge10/embeddings_pca.h5ad"

print(f"Loading {h5ad_path} to extract gene list...")
adata = sc.read_h5ad(h5ad_path)

# Get all unique perturbations
all_perturbations = adata.obs['celltype'].unique()

# Filter out UNK and sgsgNT (control)
genes_to_analyze = [p for p in all_perturbations if p not in ['UNK', 'sgsgNT']]

print(f"\nFound {len(genes_to_analyze)} genes to analyze:")
print(genes_to_analyze)

# Build command
cmd = [
    sys.executable,
    "zero_shot_perturbation_effect_prediction_chromfound.py",
    "--h5ad_path", h5ad_path,
    "--genes_of_interest"
] + genes_to_analyze

print(f"\nRunning analysis for all {len(genes_to_analyze)} genes...")
print("Command:", " ".join(cmd))

# Run the analysis
subprocess.run(cmd, check=True)


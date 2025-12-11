import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path

DATA_DIR = Path("/scratch/wkim/project-2-team-1/ChromFound-Parallel/data/sample/genetic_perturbation_data/Liscovitch_Brauer2021")
INPUT_H5AD = DATA_DIR / "tfidf_data.h5ad"
OUT_H5AD = DATA_DIR / "Liscovitch_Brauer2021_for_chromfound.h5ad"

print("Loading existing h5ad...")
adata = sc.read_h5ad(INPUT_H5AD)
print(f"Loaded adata with shape: {adata.shape}")
print(f"Obs columns: {list(adata.obs.columns)}")
print(f"Var columns: {list(adata.var.columns)}")

# ---- Parse var.index to create genomic coordinate columns ----
print("\nParsing genomic coordinates from var.index...")
chromosomes = []
starts = []
ends = []

for peak_id in adata.var.index:
    # Expected format: chr1:9848-10355
    try:
        chrom_part, pos_part = peak_id.split(':')
        start, end = pos_part.split('-')
        chromosomes.append(chrom_part)
        starts.append(int(start))
        ends.append(int(end))
    except Exception as e:
        print(f"Warning: Could not parse peak ID '{peak_id}': {e}")
        chromosomes.append('chrUnknown')
        starts.append(0)
        ends.append(0)

# Add columns to var
adata.var['#Chromosome'] = chromosomes
adata.var['hg38_Start'] = starts
adata.var['hg38_End'] = ends

print(f"Successfully parsed {len(adata.var)} peak coordinates")
print(f"  Unique chromosomes: {adata.var['#Chromosome'].nunique()}")
print(f"  Example chromosomes: {adata.var['#Chromosome'].value_counts().head(5).to_dict()}")

# ---- Ensure cell_type column exists in obs ----
print("\nProcessing obs metadata...")
if 'perturbation' in adata.obs.columns:
    # Use perturbation as cell_type
    adata.obs['cell_type'] = adata.obs['perturbation'].astype(str)
    print(f"  Created 'cell_type' column from 'perturbation'")
    print(f"  Number of unique perturbations: {adata.obs['cell_type'].nunique()}")
    print(f"  Example perturbations (top 10):")
    print(adata.obs['cell_type'].value_counts().head(10))
elif 'cell_type' not in adata.obs.columns:
    print("Warning: No 'perturbation' or 'cell_type' column found. Creating dummy cell_type.")
    adata.obs['cell_type'] = 'unknown'

# ---- Verify the data is ready for ChromFound ----
print("\nVerifying ChromFound requirements...")
required_var_cols = ['#Chromosome', 'hg38_Start', 'hg38_End']
missing_var_cols = [col for col in required_var_cols if col not in adata.var.columns]
if missing_var_cols:
    raise ValueError(f"Missing required var columns: {missing_var_cols}")
else:
    print(f"  ✓ All required var columns present: {required_var_cols}")

if 'cell_type' not in adata.obs.columns:
    raise ValueError("Missing required obs column: cell_type")
else:
    print(f"  ✓ Required obs column 'cell_type' present")

# ---- Save the formatted h5ad ----
print(f"\nSaving ChromFound-formatted h5ad to: {OUT_H5AD}")
adata.write_h5ad(OUT_H5AD)
print("✓ Successfully created ChromFound-compatible h5ad file!")

print(f"\nYou can now run the benchmark with:")
print(f"python benchmarks/scripts/run_chromfound_benchmark_notebook_pipeline_pca.py \\")
print(f"  --dataset_name Liscovitch_Brauer2021 \\")
print(f"  --dataset_path {OUT_H5AD}")

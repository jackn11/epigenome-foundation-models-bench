"""
ChromFound Benchmarking Script

This script runs the full ChromFound pipeline on the given dataset:
1. Converts cCRE identifiers to genomic coordinates
2. Preprocesses data (QC, normalization, log transform)
3. Runs ChromFound inference to generate embeddings
4. Applies PCA and K-means clustering (matching notebook approach)
5. Calculates ARI and NMI metrics
6. Saves results to benchmarks/results/chromfound/

Run this in the chromfound conda environment:
    conda activate chromfound
    python benchmarks/scripts/run_chromfound_benchmark_notebook_pipeline.py
"""
import os
import sys
import json
import scanpy as sc
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.cluster import KMeans
from scib.metrics import silhouette, silhouette_batch
from scipy.sparse import csr_matrix, vstack
from joblib import Parallel, delayed
import multiprocessing as mp
import wandb

# Conversion function is defined below

# Import ChromFound preprocessing functions
# Assuming we're running from project root, adjust path as needed
_project_root = Path(__file__).parent.parent.parent
_chromfound_path = _project_root / "ChromFound-Parallel"
sys.path.insert(0, str(_chromfound_path))
from src.data.atac_preprocess import deepen_atac_data


dataset_name = "Kanemaru2023_downsampled"
dataset_path = "data2/Kanemaru2023/Kanemaru2023-downsampled/Kanemaru2023_downsampled_10000_cells.h5ad"



def convert_ccre_to_genomic_coords(adata, inplace=True):
    """
    Convert cCRE identifiers in var.index to genomic coordinates.
    
    Parses identifiers in format "chr1:9848-10355" and adds:
    - #Chromosome: chromosome name (e.g., "chr1")
    - hg38_Start: start position as integer
    - hg38_End: end position as integer
    
    The function is idempotent - it only adds columns if they don't already exist.
    
    Args:
        adata: AnnData object with cCRE identifiers in var.index
        inplace: If True, modify adata in place. If False, return a copy.
    
    Returns:
        AnnData object with genomic coordinate columns added (or original if inplace=True)
    """
    import re
    
    if not inplace:
        adata = adata.copy()
    
    # Check if columns already exist
    required_cols = ['#Chromosome', 'hg38_Start', 'hg38_End']
    if all(col in adata.var.columns for col in required_cols):
        print("Genomic coordinate columns already exist. Skipping conversion.")
        return adata
    
    print("Converting cCRE identifiers to genomic coordinates...")
    
    # Parse cCRE identifiers from var.index
    # Format: "chr1:9848-10355" -> chromosome="chr1", start=9848, end=10355
    pattern = re.compile(r'^(chr[XY\d]+):(\d+)-(\d+)$')
    
    chromosomes = []
    starts = []
    ends = []
    
    failed_parses = []
    
    for ccre_id in adata.var.index:
        match = pattern.match(ccre_id)
        if match:
            chrom = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            chromosomes.append(chrom)
            starts.append(start)
            ends.append(end)
        else:
            failed_parses.append(ccre_id)
            # Default values for failed parses (shouldn't happen with proper cCRE format)
            chromosomes.append("chr1")
            starts.append(0)
            ends.append(0)
    
    if failed_parses:
        print(f"Warning: Failed to parse {len(failed_parses)} cCRE identifiers")
        if len(failed_parses) <= 10:
            print(f"Failed IDs: {failed_parses}")
        else:
            print(f"First 10 failed IDs: {failed_parses[:10]}")
    
    # Add columns to var
    # Use index alignment to ensure proper assignment
    adata.var['#Chromosome'] = pd.Series(chromosomes, index=adata.var.index)
    adata.var['hg38_Start'] = pd.Series(starts, dtype=int, index=adata.var.index)
    adata.var['hg38_End'] = pd.Series(ends, dtype=int, index=adata.var.index)
    
    print(f"Successfully converted {len(adata.var)} cCRE identifiers to genomic coordinates")
    print(f"  Chromosomes: {pd.Series(chromosomes).value_counts().head(5).to_dict()}")
    print(f"  Start range: {min(starts)} - {max(starts)}")
    print(f"  End range: {min(ends)} - {max(ends)}")
    
    return adata


def setup_paths(num_cell_merge=1):
    """Set up all file paths for the benchmark.
    
    Args:
        num_cell_merge: Number of cells to merge (1 or 10). Used in file naming.
    
    Returns:
        Dictionary of file paths
    """
    # Get project root (assuming script is in benchmarks/scripts/)

    
    project_root = Path(__file__).parent.parent.parent
    benchmarks_dir = project_root / "benchmarks"
    
    # Create merge-specific directory and file names
    merge_suffix = f"merge{num_cell_merge}"
    results_subdir = benchmarks_dir / "results" / "chromfound" / dataset_name / merge_suffix
    
    paths = {
        "project_root": project_root,
        "benchmarks_dir": benchmarks_dir,
        "results_dir": results_subdir,
        "input_data": benchmarks_dir / dataset_path,
        "preprocessed_data": results_subdir / f"{dataset_name}_preprocessed_{merge_suffix}.h5ad",
        "embeddings": results_subdir / "embeddings_pca.h5ad",  # Memory-efficient PCA version
        "metrics": results_subdir / "metrics.json",
        "kmeans_labels": results_subdir / "kmeans_labels.h5ad",
    }
    
    # Create all necessary directories
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["preprocessed_data"].parent.mkdir(parents=True, exist_ok=True)
    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["kmeans_labels"].parent.mkdir(parents=True, exist_ok=True)
    
    return paths


def preprocess_data(paths, data_args):
    """
    Preprocess the data: convert cCREs to genomic coordinates, verify columns.
    
    Note: Quality control is skipped as benchmark datasets are already preprocessed.
    
    Args:
        paths: Dictionary of file paths
        data_args: Dictionary of preprocessing parameters
    
    Returns:
        Path to preprocessed data file
    """
    print("\n" + "=" * 80)
    print(f"STEP 1: Loading and Preprocessing Data")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load data
    print(f"Loading data from: {paths['input_data']}")
    adata = sc.read_h5ad(paths['input_data'])
    print(f"Original shape: {adata.shape}")
    
    # Convert cCRE identifiers to genomic coordinates (idempotent)
    print("\nConverting cCRE identifiers to genomic coordinates...")
    adata = convert_ccre_to_genomic_coords(adata, inplace=True)
    
    # Ensure coordinate columns are integers (only convert if not already int)
    if 'hg38_Start' in adata.var.columns:
        if adata.var['hg38_Start'].dtype != 'int64' and adata.var['hg38_Start'].dtype != 'int32':
            # Check for NaN values before converting
            if adata.var['hg38_Start'].isna().any():
                print("Warning: hg38_Start contains NaN values. Filling with 0.")
                adata.var['hg38_Start'] = adata.var['hg38_Start'].fillna(0)
            adata.var['hg38_Start'] = adata.var['hg38_Start'].astype(int)
    if 'hg38_End' in adata.var.columns:
        if adata.var['hg38_End'].dtype != 'int64' and adata.var['hg38_End'].dtype != 'int32':
            # Check for NaN values before converting
            if adata.var['hg38_End'].isna().any():
                print("Warning: hg38_End contains NaN values. Filling with 0.")
                adata.var['hg38_End'] = adata.var['hg38_End'].fillna(0)
            adata.var['hg38_End'] = adata.var['hg38_End'].astype(int)
    
    # Verify required columns exist for ChromFound
    required_cols = ['#Chromosome', 'hg38_Start', 'hg38_End']
    missing_cols = [col for col in required_cols if col not in adata.var.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in adata.var: {missing_cols}")
    print(f"\n✓ Required columns verified: {required_cols}")
    print(f"  Data shape: {adata.shape}")
    
    # Skip cell aggregation (num_cell_merge=1) to match EpiAgent's single-cell approach
    # Or set to 1 to keep single cells
    num_cell_merge = data_args.get("num_cell_merge", 1)
    if num_cell_merge > 1:
        print(f"\nPerforming cell aggregation (num_cell_merge={num_cell_merge})...")
        adata = deepen_atac_data(adata, num_cell_merge=num_cell_merge)
        print(f"After cell aggregation: {adata.shape}")
    else:
        print("\nSkipping cell aggregation (num_cell_merge=1, single-cell mode)")
    
    # Normalize and log transform
    print("\nNormalizing and log transforming...")
    print('  Normalizing...')
    sc.pp.normalize_total(adata)
    print('  Log transforming...')
    sc.pp.log1p(adata)
    print("  ✓ Normalization and log transform complete")
    
    # Add 'celltype' column for ChromFound compatibility (ChromFound code hardcodes 'celltype')
    cell_type_col = data_args.get("cell_type_col", "cell_type")
    if 'celltype' not in adata.obs.columns and cell_type_col in adata.obs.columns:
        adata.obs['celltype'] = adata.obs[cell_type_col]
        print(f"\nAdded 'celltype' column (copied from '{cell_type_col}') for ChromFound compatibility")
    
    # Save preprocessed data (use compression for speed/size tradeoff)
    print(f"\nSaving preprocessed data to: {paths['preprocessed_data']}")
    # Use compression level 1 for faster writes (default is 4)
    adata.write_h5ad(paths['preprocessed_data'], compression='gzip', compression_opts=1)
    
    print(f"\n✓ STEP 1 COMPLETE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Preprocessed data saved: {paths['preprocessed_data']}")
    
    return paths['preprocessed_data']


def run_inference(paths, inference_config, data_args):
    """
    Run ChromFound inference to generate cell embeddings.
    
    Args:
        paths: Dictionary of file paths
        inference_config: Dictionary of inference parameters
        data_args: Dictionary of data parameters
    
    Returns:
        Path to embeddings file
    """
    print("\n" + "=" * 80)
    print(f"STEP 2: Running ChromFound Inference")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Build inference command
    # Use memory-efficient sampled PCA version for large datasets
    n_pca_components = inference_config.get("n_pca_components", 50)
    n_samples_for_pca = inference_config.get("n_samples_for_pca", 1000)  # Fit PCA on 1000 random cells
    
    inference_command = [
        sys.executable, '-m', 'src.cell_embedding_pca',
        '--local_rank', str(inference_config["device"]),
        '--data_path', str(paths['preprocessed_data']),
        '--output_path', str(paths['results_dir']),
        '--pretrain_checkpoint_path', str(inference_config["pretrain_checkpoint_path"]),
        '--pretrain_model_file', inference_config["pretrain_model_name"],
        '--pretrain_config_file', inference_config["pretrain_config_file"],
        '--batch_size', str(inference_config["batch_size"]),
        '--cell_type_col', data_args["cell_type_col"],
        '--n_pca_components', str(n_pca_components),
        '--n_samples_for_pca', str(n_samples_for_pca)  # Fast: fit PCA on sample, project all
    ]
    
    print(f"\nRunning inference command...")
    print(f"  Device: GPU {inference_config['device']}")
    print(f"  Batch size: {inference_config['batch_size']}")
    print(f"  Model: {inference_config['pretrain_model_name']}")
    print(f"  Input: {paths['preprocessed_data']}")
    print(f"  Output: {paths['results_dir']}")
    print(f"\nNote: This step may take several minutes. Progress will be shown below...")
    print("-" * 80)
    
    # Change to ChromFound directory for inference
    original_cwd = os.getcwd()
    _chromfound_path = Path(__file__).parent.parent.parent / "ChromFound-Parallel"
    
    # Set environment variable to help with CUDA memory fragmentation
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        os.chdir(str(_chromfound_path))
        result = subprocess.run(inference_command, check=True, env=env)
    finally:
        os.chdir(original_cwd)
    
    # Memory-efficient version saves PCA-reduced embeddings as embeddings_pca.h5ad
    embeddings_path = paths['results_dir'] / "embeddings_pca.h5ad"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    print("-" * 80)
    print(f"\n✓ STEP 2 COMPLETE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Embeddings saved to: {embeddings_path}")
    return embeddings_path


def cluster_and_evaluate(paths, num_cell_merge=1):
    """
    Apply PCA, K-means clustering, and calculate metrics.
    
    Uses the same approach as the notebook: PCA -> neighbors -> K-means clustering.
    
    Args:
        paths: Dictionary of file paths
        num_cell_merge: Number of cells merged (for logging)
    
    Returns:
        Dictionary with metrics (ARI, NMI, etc.)
    """
    print("\n" + "=" * 80)
    print(f"STEP 3: Clustering and Evaluation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load PCA-reduced embeddings (memory-efficient version)
    print(f"\nLoading PCA-reduced embeddings from: {paths['embeddings']}")
    adata = sc.read_h5ad(paths['embeddings'])
    
    # Check if embeddings are already PCA-reduced (from cell_embedding_pca.py)
    if 'X_pca' in adata.obsm:
        print(f"Found pre-computed PCA embeddings: {adata.obsm['X_pca'].shape}")
        embeddings = adata.obsm['X_pca']
        pca_already_done = True
    elif 'X_embedding' in adata.obsm:
        embeddings = adata.obsm['X_embedding']
        print(f"Found raw embeddings: {embeddings.shape}")
        pca_already_done = False
    else:
        raise KeyError("Neither X_pca nor X_embedding found in adata.obsm")
    
    # Get cell type column name (prefer 'cell_type' to match EpiAgent, but fallback to 'celltype')
    cell_type_col = None
    for col in ['cell_type', 'celltype']:
        if col in adata.obs.columns:
            cell_type_col = col
            break
    
    if cell_type_col is None:
        raise KeyError("Cell type column not found in adata.obs")
    
    # Create new AnnData with embeddings
    obs_df = pd.DataFrame(adata.obs[cell_type_col].tolist(), columns=[cell_type_col])
    adata_emb = sc.AnnData(X=embeddings, obs=obs_df)
    
    # Apply PCA only if not already done
    if pca_already_done:
        print("\n[3.1] Using pre-computed PCA embeddings (memory-efficient mode)")
        n_pcs = embeddings.shape[1]
        adata_emb.obsm['X_pca'] = embeddings
        print(f"  ✓ PCA already computed: {n_pcs} principal components")
    else:
        print("\n[3.1] Applying PCA (reducing dimensionality)...")
        n_pcs = 10
        sc.tl.pca(adata_emb, n_comps=n_pcs)
        print(f"  ✓ PCA complete: {n_pcs} principal components")
    
    # Compute neighbors graph (using PCA-reduced embeddings) - for UMAP visualization
    print("\n[3.2] Computing neighbors graph...")
    sc.pp.neighbors(adata_emb, n_neighbors=15, use_rep='X_pca')
    print("  ✓ Neighbors graph computed")
    
    # Compute UMAP for visualization
    print("\n[3.3] Computing UMAP...")
    sc.tl.umap(adata_emb)
    print("  ✓ UMAP computed")
    
    # K-means clustering (matching notebook approach exactly)
    true_labels = adata_emb.obs[cell_type_col].values
    num_cell_types = len(np.unique(true_labels))
    
    print(f"\n[3.4] Performing K-means clustering (n_clusters={num_cell_types})...")
    kmeans = KMeans(n_clusters=num_cell_types, random_state=23).fit(adata_emb.obsm['X_pca'])
    predicted_labels = kmeans.labels_
    adata_emb.obs['kmeans'] = predicted_labels.astype(str)
    
    n_clusters = len(np.unique(predicted_labels))
    print(f"Number of K-means clusters: {n_clusters}")
    
    # Calculate metrics (exactly as in notebook)
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
    ami_score = adjusted_mutual_info_score(true_labels, predicted_labels)
    fmi_score = fowlkes_mallows_score(true_labels, predicted_labels)
    
    # Calculate silhouette scores (same as EpiAgent)
    silhouette_score = silhouette(adata_emb, label_key=cell_type_col, embed='X_pca')
    
    # Calculate batch silhouette if batch column exists
    batch_key = None
    for col in ['batch', 'Batch', 'sample', 'Sample']:
        if col in adata_emb.obs.columns:
            batch_key = col
            break
    
    if batch_key is not None:
        silhouette_batch_score = silhouette_batch(adata_emb, label_key=cell_type_col, batch_key=batch_key, embed='X_pca')
    else:
        silhouette_batch_score = None
        print("  (No batch column found, skipping batch silhouette)")
    
    print(f"\nMetrics:")
    print(f"  ARI (Adjusted Rand Index): {ari_score:.4f}")
    print(f"  NMI (Normalized Mutual Information): {nmi_score:.4f}")
    print(f"  AMI (Adjusted Mutual Information): {ami_score:.4f}")
    print(f"  FMI (Fowlkes-Mallows Index): {fmi_score:.4f}")
    print(f"  Silhouette Score: {silhouette_score:.4f}")
    if silhouette_batch_score is not None:
        print(f"  Silhouette Batch Score: {silhouette_batch_score:.4f}")
    print(f"  Number of true cell types: {num_cell_types}")
    print(f"  Number of predicted clusters: {n_clusters}")
    
    # Save results
    metrics = {
        "model": "chromfound",
        "num_cell_merge": int(num_cell_merge),
        "ari": float(ari_score),
        "nmi": float(nmi_score),
        "ami": float(ami_score),
        "fmi": float(fmi_score),
        "silhouette": float(silhouette_score),
        "silhouette_batch": float(silhouette_batch_score) if silhouette_batch_score is not None else None,
        "n_clusters": int(n_clusters),
        "n_cells": int(adata_emb.n_obs),
        "n_cell_types": int(num_cell_types),
        "n_pcs": int(n_pcs),
        "clustering_method": "kmeans",
    }
    
    print(f"\nSaving metrics to: {paths['metrics']}")
    with open(paths['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Log to wandb if initialized
    if wandb.run is not None:
        wandb.log(metrics)
    
    # return metrics
    
    # Save K-means labels
    print(f"Saving K-means labels to: {paths['kmeans_labels']}")
    adata_emb.write_h5ad(paths['kmeans_labels'])

    
    return metrics


def main():
    """Main benchmarking pipeline."""
    print("\n" + "=" * 80)
    print("ChromFound Benchmarking Pipeline")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPipeline overview:")
    print("  1. Load data and convert cCRE identifiers to genomic coordinates")
    print("  2. Preprocess: Quality control, normalization, log transform")
    print("  3. Run ChromFound inference to generate cell embeddings")
    print("  4. Apply PCA, neighbors, UMAP, and K-means clustering")
    print("  5. Calculate ARI and NMI metrics")
    print("=" * 80)
    
    # Ask user for GPU selection
    print("\n" + "-" * 80)
    print("GPU SELECTION:")
    print("-" * 80)
    while True:
        try:
            gpu_choice = input("Enter GPU device number (e.g., 0, 1, 5): ").strip()
            gpu_device = int(gpu_choice)
            if gpu_device >= 0:
                break
            print("Invalid GPU number. Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Ask user if they want to continue from step 2 or run from scratch
    print("\n" + "-" * 80)
    print("PIPELINE OPTIONS:")
    print("  1. Run full pipeline from scratch (Step 1 → Step 2 → Step 3)")
    print("  2. Continue from Step 2 (skip preprocessing, requires preprocessed data)")
    print("-" * 80)
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    continue_from_step2 = (choice == '2')
    
    # Ask user for cell merge value
    print("\n" + "-" * 80)
    print("CELL MERGE OPTIONS:")
    print("  1. num_cell_merge=1 (single-cell mode (true zero-shot))")
    print("  2. num_cell_merge=10 (like in ChromFound paper)")
    print("-" * 80)
    
    while True:
        merge_choice = input("\nEnter choice (1 or 2): ").strip()
        if merge_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    num_cell_merge = 1 if merge_choice == '1' else 10
    
    # Setup paths with merge value
    paths = setup_paths(num_cell_merge=num_cell_merge)
    print(f"\nOutput directory: {paths['results_dir']}")
    print(f"Using num_cell_merge={num_cell_merge}")
    
    # Initialize Weights & Biases
    wandb.init(
        project="chromfound-benchmark",
        name=f"chromfound-merge{num_cell_merge}-gpu{gpu_device}_{dataset_name}",
        tags=["chromfound", f"merge{num_cell_merge}", f"gpu{gpu_device}", dataset_name],
        config={
            "model": "chromfound",
            "num_cell_merge": num_cell_merge,
            "gpu_device": gpu_device,
            "dataset": dataset_name,
            "batch_size": 1,
            "n_pcs": 50,
        }
    )
    
    # Configuration
    data_args = {
        "cell_type_col": "cell_type",
        "num_cell_merge": num_cell_merge,
    }
    
    _chromfound_path = Path(__file__).parent.parent.parent / "ChromFound-Parallel"
    inference_config = {
        "pretrain_checkpoint_path": str(_chromfound_path / "src" / "checkpoints"),
        "pretrain_model_name": "model.pt",
        "pretrain_config_file": "chromfd_pretrain.yaml",
        "batch_size": 1,  # Reduced batch size for memory
        "device": gpu_device,
        "n_pca_components": 50,  # Memory-efficient: save only 50-dim PCA instead of 1.3M-dim
        "n_samples_for_pca": 1000,  # Fit PCA on 1000 random cells (FAST), then project all
    }
    
    # Log configuration to wandb
    wandb.config.update({
        "batch_size": inference_config["batch_size"],
        "pretrain_model_name": inference_config["pretrain_model_name"],
    })
    
    # Run pipeline
    try:
        if continue_from_step2:
            # Check if preprocessed data exists
            if not paths['preprocessed_data'].exists():
                print(f"\nERROR: Preprocessed data not found at: {paths['preprocessed_data']}")
                print(f"Expected file for num_cell_merge={num_cell_merge}")
                print("Please run the full pipeline first (choose option 1)")
                sys.exit(1)
            
            print(f"\n✓ Preprocessed data found: {paths['preprocessed_data']}")
            print("  Skipping Step 1 (preprocessing already done)")
            print(f"\nUsing batch_size={inference_config['batch_size']}")
        else:
            # Step 1: Preprocess
            preprocess_data(paths, data_args)
            print(f"\nUsing batch_size={inference_config['batch_size']}")
        
        # Step 2: Run inference (COMMENTED OUT - embeddings already generated)
        run_inference(paths, inference_config, data_args)
        
        # Step 3: Cluster and evaluate
        metrics = cluster_and_evaluate(paths, num_cell_merge=num_cell_merge)
        
        print("\n" + "=" * 80)
        print("BENCHMARKING COMPLETE")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults saved to: {paths['results_dir']}")
        print(f"\nFinal Metrics (merge={num_cell_merge}):")
        print(f"  ARI (Adjusted Rand Index): {metrics['ari']:.4f}")
        print(f"  NMI (Normalized Mutual Information): {metrics['nmi']:.4f}")
        print(f"  Number of clusters: {metrics['n_clusters']}")
        print(f"  Number of cell types: {metrics['n_cell_types']}")
        print("=" * 80)
        
        # Finalize wandb run
        wandb.finish()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()


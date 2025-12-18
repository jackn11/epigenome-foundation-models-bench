import os
import sys
import json
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.cluster import KMeans
from scib.metrics import silhouette, silhouette_batch
import wandb
import re



_project_root = Path(__file__).parent.parent.parent
_chromfound_path = _project_root / "ChromFound"
sys.path.insert(0, str(_chromfound_path))

from src.data.atac_preprocess import deepen_atac_data



def get_args_parser():
    """Get arguments parser for the benchmark."""
    
    parser = argparse.ArgumentParser('ChromFound Benchmarking Script')
    
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset h5ad file')
    parser.add_argument('--project_root', type=str, required=True, help='Project root directory')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory path')
    parser.add_argument('--n_cells_target', type=int, default=None, help='Number of cells to downsample to. If None (default), no downsampling is performed.')
    parser.add_argument('--gpu_device', type=int, required=True, help='GPU device number (e.g., 0, 1, 5)')
    parser.add_argument('--continue_from_step2', action='store_true', help='Continue from Step 2 (skip preprocessing, requires preprocessed data)')
    parser.add_argument('--num_cell_merge', type=int, required=True, choices=[1, 10], help='Number of cells to merge (1 for single-cell mode, 10 for ChromFound paper mode)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--n_pca_components', type=int, default=50, help='Number of PCA components')
    parser.add_argument('--n_samples_for_pca', type=int, default=1000, help='Number of samples for PCA')
    parser.add_argument('--cell_type_col', type=str, default='cell_type', help='Column name for cell type')
    parser.add_argument('--pretrain_checkpoint_path', type=str, default=None, help='Path to pretrain checkpoint directory')
    parser.add_argument('--pretrain_model_name', type=str, default='model.pt', help='Pretrain model file name')
    parser.add_argument('--pretrain_config_file', type=str, default='chromfd_pretrain.yaml', help='Pretrain config file name')
    return parser


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
    
    if not inplace:
        adata = adata.copy()
    
    required_cols = ['#Chromosome', 'hg38_Start', 'hg38_End']
    if all(col in adata.var.columns for col in required_cols):
        print("Genomic coordinate columns already exist. Skipping conversion.")
        return adata
    
    print("Converting cCRE identifiers to genomic coordinates")
    
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
            chromosomes.append("chr1")
            starts.append(0)
            ends.append(0)
    
    if failed_parses:
        print(f"Warning: Failed to parse {len(failed_parses)} cCRE identifiers")
        if len(failed_parses) <= 10:
            print(f"Failed IDs: {failed_parses}")
        else:
            print(f"First 10 failed IDs: {failed_parses[:10]}")
    
    adata.var['#Chromosome'] = pd.Series(chromosomes, index=adata.var.index)
    adata.var['hg38_Start'] = pd.Series(starts, dtype=int, index=adata.var.index)
    adata.var['hg38_End'] = pd.Series(ends, dtype=int, index=adata.var.index)
    
    print(f"Successfully converted {len(adata.var)} cCRE identifiers to genomic coordinates")
    print(f"Chromosomes: {pd.Series(chromosomes).value_counts().head(5).to_dict()}")
    print(f"Start range: {min(starts)} - {max(starts)}")
    print(f"End range: {min(ends)} - {max(ends)}")
    
    return adata


def setup_paths(args, num_cell_merge=1):
    """Set up all file paths for the benchmark.
    
    Args:
        num_cell_merge: Number of cells to merge (1 or 10). Used in file naming.
    
    Returns:
        Dictionary of file paths
    """

    project_root = Path(args.project_root)
    benchmarks_dir = project_root / "benchmarks"
    
    merge_suffix = f"merge{num_cell_merge}"
    results_subdir = Path(args.results_dir) / merge_suffix
    
    paths = {
        "project_root": project_root,
        "benchmarks_dir": benchmarks_dir,
        "results_dir": results_subdir,
        "input_data": Path(args.dataset_path),
        "preprocessed_data": results_subdir / f"{args.dataset_name}_preprocessed_{merge_suffix}.h5ad",
        "embeddings": results_subdir / "embeddings_pca.h5ad",
        "metrics": results_subdir / "metrics.json",
        "kmeans_labels": results_subdir / "kmeans_labels.h5ad",
    }
    
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["preprocessed_data"].parent.mkdir(parents=True, exist_ok=True)
    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["kmeans_labels"].parent.mkdir(parents=True, exist_ok=True)
    
    return paths


def preprocess_data(paths, data_args, n_cells_target=None):
    """
    Preprocess the data: convert cCREs to genomic coordinates, verify columns.
    
    Note: Quality control is skipped as benchmark datasets are already preprocessed.
    
    Args:
        paths: Dictionary of file paths
        data_args: Dictionary of preprocessing parameters
    
    Returns:
        Path to preprocessed data file
    """
    print(f"\nLoading and Preprocessing Data")
    
    print(f"Loading data from: {paths['input_data']}")
    adata = sc.read_h5ad(paths['input_data'])
    print(f"Original shape: {adata.shape}")

    if n_cells_target is not None and adata.n_obs > n_cells_target:
        print(f"\nDownsampling from {adata.n_obs:,} to {n_cells_target:,} cells")
        sc.pp.subsample(adata, n_obs=n_cells_target, random_state=42)
        print(f"After downsampling: {adata.shape}")
    elif n_cells_target is not None:
        print(f"\nDataset has {adata.n_obs:,} cells (≤ {n_cells_target:,}), skipping downsampling")
    else:
        print(f"\nNo downsampling requested (n_cells_target=None), using all {adata.n_obs:,} cells")
    
    print("\nConverting cCRE identifiers to genomic coordinates")
    adata = convert_ccre_to_genomic_coords(adata, inplace=True)
    
    if 'hg38_Start' in adata.var.columns:
        if adata.var['hg38_Start'].dtype != 'int64' and adata.var['hg38_Start'].dtype != 'int32':
            if adata.var['hg38_Start'].isna().any():
                print("Warning: hg38_Start contains NaN values. Filling with 0.")
                adata.var['hg38_Start'] = adata.var['hg38_Start'].fillna(0)
            adata.var['hg38_Start'] = adata.var['hg38_Start'].astype(int)
    if 'hg38_End' in adata.var.columns:
        if adata.var['hg38_End'].dtype != 'int64' and adata.var['hg38_End'].dtype != 'int32':
            if adata.var['hg38_End'].isna().any():
                print("Warning: hg38_End contains NaN values. Filling with 0.")
                adata.var['hg38_End'] = adata.var['hg38_End'].fillna(0)
            adata.var['hg38_End'] = adata.var['hg38_End'].astype(int)
    
    required_cols = ['#Chromosome', 'hg38_Start', 'hg38_End']
    missing_cols = [col for col in required_cols if col not in adata.var.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in adata.var: {missing_cols}")
    print(f"\nRequired columns verified: {required_cols}")
    print(f"Data shape: {adata.shape}")
    
    num_cell_merge = data_args.get("num_cell_merge", 1)
    if num_cell_merge > 1:
        print(f"\nPerforming cell aggregation (num_cell_merge={num_cell_merge})")
        adata = deepen_atac_data(adata, num_cell_merge=num_cell_merge)
        print(f"After cell aggregation: {adata.shape}")
    else:
        print("\nSkipping cell aggregation (num_cell_merge=1, single-cell mode)")
    
    print("\nNormalizing")
    sc.pp.normalize_total(adata)
    print('Log transforming')
    sc.pp.log1p(adata)
    print("Normalization and log transform complete")
    
    cell_type_col = data_args.get("cell_type_col", "cell_type")
    if 'celltype' not in adata.obs.columns and cell_type_col in adata.obs.columns:
        adata.obs['celltype'] = adata.obs[cell_type_col]
        print(f"\nAdded 'celltype' column (copied from '{cell_type_col}') for ChromFound compatibility")
    
    print(f"\nSaving preprocessed data to: {paths['preprocessed_data']}")
    adata.write_h5ad(paths['preprocessed_data'], compression='gzip', compression_opts=1)
    
    print(f"Preprocessed data saved: {paths['preprocessed_data']}")
    
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
    print(f"\nRunning ChromFound Inference")

    n_pca_components = inference_config.get("n_pca_components", 50)
    n_samples_for_pca = inference_config.get("n_samples_for_pca", 1000)
    
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
        '--n_samples_for_pca', str(n_samples_for_pca)
    ]
    
    print(f"\nRunning inference command")
    print(f"Device: GPU {inference_config['device']}")
    print(f"Batch size: {inference_config['batch_size']}")
    print(f"Model: {inference_config['pretrain_model_name']}")
    print(f"Input: {paths['preprocessed_data']}")
    print(f"Output: {paths['results_dir']}")
    print(f"\nNote: This step may take several minutes. Progress will be shown below")
    
    original_cwd = os.getcwd()
    _chromfound_path = Path(__file__).parent.parent.parent / "ChromFound-Parallel"
    
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        os.chdir(str(_chromfound_path))
        result = subprocess.run(inference_command, check=True, env=env)
    finally:
        os.chdir(original_cwd)

    embeddings_path = paths['results_dir'] / "embeddings_pca.h5ad"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    print(f"\nOMPLETE")
    print(f"Embeddings saved to: {embeddings_path}")
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
    print(f"\nClustering and Evaluation")
    
    print(f"\nLoading PCA-reduced embeddings from: {paths['embeddings']}")
    adata = sc.read_h5ad(paths['embeddings'])
    
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
    
    cell_type_col = None
    for col in ['cell_type', 'celltype']:
        if col in adata.obs.columns:
            cell_type_col = col
            break
    
    if cell_type_col is None:
        raise KeyError("Cell type column not found in adata.obs")
    
    obs_df = pd.DataFrame(adata.obs[cell_type_col].tolist(), columns=[cell_type_col])
    adata_emb = sc.AnnData(X=embeddings, obs=obs_df)
    
    if pca_already_done:
        print("\n[3.1] Using pre-computed PCA embeddings (memory-efficient mode)")
        n_pcs = embeddings.shape[1]
        adata_emb.obsm['X_pca'] = embeddings
        print(f"PCA already computed: {n_pcs} principal components")
    else:
        print("\n[3.1] Applying PCA (reducing dimensionality)")
        n_pcs = 10
        sc.tl.pca(adata_emb, n_comps=n_pcs)
        print(f"PCA complete: {n_pcs} principal components")
    
    print("\n[3.2] Computing neighbors graph")
    sc.pp.neighbors(adata_emb, n_neighbors=15, use_rep='X_pca')
    print("Neighbors graph computed")
    
    print("\n[3.3] Computing UMAP")
    sc.tl.umap(adata_emb)
    print("UMAP computed")
    
    true_labels = adata_emb.obs[cell_type_col].values
    num_cell_types = len(np.unique(true_labels))
    
    print(f"\n[3.4] Performing K-means clustering (n_clusters={num_cell_types})")
    kmeans = KMeans(n_clusters=num_cell_types, random_state=23).fit(adata_emb.obsm['X_pca'])
    predicted_labels = kmeans.labels_
    adata_emb.obs['kmeans'] = predicted_labels.astype(str)
    
    n_clusters = len(np.unique(predicted_labels))
    print(f"Number of K-means clusters: {n_clusters}")
    
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
    ami_score = adjusted_mutual_info_score(true_labels, predicted_labels)
    fmi_score = fowlkes_mallows_score(true_labels, predicted_labels)
    
    silhouette_score = silhouette(adata_emb, label_key=cell_type_col, embed='X_pca')
    
    batch_key = None
    for col in adata_emb.obs.columns:
        if 'batch' in col.lower():
            batch_key = col
            break
    
    if batch_key is not None:
        silhouette_batch_score = silhouette_batch(adata_emb, label_key=cell_type_col, batch_key=batch_key, embed='X_pca')
    else:
        silhouette_batch_score = None
        print("(No batch column found, skipping batch silhouette)")
    
    print(f"\nMetrics:")
    print(f"ARI (Adjusted Rand Index): {ari_score:.4f}")
    print(f"NMI (Normalized Mutual Information): {nmi_score:.4f}")
    print(f"AMI (Adjusted Mutual Information): {ami_score:.4f}")
    print(f"FMI (Fowlkes-Mallows Index): {fmi_score:.4f}")
    print(f"Silhouette Score: {silhouette_score:.4f}")
    if silhouette_batch_score is not None:
        print(f"Silhouette Batch Score: {silhouette_batch_score:.4f}")
    print(f"Number of true cell types: {num_cell_types}")
    print(f"Number of predicted clusters: {n_clusters}")
    
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
    
    if wandb.run is not None:
        wandb.log(metrics)
    
    
    print(f"Saving K-means labels to: {paths['kmeans_labels']}")
    adata_emb.write_h5ad(paths['kmeans_labels'])

    
    return metrics


def main(args):
    """Main benchmarking pipeline."""
    print("\nChromFound Benchmarking Pipeline")
    
    gpu_device = args.gpu_device
    continue_from_step2 = args.continue_from_step2
    num_cell_merge = args.num_cell_merge
    
    paths = setup_paths(args, num_cell_merge=num_cell_merge)
    print(f"\nOutput directory: {paths['results_dir']}")
    print(f"Using num_cell_merge={num_cell_merge}")
    
    wandb.init(
        project="chromfound-benchmark",
        name=f"chromfound-merge{num_cell_merge}-gpu{gpu_device}_{args.dataset_name}",
        tags=["chromfound", f"merge{num_cell_merge}", f"gpu{gpu_device}", args.dataset_name],
        config={
            "model": "chromfound",
            "num_cell_merge": num_cell_merge,
            "gpu_device": gpu_device,
            "dataset": args.dataset_name,
            "batch_size": args.batch_size,
            "n_pcs": args.n_pca_components,
        }
    )
    
    data_args = {
        "cell_type_col": args.cell_type_col,
        "num_cell_merge": num_cell_merge,
    }
    
    _chromfound_path = Path(__file__).parent.parent.parent / "ChromFound-Parallel"
    if args.pretrain_checkpoint_path is None:
        pretrain_checkpoint_path = str(_chromfound_path / "src" / "checkpoints")
    else:
        pretrain_checkpoint_path = args.pretrain_checkpoint_path
    
    inference_config = {
        "pretrain_checkpoint_path": pretrain_checkpoint_path,
        "pretrain_model_name": args.pretrain_model_name,
        "pretrain_config_file": args.pretrain_config_file,
        "batch_size": args.batch_size,
        "device": gpu_device,
        "n_pca_components": args.n_pca_components,
        "n_samples_for_pca": args.n_samples_for_pca,
    }
    
    wandb.config.update({
        "batch_size": inference_config["batch_size"],
        "pretrain_model_name": inference_config["pretrain_model_name"],
    })
    
    try:
        if continue_from_step2:
            if not paths['preprocessed_data'].exists():
                print(f"\nERROR: Preprocessed data not found at: {paths['preprocessed_data']}")
                print(f"Expected file for num_cell_merge={num_cell_merge}")
                print("Please run the full pipeline first (choose option 1)")
                sys.exit(1)
            
            print(f"\nPreprocessed data found: {paths['preprocessed_data']}")
            print("Skipping Step 1 (preprocessing already done)")
            print(f"\nUsing batch_size={inference_config['batch_size']}")
        else:
            preprocess_data(paths, data_args, n_cells_target=args.n_cells_target)
            print(f"\nUsing batch_size={inference_config['batch_size']}")
        
        run_inference(paths, inference_config, data_args)
        
        metrics = cluster_and_evaluate(paths, num_cell_merge=num_cell_merge)
        
        print("\nBENCHMARKING COMPLETE")
        print(f"\nResults saved to: {paths['results_dir']}")
        print(f"\nFinal Metrics (merge={num_cell_merge}):")
        print(f"ARI (Adjusted Rand Index): {metrics['ari']:.4f}")
        print(f"NMI (Normalized Mutual Information): {metrics['nmi']:.4f}")
        print(f"Number of clusters: {metrics['n_clusters']}")
        print(f"Number of cell types: {metrics['n_cell_types']}")
        
        wandb.finish()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)


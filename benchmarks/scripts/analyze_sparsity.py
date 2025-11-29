"""
Analyze sparsity differences between Kanemaru dataset and ChromFound sample data.

This script compares:
1. Overall sparsity (percentage of zeros)
2. Non-zero elements per cell
3. Non-zero elements per feature
4. Memory footprint differences
"""
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse

def analyze_sparsity(adata, dataset_name):
    """Analyze sparsity characteristics of an AnnData object."""
    print(f"\n{'='*80}")
    print(f"Sparsity Analysis: {dataset_name}")
    print(f"{'='*80}")
    
    print(f"\nDataset shape: {adata.shape} (cells × features)")
    print(f"Matrix type: {'Sparse' if issparse(adata.X) else 'Dense'}")
    
    if issparse(adata.X):
        total_elements = adata.shape[0] * adata.shape[1]
        nnz = adata.X.nnz
        sparsity = 1 - (nnz / total_elements)
        
        print(f"\nSparsity Metrics:")
        print(f"  Total elements: {total_elements:,}")
        print(f"  Non-zero elements: {nnz:,}")
        print(f"  Sparsity: {sparsity:.4%} ({(sparsity*100):.2f}% zeros)")
        print(f"  Density: {(1-sparsity):.4%} ({(1-sparsity)*100:.2f}% non-zeros)")
        
        # Memory footprint
        sparse_memory_mb = adata.X.data.nbytes / (1024**2)
        print(f"\nMemory Footprint:")
        print(f"  Sparse matrix data: {sparse_memory_mb:.2f} MB")
        print(f"  Dense equivalent: {(total_elements * 4 / (1024**2)):.2f} MB")
        print(f"  Compression ratio: {(total_elements * 4 / adata.X.data.nbytes):.2f}x")
        
        # Per-cell statistics
        row_sums = np.array(adata.X.sum(axis=1)).flatten()
        row_nnz = np.array((adata.X != 0).sum(axis=1)).flatten()
        
        print(f"\nPer-Cell Statistics (non-zero elements):")
        print(f"  Mean: {row_nnz.mean():.1f}")
        print(f"  Median: {np.median(row_nnz):.1f}")
        print(f"  Min: {row_nnz.min()}")
        print(f"  Max: {row_nnz.max()}")
        print(f"  Std: {row_nnz.std():.1f}")
        
        # Per-feature statistics
        col_sums = np.array(adata.X.sum(axis=0)).flatten()
        col_nnz = np.array((adata.X != 0).sum(axis=0)).flatten()
        
        print(f"\nPer-Feature Statistics (non-zero elements):")
        print(f"  Mean: {col_nnz.mean():.1f}")
        print(f"  Median: {np.median(col_nnz):.1f}")
        print(f"  Min: {col_nnz.min()}")
        print(f"  Max: {col_nnz.max()}")
        print(f"  Std: {col_nnz.std():.1f}")
        
        # Distribution of non-zeros per cell
        print(f"\nNon-zero Distribution (cells):")
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        for p in percentiles:
            val = np.percentile(row_nnz, p)
            print(f"  {p:3d}th percentile: {val:.1f} non-zeros")
        
        return {
            'shape': adata.shape,
            'sparsity': sparsity,
            'density': 1 - sparsity,
            'nnz': nnz,
            'total_elements': total_elements,
            'sparse_memory_mb': sparse_memory_mb,
            'dense_memory_mb': total_elements * 4 / (1024**2),
            'mean_nnz_per_cell': row_nnz.mean(),
            'mean_nnz_per_feature': col_nnz.mean(),
            'median_nnz_per_cell': np.median(row_nnz),
            'median_nnz_per_feature': np.median(col_nnz),
        }
    else:
        # Dense matrix
        nnz = np.count_nonzero(adata.X)
        total_elements = adata.X.size
        sparsity = 1 - (nnz / total_elements)
        
        print(f"\nSparsity Metrics:")
        print(f"  Total elements: {total_elements:,}")
        print(f"  Non-zero elements: {nnz:,}")
        print(f"  Sparsity: {sparsity:.4%}")
        print(f"  Memory footprint: {(adata.X.nbytes / (1024**2)):.2f} MB")
        
        return {
            'shape': adata.shape,
            'sparsity': sparsity,
            'density': 1 - sparsity,
            'nnz': nnz,
            'total_elements': total_elements,
            'memory_mb': adata.X.nbytes / (1024**2),
        }


def main():
    """Compare sparsity between Kanemaru and ChromFound sample data."""
    print("\n" + "="*80)
    print("Sparsity Comparison Analysis")
    print("="*80)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    kanemaru_path = project_root / "benchmarks" / "data" / "Kanemaru2023_chromfound_preprocessed.h5ad"
    chromfound_path = project_root / "ChromFound-Parallel" / "sample_data" / "PBMC169K" / "atac_pbmc_benchmark_VIB_10xv1_1_qc_deepen_norm_log.h5ad"
    
    # Load Kanemaru data
    print(f"\nLoading Kanemaru data from: {kanemaru_path}")
    if not kanemaru_path.exists():
        print(f"ERROR: Kanemaru preprocessed file not found!")
        print(f"Please run the preprocessing step first.")
        return
    
    kanemaru = sc.read_h5ad(kanemaru_path)
    kanemaru_stats = analyze_sparsity(kanemaru, "Kanemaru2023 (preprocessed)")
    
    # Load ChromFound sample data
    print(f"\nLoading ChromFound sample data from: {chromfound_path}")
    if not chromfound_path.exists():
        print(f"WARNING: ChromFound sample data not found at expected path.")
        print(f"Trying alternative location...")
        # Try to find it
        alt_path = project_root / "ChromFound-Parallel" / "sample_data" / "PBMC169K"
        if alt_path.exists():
            files = list(alt_path.glob("*_qc_deepen_norm_log.h5ad"))
            if files:
                chromfound_path = files[0]
                print(f"Found at: {chromfound_path}")
            else:
                print(f"ERROR: ChromFound sample data not found!")
                return
        else:
            print(f"ERROR: ChromFound sample data directory not found!")
            return
    
    chromfound = sc.read_h5ad(chromfound_path)
    chromfound_stats = analyze_sparsity(chromfound, "ChromFound Sample (PBMC169K)")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nDataset Sizes:")
    print(f"  Kanemaru:    {kanemaru_stats['shape'][0]:,} cells × {kanemaru_stats['shape'][1]:,} features")
    print(f"  ChromFound:  {chromfound_stats['shape'][0]:,} cells × {chromfound_stats['shape'][1]:,} features")
    
    print(f"\nSparsity Comparison:")
    print(f"  Kanemaru sparsity:    {kanemaru_stats['sparsity']:.4%} ({(1-kanemaru_stats['sparsity'])*100:.2f}% dense)")
    print(f"  ChromFound sparsity:  {chromfound_stats['sparsity']:.4%} ({(1-chromfound_stats['sparsity'])*100:.2f}% dense)")
    
    sparsity_diff = kanemaru_stats['sparsity'] - chromfound_stats['sparsity']
    density_diff = kanemaru_stats['density'] - chromfound_stats['density']
    
    print(f"\nDifference:")
    print(f"  Sparsity difference: {sparsity_diff:+.4%} (Kanemaru is {abs(sparsity_diff):.4%} more {'sparse' if sparsity_diff > 0 else 'dense'})")
    print(f"  Density difference:  {density_diff:+.4%} (Kanemaru is {abs(density_diff):.4%} more {'dense' if density_diff > 0 else 'sparse'})")
    
    print(f"\nMemory Footprint:")
    if 'sparse_memory_mb' in kanemaru_stats:
        print(f"  Kanemaru sparse:    {kanemaru_stats['sparse_memory_mb']:.2f} MB")
        print(f"  Kanemaru dense:     {kanemaru_stats['dense_memory_mb']:.2f} MB")
    if 'sparse_memory_mb' in chromfound_stats:
        print(f"  ChromFound sparse:  {chromfound_stats['sparse_memory_mb']:.2f} MB")
        print(f"  ChromFound dense:   {chromfound_stats['dense_memory_mb']:.2f} MB")
    
    print(f"\nNon-zeros per Cell:")
    print(f"  Kanemaru mean:    {kanemaru_stats.get('mean_nnz_per_cell', 'N/A'):.1f}")
    print(f"  ChromFound mean:  {chromfound_stats.get('mean_nnz_per_cell', 'N/A'):.1f}")
    
    print(f"\nNon-zeros per Feature:")
    print(f"  Kanemaru mean:    {kanemaru_stats.get('mean_nnz_per_feature', 'N/A'):.1f}")
    print(f"  ChromFound mean:  {chromfound_stats.get('mean_nnz_per_feature', 'N/A'):.1f}")
    
    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if density_diff > 0.01:  # More than 1% difference
        print(f"\n⚠️  Kanemaru data is SIGNIFICANTLY MORE DENSE than ChromFound sample data.")
        print(f"   This means Kanemaru has {(density_diff*100):.2f}% more non-zero values.")
        print(f"   This could contribute to CUDA OOM because:")
        print(f"   1. More non-zeros = more computation per batch")
        print(f"   2. Denser matrices may use more GPU memory")
        print(f"   3. Attention mechanisms scale with non-zero elements")
    elif density_diff < -0.01:
        print(f"\n✓ Kanemaru data is MORE SPARSE than ChromFound sample data.")
        print(f"   Sparsity is likely NOT the cause of CUDA OOM.")
    else:
        print(f"\n✓ Sparsity is similar between datasets.")
        print(f"   Sparsity is likely NOT the cause of CUDA OOM.")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()


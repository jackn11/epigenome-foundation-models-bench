# Running ChromFound Benchmark

## Step-by-Step Commands

```bash
# 1. Navigate to project root
cd /home/naimer/github/project-2-team-1

# 2. Activate chromfound conda environment (created using the Readme in the ChromFound-Parallel folder)
conda activate ChromFoundBench

# 3. Run the benchmark script
python benchmarks/scripts/run_chromfound_benchmark.py
```

## What to Expect

The script will:
1. Load Kanemaru dataset
2. Convert cCRE identifiers to genomic coordinates
3. Normalize and log transform the data
4. Run ChromFound inference (this takes the longest)
5. Apply PCA, neighbors, UMAP, and Leiden clustering
6. Calculate ARI and NMI metrics
7. Save results to `benchmarks/results/chromfound/`

## Output Files

Results will be saved to:
- `benchmarks/results/chromfound/embeddings.h5ad` - Cell embeddings
- `benchmarks/results/chromfound/metrics.json` - ARI, NMI scores
- `benchmarks/results/chromfound/leiden_labels.h5ad` - Clustering results


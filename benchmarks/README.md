# Model Benchmarking Pipeline

This directory contains the benchmarking infrastructure for comparing EpiAgent and ChromFound models on the Kanemaru2023 dataset.

## Directory Structure

```
benchmarks/
├── data/
│   └── Kanemaru2023_downsampled_10000_cells.h5ad  # Input dataset (symlinked)
├── results/
│   ├── epiagent/          # EpiAgent results
│   │   ├── embeddings.h5ad
│   │   ├── metrics.json
│   │   └── leiden_labels.h5ad
│   └── chromfound/         # ChromFound results
│       ├── embeddings.h5ad
│       ├── metrics.json
│       └── leiden_labels.h5ad
├── scripts/
│   ├── run_epiagent_benchmark.py    # Run in EpiAgent conda env
│   ├── run_chromfound_benchmark.py  # Run in ChromFound conda env
│   └── compare_models.py            # Run in either env
└── plots/
    └── model_comparison.png         # Comparison visualizations
```

## Usage

1. **Run EpiAgent benchmark** (in EpiAgentBench conda env):
   ```bash
   conda activate EpiAgentBench
   python benchmarks/scripts/run_epiagent_benchmark.py
   ```

2. **Run ChromFound benchmark** (in chromfound conda env):
   ```bash
   conda activate chromfound
   python benchmarks/scripts/run_chromfound_benchmark.py
   ```

3. **Compare results** (in either env):
   ```bash
   python benchmarks/scripts/compare_models.py
   ```

## Output Format

Each model script outputs:
- `embeddings.h5ad`: Cell embeddings in AnnData format
- `metrics.json`: ARI, NMI scores and metadata
- `leiden_labels.h5ad`: Clustering results (optional)



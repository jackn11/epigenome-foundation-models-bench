"""
Continue ChromFound benchmark from Step 2 with reduced batch size.

This script skips Step 1 (preprocessing) if the preprocessed file already exists,
and runs Step 2 (inference) with batch_size=8 to avoid CUDA OOM errors.
"""
import sys
from pathlib import Path

# Add the scripts directory to path to import the main script functions
sys.path.insert(0, str(Path(__file__).parent))

from run_chromfound_benchmark import setup_paths, run_inference, cluster_and_evaluate

def main():
    """Continue from Step 2 with batch_size=8."""
    print("\n" + "=" * 80)
    print("ChromFound Benchmarking - Continue from Step 2")
    print("=" * 80)
    
    # Setup paths
    paths = setup_paths()
    
    # Check if preprocessed data exists
    if not paths['preprocessed_data'].exists():
        print(f"\nERROR: Preprocessed data not found at: {paths['preprocessed_data']}")
        print("Please run the full pipeline first (run_chromfound_benchmark.py)")
        sys.exit(1)
    
    print(f"\n✓ Preprocessed data found: {paths['preprocessed_data']}")
    print("  Skipping Step 1 (preprocessing already done)")
    
    # Configuration with reduced batch size
    data_args = {
        "cell_type_col": "cell_type",
    }
    
    _chromfound_path = Path(__file__).parent.parent.parent / "ChromFound-Parallel"
    inference_config = {
        "pretrain_checkpoint_path": str(_chromfound_path / "src" / "checkpoints"),
        "pretrain_model_name": "model.pt",
        "pretrain_config_file": "chromfd_pretrain.yaml",
        "batch_size": 2,
        "device": 5,
    }
    
    print(f"\nUsing batch_size={inference_config['batch_size']} (minimal for memory analysis)")
    
    # Run steps 2 and 3
    try:
        # Step 2: Run inference
        run_inference(paths, inference_config, data_args)
        
        # Step 3: Cluster and evaluate
        metrics = cluster_and_evaluate(paths, leiden_resolution=0.45)
        
        print("\n" + "=" * 80)
        print("BENCHMARKING COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {paths['results_dir']}")
        print(f"\nFinal Metrics:")
        print(f"  ARI (Adjusted Rand Index): {metrics['ari']:.4f}")
        print(f"  NMI (Normalized Mutual Information): {metrics['nmi']:.4f}")
        print(f"  Number of clusters: {metrics['n_clusters']}")
        print(f"  Number of cell types: {metrics['n_cell_types']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Helper script to run ChromFound perturbation analysis on ALL genes found in the h5ad file.
"""
import scanpy as sc
import numpy as np
import subprocess
import sys
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Run ChromFound perturbation analysis on all genes', add_help=False)
    parser.add_argument('--h5ad_path',
                        required=True,
                        type=str,
                        help='Path to h5ad file with embeddings')
    parser.add_argument('--output_dir',
                        required=True,
                        type=str,
                        help='Directory to save outputs')
    return parser

def main():
    args = get_args_parser().parse_args()
    
    h5ad_path = args.h5ad_path
    output_dir = args.output_dir
    
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
        "--output_dir", output_dir,
        "--genes_of_interest"
    ] + genes_to_analyze
    
    print(f"\nRunning analysis for all {len(genes_to_analyze)} genes...")
    print("Command:", " ".join(cmd))
    
    # Run the analysis
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()


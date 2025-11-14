#!/usr/bin/env python3

import argparse
import logging
import os
import scanpy as sc

logging.basicConfig(level=logging.INFO)


def merge_embeddings(output_path, num_splits, output_file="embeddings.h5ad"):
    """Merge split embedding files into a single file."""
    
    logging.info(f"Merging {num_splits} embedding splits from {output_path}")
    
    # Load all splits
    adata_list = []
    for i in range(num_splits):
        split_file = os.path.join(output_path, f"embeddings_split_{i}.h5ad")
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")
        
        logging.info(f"Loading split {i + 1}/{num_splits}: {split_file}")
        adata_split = sc.read_h5ad(split_file)
        adata_list.append(adata_split)
        logging.info(f"  - Split {i + 1} shape: {adata_split.shape}")
    
    # Concatenate all splits
    logging.info("Concatenating splits...")
    adata_merged = sc.concat(adata_list, axis=0, join='outer', merge='same')
    
    # Save merged file
    output_filepath = os.path.join(output_path, output_file)
    logging.info(f"Saving merged embeddings to {output_filepath}")
    adata_merged.write_h5ad(output_filepath)
    
    logging.info(f"Successfully merged embeddings:")
    logging.info(f"  - Total cells: {adata_merged.shape[0]}")
    logging.info(f"  - Total features: {adata_merged.shape[1]}")
    logging.info(f"  - Embedding shape: {adata_merged.obsm['X_embedding'].shape}")
    
    return adata_merged


def main():
    parser = argparse.ArgumentParser(description="Merges the split embedding files")
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='sample_data/PBMC169K/cell_embedding',
        help='Location of split embedding files'
    )
    parser.add_argument(
        '--num_splits', 
        type=int, 
        default=4,
        help='Number of splits to merge'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='embeddings.h5ad',
        help='Name of the merged output file'
    )
    parser.add_argument(
        '--keep_splits',
        action='store_true',
        help='Keep split files post-merge'
    )
    
    args = parser.parse_args()
    
    # Merge the embeddings
    merge_embeddings(args.output_path, args.num_splits, args.output_file)
    
    # Optionally remove split files
    if not args.keep_splits:
        logging.info("Removing split files...")
        for i in range(args.num_splits):
            split_file = os.path.join(args.output_path, f"embeddings_split_{i}.h5ad")
            if os.path.exists(split_file):
                os.remove(split_file)
                logging.info(f"  - Removed {split_file}")
    
    logging.info("Done!")


if __name__ == '__main__':
    main()



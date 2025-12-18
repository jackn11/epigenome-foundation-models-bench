import scanpy as sc
import pandas as pd
import numpy as np
import argparse


def check_peaks_order(file_path):
    """Check if peaks are ordered by genomic coordinates."""
    adata = sc.read_h5ad(file_path)
    var_df = adata.var.copy()
    
    chr_col = None
    start_col = None
    
    for col in var_df.columns:
        col_lower = col.lower()
        if 'chr' in col_lower or 'chrom' in col_lower:
            chr_col = col
            break
    
    for col in var_df.columns:
        col_lower = col.lower()
        if 'start' in col_lower:
            start_col = col
            break
    
    if chr_col is None or start_col is None:
        print("ERROR: Could not find genomic coordinate columns.")
        return False
    
    def get_chr_sort_key(chr_name):
        """Convert chromosome name to sortable key."""
        chr_str = str(chr_name)
        if chr_str.startswith('chr'):
            chr_num = chr_str[3:]
            try:
                if chr_num.isdigit():
                    return (0, int(chr_num))
                elif chr_num.upper() == 'X':
                    return (1, 0)
                elif chr_num.upper() == 'Y':
                    return (1, 1)
                elif chr_num.upper() == 'M':
                    return (1, 2)
                else:
                    return (2, chr_num)
            except:
                return (2, chr_num)
        else:
            return (2, chr_str)
    
    var_df['_chr_sort'] = var_df[chr_col].apply(get_chr_sort_key)
    var_df['_start'] = pd.to_numeric(var_df[start_col], errors='coerce')
    
    unsorted_count = 0
    prev_chr = None
    prev_start = None
    
    for _, row in var_df.iterrows():
        curr_chr = row['_chr_sort']
        curr_start = row['_start']
        
        if pd.isna(curr_start):
            continue
        
        if prev_chr is not None:
            if curr_chr < prev_chr or (curr_chr == prev_chr and curr_start < prev_start):
                unsorted_count += 1
        
        prev_chr = curr_chr
        prev_start = curr_start
    
    is_sorted = unsorted_count == 0
    print(f"Peaks sorted: {is_sorted} ({unsorted_count} unsorted)")
    return is_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check if peaks are ordered by genomic coordinates')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset h5ad file')
    args = parser.parse_args()
    check_peaks_order(args.dataset_path)

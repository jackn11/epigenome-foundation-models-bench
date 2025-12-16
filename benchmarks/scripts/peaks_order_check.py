#!/usr/bin/env python3
"""
Check if peaks in an h5ad file are ordered by genomic coordinates.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

def check_peaks_order(file_path):
    """Check if peaks are ordered by genomic coordinates."""
    
    print(f"Loading file: {file_path}")
    adata = sc.read_h5ad(file_path)
    
    print(f"\nDataset shape: {adata.shape}")
    print(f"Number of peaks (var): {adata.n_vars}")
    print(f"Number of cells (obs): {adata.n_obs}")
    
    # Check what columns are available in var
    print(f"\nAvailable columns in var: {list(adata.var.columns)}")
    
    # Try to find chromosome and position columns
    var_df = adata.var.copy()
    
    # Common column names for genomic coordinates
    chr_col = None
    start_col = None
    end_col = None
    
    # Check for chromosome column
    for col in var_df.columns:
        col_lower = col.lower()
        if 'chr' in col_lower or 'chrom' in col_lower:
            chr_col = col
            break
    
    # Check for start/end columns
    for col in var_df.columns:
        col_lower = col.lower()
        if 'start' in col_lower:
            start_col = col
        elif 'end' in col_lower:
            end_col = col
    
    # Also check if the index itself contains genomic coordinates (e.g., "chr1:1000-2000")
    if chr_col is None or start_col is None:
        print("\nChecking if var index contains genomic coordinates...")
        sample_indices = var_df.index[:5].tolist()
        print(f"Sample indices: {sample_indices}")
        
        # Try to parse genomic coordinates from index
        if any(':' in str(idx) and '-' in str(idx) for idx in sample_indices):
            print("Index appears to contain genomic coordinates in format 'chr:start-end'")
            # Parse coordinates from index
            parsed_coords = []
            for idx in var_df.index:
                try:
                    parts = str(idx).split(':')
                    if len(parts) == 2:
                        chrom = parts[0]
                        pos_parts = parts[1].split('-')
                        if len(pos_parts) == 2:
                            start = int(pos_parts[0])
                            end = int(pos_parts[1])
                            parsed_coords.append({
                                'chr': chrom,
                                'start': start,
                                'end': end
                            })
                        else:
                            parsed_coords.append(None)
                    else:
                        parsed_coords.append(None)
                except:
                    parsed_coords.append(None)
            
            if all(p is not None for p in parsed_coords):
                coords_df = pd.DataFrame(parsed_coords, index=var_df.index)
                chr_col = 'chr'
                start_col = 'start'
                end_col = 'end'
                var_df = pd.concat([var_df, coords_df], axis=1)
                print("Successfully parsed genomic coordinates from index")
    
    if chr_col is None or start_col is None:
        print("\nERROR: Could not find genomic coordinate columns.")
        print("Please check the var dataframe structure.")
        print("\nFirst 10 rows of var:")
        print(var_df.head(10))
        return
    
    print(f"\nUsing columns:")
    print(f"  Chromosome: {chr_col}")
    print(f"  Start: {start_col}")
    if end_col:
        print(f"  End: {end_col}")
    
    # Check if peaks are sorted
    print(f"\nChecking if peaks are sorted by genomic coordinates...")
    
    # Create a sortable representation
    # First, ensure chromosome names are sortable (handle chr1, chr2, ..., chr10, chr11, etc.)
    def get_chr_sort_key(chr_name):
        """Convert chromosome name to sortable key."""
        chr_str = str(chr_name)
        if chr_str.startswith('chr'):
            chr_num = chr_str[3:]
            try:
                # Handle numeric chromosomes
                if chr_num.isdigit():
                    return (0, int(chr_num))
                # Handle X, Y, M
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
    
    # Create sort keys
    var_df['_chr_sort'] = var_df[chr_col].apply(get_chr_sort_key)
    var_df['_start'] = pd.to_numeric(var_df[start_col], errors='coerce')
    
    # Linear scan to count unsorted peaks
    print(f"\nPerforming linear scan to check peak ordering...")
    unsorted_count = 0
    total_valid_peaks = 0
    prev_chr = None
    prev_start = None
    unsorted_examples = []
    
    for i, (idx, row) in enumerate(var_df.iterrows()):
        curr_chr = row['_chr_sort']
        curr_start = row['_start']
        
        if pd.isna(curr_start):
            continue
        
        total_valid_peaks += 1
        
        if prev_chr is not None:
            is_unsorted = False
            reason = ""
            
            # Compare chromosomes
            if curr_chr < prev_chr:
                is_unsorted = True
                reason = "chromosome order"
            elif curr_chr == prev_chr:
                # Same chromosome, check start position
                if curr_start < prev_start:
                    is_unsorted = True
                    reason = "start position on same chromosome"
            
            if is_unsorted:
                unsorted_count += 1
                if len(unsorted_examples) < 5:  # Store first 5 examples
                    unsorted_examples.append({
                        'index': i,
                        'peak_id': idx,
                        'current': f"{row[chr_col]}:{row[start_col]}",
                        'previous': f"{var_df.iloc[i-1][chr_col]}:{var_df.iloc[i-1][start_col]}",
                        'reason': reason
                    })
        
        prev_chr = curr_chr
        prev_start = curr_start
        
        # Progress indicator for large datasets
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,} peaks...")
    
    # Calculate percentage
    if total_valid_peaks > 0:
        unsorted_percent = (unsorted_count / total_valid_peaks) * 100
        sorted_percent = 100 - unsorted_percent
        
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Total valid peaks: {total_valid_peaks:,}")
        print(f"  Unsorted peaks: {unsorted_count:,} ({unsorted_percent:.4f}%)")
        print(f"  Sorted peaks: {total_valid_peaks - unsorted_count:,} ({sorted_percent:.4f}%)")
        print(f"{'='*60}")
        
        if unsorted_count == 0:
            print("\n✓ ALL PEAKS ARE SORTED by genomic coordinates!")
        else:
            print(f"\n✗ {unsorted_percent:.4f}% OF PEAKS ARE NOT SORTED")
            if unsorted_examples:
                print(f"\nFirst {len(unsorted_examples)} examples of unsorted peaks:")
                for ex in unsorted_examples:
                    print(f"  Index {ex['index']} ({ex['peak_id']}):")
                    print(f"    Current:  {ex['current']}")
                    print(f"    Previous: {ex['previous']}")
                    print(f"    Reason:   {ex['reason']}")
    else:
        print("\nERROR: No valid peaks found with genomic coordinates")
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"  Total peaks: {len(var_df)}")
    unique_chrs = var_df[chr_col].nunique()
    print(f"  Unique chromosomes: {unique_chrs}")
    print(f"  Chromosomes: {sorted(var_df[chr_col].unique())[:20]}...")  # Show first 20
    
    # Show first and last few peaks
    print(f"\nFirst 10 peaks:")
    for i in range(min(10, len(var_df))):
        row = var_df.iloc[i]
        print(f"  {i}: {row[chr_col]}:{row[start_col]}-{row.get(end_col, 'N/A')}")
    
    print(f"\nLast 10 peaks:")
    for i in range(max(0, len(var_df)-10), len(var_df)):
        row = var_df.iloc[i]
        print(f"  {i}: {row[chr_col]}:{row[start_col]}-{row.get(end_col, 'N/A')}")


if __name__ == "__main__":
    # file_path = "/home/naimer/github/project-2-team-1-gal/benchmarks/data2/Kanemaru2023/Kanemaru2023-cardiac_tissue/Kanemaru2023-cardiac_tissue-cell_by_cCRE.h5ad"
    # file_path = "/home/naimer/github/project-2-team-1-gal/benchmarks/data2/Li2023b/Li2023b-brain_tissue/Li2023b-brain_tissue-cell_by_cCRE.h5ad"
    file_path = "/home/naimer/github/project-2-team-1-gal/benchmarks/data2/Buenrostro2018/Buenrostro2018-bone_marrow_tissue/Buenrostro2018-bone_marrow_tissue-cell_by_cCRE.h5ad"
    check_peaks_order(file_path)

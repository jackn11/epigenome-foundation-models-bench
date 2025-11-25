import numpy as np
from anndata import AnnData
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from tqdm import tqdm

def construct_cell_by_ccre_matrix(intersect_file, ccre_bed_path):
    """
    Constructs a cell-by-cCRE matrix from intersect results and cCRE definitions.

    Args:
        intersect_file (str): Path to the intersect result file.
        ccre_bed_path (str): Path to the cCRE definition file.

    Returns:
        AnnData: An AnnData object containing the cell-by-cCRE matrix.
    """
    # Read cCRE definitions
    ccre_df = pd.read_csv(ccre_bed_path, sep='\t', header=None, names=['chrom', 'start', 'end'])
    ccre_df['ccre_detail'] = ccre_df['chrom'].astype(str) + ':' + ccre_df['start'].astype(str) + '-' + ccre_df['end'].astype(str)
    ccre_details = pd.Series(ccre_df['ccre_detail'].values, index=ccre_df['ccre_detail']).to_dict()

    # Read intersect file
    intersect_df = pd.read_csv(intersect_file, sep='\t', header=None)

    # Determine format and extract overlap information
    if str(intersect_df.iloc[0, 4]).startswith('chr'):
        print("Detected no count information in intersect file. Using binary overlap (count = 1).")
        intersect_ccre_details = intersect_df.iloc[:, 4].astype(str) + ':' + intersect_df.iloc[:, 5].astype(str) + '-' + intersect_df.iloc[:, 6].astype(str)
        overlaps = np.ones(intersect_df.shape[0], dtype=np.float32)
    else:
        intersect_ccre_details = intersect_df.iloc[:, 5].astype(str) + ':' + intersect_df.iloc[:, 6].astype(str) + '-' + intersect_df.iloc[:, 7].astype(str)
        overlaps = intersect_df[4].values.astype(np.float32)

    # Map cells and cCREs to categorical indices
    cells = intersect_df[3]
    used_cells = np.unique(cells).astype(str)
    cell_ids = pd.Categorical(cells, categories=used_cells).codes
    ccre_ids = pd.Categorical(intersect_ccre_details, categories=ccre_details).codes

    # Create sparse matrix
    cell_ccre_matrix = coo_matrix((overlaps, (cell_ids, ccre_ids)), shape=(len(used_cells), len(ccre_details)), dtype=np.float32).tocsr()

    # Create AnnData
    adata = AnnData(X=cell_ccre_matrix)
    adata.obs_names = used_cells
    adata.var_names = pd.Series(list(ccre_details.keys()))

    return adata

def global_TFIDF(adata, cCRE_document_frequency):
    """
    Apply a global TF-IDF transformation to the input AnnData object.
    
    Parameters:
    - adata: AnnData
        The input AnnData object containing the sparse matrix in `adata.X`.
    - cCRE_document_frequency: ndarray
        A 1D numpy array representing the document frequency for each cCRE (column).

    Returns:
    - AnnData
        A new AnnData object with TF-IDF-transformed data in `adata.X`.
    """
    # Create a copy of the input AnnData object to avoid modifying the original
    adata_copy = adata.copy()

    # Step 1: Compute the reciprocal of column sums (document frequency)
    cCRE_doc_frequency_inv = csr_matrix(np.reciprocal(cCRE_document_frequency))

    # Step 2: Compute row sums (total counts per cell)
    row_sums = adata_copy.X.sum(axis=1)
    row_sums_inv = csr_matrix(np.reciprocal(row_sums.A.flatten()))  # Convert to CSR matrix for sparse multiplication

    # Step 3: Normalize rows by row sums
    adata_copy.X = adata_copy.X.multiply(row_sums_inv.T)

    # Step 4: Normalize columns by document frequency
    adata_copy.X = adata_copy.X.multiply(cCRE_doc_frequency_inv)

    # Step 5: Apply log transformation to all non-zero elements
    adata_copy.X.data = np.log1p(10000 * adata_copy.X.data)

    return adata_copy


def global_TFIDF_with_shuffling(adata, cCRE_document_frequency):
    """
    Apply a global TF-IDF transformation to the input AnnData object and shuffle non-zero values in each row randomly.
    
    Parameters:
    - adata: AnnData
        The input AnnData object containing the sparse matrix in `adata.X`.
    - cCRE_document_frequency: ndarray
        A 1D numpy array representing the document frequency for each cCRE (column).

    Returns:
    - AnnData
        A new AnnData object with TF-IDF-transformed data in `adata.X`.
    """
    # Create a copy of the input AnnData object to avoid modifying the original
    adata_copy = adata.copy()

    #########################################################################################################################
    # START OF SHUFFLING
    #########################################################################################################################
    
    # Permute non-zero values in each row randomly (zeros stay in place)
    if not isinstance(adata_copy.X, csr_matrix):
        adata_copy.X = csr_matrix(adata_copy.X)
    
    X = adata_copy.X
    n_rows = X.shape[0]
    
    # For each row, permute non-zero values
    for i in range(n_rows):
        row_start = X.indptr[i]
        row_end = X.indptr[i + 1]
        
        if row_end > row_start:
            # Extract non-zero values for this row
            row_data = X.data[row_start:row_end]
            permuted_data = np.random.permutation(row_data)
            X.data[row_start:row_end] = permuted_data

    #########################################################################################################################
    # END OF SHUFFLING
    #########################################################################################################################

    # Step 1: Compute the reciprocal of column sums (document frequency)
    cCRE_doc_frequency_inv = csr_matrix(np.reciprocal(cCRE_document_frequency))

    # Step 2: Compute row sums (total counts per cell)
    row_sums = adata_copy.X.sum(axis=1)
    row_sums_inv = csr_matrix(np.reciprocal(row_sums.A.flatten()))  # Convert to CSR matrix for sparse multiplication

    # Step 3: Normalize rows by row sums
    adata_copy.X = adata_copy.X.multiply(row_sums_inv.T)

    # Step 4: Normalize columns by document frequency
    adata_copy.X = adata_copy.X.multiply(cCRE_doc_frequency_inv)

    # Step 5: Apply log transformation to all non-zero elements
    adata_copy.X.data = np.log1p(10000 * adata_copy.X.data)

    return adata_copy

def global_TFIDF_with_complete_shuffling(adata, cCRE_document_frequency):
    """
    Apply a global TF-IDF transformation to the input AnnData object and shuffle all values randomly.
    
    Parameters:
    - adata: AnnData
        The input AnnData object containing the sparse matrix in `adata.X`.
    - cCRE_document_frequency: ndarray
        A 1D numpy array representing the document frequency for each cCRE (column).

    Returns:
    - AnnData
        A new AnnData object with TF-IDF-transformed data in `adata.X`.
    """
    # Create a copy of the input AnnData object to avoid modifying the original
    adata_copy = adata.copy()

    #########################################################################################################################
    # START OF COMPLETE SHUFFLING
    #########################################################################################################################
    
    total_nnz = adata_copy.X.nnz
    new_data = np.empty(total_nnz, dtype=adata_copy.X.dtype)
    new_indices = np.empty(total_nnz, dtype=adata_copy.X.indices.dtype)
    new_indptr = np.zeros(adata_copy.X.shape[0] + 1, dtype=adata_copy.X.indptr.dtype)

    current_idx = 0

    for i in tqdm(range(adata_copy.X.shape[0]), desc="Shuffling rows"):
        row_start = adata_copy.X.indptr[i]
        row_end = adata_copy.X.indptr[i + 1]
        num_nnz = row_end - row_start
        
        if num_nnz > 0:
            # Get values (no copy needed)
            row_vals = adata_copy.X.data[row_start:row_end]
            
            # Randomly sample column positions (efficient for large column spaces)
            # np.random.choice is optimized for sampling small numbers from large ranges
            shuffled_cols = np.random.choice(adata_copy.X.shape[1], size=num_nnz, replace=False)
            
            # Shuffle values
            shuffled_vals = np.random.permutation(row_vals)
            
            # Sort by column while maintaining value-column correspondence
            sort_idx = np.argsort(shuffled_cols)
            new_data[current_idx:current_idx + num_nnz] = shuffled_vals[sort_idx]
            new_indices[current_idx:current_idx + num_nnz] = shuffled_cols[sort_idx]
            
            current_idx += num_nnz
        
        new_indptr[i + 1] = current_idx

    # Build the new sparse matrix
    adata_copy_shuffled = csr_matrix((new_data, new_indices, new_indptr), shape=adata_copy.X.shape)

    adata_copy.X = adata_copy_shuffled

    #########################################################################################################################
    # END OF COMPLETE SHUFFLING
    #########################################################################################################################

    # Step 1: Compute the reciprocal of column sums (document frequency)
    cCRE_doc_frequency_inv = csr_matrix(np.reciprocal(cCRE_document_frequency))

    # Step 2: Compute row sums (total counts per cell)
    row_sums = adata_copy.X.sum(axis=1)
    row_sums_inv = csr_matrix(np.reciprocal(row_sums.A.flatten()))  # Convert to CSR matrix for sparse multiplication

    # Step 3: Normalize rows by row sums
    adata_copy.X = adata_copy.X.multiply(row_sums_inv.T)

    # Step 4: Normalize columns by document frequency
    adata_copy.X = adata_copy.X.multiply(cCRE_doc_frequency_inv)

    # Step 5: Apply log transformation to all non-zero elements
    adata_copy.X.data = np.log1p(10000 * adata_copy.X.data)

    return adata_copy

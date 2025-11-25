#!/usr/bin/env python

"""
GSE194122 → EpiAgent zero-shot feature extraction on BMMC multiome ATAC
- Uses processed NeurIPS 2021 openproblems multiome_BMMC h5ad
- Extracts ATAC peaks, maps to global cCREs, builds cell x cCRE matrix
- Runs EpiAgent to get cell embeddings
- UMAP colored by cell_type and batch
"""

import gzip
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from epiagent.preprocessing import global_TFIDF
from epiagent.tokenization import tokenization
from epiagent.dataset import CellDataset, collate_fn
from epiagent.model import EpiAgent
from epiagent.inference import infer_cell_embeddings


# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------
root = Path("/scratch/wkim/project-2-team-1/EpiAgent")
data_dir = root / "external_data" / "GSE194122"

multiome_h5ad_path = data_dir / "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz"

# intermediate / output files
peaks_indexed_bed_path = data_dir / "BMMC_ATAC_peaks_indexed.bed"
peaks_ccre_overlap_path = data_dir / "BMMC_ATAC_peaks_ccre_overlaps.tsv"
X_ccre_npz_path = data_dir / "BMMC_ATAC_X_ccre.npz"
processed_adata_path = data_dir / "BMMC_ATAC_tokenized.h5ad"

# EpiAgent / global cCRE references
tmpl_h5ad_path = root / "data" / "sample" / "raw_h5ad" / "Kanemaru2023_downsampled_10000_cells.h5ad"
global_ccres_bed_path = root / "data" / "global_ccres_with_index.bed"
cCRE_df_path = root / "data" / "cCRE_document_frequency.npy"
model_path = root / "model" / "pretrained_EpiAgent.pth"


# ---------------------------------------------------------------------
# Utils: read gzipped h5ad files
# ---------------------------------------------------------------------
def read_h5ad_gz(gz_path):
    """Read a gzip-compressed h5ad file."""
    print(f"[progress] Starting to decompress and read h5ad file: {gz_path}")
    with gzip.open(gz_path, 'rb') as gz_file:
        print("[progress] Decompressing gzip file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as tmp_file:
            tmp_file.write(gz_file.read())
            tmp_path = tmp_file.name
    print("[progress] Decompression complete, reading h5ad...")
    
    try:
        adata = sc.read_h5ad(tmp_path)
        print("[progress] Finished reading h5ad file")
        return adata
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink()


# ---------------------------------------------------------------------
# Utils: ensure global cCRE BED
# ---------------------------------------------------------------------
def ensure_global_ccre_bed():
    if global_ccres_bed_path.exists():
        print(f"[info] Using existing global cCRE BED: {global_ccres_bed_path}")
        return

    print("[info] Creating global cCRE BED from Kanemaru2023_downsampled_10000_cells.h5ad ...")
    tmpl = sc.read_h5ad(tmpl_h5ad_path)
    names = pd.Index(tmpl.var_names)  # strings like "chr1:9848-10355"

    ccre_df = names.to_series().str.extract(r"^(chr[^:]+):(\d+)-(\d+)$")
    ccre_df.columns = ["chrom", "start", "end"]
    ccre_df["ccre_idx"] = np.arange(len(ccre_df))

    global_ccres_bed_path.parent.mkdir(parents=True, exist_ok=True)
    ccre_df[["chrom", "start", "end", "ccre_idx"]].to_csv(
        global_ccres_bed_path, sep="\t", header=False, index=False
    )
    print(f"[info] Saved global cCRE BED: {global_ccres_bed_path}")


# ---------------------------------------------------------------------
# Step 1: Load multiome BMMC and extract ATAC peak matrix
# ---------------------------------------------------------------------
def load_atac_from_multiome():
    print(f"[info] Loading multiome AnnData: {multiome_h5ad_path}")
    adata = read_h5ad_gz(multiome_h5ad_path)
    print("[info] Loaded:", adata)

    if "feature_types" not in adata.var.columns:
        raise ValueError("feature_types column not found in adata.var; cannot separate ATAC/GEX.")

    print("[info] feature_types value counts:")
    print(adata.var["feature_types"].value_counts())

    # Try common labels: 'Peaks' (10x style) or 'ATAC'
    unique_ft = adata.var["feature_types"].unique().tolist()
    if "Peaks" in unique_ft:
        atac_mask = adata.var["feature_types"] == "Peaks"
    elif "ATAC" in unique_ft:
        atac_mask = adata.var["feature_types"] == "ATAC"
    else:
        raise ValueError(
            f"Could not find 'Peaks' or 'ATAC' among feature_types: {unique_ft}"
        )

    adata_atac = adata[:, atac_mask].copy()
    print("[info] Subset to ATAC features:", adata_atac)

    # Get counts matrix: prefer 'counts' layer if present
    if "counts" in adata_atac.layers:
        X = adata_atac.layers["counts"]
        print("[info] Using adata_atac.layers['counts'] as ATAC matrix")
    else:
        X = adata_atac.X
        print("[info] Using adata_atac.X as ATAC matrix")

    # Ensure sparse CSR (cells x peaks)
    if not sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float32)
    else:
        X = X.tocsr().astype(np.float32)

    n_cells, n_peaks = X.shape
    print(f"[info] ATAC matrix shape (cells x peaks): {X.shape}")

    # We already inspected adata_atac.var_names and saw things like:
    #   chr1-9776-10668
    #   chr1-180726-181005
    # so we just use var_names as the peak coordinates.
    var_names_str = adata_atac.var_names.astype(str)
    peak_coords = var_names_str.to_series()
    print("[info] Using var_names as peak coordinates; examples:")
    print(peak_coords.head())

    return adata_atac, X, peak_coords


# ---------------------------------------------------------------------
# Step 2: Map peaks -> global cCREs using bedtools, build X_ccre
# ---------------------------------------------------------------------
def build_X_ccre():
    if X_ccre_npz_path.exists():
        print(f"[info] Loading existing cell x cCRE matrix: {X_ccre_npz_path}")
        return sp.load_npz(X_ccre_npz_path)

    # Load ATAC data
    adata_atac, X_peaks, peak_coords = load_atac_from_multiome()
    n_cells, n_peaks = X_peaks.shape

    # Write peaks with 0-based index to BED
    print("[info] Creating peaks BED with indices for bedtools ...")
    print("[progress] Converting peak coordinates to DataFrame...")
    peak_df = peak_coords.to_frame(name="coord")
    coord = peak_df["coord"].astype(str)

    print("[progress] Parsing peak coordinates (chr:start-end format)...")
    # Try chr:start-end
    parsed_colon = coord.str.extract(r"^(chr[^:]+):(\d+)-(\d+)$")
    mask_colon = parsed_colon.notna().all(axis=1)

    print("[progress] Parsing peak coordinates (chr-start-end format)...")
    # Try chr-start-end (your case)
    parsed_dash = coord.str.extract(r"^(chr[^-]+)-(\d+)-(\d+)$")
    mask_dash = parsed_dash.notna().all(axis=1)

    print("[progress] Combining parsed coordinates...")
    # Combine the two
    parsed_combined = pd.DataFrame(index=coord.index, columns=["chrom", "start", "end"])
    parsed_combined.loc[mask_colon, :] = parsed_colon.loc[mask_colon, :].values
    parsed_combined.loc[mask_dash, :] = parsed_dash.loc[mask_dash, :].values

    mask_any = mask_colon | mask_dash
    n_valid = mask_any.sum()
    n_total = len(coord)
    print(f"[info] Parsed {n_valid} / {n_total} peaks into chrom/start/end")

    if n_valid == 0:
        raise ValueError("No peak coordinates could be parsed into chrom/start/end")

    print("[progress] Filtering valid peaks...")
    # Keep only the valid peaks
    parsed_final = parsed_combined.loc[mask_any].copy()
    parsed_final["peak_idx0"] = np.arange(n_peaks)[mask_any.to_numpy()]

    print("[progress] Writing peaks BED file...")
    parsed_final.to_csv(peaks_indexed_bed_path, sep="\t", header=False, index=False)
    print(f"[info] Saved peaks BED with {n_valid} peaks: {peaks_indexed_bed_path}")

    # Ensure global cCRE BED exists
    ensure_global_ccre_bed()

    # Run bedtools intersect if needed
    if not peaks_ccre_overlap_path.exists():
        print("[info] Running bedtools intersect to map peaks -> cCREs ...")
        print("[progress] This may take several minutes depending on data size...")
        cmd = [
            "bedtools",
            "intersect",
            "-a",
            str(peaks_indexed_bed_path),
            "-b",
            str(global_ccres_bed_path),
            "-wa",
            "-wb",
        ]
        with open(peaks_ccre_overlap_path, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, check=True)
        print(f"[progress] Bedtools intersect complete. Saved overlaps: {peaks_ccre_overlap_path}")
    else:
        print(f"[info] Using existing overlaps: {peaks_ccre_overlap_path}")

    print("[progress] Loading overlaps from file...")
    overlaps = pd.read_csv(
        peaks_ccre_overlap_path,
        sep="\t",
        header=None,
        names=[
            "chrom_p",
            "start_p",
            "end_p",
            "peak_idx0",
            "chrom_c",
            "start_c",
            "end_c",
            "ccre_idx",
        ],
    )
    print(f"[progress] Loaded overlaps. n overlaps: {len(overlaps)}")

    # Load template to know number of cCREs
    print("[progress] Loading template h5ad to get cCRE count...")
    tmpl = sc.read_h5ad(tmpl_h5ad_path)
    n_ccre = tmpl.n_vars
    print(f"[info] n_cCRE (from template): {n_ccre}")

    # Convert X_peaks to CSC for efficient column access (per-peak)
    print("[progress] Converting X_peaks to CSC format...")
    X_peaks_csc = X_peaks.tocsc()
    print("[progress] CSC conversion complete")

    print("[info] Building cell x cCRE matrix (this may take a while) ...")
    print("[progress] Collecting (row, col, value) triplets...")
    # Collect all (row, col, value) triplets to build matrix efficiently
    rows = []
    cols = []
    data = []

    grouped = overlaps.groupby("peak_idx0")
    n_groups = len(grouped)
    print(f"[progress] Processing {n_groups} peak groups...")
    for i, (peak_idx0, df_p) in enumerate(grouped):
        if (i + 1) % 1000 == 0:
            print(f"[progress] Processed {i + 1} / {n_groups} peak groups...")
        # counts for this peak across all cells: shape (n_cells, 1)
        vec_col = X_peaks_csc[:, int(peak_idx0)]
        if vec_col.nnz == 0:
            continue

        ccre_indices = df_p["ccre_idx"].astype(int).unique()
        # Extract non-zero entries from vec_col
        vec_col_coo = vec_col.tocoo()
        # For each cCRE that overlaps with this peak, add the peak's counts
        for c_idx in ccre_indices:
            rows.extend(vec_col_coo.row)
            cols.extend([c_idx] * len(vec_col_coo.row))
            data.extend(vec_col_coo.data)

    print(f"[progress] Finished collecting triplets. Total entries: {len(data)}")
    print("[progress] Building COO matrix...")
    # Build matrix in one go using COO format (most efficient)
    if len(data) > 0:
        X_ccre = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_ccre), dtype=np.float32)
        print("[progress] Converting COO to CSR format...")
        # Convert to CSR for efficient row operations
        X_ccre = X_ccre.tocsr()
        print("[progress] CSR conversion complete")
    else:
        X_ccre = sp.csr_matrix((n_cells, n_ccre), dtype=np.float32)

    print("[info] Finished building X_ccre; shape:", X_ccre.shape)
    print("[progress] Saving X_ccre to npz file...")
    sp.save_npz(X_ccre_npz_path, X_ccre)
    print(f"[progress] Saved X_ccre to: {X_ccre_npz_path}")
    return X_ccre


# ---------------------------------------------------------------------
# Step 3: Build AnnData for EpiAgent, run TF-IDF + tokenization
# ---------------------------------------------------------------------
def build_and_tokenize_adata():
    if processed_adata_path.exists():
        print(f"[info] Loading preprocessed AnnData from: {processed_adata_path}")
        adata_ccre = sc.read_h5ad(processed_adata_path)
        return adata_ccre

    print("[info] Building cell x cCRE matrix ...")
    X_ccre = build_X_ccre()

    # Reload ATAC subset to get obs (cell_type, batch, etc.)
    print("[progress] Reloading ATAC data to get cell metadata...")
    adata_atac, _, _ = load_atac_from_multiome()
    print("[progress] ATAC data reloaded")

    print("[progress] Loading template h5ad...")
    tmpl = sc.read_h5ad(tmpl_h5ad_path)
    print("[progress] Template loaded")

    print("[info] Wrapping into AnnData (cells x cCREs) ...")
    adata_ccre = ad.AnnData(X_ccre)
    adata_ccre.var = tmpl.var.copy()  # global cCREs in correct order
    adata_ccre.obs = adata_atac.obs.copy()
    adata_ccre.obs["dataset"] = "BMMC_multiome_ATAC"

    # Apply TF-IDF + tokenization
    print("[info] Applying TF-IDF ...")
    print("[progress] Loading cCRE document frequency...")
    cCRE_document_frequency = np.load(cCRE_df_path)
    print("[progress] Computing TF-IDF (this may take a while)...")
    global_TFIDF(adata_ccre, cCRE_document_frequency)
    print("[progress] TF-IDF complete")

    print("[info] Tokenizing ...")
    print("[progress] Tokenizing cell sentences (this may take a while)...")
    tokenization(adata_ccre)
    print("[progress] Tokenization complete")

    processed_adata_path.parent.mkdir(parents=True, exist_ok=True)
    print("[progress] Writing processed AnnData to disk...")
    adata_ccre.write(processed_adata_path)
    print(f"[progress] Saved tokenized AnnData to: {processed_adata_path}")
    return adata_ccre


# ---------------------------------------------------------------------
# Step 4: Run EpiAgent to get embeddings
# ---------------------------------------------------------------------
def run_epiaent_embeddings(adata_ccre: ad.AnnData):
    print("[info] Preparing dataloader for EpiAgent ...")
    print("[progress] Creating CellDataset...")
    cell_dataset = CellDataset(cell_sentences=adata_ccre.obs["cell_sentences"].tolist())
    print("[progress] Creating DataLoader...")
    dataloader = DataLoader(
        cell_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    print("[progress] DataLoader created")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    print("[progress] Initializing EpiAgent model...")
    model = EpiAgent(
        vocab_size=1355449,
        num_layers=18,
        embedding_dim=512,
        num_attention_heads=8,
        max_rank_embeddings=8192,
        use_flash_attn=True,
        pos_weight_for_RLM=torch.tensor(1.0),
        pos_weight_for_CCA=torch.tensor(1.0),
    )
    print("[progress] Loading pretrained model weights...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[progress] Model loaded, moving to device...")
    model = model.to(device)
    model.eval()
    print("[progress] Model ready")

    print("[info] Inferring cell embeddings with EpiAgent (this may take a while) ...")
    embeddings = infer_cell_embeddings(model, device, dataloader)
    print("[progress] Embedding inference complete")
    adata_ccre.obsm["cell_embeddings"] = embeddings
    return adata_ccre


# ---------------------------------------------------------------------
# Step 5: UMAP + visualization
# ---------------------------------------------------------------------
def run_umap_and_plot(adata_ccre: ad.AnnData):
    print("[info] Running neighbors/UMAP on EpiAgent embeddings ...")
    print("[progress] Computing neighbors graph (this may take a while)...")
    sc.pp.neighbors(adata_ccre, use_rep="cell_embeddings")
    print("[progress] Neighbors graph complete")
    print("[progress] Computing UMAP (this may take a while)...")
    sc.tl.umap(adata_ccre)
    print("[progress] UMAP complete")

    # UMAP colored by true cell type
    if "cell_type" in adata_ccre.obs.columns:
        print("[progress] Plotting UMAP colored by cell_type...")
        sc.pl.umap(adata_ccre, color="cell_type", save="_GSE194122_BMMC_EpiAgent_celltype.png")
        print("[progress] Cell type plot saved")
    else:
        print("[warn] 'cell_type' not found in obs; available columns:")
        print(adata_ccre.obs.columns)

    # UMAP colored by batch (donor/batch structure)
    if "batch" in adata_ccre.obs.columns:
        print("[progress] Plotting UMAP colored by batch...")
        sc.pl.umap(adata_ccre, color="batch", save="_GSE194122_BMMC_EpiAgent_batch.png")
        print("[progress] Batch plot saved")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("[START] GSE194122 BMMC multiome → EpiAgent pipeline")
    print("=" * 80)
    
    print("\n[STEP 1/3] Building and tokenizing AnnData...")
    adata_ccre = build_and_tokenize_adata()
    print("[STEP 1/3] Complete\n")
    
    print("\n[STEP 2/3] Running EpiAgent embeddings...")
    adata_ccre = run_epiaent_embeddings(adata_ccre)
    print("[STEP 2/3] Complete\n")
    
    print("\n[STEP 3/3] Computing UMAP and generating plots...")
    run_umap_and_plot(adata_ccre)
    print("[STEP 3/3] Complete\n")
    
    print("=" * 80)
    print("[DONE] GSE194122 BMMC multiome → EpiAgent pipeline finished.")
    print("=" * 80)

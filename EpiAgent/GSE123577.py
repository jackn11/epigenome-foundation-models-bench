# GSE123577.py
# Zero-shot EpiAgent embeddings on Buenrostro PBMC scATAC (GSE123577)

import subprocess
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
data_dir = root / "external_data" / "GSE123577"

counts_path = data_dir / "GSE123577_pbmc_countsData.csv.gz"
peaks_bed_gz_path = data_dir / "GSE123577_pbmc_peaks.bed.gz"
cell_meta_path = data_dir / "GSE123577_pbmc_cellData.tsv.gz"
barcode_translate_path = data_dir / "GSE123577_pbmc.barcodeTranslate.tsv.gz"  # not yet used

processed_adata_path = data_dir / "pbmc_tokenized.h5ad"
X_ccre_npz_path = data_dir / "pbmc_X_ccre.npz"
peaks_indexed_bed_path = data_dir / "pbmc_peaks_indexed.bed"
peaks_ccre_overlap_path = data_dir / "pbmc_peaks_ccre_overlaps.tsv"

global_ccres_bed_path = root / "data" / "global_ccres_with_index.bed"

tmpl_h5ad_path = root / "data" / "sample" / "raw_h5ad" / "Kanemaru2023_downsampled_10000_cells.h5ad"
cCRE_df_path = root / "data" / "cCRE_document_frequency.npy"
model_path = root / "model" / "pretrained_EpiAgent.pth"


# ---------------------------------------------------------------------
# Step 0: Make global_ccres_with_index.bed if it doesn't exist
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
# Step 1: Build cell x cCRE matrix from Buenrostro PBMC data
# ---------------------------------------------------------------------
def build_X_ccre():
    if X_ccre_npz_path.exists():
        print(f"[info] Loading existing cell x cCRE matrix: {X_ccre_npz_path}")
        return sp.load_npz(X_ccre_npz_path)

    print("[info] Loading countsData as COO (peak_idx, cell_idx, count) ...")
    # countsData is whitespace-separated: "peak_idx cell_idx count"
    coo_df = pd.read_csv(
        counts_path,
        delim_whitespace=True,
        header=0,
        names=["peak_idx", "cell_idx", "count"],
    )
    print(f"[info] countsData rows (nonzero entries): {len(coo_df)}")

    n_peaks = int(coo_df["peak_idx"].max())
    n_cells = int(coo_df["cell_idx"].max())
    print(f"[info] n_peaks (max peak_idx in countsData): {n_peaks}")
    print(f"[info] n_cells (from countsData): {n_cells}")

    print("[info] Loading peaks BED ...")
    peaks_df = pd.read_csv(
        peaks_bed_gz_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end"],
    )
    n_peaks_bed = peaks_df.shape[0]
    print("[info] peaks BED shape:", peaks_df.shape)

    if n_peaks_bed < n_peaks:
        raise ValueError(
            f"peaks.bed has fewer rows ({n_peaks_bed}) than max peak_idx in countsData ({n_peaks})."
        )

    # We adopt the BED as ground truth for number of peaks.
    # Peaks with index > max(peak_idx) in countsData will just have zero counts.
    n_peaks_full = n_peaks_bed

    # Convert to sparse peaks x cells matrix (rows=peaks, cols=cells)
    print("[info] Building sparse peaks x cells matrix ...")
    peak_idx0 = coo_df["peak_idx"].astype(int) - 1  # 0-based
    cell_idx0 = coo_df["cell_idx"].astype(int) - 1  # 0-based
    vals = coo_df["count"].astype(np.float32)

    X_peaks = sp.coo_matrix(
        (vals, (peak_idx0, cell_idx0)), shape=(n_peaks_full, n_cells)
    ).tocsr()
    del coo_df  # free memory
    print("[info] X_peaks shape:", X_peaks.shape, "(peaks x cells)")

    # Add a 0-based peak index for bedtools
    print("[info] Writing indexed peaks BED for bedtools ...")
    peaks_indexed = peaks_df.copy()
    peaks_indexed["peak_idx0"] = np.arange(n_peaks_bed)
    peaks_indexed.to_csv(
        peaks_indexed_bed_path, sep="\t", header=False, index=False
    )

    # Ensure global cCRE BED exists
    ensure_global_ccre_bed()

    # bedtools intersect: peaks x cCRE
    if not peaks_ccre_overlap_path.exists():
        print("[info] Running bedtools intersect to map peaks -> cCREs ...")
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
        print(f"[info] Saved overlaps: {peaks_ccre_overlap_path}")
    else:
        print(f"[info] Using existing overlap file: {peaks_ccre_overlap_path}")

    print("[info] Loading overlaps ...")
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
    print(f"[info] n overlaps: {len(overlaps)}")

    # Determine number of cCREs from template
    tmpl = sc.read_h5ad(tmpl_h5ad_path)
    n_ccre = tmpl.n_vars
    print(f"[info] n_cCRE (from template): {n_ccre}")

    # Initialize cell x cCRE matrix
    print("[info] Building cell x cCRE matrix (this may take a while) ...")
    X_ccre = sp.csr_matrix((n_cells, n_ccre), dtype=np.float32)

    grouped = overlaps.groupby("peak_idx0")
    for peak_idx0, df_p in grouped:
        # counts for this peak across cells -> shape (n_cells,)
        vec = X_peaks[peak_idx0, :].toarray().ravel()
        if vec.sum() == 0:
            continue
        ccre_indices = df_p["ccre_idx"].astype(int).unique()
        v = sp.csr_matrix(vec).T  # (n_cells, 1)
        for c_idx in ccre_indices:
            X_ccre[:, c_idx] += v

    print("[info] Finished building X_ccre; shape:", X_ccre.shape)
    sp.save_npz(X_ccre_npz_path, X_ccre)
    print(f"[info] Saved X_ccre to: {X_ccre_npz_path}")
    return X_ccre


# ---------------------------------------------------------------------
# Step 2: Build AnnData, attach minimal metadata, TF-IDF + tokenization
# ---------------------------------------------------------------------
def build_and_tokenize_adata():
    if processed_adata_path.exists():
        print(f"[info] Loading preprocessed AnnData from: {processed_adata_path}")
        adata = sc.read_h5ad(processed_adata_path)
        return adata

    X_ccre = build_X_ccre()
    n_cells = X_ccre.shape[0]

    # For now, index cells as "cell_1" ... "cell_N"
    cell_ids = [f"cell_{i+1}" for i in range(n_cells)]

    # Load cell metadata (we'll at least keep it; exact schema might need inspection)
    print("[info] Loading PBMC cell metadata (for later label use) ...")
    try:
        cell_meta = pd.read_csv(cell_meta_path, sep="\t")
        print("[info] cell_meta shape:", cell_meta.shape)
        print("[info] cell_meta columns:", list(cell_meta.columns)[:10])
    except Exception as e:
        print("[warn] Could not parse cell metadata cleanly:", e)
        cell_meta = pd.DataFrame(index=range(1, n_cells + 1))

    # Build AnnData
    tmpl = sc.read_h5ad(tmpl_h5ad_path)

    print("[info] Wrapping into AnnData ...")
    adata = ad.AnnData(X_ccre)
    adata.var = tmpl.var.copy()
    adata.obs = pd.DataFrame(index=cell_ids)
    adata.obs["cell_idx"] = np.arange(1, n_cells + 1)
    adata.obs["dataset"] = "Buenrostro_PBMC"

    # TODO: After inspecting cell_meta, join useful columns to adata.obs.

    # TF-IDF + tokenization
    print("[info] Applying TF-IDF ...")
    cCRE_document_frequency = np.load(cCRE_df_path)
    global_TFIDF(adata, cCRE_document_frequency)

    print("[info] Tokenizing ...")
    tokenization(adata)

    processed_adata_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(processed_adata_path)
    print(f"[info] Saved tokenized AnnData to: {processed_adata_path}")
    return adata


# ---------------------------------------------------------------------
# Step 3: Run EpiAgent to get embeddings
# ---------------------------------------------------------------------
def run_epiaent_embeddings(adata: ad.AnnData):
    print("[info] Preparing dataloader for EpiAgent ...")
    cell_dataset = CellDataset(cell_sentences=adata.obs["cell_sentences"].tolist())
    dataloader = DataLoader(
        cell_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

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
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("[info] Inferring cell embeddings with EpiAgent ...")
    embeddings = infer_cell_embeddings(model, device, dataloader)
    adata.obsm["cell_embeddings"] = embeddings
    return adata


# ---------------------------------------------------------------------
# Step 4: UMAP + basic visualization
# ---------------------------------------------------------------------
def run_umap_and_plot(adata: ad.AnnData):
    print("[info] Running neighbors/UMAP on EpiAgent embeddings ...")
    sc.pp.neighbors(adata, use_rep="cell_embeddings")
    sc.tl.umap(adata)

    # For now, color by cell_idx; we'll wire in real labels next pass
    sc.pl.umap(adata, color="cell_idx", save="_GSE123577_PBMC_EpiAgent.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    adata = build_and_tokenize_adata()
    adata = run_epiaent_embeddings(adata)
    run_umap_and_plot(adata)
    print("[done] GSE123577 EpiAgent pipeline finished.")
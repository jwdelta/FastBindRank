#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.decomposition import PCA


def fp_to_numpy(fp, nbits: int) -> np.ndarray:
    arr = np.zeros((nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def plot_pca(X, df_plot, out_pdf, fingerprint_name, title_suffix=""):
    pca = PCA(n_components=2, random_state=1)
    emb = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    df_plot = df_plot.copy()
    df_plot["PC1"] = emb[:, 0]
    df_plot["PC2"] = emb[:, 1]

    plt.figure(figsize=(6, 5))
    plt.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["cluster_id"].astype(float), s=4, alpha=0.6)
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    plt.title(f"PCA ({fingerprint_name}) {title_suffix}")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-log", required=True)
    parser.add_argument("--pca-figure", required=True)
    parser.add_argument("--prob-threshold", type=float, required=True)
    parser.add_argument("--logic50-threshold", type=float, required=True)
    parser.add_argument("--drug-like", default="Yes")
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    smiles_col = "SMILES"
    prob_col = "Boltz2_pred_prob"
    logic50_col = "Boltz2_pred_log10IC50"
    drug_like_col = "Drug_like"

    mask = (
        (df[prob_col].astype(float) > args.prob_threshold)
        & (df[logic50_col].astype(float) < args.logic50_threshold)
        & (df[drug_like_col].astype(str) == args.drug_like)
        & (df[smiles_col].notna())
    )

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.fingerprint_radius, fpSize=args.fingerprint_bits
    )

    fps = []
    row_idx = []
    X_rows = []
    for idx, smiles in zip(df.loc[mask].index.tolist(), df.loc[mask, smiles_col].astype(str).tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        fps.append(fp)
        row_idx.append(idx)
        X_rows.append(fp_to_numpy(fp, args.fingerprint_bits))

    df["cluster_id"] = pd.NA
    df["cluster_size"] = pd.NA
    df["passed_filter"] = mask
    df["tanimoto_cutoff"] = args.similarity_threshold

    if not fps:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_file, index=False)
        Path(args.summary_log).write_text("No valid molecules for clustering.\n")
        return

    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(
        dists,
        nPts=len(fps),
        distThresh=1.0 - args.similarity_threshold,
        isDistData=True,
    )

    for cid, members in enumerate(clusters, start=1):
        for member in members:
            df.loc[row_idx[member], "cluster_id"] = cid
            df.loc[row_idx[member], "cluster_size"] = len(members)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)

    cs = df.loc[df["passed_filter"] == True, "cluster_size"].dropna().astype(int)
    with Path(args.summary_log).open("w") as handle:
        handle.write("=== Butina clustering summary ===\n")
        handle.write(f"Timestamp: {datetime.now()}\n")
        handle.write(f"num_clusters_total\t{len(clusters)}\n")
        handle.write(f"num_clustered_molecules\t{cs.shape[0]}\n")
        handle.write(f"largest_cluster_size\t{cs.max() if not cs.empty else 'NA'}\n")

    X = np.vstack(X_rows)
    df_plot = df.loc[row_idx].copy()
    plot_pca(
        X,
        df_plot,
        args.pca_figure,
        f"ECFP{args.fingerprint_radius * 2}_{args.fingerprint_bits}",
        title_suffix=f"(n={len(df_plot)})",
    )


if __name__ == "__main__":
    main()

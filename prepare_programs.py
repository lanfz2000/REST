#!/usr/bin/env python3
"""Build the Hallmark, KEGG, and PanglaoDB program bank used by REST."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_gmt(path: Path, source: str, min_genes: int) -> list[tuple[str, str, str | None, set[str]]]:
    programs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        genes = {gene.strip().upper() for gene in fields[2:] if gene.strip()}
        if len(genes) >= min_genes:
            programs.append((fields[0], source, None, genes))
    return programs


def panglao_programs(path: Path, min_genes: int) -> list[tuple[str, str, str, set[str]]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"species", "official gene symbol", "cell type", "canonical marker"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PanglaoDB table is missing: {sorted(missing)}")
    frame = frame[frame["species"].isin(["Hs", "Mm Hs"])]
    frame = frame[frame["canonical marker"].fillna(0).astype(int) == 1].copy()
    frame["gene"] = frame["official gene symbol"].astype(str).str.strip().str.upper()
    programs = []
    for cell_type, group in frame.groupby("cell type", sort=True):
        genes = set(group["gene"]) - {""}
        if len(genes) < min_genes:
            continue
        name = "PANGLAO_CELL_" + re.sub(r"[^A-Z0-9]+", "_", str(cell_type).upper()).strip("_")
        programs.append((name, "PanglaoDBCanonical", str(cell_type), genes))
    return programs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hallmark-gmt", type=Path, required=True)
    parser.add_argument("--kegg-gmt", type=Path, required=True)
    parser.add_argument("--panglao-tsv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--min-genes", type=int, default=3)
    args = parser.parse_args()

    programs = read_gmt(args.hallmark_gmt, "Hallmark", args.min_genes)
    programs += read_gmt(args.kegg_gmt, "KEGG", args.min_genes)
    programs += panglao_programs(args.panglao_tsv, args.min_genes)
    if not programs:
        raise ValueError("No programs passed filtering")
    genes = sorted(set().union(*(members for _, _, _, members in programs)))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    rows, columns = [], []
    for row, (_, _, _, members) in enumerate(programs):
        rows.extend([row] * len(members))
        columns.extend(gene_index[gene] for gene in sorted(members))
    membership = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(programs), len(genes)),
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.output_prefix.with_suffix(".npz"), membership, compressed=True)
    metadata = {
        "shape": list(membership.shape),
        "genes": genes,
        "pathways": [name for name, _, _, _ in programs],
        "pathway_sources": [source for _, source, _, _ in programs],
        "panglao_cell_labels": [label for _, _, label, _ in programs],
        "pathway_sizes": [len(members) for _, _, _, members in programs],
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {membership.shape[0]} programs over {membership.shape[1]} genes")


if __name__ == "__main__":
    main()

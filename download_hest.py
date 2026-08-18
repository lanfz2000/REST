#!/usr/bin/env python3
"""Download paired HEST-1k patches and spatial transcriptomics files."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", nargs="+", help="HEST sample IDs, for example TENX95")
    parser.add_argument("--all", action="store_true", help="Download all patches and ST files")
    args = parser.parse_args()
    if bool(args.ids) == bool(args.all):
        parser.error("Specify either --ids or --all")

    patterns = ["HEST_v1_3_0.csv"]
    if args.all:
        patterns += ["patches/**", "st/**", "metadata/**"]
    else:
        for sample_id in args.ids:
            patterns += [
                f"patches/{sample_id}.h5",
                f"st/{sample_id}.h5ad",
                f"metadata/{sample_id}.json",
            ]
    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        local_dir=args.output,
        allow_patterns=patterns,
    )


if __name__ == "__main__":
    main()

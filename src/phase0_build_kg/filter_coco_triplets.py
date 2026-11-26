"""
Phase 0 (Step 3): Filter Merged Triplets to Canonical MS-COCO / COCO-Stuff Vocabulary.

Retains triplets whose resolved `subject` and `object` both belong to the canonical
entity inventory (`knowledge_graph/coco_alias_v2.txt`) and aggregates raw surface counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase0_build_kg.merge_kb_triplets import build_alias_lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter merged triplets to canonical MS-COCO / COCO-Stuff classes.")
    parser.add_argument("--merged-csv", type=Path, default=Path("outputs/kg_build/alias_merged_triplets.csv.gz"))
    parser.add_argument("--alias-file", type=Path, default=ROOT / "knowledge_graph" / "coco_alias_v2.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kg_build"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, coco_classes = build_alias_lookup(args.alias_file)

    df = pd.read_csv(args.merged_csv, dtype=str, compression="infer")
    for col in ("freq_coco", "freq_vg", "freq_combined"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    mask = (
        df["subject"].str.strip().str.lower().isin(coco_classes)
        & df["object"].str.strip().str.lower().isin(coco_classes)
    )
    kept = df[mask].copy()

    coco_only = (
        kept.groupby(["subject", "predicate", "object"], sort=False)[["freq_coco", "freq_vg", "freq_combined"]]
        .sum()
        .reset_index()
        .sort_values(["subject", "predicate", "object"])
        .reset_index(drop=True)
    )
    out_csv = args.output_dir / "coco_only_triplets.csv"
    coco_only.to_csv(out_csv, index=False)
    print(f"[Filter COCO] Retained {len(coco_only):,} canonical triplets -> {out_csv}")


if __name__ == "__main__":
    main()

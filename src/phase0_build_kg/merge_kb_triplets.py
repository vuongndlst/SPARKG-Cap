"""
Phase 0 (Step 2): Merge MS-COCO Training Triplets and Visual Genome Triplets.

Resolves surface entity mentions to canonical MS-COCO / COCO-Stuff classes via longest
left-anchored n-gram matching (`knowledge_graph/coco_alias_v2.txt`), while strictly
filtering out any image belonging to the Karpathy `val` or `test` splits.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import get_heldout_coco_ids

_QUOTE_RE = re.compile(r'["\']')


def build_alias_lookup(path: Path) -> Tuple[Dict[str, str], Set[str]]:
    """Load `lowercase_alias -> canonical_class` mapping and canonical class set."""
    lookup: Dict[str, str] = {}
    canonical_classes: Set[str] = set()
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            canonical, rest = line.split(":", 1)
            canonical_clean = canonical.strip().lower()
            if not canonical_clean:
                continue
            canonical_classes.add(canonical_clean)
            lookup[canonical_clean] = canonical_clean
            if rest.strip() == "(no aliases)":
                continue
            for alias in rest.split(","):
                a = alias.strip().lower()
                if a:
                    lookup[a] = canonical_clean
    return lookup, canonical_classes


def build_resolver(lookup: Dict[str, str]) -> Callable[[str], str]:
    """Longest left-anchored n-gram resolver with memoization."""
    cache: Dict[str, str] = {}

    def _candidates(words: List[str]):
        n = len(words)
        for length in range(n, 0, -1):
            for start in range(n - length + 1):
                yield " ".join(words[start : start + length])

    def resolve(raw: str) -> str:
        if not raw:
            return ""
        if raw in cache:
            return cache[raw]
        cleaned = _QUOTE_RE.sub("", raw).strip().lower()
        words = cleaned.split()
        result = cleaned
        for cand in _candidates(words):
            if cand in lookup:
                result = lookup[cand]
                break
        cache[raw] = result
        return result

    return resolve


def stream_coco_triplets(path: Path, heldout_coco_ids: Set[int], chunksize: int = 100_000) -> Counter:
    cnt: Counter = Counter()
    if not path.is_file():
        return cnt
    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    use_cols = [c for c in ["image_id", "subject", "predicate", "object", "subject_coco_class", "object_coco_class"] if c in header_cols]
    for chunk in pd.read_csv(path, dtype=str, usecols=use_cols, chunksize=chunksize, compression="infer"):
        chunk.fillna("", inplace=True)
        if "image_id" in chunk.columns and heldout_coco_ids:
            img_nums = pd.to_numeric(chunk["image_id"], errors="coerce").fillna(-1).astype(int)
            chunk = chunk[~img_nums.isin(heldout_coco_ids)]
        subj = chunk["subject_coco_class"].where(chunk.get("subject_coco_class", pd.Series("", index=chunk.index)).ne(""), chunk["subject"]).str.strip()
        obj = chunk["object_coco_class"].where(chunk.get("object_coco_class", pd.Series("", index=chunk.index)).ne(""), chunk["object"]).str.strip()
        pred = chunk["predicate"].str.strip().str.lower()
        mask = subj.ne("") & pred.ne("") & obj.ne("")
        for s, p, o in zip(subj[mask], pred[mask], obj[mask]):
            cnt[(s.lower(), p, o.lower())] += 1
    return cnt


def stream_vg_triplets(path: Path, chunksize: int = 100_000) -> Counter:
    cnt: Counter = Counter()
    if not path.is_file():
        return cnt
    for chunk in pd.read_csv(path, dtype=str, usecols=["subject", "predicate", "object"], chunksize=chunksize, compression="infer"):
        chunk.fillna("", inplace=True)
        subj = chunk["subject"].str.strip().str.lower()
        pred = chunk["predicate"].str.strip().str.lower()
        obj = chunk["object"].str.strip().str.lower()
        mask = subj.ne("") & pred.ne("") & obj.ne("")
        for s, p, o in zip(subj[mask], pred[mask], obj[mask]):
            cnt[(s, p, o)] += 1
    return cnt


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Karpathy-train COCO and Visual Genome triplets.")
    parser.add_argument("--coco-csv", type=Path, default=Path("outputs/kg_build/coco_triplets_dataset.csv"))
    parser.add_argument("--vg-csv", type=Path, default=Path("outputs/kg_build/vg_triplets_dataset.csv"))
    parser.add_argument("--karpathy-coco", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--alias-file", type=Path, default=ROOT / "knowledge_graph" / "coco_alias_v2.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kg_build"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout_ids = get_heldout_coco_ids(args.karpathy_coco) if args.karpathy_coco.is_file() else set()
    coco_cnt = stream_coco_triplets(args.coco_csv, heldout_ids)
    vg_cnt = stream_vg_triplets(args.vg_csv)

    all_keys = set(coco_cnt) | set(vg_cnt)
    raw_rows = [(s, p, o, coco_cnt.get((s, p, o), 0), vg_cnt.get((s, p, o), 0)) for s, p, o in all_keys]
    raw_df = pd.DataFrame(raw_rows, columns=["subject", "predicate", "object", "freq_coco", "freq_vg"])
    raw_df["freq_combined"] = raw_df["freq_coco"] + raw_df["freq_vg"]
    raw_df = raw_df.sort_values(["subject", "predicate", "object"]).reset_index(drop=True)
    raw_df.to_csv(args.output_dir / "raw_merged_triplets.csv.gz", index=False, compression="gzip")

    lookup, _ = build_alias_lookup(args.alias_file)
    resolve = build_resolver(lookup)
    alias_rows = [
        (resolve(s), p, resolve(o), s, o, coco_cnt.get((s, p, o), 0), vg_cnt.get((s, p, o), 0))
        for s, p, o in all_keys
    ]
    alias_df = pd.DataFrame(
        alias_rows,
        columns=["subject", "predicate", "object", "raw_subject", "raw_object", "freq_coco", "freq_vg"],
    )
    alias_df["freq_combined"] = alias_df["freq_coco"] + alias_df["freq_vg"]
    alias_df.to_csv(args.output_dir / "alias_merged_triplets.csv.gz", index=False, compression="gzip")

    pred_df = (
        alias_df.groupby("predicate")
        .agg(
            canonical_triplets=("predicate", "count"),
            freq_coco=("freq_coco", "sum"),
            freq_vg=("freq_vg", "sum"),
            freq_combined=("freq_combined", "sum"),
            unique_subjects=("subject", "nunique"),
            unique_objects=("object", "nunique"),
        )
        .sort_values("freq_combined", ascending=False)
        .reset_index()
    )
    pred_df.to_csv(args.output_dir / "predicate_frequency.csv", index=False)
    print(f"[Merge KB] Saved {len(raw_df):,} merged triplets to {args.output_dir}")


if __name__ == "__main__":
    main()

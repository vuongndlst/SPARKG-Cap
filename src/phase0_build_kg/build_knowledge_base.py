"""
Phase 0 (Step 5): End-to-End Construction of the Corpus-Level Knowledge Graph (G_K).

Builds `knowledge_base_v2.csv` from MS-COCO Karpathy `train` triplets and Visual Genome
region graphs while strictly excluding all Karpathy `val` and `test` images (including
cross-dataset Visual Genome `coco_id` links) and applying Pointwise Mutual Information
filtering (PMI >= 0.0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import get_heldout_coco_ids
from src.phase0_build_kg.classify_predicate_geometry import classify_predicate_rule, load_geo_alias_map
from src.phase0_build_kg.extract_vg_triplets import load_vg_heldout_ids
from src.phase0_build_kg.merge_kb_triplets import build_alias_lookup, build_resolver


def load_domain_types(things_txt: Path, stuff_txt: Path) -> Dict[str, str]:
    domain: Dict[str, str] = {}
    if things_txt.is_file():
        for line in things_txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                domain[line.strip().lower()] = "thing"
    if stuff_txt.is_file():
        for line in stuff_txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                domain[line.strip().lower()] = "stuff"
    return domain


def build_knowledge_base(
    karpathy_coco: Path,
    coco_triplets_csv: Optional[Path],
    vg_zip: Optional[Path],
    vg_image_data: Optional[Path],
    alias_path: Path,
    geo_alias_path: Path,
    things_txt: Path,
    stuff_txt: Path,
    output_csv: Path,
) -> None:
    t0 = time.time()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    heldout_coco_ids: Set[int] = get_heldout_coco_ids(karpathy_coco) if karpathy_coco.is_file() else set()
    blocked_vg_ids: Set[int] = load_vg_heldout_ids(vg_image_data, heldout_coco_ids)
    print(f"[Build G_K] Karpathy held-out exclusion: {len(heldout_coco_ids):,} COCO IDs | {len(blocked_vg_ids):,} VG IDs")

    lookup, coco_classes = build_alias_lookup(alias_path)
    resolve = build_resolver(lookup)
    geo_map = load_geo_alias_map(geo_alias_path)
    domain_map = load_domain_types(things_txt, stuff_txt)

    kb_counter: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: {"coco": 0, "vg": 0})

    # 1. Stream MS-COCO training triplets (excluding held-out Karpathy val/test IDs)
    if coco_triplets_csv and coco_triplets_csv.is_file():
        header_cols = pd.read_csv(coco_triplets_csv, nrows=0).columns.tolist()
        use_cols = [c for c in ["image_id", "subject", "predicate", "object", "subject_coco_class", "object_coco_class"] if c in header_cols]
        for chunk in pd.read_csv(coco_triplets_csv, dtype=str, usecols=use_cols, chunksize=100_000, compression="infer"):
            chunk.fillna("", inplace=True)
            if "image_id" in chunk.columns and heldout_coco_ids:
                nums = pd.to_numeric(chunk["image_id"], errors="coerce").fillna(-1).astype(int)
                chunk = chunk[~nums.isin(heldout_coco_ids)]
            s_series = chunk["subject_coco_class"].where(chunk.get("subject_coco_class", pd.Series("", index=chunk.index)).ne(""), chunk["subject"]).str.strip()
            o_series = chunk["object_coco_class"].where(chunk.get("object_coco_class", pd.Series("", index=chunk.index)).ne(""), chunk["object"]).str.strip()
            p_series = chunk["predicate"].str.strip().str.lower()
            mask = s_series.ne("") & p_series.ne("") & o_series.ne("")
            for raw_s, p, raw_o in zip(s_series[mask], p_series[mask], o_series[mask]):
                s_can, o_can = resolve(raw_s), resolve(raw_o)
                if s_can in coco_classes and o_can in coco_classes:
                    kb_counter[(s_can, p, o_can)]["coco"] += 1

    # 2. Stream Visual Genome region graphs (excluding Karpathy val/test coco_id links)
    if vg_zip and vg_zip.is_file():
        with zipfile.ZipFile(vg_zip, "r") as zf, zf.open("region_graphs.json") as f_in:
            for img_data in json.load(f_in):
                if int(img_data.get("image_id", -1)) in blocked_vg_ids:
                    continue
                for region in img_data.get("regions", []):
                    obj_map = {}
                    for obj in region.get("objects", []):
                        oid = obj.get("object_id")
                        names = obj.get("names") or []
                        obj_map[oid] = (names[0] if names else obj.get("name", "unknown")).lower().strip()
                    for rel in region.get("relationships", []):
                        raw_s = obj_map.get(rel.get("subject_id"), "unknown")
                        raw_o = obj_map.get(rel.get("object_id"), "unknown")
                        p = str(rel.get("predicate", "")).lower().strip()
                        if raw_s != "unknown" and raw_o != "unknown" and p:
                            s_can, o_can = resolve(raw_s), resolve(raw_o)
                            if s_can in coco_classes and o_can in coco_classes:
                                kb_counter[(s_can, p, o_can)]["vg"] += 1

    if not kb_counter:
        raise RuntimeError("No triplets collected; verify --coco-triplets or --vg-zip paths.")

    rows = [
        {
            "subject": s,
            "predicate": p,
            "object": o,
            "freq_coco": c["coco"],
            "freq_vg": c["vg"],
            "frequency": c["coco"] + c["vg"],
            "sources": "coco+vg" if (c["coco"] > 0 and c["vg"] > 0) else ("coco" if c["coco"] > 0 else "vg"),
        }
        for (s, p, o), c in kb_counter.items()
    ]
    df = pd.DataFrame(rows)

    # 3. Pointwise Mutual Information filter: retain PMI(subject, object) >= 0.0
    total_freq = float(df["frequency"].sum())
    p_subj = df.groupby("subject")["frequency"].sum() / total_freq
    p_obj = df.groupby("object")["frequency"].sum() / total_freq
    pair_freq = df.groupby(["subject", "object"])["frequency"].sum()
    p_pair = pair_freq / total_freq
    pmi_vals = np.log2(
        (p_pair.values + 1e-10)
        / (p_subj[pair_freq.index.get_level_values(0)].values * p_obj[pair_freq.index.get_level_values(1)].values + 1e-10)
    )
    pmi_df = pd.DataFrame(
        {"subject": pair_freq.index.get_level_values(0), "object": pair_freq.index.get_level_values(1), "pmi": pmi_vals}
    )
    df = df.merge(pmi_df, on=["subject", "object"], how="left")
    df = df[df["pmi"] >= 0.0].copy()

    # 4. Classify predicate type and canonical 2D spatial geometry
    unique_preds = df["predicate"].unique()
    ptype_map, sgeo_map = {}, {}
    for p in unique_preds:
        pt, sg = classify_predicate_rule(p, geo_map)
        ptype_map[p] = pt
        sgeo_map[p] = sg

    df["rel_type"] = df["predicate"].map(ptype_map)
    df["spatial_geo_rel"] = df["predicate"].map(sgeo_map)
    df["geo_rel"] = df["predicate"]
    df["coco_class"] = df["subject"]
    df["class_type"] = df["coco_class"].map(lambda x: domain_map.get(x, "thing"))
    df["role"] = "subject"
    df["partner_canonical"] = df["object"]
    df["partner_coco_type"] = df["partner_canonical"].map(lambda x: domain_map.get(x, "thing"))

    cols = [
        "coco_class", "class_type", "sources", "role", "predicate",
        "rel_type", "geo_rel", "spatial_geo_rel",
        "partner_canonical", "partner_coco_type", "frequency",
    ]
    df = df.sort_values("frequency", ascending=False)[cols].reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"[Build G_K] Saved {len(df):,} PMI-filtered canonical triplets to {output_csv} in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SPARKG-Cap corpus-level Knowledge Graph G_K.")
    parser.add_argument("--karpathy-coco", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--coco-triplets", type=Path, default=Path("outputs/kg_build/coco_triplets_dataset.csv"))
    parser.add_argument("--vg-zip", type=Path, default=Path("data/visual_genome/region_graphs.json.zip"))
    parser.add_argument("--vg-image-data", type=Path, default=Path("data/visual_genome/image_data.json"))
    parser.add_argument("--alias-file", type=Path, default=ROOT / "knowledge_graph" / "coco_alias_v2.txt")
    parser.add_argument("--geo-alias", type=Path, default=ROOT / "knowledge_graph" / "geo_rel_alias.txt")
    parser.add_argument("--things-txt", type=Path, default=ROOT / "statistics" / "foreground_things_coco.txt")
    parser.add_argument("--stuff-txt", type=Path, default=ROOT / "statistics" / "background_stuff_coco.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "knowledge_graph" / "knowledge_base_v2.csv")
    args = parser.parse_args()

    build_knowledge_base(
        karpathy_coco=args.karpathy_coco,
        coco_triplets_csv=args.coco_triplets,
        vg_zip=args.vg_zip,
        vg_image_data=args.vg_image_data,
        alias_path=args.alias_file,
        geo_alias_path=args.geo_alias,
        things_txt=args.things_txt,
        stuff_txt=args.stuff_txt,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()

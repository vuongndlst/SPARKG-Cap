"""
Phase 0 (Step 1): Extract Relational Triplets from Visual Genome Region Graphs.

Streams `region_graphs.json.zip` while strictly excluding any Visual Genome image
whose `coco_id` cross-links to the held-out MS-COCO Karpathy `val` or `test` splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import get_heldout_coco_ids


def load_vg_heldout_ids(vg_image_data: Optional[Path], heldout_coco_ids: Set[int]) -> Set[int]:
    """Identify Visual Genome `image_id`s linked to Karpathy `val` or `test` COCO images."""
    if not vg_image_data or not vg_image_data.is_file() or not heldout_coco_ids:
        return set()
    data = json.loads(vg_image_data.read_bytes())
    blocked: Set[int] = set()
    for row in data:
        cid = row.get("coco_id")
        if cid is not None and int(cid) in heldout_coco_ids:
            blocked.add(int(row["image_id"]))
    return blocked


def extract_vg_triplets(
    vg_zip: Path,
    output_csv: Path,
    karpathy_coco: Optional[Path] = None,
    vg_image_data: Optional[Path] = None,
) -> None:
    heldout_coco = get_heldout_coco_ids(karpathy_coco) if karpathy_coco and karpathy_coco.is_file() else set()
    blocked_vg_ids = load_vg_heldout_ids(vg_image_data, heldout_coco)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predicates: Counter = Counter()
    kept_images = 0
    skipped_heldout = 0
    total_triplets = 0

    with zipfile.ZipFile(vg_zip, "r") as zf, zf.open("region_graphs.json") as f_in:
        all_data = json.load(f_in)
        with output_csv.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["image_id", "phrase", "subject", "predicate", "object"])

            for img_data in all_data:
                img_id = int(img_data.get("image_id", -1))
                if img_id in blocked_vg_ids:
                    skipped_heldout += 1
                    continue
                kept_images += 1

                for region in img_data.get("regions", []):
                    phrase = region.get("phrase", "").replace("\n", " ").strip()
                    obj_map: Dict[int, str] = {}
                    for obj in region.get("objects", []):
                        oid = obj.get("object_id")
                        names = obj.get("names") or []
                        name = names[0] if names else obj.get("name", "unknown")
                        if oid is not None:
                            obj_map[oid] = str(name).lower().strip()

                    for rel in region.get("relationships", []):
                        subj = obj_map.get(rel.get("subject_id"), "")
                        obj_name = obj_map.get(rel.get("object_id"), "")
                        pred = str(rel.get("predicate", "")).lower().strip()
                        if subj and obj_name and pred and subj != "unknown" and obj_name != "unknown":
                            writer.writerow([img_id, phrase, subj, pred, obj_name])
                            predicates[pred] += 1
                            total_triplets += 1

    print(
        f"[VG Extract] Retained {kept_images:,} images (excluded {skipped_heldout:,} Karpathy val/test links) "
        f"-> {total_triplets:,} triplets ({len(predicates):,} unique predicates) saved to {output_csv}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Visual Genome triplets excluding Karpathy val/test splits.")
    parser.add_argument("--vg-zip", type=Path, default=Path("data/visual_genome/region_graphs.json.zip"))
    parser.add_argument("--vg-image-data", type=Path, default=Path("data/visual_genome/image_data.json"))
    parser.add_argument("--karpathy-coco", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/kg_build/vg_triplets_dataset.csv"))
    args = parser.parse_args()
    extract_vg_triplets(args.vg_zip, args.output, args.karpathy_coco, args.vg_image_data)


if __name__ == "__main__":
    main()

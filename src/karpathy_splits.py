"""
Official Karpathy Split Loader for MS-COCO and Flickr30K.

Provides a unified interface across all pipeline stages (Knowledge Graph construction,
Scene Graph generation, CLIP enrichment, Caption generation, and Evaluation) so that
every stage resolves image IDs, file paths, and reference captions strictly from
the standard Karpathy split manifests (`dataset_coco.json` and `dataset_flickr30k.json`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# Karpathy split alias mapping ('restval' belongs to the standard 113,287-image training split)
SPLIT_MAP_TRAIN = {"train", "restval"}


@dataclass(frozen=True)
class KarpathyItem:
    """Single image entry resolved from a Karpathy split manifest."""
    dataset: str
    split: str
    image_id: str          # Zero-padded 12-digit COCO ID or Flickr30K numeric stem
    numeric_id: Optional[int]
    filename: str          # Original filename in manifest
    filepath: str          # Relative subfolder in manifest (e.g., 'val2014' or '')
    references: Tuple[str, ...]


def normalize_split_query(split: str) -> Set[str]:
    """Map user split argument ('train', 'val', 'test') to manifest split tags."""
    s = split.strip().lower()
    if s == "train":
        return set(SPLIT_MAP_TRAIN)
    if s in {"val", "test"}:
        return {s}
    if s == "all":
        return {"train", "restval", "val", "test"}
    raise ValueError(f"Unsupported Karpathy split {split!r}; expected 'train', 'val', 'test', or 'all'.")


def format_coco_stem(cocoid: int) -> str:
    """Standard 12-digit zero-padded MS-COCO stem (e.g., '000000391895')."""
    return f"{int(cocoid):012d}"


def load_karpathy_manifest(
    manifest_path: Path | str,
    dataset: Optional[str] = None,
    split: str = "test",
) -> List[KarpathyItem]:
    """
    Load items for a target split ('train', 'val', 'test') from a Karpathy JSON manifest.

    Supports both official Andrej Karpathy `dataset_coco.json` / `dataset_flickr30k.json`
    (`{"images": [{"split": ..., "filename": ..., "cocoid": ..., "sentences": [{"raw": ...}]}]}`)
    and pre-extracted reference lists (`[{"image_id": ..., "captions": [...]}]`).
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Karpathy split manifest not found: {path}")

    raw = json.loads(path.read_bytes())
    wanted_splits = normalize_split_query(split)

    # Infer dataset if omitted
    ds = (dataset or raw.get("dataset") if isinstance(raw, dict) else dataset) or ""
    ds_norm = "flickr30k" if "flickr" in str(ds).lower() or "flickr" in path.name.lower() else "coco"

    entries = raw["images"] if isinstance(raw, dict) and "images" in raw else raw
    if not isinstance(entries, list):
        raise ValueError(f"Unrecognized Karpathy manifest structure in {path}")

    items: List[KarpathyItem] = []
    for entry in entries:
        entry_split = str(entry.get("split", "test")).strip().lower()
        if isinstance(raw, dict) and "images" in raw and entry_split not in wanted_splits:
            continue

        filename = str(entry.get("filename") or entry.get("file_name") or "").strip()
        filepath = str(entry.get("filepath") or "").strip()

        if ds_norm == "coco":
            cid = entry.get("cocoid", entry.get("image_id"))
            if cid is None and filename:
                digits = "".join(ch for ch in Path(filename).stem if ch.isdigit())
                cid = int(digits)
            num_id = int(cid)
            stem = format_coco_stem(num_id)
            if not filename:
                filename = f"{stem}.jpg"
        else:
            stem_raw = Path(filename).stem if filename else str(entry.get("image_id", entry.get("imgid", ""))).strip()
            stem = stem_raw
            try:
                num_id = int(stem_raw)
            except ValueError:
                num_id = None
            if not filename:
                filename = f"{stem}.jpg"

        if "sentences" in entry:
            refs = tuple(s["raw"].strip() for s in entry["sentences"] if s.get("raw", "").strip())
        elif "captions" in entry:
            refs = tuple(str(c).strip() for c in entry["captions"] if str(c).strip())
        else:
            refs = ()

        items.append(
            KarpathyItem(
                dataset=ds_norm,
                split="train" if entry_split in SPLIT_MAP_TRAIN else entry_split,
                image_id=stem,
                numeric_id=num_id,
                filename=filename,
                filepath=filepath,
                references=refs,
            )
        )

    items.sort(key=lambda x: (x.numeric_id if x.numeric_id is not None else 0, x.image_id))
    return items


def load_karpathy_references(
    manifest_path: Path | str,
    dataset: Optional[str] = None,
    split: str = "test",
) -> Tuple[Dict[str, List[str]], Dict[str, Optional[int]]]:
    """Return `(gt_dict, id_to_numeric)` keyed by canonical `image_id` stem."""
    items = load_karpathy_manifest(manifest_path, dataset=dataset, split=split)
    gt_dict: Dict[str, List[str]] = {it.image_id: list(it.references) for it in items}
    id_map: Dict[str, Optional[int]] = {it.image_id: it.numeric_id for it in items}
    return gt_dict, id_map


def get_heldout_coco_ids(manifest_path: Path | str) -> Set[int]:
    """Return all COCO numeric image IDs belonging to Karpathy `val` or `test` splits."""
    val_items = load_karpathy_manifest(manifest_path, dataset="coco", split="val")
    test_items = load_karpathy_manifest(manifest_path, dataset="coco", split="test")
    return {it.numeric_id for it in (val_items + test_items) if it.numeric_id is not None}


def resolve_image_path(item: KarpathyItem, img_root: Path | str) -> Optional[Path]:
    """
    Locate an image on disk across standard MS-COCO and Flickr30K directory layouts:
      - `<img_root>/<filepath>/<filename>` (e.g., `data/coco/val2014/COCO_val2014_000000391895.jpg`)
      - `<img_root>/<filename>`
      - `<img_root>/<image_id>.jpg` (e.g., COCO 2017 12-digit naming `000000391895.jpg`)
      - `<img_root>/{train2017,val2017,train2014,val2014,images,flickr30k-images}/...`
    """
    root = Path(img_root)
    candidates: List[Path] = []
    if item.filepath:
        candidates.append(root / item.filepath / item.filename)
        candidates.append(root / item.filepath / f"{item.image_id}.jpg")
    candidates.append(root / item.filename)
    for ext in (".jpg", ".jpeg", ".png"):
        candidates.append(root / f"{item.image_id}{ext}")
    for sub in ("images", "val2014", "train2014", "val2017", "train2017", "flickr30k-images", "flickr30k_images"):
        candidates.append(root / sub / item.filename)
        candidates.append(root / sub / f"{item.image_id}.jpg")

    for cand in candidates:
        if cand.is_file():
            return cand
    return None

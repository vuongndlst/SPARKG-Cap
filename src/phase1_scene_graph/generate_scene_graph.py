"""
Phase 1A: Panoptic Entity Detection, Salience Partitioning, and K1-K4 Scene Graph Construction.

Implements Stages III.A-III.C of SPARKG-Cap over official Karpathy splits:
  1. Instance detection (Mask R-CNN R50-FPN over 80 `thing` classes) and semantic
     segmentation (DeepLabV3 R50-D8 over 91 `stuff` classes).
  2. Closed-form Foreground (FGS) and Background (BGS) salience scoring using
     corpus-derived criterion weights (`statistics/`) and consistency rules R0-R5.
  3. Canonical 2D bounding-box spatial geometry (`on`, `in`, `contains`, `above`,
     `under`, `next_to`, `overlap`) mapped through materialized knowledge-graph
     views K1-K4 (`knowledge_graph/knowledge_base_v2.csv`) along cascade L1-L5,
     with semantic rescue (`Lsem`), foreground orphan rescue (`Lfg`), and degree caps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import multiprocessing as mp
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import KarpathyItem, load_karpathy_manifest, resolve_image_path

try:
    from mmdet.apis import inference_detector, init_detector
    from mmseg.apis import inference_model as inference_seg_model, init_model as init_seg_model
    MMLAB_OK = True
except Exception:
    MMLAB_OK = False

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kw): self._it = iterable or []
        def __iter__(self): return iter(self._it)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Default model configuration & asset paths
DEFAULT_DET_CONFIG = ROOT / "configs" / "mask_rcnn" / "mask-rcnn_r50_fpn_1x_coco.py"
DEFAULT_DET_CKPT = ROOT / "checkpoints" / "mask_rcnn_r50_fpn_1x_coco.pth"
DEFAULT_SEG_CONFIG = ROOT / "configs" / "mmseg" / "deeplabv3_r50-d8_4xb4-160k_coco-stuff164k-512x512.py"
DEFAULT_SEG_CKPT = ROOT / "checkpoints" / "deeplabv3_r50-d8_512x512_4x4_160k_coco-stuff164k.pth"

DEFAULT_KB_PATH = ROOT / "knowledge_graph" / "knowledge_base_v2.csv"
DEFAULT_THING_TXT = ROOT / "statistics" / "foreground_things_coco.txt"
DEFAULT_STUFF_TXT = ROOT / "statistics" / "background_stuff_coco.txt"
DEFAULT_THING_PRIOR = ROOT / "statistics" / "coco_caption_foreground_stats_trainval.csv"
DEFAULT_STUFF_PRIOR = ROOT / "statistics" / "cocostuff_caption_foreground_stats_trainval.csv"
DEFAULT_FG_WEIGHTS = ROOT / "statistics" / "derived_weights_coco_things_stuff_penalty_gamma.csv"
DEFAULT_BG_WEIGHTS = ROOT / "statistics" / "derived_bg_weights_farcenter_penalty_gamma.csv"

# Fixed thresholds (Table 2 / docs/PARAMETERS_AND_WEIGHTS.md)
TAU_FG = 0.55
TAU_BG = 0.15
TAU_BG_SOFT = 0.10
DET_CONF_MIN_FG = 0.45
BG_AREA_RATIO_THR = 0.40

BBOX_SCORE_THR = 0.55
STUFF_AREA_THR = 500
DUP_IOU_THR = 0.75
DUP_IOMIN_THR = 0.85
MAX_RELS_PER_IMG = 30
MIN_EDGE_WEIGHT = 0.70

ENABLE_PASS2 = True
PASS2_MIN_LVLS = {"L1", "L2"}
NEXT_TO_MIN_LEVEL: Optional[str] = "L1"
NEXT_TO_ACTION_ONLY: bool = False
MAX_DEGREE_PER_NODE: Optional[int] = 3
DROP_BG_BG_EDGES: bool = True

INFER_TOP_K = 5
INFER_MIN_FREQ = 3

_ON_VERT_GAP = 0.25
_ON_HORIZ_OVER = 0.25
_IN_COVER = 0.95
_OVERLAP_IOU = 0.10
_DIST_FACTOR = 1.5

_NODE_COLOR = {"fg": "#2ECC71", "bg": "#F1C40F", "objects": "#5DADE2"}
_NODE_COLOR_DEFAULT = "#BDC3C7"

_SPATIAL_LABEL: Dict[str, str] = {
    "on": "on", "in": "in", "above": "above", "under": "under",
    "next_to": "next to", "overlap": "near", "unknown": "near", "contains": "containing",
}
_LVL_META = {
    "L1": {"rank": 0, "conf": 1.00, "wt": 1.00},
    "L2": {"rank": 1, "conf": 0.85, "wt": 0.85},
    "L3": {"rank": 2, "conf": 0.70, "wt": 0.70},
    "L4": {"rank": 3, "conf": 0.50, "wt": 0.50},
    "L5": {"rank": 4, "conf": 0.30, "wt": 0.30},
}
_LVL_RANK = {k: v["rank"] for k, v in _LVL_META.items()}
_ACTIVE_VERBS = {"holding", "wearing", "riding", "playing with", "eating"}

_FORBIDDEN_RULES = {
    "wearing": ["playingfield", "sky", "tree", "grass", "road", "pavement", "building", "wall", "snow", "water", "ground", "dirt", "mountain", "fence", "person", "cat", "dog"],
    "riding": ["playingfield", "sky", "tree", "grass", "road", "pavement", "building", "wall", "snow", "water", "ground", "dirt", "mountain", "fence", "person"],
    "flying": ["playingfield", "tree", "grass", "road", "pavement", "building", "wall", "snow", "water", "ground", "dirt", "person", "fence", "tennis racket", "baseball bat", "skis", "snowboard", "cell phone"],
    "sitting at": ["tennis racket", "sports ball", "baseball bat", "skis", "snowboard", "frisbee", "kite", "skateboard", "cell phone"],
    "holding": ["playingfield", "sky", "tree", "grass", "road", "pavement", "building", "wall", "snow", "water", "ground", "dirt", "mountain", "fence", "sky-other", "wall-other", "wall-concrete"],
    "filled with": ["person", "dog", "cat", "bird", "horse"],
    "in": ["person", "dog", "cat"],
}
_STUFF_SUFFIXES = ("-other", "-stuff", "-merged", "-tile", "-wood", "-marble", "-stone", "-concrete", "-brick", "-panel", "-blind")


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _normalize_class(cls: str) -> str:
    cls = cls.replace("_", " ")
    for suf in _STUFF_SUFFIXES:
        if cls.endswith(suf):
            return cls[: -len(suf)]
    return cls


def _kb_candidates(c: str) -> List[str]:
    cn = _normalize_class(c)
    return [c, cn] if cn != c else [c]


def compute_geo_rel(box_a: List[float], box_b: List[float]) -> str:
    """Compute canonical 2D bounding-box geometry (`on`, `in`, `contains`, `above`, `under`, `next_to`, `overlap`, `unknown`)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    aa = max(1.0, (xa2 - xa1) * (ya2 - ya1))
    ab = max(1.0, (xb2 - xb1) * (yb2 - yb1))
    wa, ha = xa2 - xa1, ya2 - ya1
    wb, hb = xb2 - xb1, yb2 - yb1
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    if inter / aa > _IN_COVER:
        return "in"
    if inter / ab > _IN_COVER:
        return "overlap" if (ab / aa) < 0.25 else "contains"

    cax, cay = (xa1 + xa2) / 2.0, (ya1 + ya2) / 2.0
    cbx, cby = (xb1 + xb2) / 2.0, (yb1 + yb2) / 2.0
    dx, dy = cbx - cax, cby - cay
    h_ov = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    h_ratio = h_ov / (min(wa, wb) + 1e-6)
    iou = inter / (aa + ab - inter + 1e-6)

    ref_h = max(ha, hb)
    v_gap = yb1 - ya2
    if cay < cby and (-ref_h * 0.15 < v_gap < ref_h * _ON_VERT_GAP) and h_ratio > _ON_HORIZ_OVER and iou < 0.30:
        return "on"
    if iou > _OVERLAP_IOU:
        return "overlap"
    if math.hypot(dx, dy) > max(wa, wb, ha, hb) * _DIST_FACTOR:
        return "unknown"
    if abs(dy) > abs(dx):
        return "above" if dy > 0 else "under"
    return "next_to"


def load_prior_auto(csv_path: Path) -> Dict[str, float]:
    prior: Dict[str, float] = {}
    if not csv_path.is_file():
        return prior
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        class_col = next((c for c in cols if c.lower() in {"class", "category", "name"}), None)
        ratio_col = next((c for c in cols if any(k in c.lower() for k in ("fg_ratio", "fg %", "ratio", "prior", "fg_percent"))), None)
        if not class_col or not ratio_col:
            return prior
        for row in reader:
            cls = _norm_name(row[class_col])
            try:
                val = float(row[ratio_col])
            except (ValueError, TypeError):
                continue
            if "%" in ratio_col or "percent" in ratio_col.lower():
                val /= 100.0
            prior[cls] = float(np.clip(val, 0.0, 1.0))
    return prior


def load_weights(csv_path: Path) -> Dict[str, float]:
    fallback = {"size": 0.25, "center": 0.25, "semantic": 0.25, "relation": 0.25}
    if not csv_path.is_file():
        return fallback
    w: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        crit_col = next((c for c in cols if c.lower() == "criterion"), None)
        w_col = next((c for c in cols if c.lower() == "weight"), None)
        if not crit_col or not w_col:
            return fallback
        for row in reader:
            try:
                w[_norm_name(row[crit_col])] = float(row[w_col])
            except (ValueError, TypeError):
                pass
    for k in ("size", "center", "semantic", "relation"):
        w.setdefault(k, 0.05 if k == "relation" else 0.0)
    s = sum(w.values())
    return {k: v / s for k, v in w.items()} if s > 0 else fallback


def decide_label(fgs: float, bgs: float) -> str:
    if fgs >= TAU_FG and fgs >= bgs:
        return "fg"
    if bgs >= TAU_BG and bgs > fgs:
        return "bg"
    return "objects"


def recheck_label(
    label_raw: str,
    cls: str,
    fgs: float,
    bgs: float,
    thing_set: set,
    stuff_set: set,
    det_conf: Optional[float] = None,
    area_ratio: float = 0.0,
) -> Tuple[str, Optional[str]]:
    in_things = cls in thing_set
    in_stuff = cls in stuff_set
    if in_stuff and not in_things and area_ratio >= BG_AREA_RATIO_THR and label_raw != "bg":
        return "bg", f"R5:{label_raw}->bg"
    if label_raw == "fg" and det_conf is not None and det_conf < DET_CONF_MIN_FG:
        return "objects", "R0:fg->objects"
    if label_raw == "fg" and in_stuff and not in_things:
        return "objects", "R1:fg->objects"
    if label_raw == "fg" and not in_things:
        cls_norm = _normalize_class(cls)
        if cls_norm in stuff_set or any(cls.startswith(s + "-") or cls.startswith(s + "_") for s in stuff_set):
            return "bg", "R1b:fg->bg"
    if label_raw == "bg" and in_things and not in_stuff:
        return "objects", "R2:bg->objects"
    if label_raw == "objects" and in_stuff and not in_things:
        if bgs > fgs and bgs >= TAU_BG_SOFT:
            return "bg", "R3:objects->bg"
        return "bg", "R4:objects->bg"
    return label_raw, None


def compute_fgbg_scores(
    objects: List[dict],
    img_h: int,
    img_w: int,
    thing_prior: Dict[str, float],
    stuff_prior: Dict[str, float],
    w_fg: Dict[str, float],
    w_bg: Dict[str, float],
    thing_set: Optional[set] = None,
    stuff_set: Optional[set] = None,
) -> List[dict]:
    t_set = thing_set or set()
    s_set = stuff_set or set()
    out: List[dict] = []
    icx, icy = img_w * 0.5, img_h * 0.5
    dmax = math.hypot(icx, icy) + 1e-9

    for d in objects:
        box = d["box"]
        cls = _norm_name(d.get("class", "unknown"))
        otyp = d.get("type", "thing")
        s_size = float(np.clip((box[2] - box[0]) * (box[3] - box[1]) / max(1, img_h * img_w), 0.0, 1.0))
        cx, cy = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
        s_center = float(np.clip(1.0 - math.hypot(cx - icx, cy - icy) / dmax, 0.0, 1.0))

        deg = 0
        for other in objects:
            if other.get("id") == d["id"]:
                continue
            ob = other["box"]
            if max(0, min(box[2], ob[2]) - max(box[0], ob[0])) * max(0, min(box[3], ob[3]) - max(box[1], ob[1])) > 0:
                deg += 1
        s_rel = float(deg / (deg + 5.0))

        sem_fg = float(thing_prior.get(cls, 0.5))
        sem_bg = float(stuff_prior.get(cls, 0.5))
        fgs = float(w_fg["size"] * s_size + w_fg["center"] * s_center + w_fg["semantic"] * sem_fg + w_fg["relation"] * s_rel)
        bgs = float(w_bg["size"] * s_size + w_bg["center"] * (1.0 - s_center) + w_bg["semantic"] * sem_bg + w_bg["relation"] * s_rel)

        label_raw = decide_label(fgs, bgs)
        label_final, reason = recheck_label(label_raw, cls, fgs, bgs, t_set, s_set, det_conf=d.get("det_conf"), area_ratio=s_size)

        nd = dict(d)
        nd.update(
            FGS=round(fgs, 4), BGS=round(bgs, 4), label=label_final, label_raw=label_raw,
            recheck_reason=reason, s_size=round(s_size, 4), s_center=round(s_center, 4),
            s_rel=round(s_rel, 4), sem_fg=round(sem_fg, 4), sem_bg=round(sem_bg, 4), area_ratio=round(s_size, 4),
        )
        out.append(nd)
    return out


class RelationMapper:
    """Materializes K1-K4 lookup views and Lsem semantic rescue tables from `knowledge_base_v2.csv`."""

    def __init__(self, csv_path: Path):
        self.kb_exact: Dict[Tuple, str] = {}
        self.kb_subj_geo: Dict[Tuple, str] = {}
        self.kb_obj_geo: Dict[Tuple, str] = {}
        self.kb_geo_label: Dict[str, str] = {}
        self.kb_exact_rtype: Dict[Tuple, str] = {}
        self.kb_subj_geo_rtype: Dict[Tuple, str] = {}
        self.kb_obj_geo_rtype: Dict[Tuple, str] = {}
        self.kb_pred_exact: Dict[Tuple, str] = {}
        self.kb_pred_subj: Dict[str, str] = {}
        self._df_clean: Optional[pd.DataFrame] = None
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        if not csv_path.is_file():
            log.warning("Knowledge base CSV not found: %s", csv_path)
            return
        df = pd.read_csv(csv_path, low_memory=False, compression="infer")
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ("coco_class", "role", "predicate", "spatial_geo_rel", "geo_rel", "partner_canonical"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        df["frequency"] = pd.to_numeric(df.get("frequency", 1), errors="coerce").fillna(1)
        df = df[~df["partner_canonical"].str.match(r"^\d+$")].copy()
        self._df_clean = df

        valid_geo = {"in", "on", "above", "under", "next_to", "overlap"}
        df_known = df[df["spatial_geo_rel"].isin(valid_geo)].copy()
        subj_mask = df_known["role"].isin(["subject"])
        df_known["_subj"] = df_known["coco_class"].where(subj_mask, df_known["partner_canonical"])
        df_known["_obj"] = df_known["partner_canonical"].where(subj_mask, df_known["coco_class"])
        df_known["_spatial"] = df_known["spatial_geo_rel"]
        df_known["_geo_raw"] = df_known["geo_rel"]
        df_known["_rel_pri"] = df_known["rel_type"].map({"action": 0, "spatial": 1}).fillna(2).astype(int)
        ds = df_known.sort_values(["_rel_pri", "frequency"], ascending=[True, False])

        l1 = ds.drop_duplicates(["_subj", "_obj", "_spatial"], keep="first")
        self.kb_exact = dict(zip(zip(l1["_subj"], l1["_obj"], l1["_spatial"]), l1["predicate"]))
        self.kb_exact_rtype = dict(zip(zip(l1["_subj"], l1["_obj"], l1["_spatial"]), l1["rel_type"]))

        l2 = ds.drop_duplicates(["_subj", "_spatial"], keep="first")
        self.kb_subj_geo = dict(zip(zip(l2["_subj"], l2["_spatial"]), l2["predicate"]))
        self.kb_subj_geo_rtype = dict(zip(zip(l2["_subj"], l2["_spatial"]), l2["rel_type"]))

        l3 = ds.drop_duplicates(["_obj", "_spatial"], keep="first")
        self.kb_obj_geo = dict(zip(zip(l3["_obj"], l3["_spatial"]), l3["predicate"]))
        self.kb_obj_geo_rtype = dict(zip(zip(l3["_obj"], l3["_spatial"]), l3["rel_type"]))

        l4 = ds.drop_duplicates(["_spatial"], keep="first")
        self.kb_geo_label = dict(zip(l4["_spatial"], l4["_geo_raw"]))

        noise = {"of", "has", "and", "with", "have", "by", "at", "to", "for", "or", "a", "an", "the", "is", "in", "on", "from", "not"}
        df_unk = df[(df["spatial_geo_rel"] == "unknown") & (df["rel_type"] == "action") & (df["predicate"].str.len() > 3) & (~df["predicate"].isin(noise))].copy()
        if len(df_unk) > 0:
            s_mask = df_unk["role"].isin(["subject"])
            df_unk["_subj"] = df_unk["coco_class"].where(s_mask, df_unk["partner_canonical"])
            df_unk["_obj"] = df_unk["partner_canonical"].where(s_mask, df_unk["coco_class"])
            ds_unk = df_unk.sort_values("frequency", ascending=False)
            lp1 = ds_unk.drop_duplicates(["_subj", "_obj"], keep="first")
            self.kb_pred_exact = dict(zip(zip(lp1["_subj"], lp1["_obj"]), lp1["predicate"]))
            lp2 = ds_unk.drop_duplicates(["_subj"], keep="first")
            self.kb_pred_subj = dict(zip(lp2["_subj"], lp2["predicate"]))


def _lookup(
    mapper: RelationMapper, s: str, o: str, spatial_geo: str
) -> Tuple[str, str, float, str]:
    sc_list = _kb_candidates(s)
    oc_list = _kb_candidates(o)
    for sc in sc_list:
        for oc in oc_list:
            if (sc, oc, spatial_geo) in mapper.kb_exact:
                return mapper.kb_exact[(sc, oc, spatial_geo)], "L1", 1.00, mapper.kb_exact_rtype.get((sc, oc, spatial_geo), "spatial")
    for sc in sc_list:
        if (sc, spatial_geo) in mapper.kb_subj_geo:
            return mapper.kb_subj_geo[(sc, spatial_geo)], "L2", 0.85, mapper.kb_subj_geo_rtype.get((sc, spatial_geo), "spatial")
    for oc in oc_list:
        if (oc, spatial_geo) in mapper.kb_obj_geo:
            return mapper.kb_obj_geo[(oc, spatial_geo)], "L3", 0.70, mapper.kb_obj_geo_rtype.get((oc, spatial_geo), "spatial")
    raw_geo = mapper.kb_geo_label.get(spatial_geo)
    if raw_geo:
        return raw_geo, "L4", 0.50, "fallback"
    return _SPATIAL_LABEL.get(spatial_geo, spatial_geo), "L5", 0.30, "fallback"


def filter_dups_and_noise(dets: List[dict]) -> List[dict]:
    clean = []
    for d in dets:
        if d.get("type") == "stuff":
            clean.append(d)
            continue
        x1, y1, x2, y2 = d["box"]
        w, h = x2 - x1, y2 - y1
        if w < 15 or h < 15 or (y1 <= 5 and h < 30):
            continue
        clean.append(d)
    dets = sorted(clean, key=lambda x: (x.get("det_conf") or x.get("score", 0)), reverse=True)
    kept = []
    for d in dets:
        dup = False
        for k in kept:
            if d["class"] != k["class"]:
                continue
            xi1, yi1 = max(d["box"][0], k["box"][0]), max(d["box"][1], k["box"][1])
            xi2, yi2 = min(d["box"][2], k["box"][2]), min(d["box"][3], k["box"][3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            aa = max(1, (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            ab = max(1, (k["box"][2] - k["box"][0]) * (k["box"][3] - k["box"][1]))
            if inter / (aa + ab - inter + 1e-6) > DUP_IOU_THR or inter / (min(aa, ab) + 1e-6) > DUP_IOMIN_THR:
                dup = True
                break
        if not dup:
            kept.append(d)
    for idx, d in enumerate(kept, 1):
        d["id"] = idx
    return kept


def build_scene_graph_from_detections(
    image_name: str,
    split: str,
    img_hw: Tuple[int, int],
    raw_objs: List[dict],
    mapper: RelationMapper,
    co_occ: Dict[str, Dict[str, float]],
    thing_set: set,
    stuff_set: set,
    thing_prior: Dict[str, float],
    stuff_prior: Dict[str, float],
    w_fg: Dict[str, float],
    w_bg: Dict[str, float],
) -> dict:
    img_h, img_w = img_hw
    raw_dets = filter_dups_and_noise(raw_objs)
    cls_lower = {d["id"]: d["class"].lower().strip() for d in raw_dets}
    scored = compute_fgbg_scores(raw_dets, img_h, img_w, thing_prior, stuff_prior, w_fg, w_bg, thing_set, stuff_set)

    def _in_kb(c: str) -> bool:
        for cand in _kb_candidates(c):
            for kb in (mapper.kb_exact, mapper.kb_subj_geo, mapper.kb_obj_geo):
                if any(k[0] == cand for k in kb):
                    return True
        return False

    dets_for_rel = [
        d for d in scored
        if d.get("type") == "thing" or _normalize_class(cls_lower[d["id"]]) == cls_lower[d["id"]] or _in_kb(cls_lower[d["id"]])
    ]

    def _check_forbidden(pred: str, lvl: str, edge_w: float, rel_type: str, oc: str, tid: int):
        if pred in _FORBIDDEN_RULES and any(f in oc for f in _FORBIDDEN_RULES[pred]):
            return None, None, None, None
        if pred in _ACTIVE_VERBS:
            target = next((x for x in dets_for_rel if x["id"] == tid), None)
            if target and target.get("type") == "stuff":
                return None, None, None, None
        if pred == "filled with" and oc == "person":
            return None, None, None, None
        return pred, lvl, edge_w, rel_type

    def _make_rel(sid, tid, pred, lvl, edge_w, spatial, geo_raw, rel_type="spatial"):
        return {
            "source_id": sid, "target_id": tid, "predicate": pred, "rel_type": rel_type,
            "spatial_geo_rel": spatial, "geo_rel": geo_raw, "level": lvl,
            "confidence": round(edge_w, 3), "edge_weight": round(edge_w, 3),
        }

    def _sem_lookup(sc: str, oc: str) -> Optional[str]:
        for csc in _kb_candidates(sc):
            for coc in _kb_candidates(oc):
                if (csc, coc) in mapper.kb_pred_exact:
                    return mapper.kb_pred_exact[(csc, coc)]
        for coc in _kb_candidates(oc):
            for csc in _kb_candidates(sc):
                if (coc, csc) in mapper.kb_pred_exact:
                    return mapper.kb_pred_exact[(coc, csc)]
        for csc in _kb_candidates(sc):
            if csc in mapper.kb_pred_subj:
                return mapper.kb_pred_subj[csc]
        return None

    relations: List[dict] = []
    n = len(dets_for_rel)

    if n == 1:
        solo = dets_for_rel[0]
        sc = cls_lower[solo["id"]]
        stuff_dets = [d for d in scored if d.get("type") == "stuff" and d["id"] != solo["id"]]
        best_rank, best_rel = 9, None
        for sd in stuff_dets:
            oc = cls_lower[sd["id"]]
            geo = compute_geo_rel(solo["box"], sd["box"])
            if geo == "unknown":
                continue
            g, sc_q, oc_q, sid, tid = (
                ("in", oc, sc, sd["id"], solo["id"]) if geo == "contains" else (geo, sc, oc, solo["id"], sd["id"])
            )
            pred, lvl, ew, rtype = _lookup(mapper, sc_q, oc_q, g)
            if lvl in {"L4", "L5"} or ew < MIN_EDGE_WEIGHT:
                continue
            pred, lvl, ew, rtype = _check_forbidden(pred, lvl, ew, rtype, oc_q, tid)
            if pred and _LVL_RANK.get(lvl, 9) < best_rank:
                best_rank = _LVL_RANK[lvl]
                best_rel = _make_rel(sid, tid, pred, lvl, ew, g, mapper.kb_geo_label.get(g, g), rtype)
        if best_rel:
            relations.append(best_rel)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                a, b = dets_for_rel[i], dets_for_rel[j]
                if a.get("type") == "stuff" and b.get("type") == "stuff":
                    continue
                sc0, oc0 = cls_lower[a["id"]], cls_lower[b["id"]]
                geo_ab = compute_geo_rel(a["box"], b["box"])
                geo_ba = compute_geo_rel(b["box"], a["box"])
                best_rank, best_rel = 9, None

                for geo, s_id, t_id in ((geo_ab, a["id"], b["id"]), (geo_ba, b["id"], a["id"])):
                    if geo == "unknown":
                        continue
                    sc, oc = cls_lower[s_id], cls_lower[t_id]
                    g, sid, tid = ("in", t_id, s_id) if geo == "contains" else (geo, s_id, t_id)
                    if geo == "contains":
                        sc, oc = oc, sc
                    pred, lvl, ew, rtype = _lookup(mapper, sc, oc, g)
                    if lvl in {"L4", "L5"} or ew < MIN_EDGE_WEIGHT:
                        continue
                    pred, lvl, ew, rtype = _check_forbidden(pred, lvl, ew, rtype, oc, tid)
                    if pred and _LVL_RANK.get(lvl, 9) < best_rank:
                        best_rank = _LVL_RANK[lvl]
                        best_rel = _make_rel(sid, tid, pred, lvl, ew, g, mapper.kb_geo_label.get(g, g), rtype)

                if ENABLE_PASS2 and best_rel is None and (a.get("type") == "thing" or b.get("type") == "thing"):
                    for sid, tid, sc, oc, g_raw in ((a["id"], b["id"], sc0, oc0, geo_ab), (b["id"], a["id"], oc0, sc0, geo_ba)):
                        g = "in" if g_raw == "contains" else ("next_to" if g_raw == "unknown" else g_raw)
                        pred, lvl, ew, rtype = _lookup(mapper, sc, oc, g)
                        if lvl not in PASS2_MIN_LVLS:
                            continue
                        p2, l2, w2, rt2 = _check_forbidden(pred, lvl, ew, rtype, oc, tid)
                        if p2 and round(w2 * 0.75, 3) >= MIN_EDGE_WEIGHT and _LVL_RANK.get(l2, 9) < best_rank:
                            best_rank = _LVL_RANK[l2]
                            best_rel = _make_rel(sid, tid, p2, f"{l2}_p2", round(w2 * 0.75, 3), g, mapper.kb_geo_label.get(g, g), rt2)

                if best_rel is None:
                    for sc, oc, s_id, t_id in ((sc0, oc0, a["id"], b["id"]), (oc0, sc0, b["id"], a["id"])):
                        sem = _sem_lookup(sc, oc)
                        if sem:
                            p_c, _, _, _ = _check_forbidden(sem, "Lsem", 0.65, "action", oc, t_id)
                            if p_c:
                                g_ann = geo_ab if geo_ab not in {"unknown", "contains"} else (geo_ba if geo_ba not in {"unknown", "contains"} else "unknown")
                                best_rel = _make_rel(s_id, t_id, p_c, "Lsem", 0.65, g_ann, p_c, "action")
                                break
                if best_rel:
                    relations.append(best_rel)

    id_to_lbl = {d["id"]: d.get("label", "objects") for d in scored}
    lvl_order = {"L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4, "Lsem": 2, "Lfg": 5}
    if NEXT_TO_MIN_LEVEL is not None or NEXT_TO_ACTION_ONLY:
        min_rank = lvl_order.get(NEXT_TO_MIN_LEVEL, 99) if NEXT_TO_MIN_LEVEL else 99
        relations = [
            r for r in relations
            if r.get("spatial_geo_rel") != "next_to"
            or (
                (not NEXT_TO_ACTION_ONLY or r.get("rel_type") == "action")
                and lvl_order.get(r.get("level", "L5").split("_")[0], 99) <= min_rank
            )
        ]

    if DROP_BG_BG_EDGES:
        relations = [
            r for r in relations
            if not (id_to_lbl.get(r["source_id"]) == "bg" and id_to_lbl.get(r["target_id"]) == "bg")
        ]

    def _rel_score(r: dict) -> float:
        sl, tl = id_to_lbl.get(r["source_id"], "objects"), id_to_lbl.get(r["target_id"], "objects")
        bonus = 0.10 if sl == "fg" and tl == "fg" else (0.05 if "fg" in (sl, tl) else 0.0)
        act_b = 0.05 if r.get("rel_type") == "action" else 0.0
        return float(r["edge_weight"]) + bonus + act_b

    relations = sorted(relations, key=_rel_score, reverse=True)[:MAX_RELS_PER_IMG]
    if MAX_DEGREE_PER_NODE is not None:
        deg_map: Dict[int, int] = defaultdict(int)
        capped = []
        for r in relations:
            sid, tid = r["source_id"], r["target_id"]
            if deg_map[sid] < MAX_DEGREE_PER_NODE and deg_map[tid] < MAX_DEGREE_PER_NODE:
                capped.append(r)
                deg_map[sid] += 1
                deg_map[tid] += 1
        relations = capped

    connected_ids = {r["source_id"] for r in relations} | {r["target_id"] for r in relations}
    geo_pri_fg = {"on": 0, "in": 1, "overlap": 2, "above": 3, "under": 4, "next_to": 5}
    for d in dets_for_rel:
        if d["id"] in connected_ids or d.get("label") != "fg":
            continue
        sc = cls_lower[d["id"]]
        best_gp, fb_rel = 9, None
        for other in dets_for_rel:
            if other["id"] == d["id"]:
                continue
            geo = compute_geo_rel(d["box"], other["box"])
            if geo == "unknown":
                continue
            oc = cls_lower[other["id"]]
            g = "in" if geo == "contains" else geo
            sid = other["id"] if geo == "contains" else d["id"]
            tid = d["id"] if geo == "contains" else other["id"]
            oc_q = sc if geo == "contains" else oc
            lbl = _SPATIAL_LABEL.get(g, g)
            p_c, _, _, _ = _check_forbidden(lbl, "Lfg", 0.40, "spatial", oc_q, tid)
            if p_c and geo_pri_fg.get(geo, 7) < best_gp:
                best_gp = geo_pri_fg.get(geo, 7)
                fb_rel = _make_rel(sid, tid, p_c, "Lfg", 0.40, g, lbl, "spatial")
        if fb_rel:
            relations.append(fb_rel)
            connected_ids.update((fb_rel["source_id"], fb_rel["target_id"]))

    final_objects = [d for d in scored if d["id"] in connected_ids]
    seen_cls = {d["class"].lower().strip() for d in scored}
    sc_map: Dict[str, float] = defaultdict(float)
    for c in seen_cls:
        for p, f in co_occ.get(c, {}).items():
            if p not in seen_cls and f >= INFER_MIN_FREQ:
                sc_map[p] += f
    suggested = [{"class": c, "score": round(float(s), 2)} for c, s in sorted(sc_map.items(), key=lambda x: -x[1])[:INFER_TOP_K]]

    return {
        "image_name": image_name,
        "split": split,
        "img_hw": [img_h, img_w],
        "objects": final_objects,
        "relations": relations,
        "suggested_entities": suggested,
        "label_summary": {
            "fg": sum(1 for d in final_objects if d.get("label") == "fg"),
            "bg": sum(1 for d in final_objects if d.get("label") == "bg"),
            "objects": sum(1 for d in final_objects if d.get("label") == "objects"),
            "total_sg": len(final_objects),
            "orphan_count": len(scored) - len(final_objects),
            "n_action_rels": sum(1 for r in relations if r.get("rel_type") == "action"),
            "n_spatial_rels": sum(1 for r in relations if r.get("rel_type") == "spatial"),
        },
    }


def run_karpathy_pipeline(args: argparse.Namespace) -> None:
    if not MMLAB_OK:
        raise RuntimeError("OpenMMLab (`mmdet` and `mmsegmentation`) is required to run raw image detection.")

    items = load_karpathy_manifest(args.karpathy_json, dataset=args.dataset, split=args.split)
    if args.limit:
        items = items[: args.limit]

    sg_dir = args.output_dir / "sg"
    temp_dir = args.output_dir / "temp"
    sg_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    thing_cls = [l.strip() for l in args.things_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    stuff_cls = [l.strip() for l in args.stuff_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    thing_set = {_norm_name(c) for c in thing_cls}
    stuff_set = {_norm_name(c) for c in stuff_cls}

    thing_prior = load_prior_auto(args.thing_prior)
    stuff_prior = load_prior_auto(args.stuff_prior)
    w_fg = load_weights(args.fg_weights)
    w_bg = load_weights(args.bg_weights)
    mapper = RelationMapper(args.kb_path)

    co_occ: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if mapper._df_clean is not None:
        agg = mapper._df_clean.groupby(["coco_class", "partner_canonical"], sort=False)["frequency"].sum().reset_index()
        for row in agg.itertuples(index=False):
            co_occ[row.coco_class][row.partner_canonical] += float(row.frequency)
            co_occ[row.partner_canonical][row.coco_class] += float(row.frequency)

    resolved_items: List[Tuple[KarpathyItem, Path]] = []
    for it in items:
        if (sg_dir / f"{it.image_id}_sg.json").is_file():
            continue
        p = resolve_image_path(it, args.img_root)
        if p is not None:
            resolved_items.append((it, p))

    if not resolved_items:
        log.info("All %d Karpathy %s scene graphs already exist in %s.", len(items), args.split, sg_dir)
        return

    device = args.device
    det_model = init_detector(str(args.det_config), str(args.det_ckpt), device=device)
    det_classes = det_model.dataset_meta.get("classes", thing_cls) if hasattr(det_model, "dataset_meta") else thing_cls

    for it, img_path in tqdm(resolved_items, desc=f"[Stage 1 MMDet {args.dataset}/{args.split}]"):
        det_cache = temp_dir / f"{it.image_id}_det.json"
        if det_cache.is_file():
            continue
        r = inference_detector(det_model, str(img_path))
        pred = r.pred_instances
        dets = []
        for bbox, lbl, score in zip(pred.bboxes.cpu().numpy(), pred.labels.cpu().numpy(), pred.scores.cpu().numpy()):
            if score >= BBOX_SCORE_THR:
                cls = det_classes[lbl] if lbl < len(det_classes) else f"thing_{lbl}"
                dets.append({"class": cls, "box": [int(x) for x in bbox], "type": "thing", "det_conf": round(float(score), 4), "seg_conf": None, "score": round(float(score), 4)})
        det_cache.write_text(json.dumps(dets, separators=(",", ":")), encoding="utf-8")
    del det_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    seg_model = init_seg_model(str(args.seg_config), str(args.seg_ckpt), device=device)
    seg_classes = seg_model.dataset_meta.get("classes", None) if hasattr(seg_model, "dataset_meta") else None

    for it, img_path in tqdm(resolved_items, desc=f"[Stage 2 MMSeg + Stage 3 SG {args.dataset}/{args.split}]"):
        det_cache = temp_dir / f"{it.image_id}_det.json"
        raw = json.loads(det_cache.read_text(encoding="utf-8")) if det_cache.is_file() else []
        sr = inference_seg_model(seg_model, str(img_path))
        msk = sr.pred_sem_seg.data[0].cpu().numpy()
        img_h, img_w = msk.shape[:2]

        for lbl in np.unique(msk):
            if lbl == 255 or lbl < 80:
                continue
            cls = seg_classes[lbl] if seg_classes and lbl < len(seg_classes) else (stuff_cls[lbl - 80] if (lbl - 80) < len(stuff_cls) else f"stuff_{lbl}")
            m = (msk == lbl).astype(np.uint8)
            if m.sum() < STUFF_AREA_THR:
                continue
            yi, xi = np.where(m > 0)
            raw.append({"class": cls, "box": [int(xi.min()), int(yi.min()), int(xi.max()), int(yi.max())], "type": "stuff", "det_conf": None, "seg_conf": 1.0, "score": 1.0})

        sg = build_scene_graph_from_detections(
            image_name=f"{it.image_id}.jpg",
            split=f"karpathy_{args.split}",
            img_hw=(int(img_h), int(img_w)),
            raw_objs=raw,
            mapper=mapper,
            co_occ=co_occ,
            thing_set=thing_set,
            stuff_set=stuff_set,
            thing_prior=thing_prior,
            stuff_prior=stuff_prior,
            w_fg=w_fg,
            w_bg=w_bg,
        )
        (sg_dir / f"{it.image_id}_sg.json").write_text(json.dumps(sg, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SPARKG-Cap Phase 1A: Panoptic Scene Graph Generation on Karpathy Splits.")
    parser.add_argument("--dataset", choices=["coco", "flickr30k"], default="coco")
    parser.add_argument("--karpathy-json", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--img-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coco_karpathy_test"))
    parser.add_argument("--kb-path", type=Path, default=DEFAULT_KB_PATH)
    parser.add_argument("--things-txt", type=Path, default=DEFAULT_THING_TXT)
    parser.add_argument("--stuff-txt", type=Path, default=DEFAULT_STUFF_TXT)
    parser.add_argument("--thing-prior", type=Path, default=DEFAULT_THING_PRIOR)
    parser.add_argument("--stuff-prior", type=Path, default=DEFAULT_STUFF_PRIOR)
    parser.add_argument("--fg-weights", type=Path, default=DEFAULT_FG_WEIGHTS)
    parser.add_argument("--bg-weights", type=Path, default=DEFAULT_BG_WEIGHTS)
    parser.add_argument("--det-config", type=Path, default=DEFAULT_DET_CONFIG)
    parser.add_argument("--det-ckpt", type=Path, default=DEFAULT_DET_CKPT)
    parser.add_argument("--seg-config", type=Path, default=DEFAULT_SEG_CONFIG)
    parser.add_argument("--seg-ckpt", type=Path, default=DEFAULT_SEG_CKPT)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_karpathy_pipeline(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

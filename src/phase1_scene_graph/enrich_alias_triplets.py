"""
Phase 1B: Multi-Modal Alias, Attribute, and T1/T2 Triplet Scene Graph Enrichment.

Implements Stage III.D of SPARKG-Cap over official Karpathy splits:
  1. GrabCut foreground mask refinement inside detected bounding boxes.
  2. Zero-shot CLIP ViT-B/32 classification for fine-grained lexical aliases
     (`knowledge_graph/coco_alias_clip.txt`), dominant object colors, and human
     torso clothing items/colors.
  3. Two-tier relational triplet enrichment (`T1` exact canonical match, `T2` alias
     match) from `knowledge_graph/raw_merged_triplets.csv.gz` subject to physical,
     geometric, scale, and semantic compatibility constraints (`_is_forbidden`).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import KarpathyItem, load_karpathy_manifest, resolve_image_path

try:
    import clip as _clip_lib
    CLIP_OK = True
except ImportError:
    CLIP_OK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_ALIAS_PATH = ROOT / "knowledge_graph" / "coco_alias_clip.txt"
DEFAULT_MERGED_KB_PATH = ROOT / "knowledge_graph" / "raw_merged_triplets.csv.gz"

CLIP_MODEL = "ViT-B/32"
IMG_BATCH_SIZE = 64
MIN_CROP_SIZE = 32
ALIAS_CONF_THR = 0.45
CLIP_CHUNK_THRESHOLD = 12
CLIP_CHUNK_SIZE = 10
CLIP_CHUNK_CONF_THR = 0.27
COLOR_CONF_THR = 0.35

TRIPLET_TOP_K = 3
TRIPLET_MIN_COUNT = 2
TRIPLET_MAX_RELS = 30
TRIPLET_MIN_WEIGHT = 0.55
_TRIP_WT = {"T1": 0.92, "T2": 0.80}

USE_GRABCUT = True
GRABCUT_ITER = 2
GRABCUT_MAX_DIM = 512
GRABCUT_MIN_BBOX_DIM = 50
GRABCUT_FG_MIN_RATIO = 0.05

DETECT_CLOTHING = True
CLOTHING_CONF_THR = 0.45
CLOTHING_COLOR_CONF_THR = 0.35
MIN_PERSON_BBOX_AREA = 15000
MIN_COLOR_BBOX_AREA = 10000
TORSO_TOP_FRAC = 0.15
TORSO_BOT_FRAC = 0.65

COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "brown", "white", "gray", "black", "pink"]
CLOTHING_ITEMS = ["shirt", "suit", "pants", "jeans", "dress", "bikini"]
CLOTHING_COLORS = ["red", "blue", "green", "yellow", "white", "black", "pink", "orange", "brown", "gray", "purple", "striped"]

_ENT_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
_ENT_PETS = {"cat", "dog"}
_ENT_VEHICLES = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "airplane"}
_ENT_FURNITURE = {"bed", "chair", "couch", "bench", "dining table"}
_ENT_CAN_FLY = {"bird", "airplane", "kite", "frisbee", "sports ball"}
_ENT_CAN_SWIM = {"boat", "bird", "person", "dog", "bear", "elephant", "surfboard"}
_ENT_CAN_BE_RIDDEN = {"bicycle", "motorcycle", "horse", "elephant", "surfboard", "skateboard", "skis", "snowboard", "boat", "car", "bus", "train", "bench"}

_PRED_FLYING = {"flying", "flying in", "flying over", "soaring in", "fly in"}
_PRED_SWIMMING = {"swimming", "swimming in", "floating in", "floating on", "sailing", "sailing on"}
_PRED_PARKING = {"parked", "parked on", "parked in", "parked at", "parking"}
_PRED_WEARING = {"wearing", "wears"}
_PRED_HOLDING = {"holding", "holds", "carrying"}
_PRED_RIDING = {"riding", "rides", "riding on"}
_PRED_DRIVING = {"driving", "drives", "driving on", "driving down"}
_PRED_EATING = {"eating", "eats"}
_PRED_SITTING = {"sitting on", "sits on", "lying on", "laying on"}
_PRED_CONTACT = _PRED_WEARING | _PRED_HOLDING | _PRED_RIDING | _PRED_EATING | _PRED_SITTING

_MALE_PERSON_ALIASES = frozenset({"man", "young man", "elderly man", "older man", "boy", "young boy", "little boy", "guy"})
_FEMALE_PERSON_ALIASES = frozenset({"woman", "young woman", "elderly woman", "older woman", "girl", "young girl", "little girl", "lady"})

CLOTHING_GENDER_RULES: Dict[str, Set[str]] = {"male": {"dress", "skirt", "bikini"}}
RELATION_GENDER_RULES: Dict[str, Dict[str, Set[str]]] = {
    "female": {"wearing": {"tie", "necktie"}},
    "male": {"wearing": {"dress", "skirt", "bikini", "handbag", "purse"}, "holding": {"handbag", "purse"}},
}
NO_CLOTHING_ALIASES = {
    "football player", "baseball player", "tennis player", "basketball player", "soccer player",
    "volleyball player", "hockey player", "rugby player", "skier", "snowboarder", "surfer",
    "skateboarder", "rider", "cyclist", "biker", "jockey",
}
NO_CLIP_CLASSES = {
    "fork", "knife", "spoon", "mouse", "keyboard", "remote", "toaster", "microwave", "hair drier",
    "toothbrush", "scissors", "skis", "snowboard", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "frisbee", "kite", "stop sign", "parking meter", "zebra", "giraffe",
    "elephant", "cow", "umbrella", "banana", "donut", "broccoli", "carrot", "laptop", "toilet",
    "sink", "pizza", "bench", "horse", "sheep", "bus", "truck", "clock", "clothes", "clothing",
}
NO_COLOR_CLASSES = {
    "elephant", "bear", "zebra", "giraffe", "tv", "laptop", "cell phone", "microwave", "oven",
    "refrigerator", "toaster", "sink", "toilet", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "potted plant", "wine glass", "fork", "knife",
    "spoon", "mouse", "keyboard", "remote", "hair drier", "toothbrush", "scissors", "skis", "snowboard",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "stop sign",
    "parking meter", "umbrella",
}
STUFF_CLIP_CLASSES = {"playingfield"}
_FORBIDDEN_OBJ_RULES: Dict[str, Set[str]] = {
    "wearing": {"playingfield", "sky", "tree", "grass", "road", "building", "wall", "snow", "water", "ground", "mountain", "fence", "person", "car"},
    "riding": {"playingfield", "sky", "tree", "grass", "road", "building", "wall", "snow", "water", "ground", "mountain", "fence", "person"},
    "holding": {"playingfield", "sky", "tree", "grass", "road", "building", "wall", "snow", "water", "ground", "mountain", "fence"},
    "filled with": {"person", "dog", "cat", "bird", "horse"},
}

_RE_SPACES = re.compile(r"\s+")


def _n(s: str) -> str:
    return _RE_SPACES.sub(" ", s.replace("_", " ").lower().strip())


def _get_person_gender(alias: str) -> str:
    a = _n(alias)
    if a in _MALE_PERSON_ALIASES:
        return "male"
    if a in _FEMALE_PERSON_ALIASES:
        return "female"
    return "unknown"


def _bboxes_intersect(box_a: List[float], box_b: List[float]) -> bool:
    return min(box_a[2], box_b[2]) > max(box_a[0], box_b[0]) and min(box_a[3], box_b[3]) > max(box_a[1], box_b[1])


def _bbox_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class AliasMapper:
    """Loads fine-grained CLIP candidate aliases per canonical class (`coco_alias_clip.txt`)."""

    def __init__(self, path: Path):
        self.cls_to_aliases: Dict[str, List[str]] = {}
        self.alias_to_cls: Dict[str, str] = {}
        if path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                cls_part, alias_part = line.split(":", 1)
                cls = _n(cls_part)
                aliases = [_n(a) for a in alias_part.split(",") if _n(a) and _n(a) != "(no aliases)"]
                self.cls_to_aliases[cls] = aliases
                self.alias_to_cls[cls] = cls
                for a in aliases:
                    self.alias_to_cls.setdefault(a, cls)

    def get_aliases(self, cls: str, max_n: int = 30) -> List[str]:
        base = cls.split("-")[0] if "-" in cls else cls
        aliases = self.cls_to_aliases.get(_n(cls)) or self.cls_to_aliases.get(_n(base)) or []
        short = [a for a in aliases if len(a.split()) <= 3]
        long_ = [a for a in aliases if len(a.split()) > 3]
        return (short + long_)[:max_n] or [cls]

    def map(self, text: str) -> Optional[str]:
        t = _n(text)
        if t in self.alias_to_cls:
            return self.alias_to_cls[t]
        words = t.split()
        for k in (2, 3):
            ph = " ".join(words[:k])
            if ph in self.alias_to_cls:
                return self.alias_to_cls[ph]
        for w in words:
            if w in self.alias_to_cls:
                return self.alias_to_cls[w]
        return None

    @staticmethod
    def stuff_base(cls: str) -> str:
        return cls.split("-")[0] if "-" in cls else cls

    def stuff_aliases(self, cls: str) -> List[str]:
        base = self.stuff_base(cls)
        return self.cls_to_aliases.get(_n(cls)) or self.cls_to_aliases.get(_n(base)) or [base]


class TripletKB:
    """Two-tier (`T1`/`T2`) corpus triplet lookup supporting both `.csv` and `.csv.gz`."""

    def __init__(self, merged_path: Path, alias: AliasMapper):
        self._alias = alias
        self._kb: Dict[Tuple[str, str], Dict[str, int]] = {}
        actual = merged_path
        if not actual.is_file() and merged_path.with_suffix(merged_path.suffix + ".gz").is_file():
            actual = merged_path.with_suffix(merged_path.suffix + ".gz")
        elif not actual.is_file() and merged_path.suffix == ".gz" and merged_path.with_suffix("").is_file():
            actual = merged_path.with_suffix("")
        if not actual.is_file():
            log.warning("Merged triplet file not found: %s", merged_path)
            return

        t0 = time.time()
        df = pd.read_csv(actual, dtype=str, low_memory=False, na_filter=False, on_bad_lines="skip", compression="infer")
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ("subject", "predicate", "object"):
            if col in df.columns:
                df[col] = df[col].str.lower().str.strip()
        df["freq_combined"] = pd.to_numeric(df.get("freq_combined", 0), errors="coerce").fillna(0).astype(int)
        sub = df[(df["subject"].str.len() > 0) & (df["object"].str.len() > 0) & (df["predicate"].str.len() > 0)]
        for s, o, p, cnt in zip(sub["subject"], sub["object"], sub["predicate"], sub["freq_combined"]):
            ci = int(cnt)
            if ci < 1:
                continue
            bucket = self._kb.setdefault((s, o), {})
            if p not in bucket or ci > bucket[p]:
                bucket[p] = ci
        log.info("[TripletKB] Loaded %d entity pairs from %s (%.1fs)", len(self._kb), actual.name, time.time() - t0)

    def query(
        self, sc: str, oc: str, sa: str = "", oa: str = "",
        top_k: int = TRIPLET_TOP_K, min_count: int = TRIPLET_MIN_COUNT,
    ) -> List[Tuple[str, int, str]]:
        results: Dict[str, Tuple[int, str]] = {}

        def _merge(key: Tuple[str, str], level: str) -> None:
            entry = self._kb.get(key)
            if not entry:
                return
            for pred, cnt in entry.items():
                if cnt >= min_count and (pred not in results or cnt > results[pred][0]):
                    results[pred] = (cnt, level)

        _merge((sc, oc), "T1")
        if sa and sa != sc:
            _merge((self._alias.map(sa) or sa, oc), "T2")
        if oa and oa != oc:
            _merge((sc, self._alias.map(oa) or oa), "T2")
        if sa and sa != sc and oa and oa != oc:
            _merge((self._alias.map(sa) or sa, self._alias.map(oa) or oa), "T2")

        lvl_rank = {"T1": 0, "T2": 1}
        ranked = sorted(((p, c, l) for p, (c, l) in results.items()), key=lambda x: (lvl_rank.get(x[2], 9), -x[1]))
        return ranked[:top_k]


def is_forbidden_relation(
    pred: str,
    obj_cls: str,
    subj_cls: str = "",
    subj_alias: str = "",
    obj_alias: str = "",
    subj_box: Optional[List[float]] = None,
    obj_box: Optional[List[float]] = None,
) -> bool:
    """Verify physical, geometric, scale, gender, and ontological compatibility of a candidate triplet."""
    p = pred.lower()
    if subj_box and obj_box:
        if p in _PRED_CONTACT and not _bboxes_intersect(subj_box, obj_box):
            return True
        if p in _PRED_HOLDING and _bbox_area(obj_box) > _bbox_area(subj_box) * 2.5:
            return True
        if p in _PRED_RIDING and _bbox_area(obj_box) < _bbox_area(subj_box) * 0.1:
            return True

    if subj_cls == "person":
        gender = _get_person_gender(subj_alias or subj_cls)
        if gender in RELATION_GENDER_RULES:
            blocked = RELATION_GENDER_RULES[gender].get(p, set())
            if obj_cls in blocked or obj_alias in blocked:
                return True

    if p in _PRED_FLYING and subj_cls not in _ENT_CAN_FLY:
        return True
    if p in _PRED_SWIMMING and subj_cls not in _ENT_CAN_SWIM:
        return True
    if p in (_PRED_WEARING | _PRED_HOLDING | _PRED_DRIVING):
        if subj_cls != "person":
            return True
        if p in _PRED_HOLDING and obj_cls in (_ENT_VEHICLES | _ENT_FURNITURE | {"elephant", "horse", "cow", "zebra", "bear", "giraffe"}):
            return True
    if p in _PRED_RIDING and (subj_cls != "person" or obj_cls not in _ENT_CAN_BE_RIDDEN):
        return True
    if p in _PRED_EATING and subj_cls not in ({"person"} | _ENT_ANIMALS):
        return True
    if p in _PRED_SITTING:
        if subj_cls not in ({"person"} | _ENT_PETS | _ENT_ANIMALS):
            return True
        if obj_cls in {"cup", "apple", "bottle", "bowl", "fork", "knife", "toothbrush", "tie", "scissors", "donut", "hot dog", "pizza", "carrot", "broccoli", "cell phone", "remote", "mouse"}:
            return True
    if p in _PRED_PARKING and subj_cls not in _ENT_VEHICLES:
        return True
    if obj_cls in _FORBIDDEN_OBJ_RULES.get(p, set()):
        return True
    return False


def build_triplet_sg(sg_att: dict, kb: TripletKB, alias_mapper: AliasMapper) -> dict:
    """Upgrade an attribute-enriched scene graph with T1/T2 predicates and `<ATTR>` clothing nodes."""
    sg_new = dict(sg_att)
    objects = copy.deepcopy(sg_att.get("objects", []))
    sg_new["objects"] = objects
    if len(objects) < 2:
        sg_new["_triplet_version"] = True
        return sg_new

    id_to_lbl = {o["id"]: o.get("label", "objects") for o in objects}
    kb2_rels = {(r["source_id"], r["target_id"]): r for r in sg_att.get("relations", [])}
    new_rels: List[dict] = []

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if a.get("label") == "bg" and b.get("label") == "bg":
                continue
            sc, oc = _n(a.get("class", "")), _n(b.get("class", ""))
            sa, oa = _n(a.get("alias", "")), _n(b.get("alias", ""))
            sa_list = [_n(x) for x in alias_mapper.stuff_aliases(sc) if x] if sc in STUFF_CLIP_CLASSES else [sa or sc]
            oa_list = [_n(x) for x in alias_mapper.stuff_aliases(oc) if x] if oc in STUFF_CLIP_CLASSES else [oa or oc]
            box_a, box_b = a.get("box", [0, 0, 0, 0]), b.get("box", [0, 0, 0, 0])
            existing = kb2_rels.get((a["id"], b["id"])) or kb2_rels.get((b["id"], a["id"]))
            spatial_hint = existing.get("spatial_geo_rel", "") if existing else ""

            best_rel, best_score = None, -1.0
            for sid, tid, s_cls, o_cls, s_aliases, o_aliases, sb, ob in (
                (a["id"], b["id"], sc, oc, sa_list, oa_list, box_a, box_b),
                (b["id"], a["id"], oc, sc, oa_list, sa_list, box_b, box_a),
            ):
                for s_alias in s_aliases:
                    for o_alias in o_aliases:
                        for pred, count, level in kb.query(s_cls, o_cls, s_alias, o_alias):
                            if is_forbidden_relation(pred, o_cls, subj_cls=s_cls, subj_alias=s_alias, obj_alias=o_alias, subj_box=sb, obj_box=ob):
                                continue
                            wt = _TRIP_WT.get(level, 0.60)
                            if spatial_hint and spatial_hint in pred.split():
                                wt = min(wt + 0.05, 1.0)
                            if wt < TRIPLET_MIN_WEIGHT:
                                continue
                            score = wt + count * 0.0005
                            if score > best_score:
                                best_score = score
                                best_rel = {
                                    "source_id": sid,
                                    "target_id": tid,
                                    "source_alias": s_alias or s_cls,
                                    "target_alias": o_alias or o_cls,
                                    "predicate": pred,
                                    "rel_type": "spatial",
                                    "spatial_geo_rel": existing.get("spatial_geo_rel", "") if existing else "",
                                    "geo_rel": existing.get("geo_rel", "") if existing else "",
                                    "triplet_level": level,
                                    "triplet_count": count,
                                    "triplet_source": "merged_kb",
                                    "confidence": round(wt, 3),
                                    "edge_weight": round(wt, 3),
                                }
            if best_rel is None and existing is not None:
                fb = dict(existing)
                fb["triplet_level"] = "KB_v2"
                fb["triplet_source"] = "fallback"
                best_rel = fb
            if best_rel is not None:
                new_rels.append(best_rel)

    def _trip_score(r: dict) -> float:
        sl, tl = id_to_lbl.get(r["source_id"], "objects"), id_to_lbl.get(r["target_id"], "objects")
        b = 0.10 if sl == "fg" and tl == "fg" else (0.05 if "fg" in (sl, tl) else 0.0)
        lb = {"T1": 0.06, "T2": 0.04, "T3": 0.02, "T4": 0.0}.get(r.get("triplet_level", "T4"), 0.0)
        return float(r["edge_weight"]) + b + lb

    if len(new_rels) > TRIPLET_MAX_RELS:
        new_rels = sorted(new_rels, key=_trip_score, reverse=True)[:TRIPLET_MAX_RELS]

    connected = {r["source_id"] for r in new_rels} | {r["target_id"] for r in new_rels}
    next_id = max((o["id"] for o in objects), default=0) + 1
    cloth_objs, cloth_rels = [], []

    for obj in objects:
        attrs = obj.get("attributes") or {}
        if "clothing" in attrs:
            ctype = attrs["clothing"]
            ccolor = attrs.get("clothing_color", "")
            cconf = float(attrs.get("clothing_conf", 0.95))
            cobj = {
                "id": next_id, "class": ctype, "alias": ctype, "alias_conf": round(cconf, 3),
                "label": "clth", "type": "thing", "box": obj["box"], "attributes": {}, "is_clothing_node": True,
            }
            if ccolor:
                cobj["attributes"]["color"] = ccolor
            cloth_objs.append(cobj)
            cloth_rels.append({
                "source_id": obj["id"], "target_id": next_id,
                "source_alias": obj.get("alias") or obj.get("class", ""),
                "target_alias": ctype, "predicate": "wearing", "rel_type": "action",
                "triplet_level": "attribute", "triplet_source": "clip_zero_shot", "edge_weight": cconf,
            })
            connected.update((obj["id"], next_id))
            next_id += 1

    sg_new["objects"] = [o for o in objects if o["id"] in connected] + cloth_objs
    sg_new["relations"] = new_rels + cloth_rels
    sg_new["_triplet_version"] = True
    sg_new["_triplet_n_rels"] = len(sg_new["relations"])
    sg_new["_fallback_rels"] = sum(1 for r in new_rels if r.get("triplet_source") == "fallback")
    return sg_new


def grabcut_foreground_mask(img_bgr: np.ndarray, box: List[int], n_iter: int = GRABCUT_ITER) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < GRABCUT_MIN_BBOX_DIM or bh < GRABCUT_MIN_BBOX_DIM:
        return None
    scale = min(1.0, GRABCUT_MAX_DIM / max(h, w, 1))
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        work = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        sx1, sy1, sx2, sy2 = int(x1 * scale), int(y1 * scale), min(nw, int(x2 * scale) + 1), min(nh, int(y2 * scale) + 1)
    else:
        work = img_bgr
        sx1, sy1, sx2, sy2 = x1, y1, x2, y2
    sbw, sbh = sx2 - sx1, sy2 - sy1
    if sbw < 5 or sbh < 5:
        return None
    mask = np.zeros(work.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mx, my = max(2, int(sbw * 0.03)), max(2, int(sbh * 0.03))
    rx1, ry1 = max(0, sx1 - mx), max(0, sy1 - my)
    rx2, ry2 = min(work.shape[1], sx2 + mx), min(work.shape[0], sy2 + my)
    try:
        cv2.grabCut(work, mask, (rx1, ry1, rx2 - rx1, ry2 - ry1), bgd, fgd, n_iter, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg_work = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg_small = fg_work[sy1:sy2, sx1:sx2]
    if float(fg_small.sum() / 255) / max(fg_small.size, 1) < GRABCUT_FG_MIN_RATIO:
        return None
    return cv2.resize(fg_small, (bw, bh), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else fg_small


class ClipEnricher:
    """Batched zero-shot CLIP ViT-B/32 visual encoder for entity aliases, object colors, and clothing."""

    def __init__(self, alias_mapper: AliasMapper, device: str):
        self.device = device
        self.alias_mapper = alias_mapper
        self.dtype = torch.float16 if device != "cpu" else torch.float32
        self.model, self.preprocess = _clip_lib.load(CLIP_MODEL, device=device)
        self.model.eval()
        self.alias_feat: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
        self.alias_chunks: Dict[str, List[Tuple[torch.Tensor, List[str]]]] = {}
        self.color_feat: Dict[str, torch.Tensor] = {}
        self.cloth_type_feat: Optional[torch.Tensor] = None
        self.cloth_base_feat: Optional[torch.Tensor] = None
        self.cloth_color_feats: Dict[str, torch.Tensor] = {}
        self._precompute_text_features()

    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        tokens = _clip_lib.tokenize(prompts, truncate=True).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1).to(self.dtype)

    def _precompute_text_features(self) -> None:
        with torch.inference_mode():
            for cls_name in self.alias_mapper.cls_to_aliases:
                aliases = self.alias_mapper.get_aliases(cls_name, 80)
                prompts = [f"a photo of {a}" if not a.startswith("a photo") else a for a in aliases]
                if len(aliases) > CLIP_CHUNK_THRESHOLD:
                    chunks = []
                    for s in range(0, len(aliases), CLIP_CHUNK_SIZE):
                        chunks.append((self._encode_texts(prompts[s : s + CLIP_CHUNK_SIZE]), aliases[s : s + CLIP_CHUNK_SIZE]))
                    self.alias_chunks[cls_name] = chunks
                else:
                    self.alias_feat[cls_name] = (self._encode_texts(prompts), aliases)

            for cls_name in set(self.alias_feat) | set(self.alias_chunks):
                if cls_name not in NO_COLOR_CLASSES:
                    self.color_feat[cls_name] = self._encode_texts([f"a photo of a {c} {cls_name}." for c in COLORS])

            self.cloth_base_feat = self._encode_texts(["a photo of a person"])
            self.cloth_type_feat = self._encode_texts([f"a person wearing a {it}" for it in CLOTHING_ITEMS])
            for it in CLOTHING_ITEMS:
                self.cloth_color_feats[it] = self._encode_texts([f"a person wearing a {c} {it}" for c in CLOTHING_COLORS])

    def _encode_crop(self, crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        t = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return F.normalize(self.model.encode_image(t), dim=-1).float()

    def enrich_image_objects(self, objects: List[dict], img_bgr: np.ndarray) -> None:
        h, w = img_bgr.shape[:2]
        for obj in objects:
            cls = _n(obj.get("class", ""))
            if obj.get("type") == "stuff" and cls not in STUFF_CLIP_CLASSES:
                obj["alias"] = self.alias_mapper.stuff_base(cls)
                obj["alias_conf"] = 1.0
                continue

            x1, y1, x2, y2 = [int(v) for v in obj.get("box", [0, 0, 0, 0])]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < MIN_CROP_SIZE or bh < MIN_CROP_SIZE:
                continue

            fg_mask = grabcut_foreground_mask(img_bgr, [x1, y1, x2, y2]) if USE_GRABCUT else None
            crop = img_bgr[y1:y2, x1:x2].copy()
            if fg_mask is not None and fg_mask.shape[:2] == crop.shape[:2]:
                crop[fg_mask == 0] = [128, 128, 128]

            # 1. Alias classification
            if cls not in NO_CLIP_CLASSES and (cls in self.alias_feat or cls in self.alias_chunks):
                feat = self._encode_crop(crop)
                if feat is not None:
                    if cls in self.alias_chunks:
                        best_a, best_r, best_c = None, -1e9, 0.0
                        for c_feats, c_aliases in self.alias_chunks[cls]:
                            raw = (100.0 * feat @ c_feats.float().T).squeeze(0)
                            probs = raw.softmax(dim=-1)
                            k = int(probs.argmax().item())
                            if float(raw[k]) > best_r:
                                best_r, best_a, best_c = float(raw[k]), c_aliases[k], float(probs[k])
                        if best_a and best_c >= CLIP_CHUNK_CONF_THR:
                            obj["alias"] = best_a
                            obj["alias_conf"] = round(best_c, 3)
                    else:
                        t_feats, aliases = self.alias_feat[cls]
                        probs = (100.0 * feat @ t_feats.float().T).squeeze(0).softmax(dim=-1)
                        k = int(probs.argmax().item())
                        if float(probs[k]) >= ALIAS_CONF_THR:
                            obj["alias"] = aliases[k]
                            obj["alias_conf"] = round(float(probs[k]), 3)

            # 2. Person clothing vs. object color
            area = bw * bh
            if cls == "person" and DETECT_CLOTHING and area >= MIN_PERSON_BBOX_AREA:
                alias_cur = _n(obj.get("alias", ""))
                if alias_cur not in NO_CLOTHING_ALIASES and self.cloth_type_feat is not None and self.cloth_base_feat is not None:
                    top, bot = y1 + int(bh * TORSO_TOP_FRAC), y1 + int(bh * TORSO_BOT_FRAC)
                    if bot - top >= 20:
                        torso = crop[top - y1 : bot - y1, :]
                        tfeat = self._encode_crop(torso)
                        if tfeat is not None:
                            raw = (100.0 * tfeat @ self.cloth_type_feat.float().T).squeeze(0)
                            probs = raw.softmax(dim=-1)
                            base_sim = float((100.0 * tfeat @ self.cloth_base_feat.float().T).item())
                            k = int(probs.argmax().item())
                            citem = CLOTHING_ITEMS[k]
                            gender = _get_person_gender(alias_cur)
                            if float(probs[k]) >= CLOTHING_CONF_THR and float(raw[k]) >= base_sim + 0.5 and citem not in CLOTHING_GENDER_RULES.get(gender, set()):
                                attrs = obj.setdefault("attributes", {})
                                attrs["clothing"] = citem
                                attrs["clothing_conf"] = round(float(probs[k]), 3)
                                c_probs = (100.0 * tfeat @ self.cloth_color_feats[citem].float().T).squeeze(0).softmax(dim=-1)
                                ck = int(c_probs.argmax().item())
                                if float(c_probs[ck]) >= CLOTHING_COLOR_CONF_THR:
                                    attrs["clothing_color"] = CLOTHING_COLORS[ck]
            elif cls != "person" and cls in self.color_feat and area >= MIN_COLOR_BBOX_AREA:
                feat = self._encode_crop(crop)
                if feat is not None:
                    probs = (100.0 * feat @ self.color_feat[cls].float().T).squeeze(0).softmax(dim=-1)
                    ck = int(probs.argmax().item())
                    if float(probs[ck]) >= COLOR_CONF_THR:
                        obj.setdefault("attributes", {})["color"] = COLORS[ck]


def main() -> None:
    parser = argparse.ArgumentParser(description="SPARKG-Cap Phase 1B: Alias, Attribute & T1/T2 Triplet Enrichment.")
    parser.add_argument("--dataset", choices=["coco", "flickr30k"], default="coco")
    parser.add_argument("--karpathy-json", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--img-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--sg-root", type=Path, default=Path("outputs/coco_karpathy_test"))
    parser.add_argument("--alias-path", type=Path, default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--merged-kb", type=Path, default=DEFAULT_MERGED_KB_PATH)
    parser.add_argument("--only", choices=["both", "att", "triplets"], default="both")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = load_karpathy_manifest(args.karpathy_json, dataset=args.dataset, split=args.split)
    if args.limit:
        items = items[: args.limit]

    sg_dir = args.sg_root / "sg"
    att_dir = args.sg_root / "sg_with_att"
    trip_dir = args.sg_root / "sg_with_triplets"
    att_dir.mkdir(parents=True, exist_ok=True)
    trip_dir.mkdir(parents=True, exist_ok=True)

    alias_mapper = AliasMapper(args.alias_path)
    triplet_kb = TripletKB(args.merged_kb, alias_mapper)

    enricher: Optional[ClipEnricher] = None
    if args.only in {"both", "att"}:
        if not CLIP_OK:
            raise RuntimeError("OpenAI `clip` package is required for --only both/att (`pip install git+https://github.com/openai/CLIP.git`).")
        enricher = ClipEnricher(alias_mapper, device=args.device)

    for it in tqdm(items, desc=f"[Phase 1B Enrichment {args.dataset}/{args.split}]"):
        sg_path = sg_dir / f"{it.image_id}_sg.json"
        att_path = att_dir / f"{it.image_id}_sg.json"
        trip_path = trip_dir / f"{it.image_id}_sg.json"

        if args.only in {"both", "att"}:
            if not sg_path.is_file():
                continue
            sg_data = json.loads(sg_path.read_text(encoding="utf-8"))
            img_path = resolve_image_path(it, args.img_root)
            if img_path is not None and enricher is not None:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    enricher.enrich_image_objects(sg_data.get("objects", []), img_bgr)
            sg_data["_alias_enriched"] = True
            att_path.write_text(json.dumps(sg_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            sg_att = sg_data
        else:
            if not att_path.is_file():
                continue
            sg_att = json.loads(att_path.read_text(encoding="utf-8"))

        if args.only in {"both", "triplets"}:
            sg_trip = build_triplet_sg(sg_att, triplet_kb, alias_mapper)
            trip_path.write_text(json.dumps(sg_trip, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()

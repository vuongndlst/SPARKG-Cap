"""
Phase 2: Unified SPARKG-Cap Caption Generation over Official Karpathy Splits.

Supports two LLM generation backends sharing identical context serialization (C_I),
system prompts, training-split 4-gram phrase boosters, few-shot exemplars, and
deterministic post-processing:
  1. `--backend openai` (Default): OpenAI `gpt-4o-mini` (`temperature=0`) with
     asynchronous key-pool rotation and automatic rate-limit backoff.
  2. `--backend qwen` (Optional): Open-weight local `Qwen/Qwen2.5-14B-Instruct`
     via Hugging Face `transformers` (with optional `--load-in-4bit` NF4 BF16
     quantization and automatic OOM batch-size halving).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import load_karpathy_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@dataclass
class Config:
    backend: str = "openai"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    qwen_model: str = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    load_in_4bit: bool = False
    qwen_batch_size: int = 16
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "10"))
    temperature: float = 0.0
    max_tokens_coco: int = 200
    max_tokens_flickr: int = 250
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_base_delay: float = 2.0
    rel_conf_threshold: float = 0.50
    max_rels_trip: int = 10
    alias_conf_min: float = 0.30


CFG = Config()

ANIMATE_CLASSES = frozenset({
    "person", "man", "woman", "boy", "girl", "child", "baby", "player",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "monkey",
})
ACTION_PREDICATES = frozenset({
    "holding", "eating", "riding", "wearing", "looking at", "playing", "talking to", "walking on",
    "operating", "using", "catching", "throwing", "driving", "carrying", "kicking", "hitting",
    "jumping over", "climbing", "feeding", "petting", "chasing", "pulling", "pushing", "swinging",
})
SPATIAL_PREDICATES = frozenset({
    "on", "in", "under", "next to", "behind", "in front of", "above", "near", "beside", "across from",
})

COCO_ALIAS_MAP: Dict[str, str] = {
    "commode": "toilet", "washbasin": "sink", "basin": "sink", "cooker": "oven",
    "bovine": "cow", "cattle": "cow", "bull": "cow", "heifer": "cow", "steer": "cow",
    "ewe": "sheep", "ram": "sheep", "lamb": "sheep",
    "stallion": "horse", "foal": "horse", "equine": "horse", "mare": "horse", "colt": "horse", "pony": "horse",
    "feline": "cat", "kitty": "cat", "kitten": "cat", "tabby": "cat",
    "terrier": "dog", "dachshund": "dog", "poodle": "dog", "hound": "dog", "labrador": "dog",
    "retriever": "dog", "bulldog": "dog", "shepherd": "dog",
    "hawk": "bird", "robin": "bird", "sparrow": "bird", "crow": "bird", "parrot": "bird",
    "pelican": "bird", "heron": "bird", "woodpecker": "bird", "hummingbird": "bird", "pigeon": "bird", "fowl": "bird",
    "minibus": "bus", "transit bus": "bus", "shuttle": "bus",
    "lorry": "truck", "dump truck": "truck", "pickup truck": "truck", "pickup": "truck",
    "taxicab": "taxi", "scooter": "motorcycle", "locomotive": "train", "trolley": "train", "tram": "train",
    "airliner": "airplane", "aeroplane": "airplane", "speedboat": "boat", "watercraft": "boat", "vessel": "boat",
    "convertible": "car", "sedan": "car", "SUV": "car",
    "park bench": "bench", "park seat": "bench", "settee": "couch", "loveseat": "couch", "davenport": "couch",
    "office chair": "chair", "folding chair": "chair", "armchair": "chair",
    "king bed": "bed", "queen bed": "bed", "bunk bed": "bed", "bunk": "bed", "cot": "bed",
    "flatbread": "pizza", "pizza pie": "pizza", "pie": "pizza", "pastry": "cake", "birthday cake": "cake", "cupcake": "cake",
    "hoagie": "sandwich", "sub": "sandwich", "mixing bowl": "bowl", "salad bowl": "bowl", "carafe": "bottle",
    "cravat": "tie", "necktie": "tie", "daypack": "backpack", "rucksack": "backpack", "knapsack": "backpack",
    "baggage": "luggage", "wall clock": "clock", "timepiece": "clock", "alarm clock": "clock",
    "sports ball": "ball", "playing field": "field", "potted plant": "potted plant", "fire hydrant": "fire hydrant",
    "teddy bear": "teddy bear", "dining table": "table", "mandarin": "orange", "tangerine": "orange", "citrus": "orange",
}
ROLE_TO_GENERIC = {"chef": "person", "worker": "person", "guard": "person", "officer": "person", "soldier": "person", "airman": "person"}
KEEP_ROLES_WITH_EQUIPMENT: Dict[str, set] = {
    "tennis player": {"tennis racket", "racket", "racquet"},
    "baseball player": {"baseball bat", "bat", "baseball glove", "glove"},
    "player": {"bat", "glove", "racket", "ball"},
    "surfer": {"surfboard"}, "skier": {"skis", "ski"}, "snowboarder": {"snowboard"},
    "skateboarder": {"skateboard"}, "cyclist": {"bicycle", "bike"},
}

FLICKR_PERSON_ALIAS: Dict[str, str] = {
    "elderly man": "man", "older man": "man", "old man": "man",
    "elderly woman": "woman", "older woman": "woman", "old woman": "woman",
    "young man": "man", "young woman": "woman",
    "adult": "person", "guy": "man", "gentleman": "man",
    "lady": "woman", "female": "woman", "male": "man",
    "toddler": "child", "infant": "baby", "kid": "child",
    "football player": "man", "baseball player": "man", "tennis player": "man", "soccer player": "man",
    "skier": "person", "snowboarder": "person", "surfer": "person",
    "skateboarder": "person", "rider": "person", "cyclist": "person",
    "chef": "man", "worker": "man", "guard": "man", "officer": "man", "soldier": "man", "airman": "man",
}
FLICKR_OBJECT_ALIAS: Dict[str, str] = {
    "feline": "cat", "kitty": "cat", "kitten": "cat", "tabby": "cat",
    "terrier": "dog", "dachshund": "dog", "poodle": "dog", "hound": "dog",
    "labrador": "dog", "retriever": "dog", "bulldog": "dog", "shepherd": "dog",
    "hawk": "bird", "robin": "bird", "sparrow": "bird", "crow": "bird", "parrot": "bird", "pelican": "bird", "pigeon": "bird", "fowl": "bird",
    "stallion": "horse", "foal": "horse", "equine": "horse", "mare": "horse", "colt": "horse", "pony": "horse",
    "ewe": "sheep", "ram": "sheep", "lamb": "sheep",
    "bovine": "cow", "cattle": "cow", "bull": "cow", "heifer": "cow",
    "scooter": "motorcycle", "locomotive": "train", "trolley": "train",
    "convertible": "car", "sedan": "car", "SUV": "car",
    "speedboat": "boat", "watercraft": "boat", "vessel": "boat", "yacht": "boat",
    "park bench": "bench", "park seat": "bench",
    "dining table": "table", "settee": "couch", "loveseat": "couch",
    "cravat": "tie", "necktie": "tie", "daypack": "backpack", "rucksack": "backpack", "knapsack": "backpack",
}

_NUM_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}
_ARTIFACT_PAT = re.compile(r"^\d[a-z]")


def _apply_coco_alias(name: str, all_class_names: Optional[set] = None) -> str:
    if not name:
        return name
    nl = name.lower().strip()
    if nl in COCO_ALIAS_MAP:
        return COCO_ALIAS_MAP[nl]
    if nl in ROLE_TO_GENERIC:
        return ROLE_TO_GENERIC[nl]
    if nl in KEEP_ROLES_WITH_EQUIPMENT:
        equip = KEEP_ROLES_WITH_EQUIPMENT[nl]
        if all_class_names and (equip & all_class_names):
            return nl
        return ROLE_TO_GENERIC.get(nl, "person")
    parts = nl.split()
    if len(parts) >= 2 and parts[-1] in COCO_ALIAS_MAP:
        return COCO_ALIAS_MAP[parts[-1]]
    return nl


def _apply_flickr_alias(name: str, all_class_names: Optional[set] = None) -> str:
    if not name:
        return name
    nl = name.lower().strip()
    if nl in FLICKR_PERSON_ALIAS:
        return FLICKR_PERSON_ALIAS[nl]
    if nl in FLICKR_OBJECT_ALIAS:
        return FLICKR_OBJECT_ALIAS[nl]
    parts = nl.split()
    if len(parts) >= 2 and parts[-1] in FLICKR_OBJECT_ALIAS:
        return FLICKR_OBJECT_ALIAS[parts[-1]]
    return nl


def deduplicate_relations(raw: list) -> list:
    best: Dict[tuple, dict] = {}
    for rel in raw:
        key = (rel["source_id"], rel["target_id"])
        conf = rel.get("confidence", rel.get("edge_weight", 1.0))
        if key not in best or conf > best[key].get("confidence", best[key].get("edge_weight", 0)):
            best[key] = rel
    seen_sp: set = set()
    out = []
    for rel in best.values():
        if rel["predicate"].lower() in SPATIAL_PREDICATES:
            pair = frozenset((rel["source_id"], rel["target_id"]))
            if pair in seen_sp:
                continue
            seen_sp.add(pair)
        out.append(rel)
    return out


def _compact_list(items: List[str]) -> str:
    if not items:
        return "None"
    counts = Counter(items)
    parts = []
    for item, cnt in counts.most_common():
        if cnt == 1:
            parts.append(item)
        else:
            n = _NUM_WORDS.get(cnt, str(cnt))
            plural = item if item.endswith("s") else item + "s"
            parts.append(f"{n} {plural}")
    return ", ".join(parts)


def _compute_salience(obj: dict) -> float:
    area = obj.get("area_ratio", 0.0)
    fgs = obj.get("FGS", 0.0)
    s_ctr = obj.get("s_center", 0.0)
    label = obj.get("label", "bg")
    otype = obj.get("type", "thing")
    label_bonus = {"fg": 0.20, "objects": 0.08, "bg": 0.0}.get(label, 0.0)
    type_mult = 1.0 if otype == "thing" else 0.5
    area_mult = 0.30 if area < 0.003 else (0.70 if area < 0.015 else 1.0)
    return max(0.0, (area * 0.50 + fgs * 0.30 + s_ctr * 0.10 + label_bonus) * type_mult * area_mult)


def _is_focus_object(obj: dict) -> bool:
    if obj.get("type", "thing") != "thing" or obj.get("is_clothing_node"):
        return False
    label = obj.get("label", "bg")
    area = obj.get("area_ratio", 0.0)
    fgs = obj.get("FGS", 0.0)
    if area < 0.003:
        return False
    return label == "fg" or area >= 0.08 or (fgs >= 0.45 and label in ("fg", "objects")) or fgs >= 0.65


def _size_tag(area_ratio: float) -> str:
    if area_ratio >= 0.30:
        return " [large]"
    if area_ratio >= 0.12:
        return " [medium]"
    return ""


def _stratify_objects(objects: list, obj_dict: Dict[int, dict]) -> Tuple[list, list, list]:
    candidates = sorted([o for o in objects if _is_focus_object(o)], key=lambda x: -_compute_salience(x))
    setting_objs = [o for o in objects if not _is_focus_object(o) and not o.get("is_clothing_node")]

    if not candidates:
        things = [o for o in objects if o.get("type") == "thing" and not o.get("is_clothing_node")]
        if not things:
            return [], [], setting_objs
        best = max(things, key=_compute_salience)
        same_class = [o for o in things if o.get("class") == best.get("class", "")]
        return sorted(same_class, key=lambda x: -_compute_salience(x)), [], setting_objs

    primary = [candidates[0]]
    secondary = []
    max_sal = _compute_salience(candidates[0])
    max_area = candidates[0].get("area_ratio", 0.0)
    p1_desc = obj_dict[candidates[0]["id"]]["desc"]
    p1_cls = candidates[0].get("class", "")

    for c in candidates[1:]:
        sal = _compute_salience(c)
        area = c.get("area_ratio", 0.0)
        c_desc = obj_dict[c["id"]]["desc"]
        c_cls = c.get("class", "")
        same_alias = p1_desc == c_desc
        demote = False
        if sal < max_sal * (0.15 if same_alias else 0.35):
            demote = True
        elif c_cls == p1_cls and max_area > area * (8.0 if same_alias else 4.0):
            demote = True
        elif p1_cls == "person" and c_cls == "person":
            generic = {"person", "man", "woman", "boy", "girl", "people", "child", "guy", "lady"}
            if p1_desc not in generic and c_desc in generic:
                demote = True
        (secondary if demote else primary).append(c)

    return primary, secondary, setting_objs


def _focus_line(focus_objs: list, obj_dict: Dict[int, dict]) -> str:
    if not focus_objs:
        return "none"
    class_groups: Dict[str, list] = {}
    for o in focus_objs:
        class_groups.setdefault(o.get("class", "object"), []).append(o)

    parts = []
    for cls, group in class_groups.items():
        tag = _size_tag(max(o.get("area_ratio", 0.0) for o in group))
        n = len(group)

        def _disp(o: dict) -> str:
            od = obj_dict.get(o.get("id"), {})
            col, nm = od.get("color", ""), od.get("desc", cls)
            return f"{col} {nm}".strip() if col else nm

        displays = [_disp(o) for o in group]
        if n == 1:
            parts.append(f"{displays[0]}{tag}")
        elif len(set(displays)) == 1:
            disp = displays[0]
            plural = disp if disp.endswith("s") else disp + "s"
            parts.append(f"{_NUM_WORDS.get(n, str(n))} {plural}{tag}")
        elif n == 2:
            ranked = sorted(group, key=lambda x: -_compute_salience(x))
            parts.append(", ".join(_disp(o) for o in ranked))
        else:
            plural = cls if cls.endswith("s") else cls + "s"
            parts.append(f"{_NUM_WORDS.get(n, str(n))} {plural}{tag}")
    return ", ".join(parts)


_LEVEL_RANK = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "KB_v2": 4, "attribute": 0}
_LEVEL_LABEL = {"T1": "COCO★★★", "T2": "COCO★★", "T3": "VG★★", "T4": "VG★", "KB_v2": "spatial", "attribute": "ATTR"}


def _build_rel_buckets(
    relations: list, obj_dict: Dict[int, dict], primary_ids: set, secondary_ids: set,
    max_actions: int = 5, max_spatial: int = 4,
) -> Tuple[List[str], List[str]]:
    high_conf = [r for r in relations if r.get("confidence", r.get("edge_weight", 1.0)) >= CFG.rel_conf_threshold]
    deduped = deduplicate_relations(high_conf)

    entries = []
    for rel in deduped:
        s_id, t_id = rel.get("source_id"), rel.get("target_id")
        if s_id not in obj_dict or t_id not in obj_dict:
            continue
        pred = rel["predicate"]
        level = rel.get("triplet_level", "KB_v2")
        conf = rel.get("confidence", rel.get("edge_weight", 0.5))
        count = rel.get("triplet_count", 0)
        inv_prim = bool(primary_ids & {s_id, t_id})
        inv_sec = bool(secondary_ids & {s_id, t_id})
        cross_tier = bool((s_id in primary_ids and t_id in secondary_ids) or (t_id in primary_ids and s_id in secondary_ids))
        is_act = rel.get("rel_type") == "action" or pred.lower() in ACTION_PREDICATES
        lvl_r = _LEVEL_RANK.get(level, 4)

        if is_act:
            pri = -1 if cross_tier else (0 if inv_prim else (1 if inv_sec else 2))
        else:
            pri = 3 if cross_tier else (4 if inv_prim else (5 if inv_sec else 6))
        entries.append((pri, lvl_r, -conf, -count, rel, s_id, t_id, pred, level))

    entries.sort(key=lambda x: x[:4])
    actions: List[str] = []
    spatial: List[str] = []
    seen: set = set()
    class_pair_count: Dict[tuple, int] = {}
    action_pairs: set = set()

    for *_, rel, s_id, t_id, pred, level in entries:
        s, t = obj_dict[s_id], obj_dict[t_id]
        s_disp = rel.get("source_alias") or s["desc"]
        t_disp = rel.get("target_alias") or t["desc"]
        if s.get("color"):
            s_disp = f"{s['color']} {s_disp}"
        if t.get("color"):
            t_disp = f"{t['color']} {t_disp}"

        text = f"[{s_disp}] {pred} [{t_disp}]"
        if text in seen:
            continue
        seen.add(text)

        cp_key = (s.get("class", ""), pred.lower(), t.get("class", ""))
        if class_pair_count.get(cp_key, 0) >= 2:
            continue

        is_act = rel.get("rel_type") == "action" or pred.lower() in ACTION_PREDICATES
        pair_key = (s_id, t_id)
        if is_act:
            action_pairs.add(pair_key)
        elif pair_key in action_pairs:
            continue
        class_pair_count[cp_key] = class_pair_count.get(cp_key, 0) + 1

        tag = _LEVEL_LABEL.get(level, level)
        if level not in ("KB_v2", "attribute"):
            tag += f"|cnt={rel.get('triplet_count', 0)}"
        text_with_tag = f"{text} <{tag}>" if level in ("T1", "T2", "T3", "attribute") else text

        if is_act and len(actions) < max_actions:
            actions.append(text_with_tag)
        elif not is_act and len(spatial) < max_spatial:
            spatial.append(text_with_tag)
        if len(actions) >= max_actions and len(spatial) >= max_spatial:
            break

    return actions, spatial


def _assemble_context(primary_line: str, secondary_line: str, setting_descs: List[str], actions: List[str], spatial: List[str]) -> str:
    lines = [f"PRIMARY FOCUS: {primary_line or 'none'}"]
    if secondary_line and secondary_line != "none":
        lines.append(f"SECONDARY: {secondary_line}")
    lines.append("SETTING: " + (_compact_list(setting_descs) if setting_descs else "none"))
    if actions:
        lines.append("ACTIONS:")
        lines.extend(f"  {r}" for r in actions)
    if spatial:
        lines.append("SPATIAL:")
        lines.extend(f"  {r}" for r in spatial)
    if not actions and not spatial:
        lines.append("RELATIONS: none")
    return "\n".join(lines)


def build_context_raw(sg_data: dict) -> str:
    objects = sg_data.get("objects", []) + sg_data.get("orphan_objects", [])
    relations = sg_data.get("relations", [])
    obj_dict = {
        o["id"]: {"desc": o.get("class", "object"), "class": o.get("class", "object"), "type": o.get("type", "thing"), "label": o.get("label", "bg"), "area": o.get("area_ratio", 0.0), "color": ""}
        for o in objects
    }
    p_objs, s_objs, set_objs = _stratify_objects(objects, obj_dict)
    p_ids, s_ids = {o["id"] for o in p_objs}, {o["id"] for o in s_objs}
    setting_descs = [o.get("class", "object") for o in sorted(set_objs, key=lambda x: 0 if x.get("type") == "stuff" else 1)]
    actions, spatial = _build_rel_buckets(relations, obj_dict, p_ids, s_ids, max_actions=5, max_spatial=4)
    return _assemble_context(_focus_line(p_objs, obj_dict), _focus_line(s_objs, obj_dict), setting_descs, actions, spatial)


def build_context_triplets(sg_data: dict, dataset: str = "coco") -> str:
    objects = sg_data.get("objects", []) + sg_data.get("orphan_objects", [])
    relations = sg_data.get("relations", [])
    all_names = {o.get("class", "").lower() for o in objects} | {(o.get("alias") or "").lower() for o in objects}
    alias_fn = _apply_flickr_alias if dataset == "flickr30k" else _apply_coco_alias

    obj_dict: Dict[int, dict] = {}
    for o in objects:
        raw_cls = o.get("class", "object")
        alias = o.get("alias") or o.get("fine_label", "")
        color = (o.get("attributes") or {}).get("color", "")
        display = alias_fn(alias if alias else raw_cls, all_names)
        color_pfx = color if (color and raw_cls not in ANIMATE_CLASSES) else ""
        obj_dict[o["id"]] = {
            "desc": display, "class": raw_cls, "type": o.get("type", "thing"),
            "label": o.get("label", "bg"), "area": o.get("area_ratio", 0.0), "color": color_pfx,
        }

    p_objs, s_objs, set_objs = _stratify_objects(objects, obj_dict)
    p_ids, s_ids = {o["id"] for o in p_objs}, {o["id"] for o in s_objs}
    setting_descs = []
    for o in sorted(set_objs, key=lambda x: 0 if x.get("type") == "stuff" else 1):
        od = obj_dict[o["id"]]
        setting_descs.append(f"{od['color']} {od['desc']}".strip() if od["color"] else od["desc"])

    actions, spatial = _build_rel_buckets(relations, obj_dict, p_ids, s_ids, max_actions=CFG.max_rels_trip, max_spatial=4)
    return _assemble_context(_focus_line(p_objs, obj_dict), _focus_line(s_objs, obj_dict), setting_descs, actions, spatial)


def load_ngram_boosters(filepath: Path, dataset: str = "coco", top_n: int = 80, min_count: int = 50) -> List[str]:
    if not filepath.is_file():
        return []
    ds_key = "COCO" if dataset.lower() == "coco" else "Flickr"
    boosters: Dict[str, int] = {}
    in_section = False
    for raw_line in filepath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("---"):
            in_section = ds_key in line
            continue
        if not in_section or ":" not in line:
            continue
        c_str, ngram = line.split(":", 1)
        c_str, ngram = c_str.strip(), ngram.strip()
        if c_str.isdigit() and ngram and not _ARTIFACT_PAT.match(ngram) and int(c_str) >= min_count:
            boosters[ngram] = max(boosters.get(ngram, 0), int(c_str))
    return [ng for ng, _ in sorted(boosters.items(), key=lambda kv: -kv[1])[:top_n]]


_DEFAULT_BOOSTERS_COCO = [
    "on top of a", "in front of a", "a group of people", "sitting on top of", "a close up of",
    "on the side of", "a man in a", "standing next to a", "close up of a", "next to each other",
    "a black and white", "the side of a", "in the middle of", "sitting next to a", "a man riding a",
    "is sitting on a", "sitting at a table", "standing in front of", "a man holding a", "on a tennis court",
    "a body of water", "sitting on a bench", "on a city street", "a plate of food", "at a table with",
]
_DEFAULT_BOOSTERS_FLICKR = [
    "a group of people", "in front of a", "a man in a", "a crowd of people",
    "group of people are", "a woman in a", "a group of men", "in the middle of",
    "a man wearing a", "standing in front of", "walking down the street",
    "in a white shirt", "a man with a", "group of people in", "a man and a",
    "man and a woman", "in a blue shirt", "in a black shirt", "in a red shirt",
]


def build_prompts_for_dataset(dataset: str) -> Tuple[str, str, str, str]:
    """Return `(system_prompt, user_template, fallback_sys, fallback_tpl)` for `coco` or `flickr30k`."""
    if dataset == "coco":
        boosters = load_ngram_boosters(ROOT / "statistics" / "tmp_4gram_coco_v8.txt", "coco", top_n=80, min_count=50) or _DEFAULT_BOOSTERS_COCO
        booster_lines = ",\n  ".join(f'"{b}"' for b in boosters[:40])
        vocab_block = f"""
VOCABULARY (calibrated to COCO 2017 GT — median 10 words, avg 10.5):
• SUBJECTS — use COCO 80 class names ONLY:
  - People: "a man" / "a woman" / "a person" (prefer gender when clear)
  - Counts: "Two men" / "Three women" / "Several people" — ALWAYS count ≥ 2
  - ★ BANNED: "persons" / "individual" / "male" / "female" → use man/woman/person/people
• VERBS: Noun Phrases with -ING: riding, sitting, standing, holding, walking, playing, eating.
• 4-GRAM BOOSTERS — Use these EXACT phrases whenever applicable:
  {booster_lines}
"""
        sanity = """
SEMANTIC CHECK (apply before writing):
  A. Action + object coherent? If not → drop relation, use generic description.
  B. SETTING matches FOCUS? Use SETTING to resolve ambiguous labels.
  C. PRIMARY FOCUS is none? → describe SETTING (sky, grass, street scene).
     ★ NEVER output "blank scene" or "no visible objects" — always describe."""
        think_trip = """
OUTPUT FORMAT — EXACTLY TWO LINES, no exceptions:
Line 1 → THINK: <max 3 words — e.g. "ok" / "fix count" / "skip spatial">
Line 2 → CAPTION: <your caption, TARGET 10-12 words>

★ RULES for THINK line:
  - Max 3 words. No sentences. No periods.
  - Good: "ok" / "fix alias" / "skip noise" / "keep alias" / "fix count"
★ RULES for CAPTION line:
  - TARGET 10-12 words (COCO GT median = 10, avg = 10.5).
  - Be CONCISE: match real COCO annotator style exactly.
  - Do NOT start CAPTION on same line as THINK"""
        sys_trip = (
            "You are an expert image caption annotator for COCO 2017.\n"
            "You receive a triplet scene graph. Output EXACTLY ONE fluent English sentence.\n"
            "TARGET LENGTH: 10-12 words (GT median = 10). Be CONCISE like real COCO annotators."
            + sanity + vocab_block + think_trip + "\n\n"
            "━━━ CLOTHING RULES ━━━\n"
            "If ACTIONS contains 'wearing' [clothing] <ATTR>, MERGE into noun phrase:\n"
            "  ✓ 'A man in a red shirt holding a pan...'\n"
            "  ✗ 'A man wearing a red shirt is holding a pan...'\n\n"
            "━━━ ALIAS & VOCAB (CRITICAL for CIDEr) ━━━\n"
            "BANNED: feline, commode, washbasin, persons, individuals, transit bus.\n"
            "DOG BREEDS → dog. BIRD SPECIES → bird. FOOD TYPES: use COCO class names.\n\n"
            "CAPTION RULES:\n"
            "1. Start: A/An/The or count (Two/Three/Several).\n"
            "2. Make PRIMARY FOCUS the subject. Use COCO★★★ predicates.\n"
            "3. TARGET 10-12 words. Max 13. Use -ING verbs.\n"
            "4. Prefer 4-gram phrases from boosters when they fit naturally.\n"
            "5. BANNED: There is/are, The image, This image, persons."
        )
        few_trip = """Triplet SG — THINK (≤3w) then CAPTION (10-12w, concise COCO GT style) on separate lines:

• PRIMARY FOCUS: person [medium], laptop [medium]
  SECONDARY: none
  SETTING: table, window
  ACTIONS:
    [person] using [laptop]  <COCO★★★|cnt=634>
    [person] sitting at [table]  <COCO★★★|cnt=2000>
  THINK: ok
  CAPTION: A person sitting at a table using a laptop .

• PRIMARY FOCUS: Two zebras [medium]
  SECONDARY: none
  SETTING: fence, grass
  ACTIONS:
    [zebra] standing next to [zebra]  <COCO★★★|cnt=194>
    [zebra] in [grass]  <COCO★★★|cnt=633>
  THINK: count=2
  CAPTION: Two zebras standing next to a fence in the grass .

• PRIMARY FOCUS: man [medium]
  SECONDARY: none
  SETTING: kitchen, counter
  ACTIONS:
    [man] wearing [white shirt]  <ATTR>
    [man] holding [pan]  <COCO★★★|cnt=400>
  THINK: merge clothing
  CAPTION: A man in a white shirt holding a pan in a kitchen .

• PRIMARY FOCUS: cat [large]
  SECONDARY: none
  SETTING: couch, blanket
  ACTIONS:
    [cat] sitting on [couch]  <COCO★★★|cnt=903>
  THINK: ok
  CAPTION: A cat sitting on top of a couch next to a blanket .

Now generate for this triplet scene graph:
{sg_text}

THINK:"""
        sys_raw = (
            "You are an expert image caption annotator for the COCO benchmark.\n"
            "Output EXACTLY ONE fluent English noun phrase or short sentence of 10-15 words."
            + sanity + vocab_block + "\n\nOUTPUT FORMAT — EXACTLY TWO LINES:\nLine 1 → THINK: <max 3 words>\nLine 2 → CAPTION: <your caption>\n"
        )
        few_raw = "Now generate for this scene graph:\n{sg_text}\n\nTHINK:"
        return sys_trip, few_trip, sys_raw, few_raw

    boosters = load_ngram_boosters(ROOT / "statistics" / "tmp_4gram_flickr_v3.txt", "flickr", top_n=25, min_count=40) or _DEFAULT_BOOSTERS_FLICKR
    booster_lines = ",\n  ".join(f'"{b}"' for b in boosters[:16])
    vocab_flickr = f"""
VOCABULARY (calibrated to Flickr30k GT — median 12 words):
• SUBJECTS — use GT-aligned nouns ONLY:
  - Adults: "a man" / "a woman" (★ NEVER "elderly man" / "older man" / "young man")
  - Groups: "a group of people" / "a crowd of people" / "two men" / "a group of women"
  - Children: "a boy" / "a girl" / "a child" / "a little girl" / "a little boy"
  - ★ BANNED: "persons" → "people/men/women", "elderly" prefix, "older" prefix
• CLOTHING & COLORS (CRITICAL):
  - You MUST explicitly describe the color of ANY object that has a color attribute.
  - "in a [color] shirt/jacket/dress" (GT dominant pattern) for people.
  - For objects: "a [color] [object]", e.g. "a blue car", "a red ball".
• VERBS: bare -ING form: standing, sitting, wearing, playing, holding, walking, running, riding.
• 4-GRAM BOOSTERS — Use these EXACT phrases whenever applicable:
  {booster_lines}
"""
    sys_flickr = f"""You are annotating Flickr30k, a photo dataset.
You receive a triplet scene graph. Output EXACTLY ONE fluent English sentence.
TARGET LENGTH: 10-15 words.
SEMANTIC CHECK (apply before writing):
  A. Action + object coherent? If not → drop, use generic description.
  B. SETTING matches FOCUS? Use SETTING to resolve ambiguous labels.
  C. PRIMARY FOCUS is none? → describe most likely scene from SETTING objects.

{vocab_flickr}

OUTPUT FORMAT — EXACTLY TWO LINES, no exceptions:
Line 1 → THINK: <max 3 words>
Line 2 → CAPTION: <your caption, TARGET 10-15 words>

━━━ COLOR RULES (CRITICAL) ━━━
★ You MUST describe the colors of objects and clothing if provided in the SG!
Merge 'wearing' [clothing] <ATTR> into noun phrase: 'A man in a red shirt...'
For other items, mention the color directly: 'holding a green umbrella...'.

━━━ PERSON RULES (CRITICAL for BLEU/CIDEr) ━━━
★ ALWAYS use 'man'/'woman' — NEVER 'elderly man', 'older man', 'young man'.
★ Groups: 'a group of people', 'a crowd of people' (NOT counts like '13 people').
★ Two people: 'A man and a woman' / 'Two men' (use number words for 2-5).

FLICKR30K RULES:
1. Start: A/An/The or count (Two/Three/Several/A group of).
2. Use GT patterns: 'A man in a [color] shirt [action]...' or 'A group of people [action]...'.
3. Bare -ING verbs: 'A man riding...' not 'A man rides...'.
4. OMIT EMPTY SETTINGS: If SETTING is 'none', skip it. NEVER say 'unspecified location'.
5. BANNED: There is/are, The image, This image, persons, elderly, older, rider.
"""
    few_flickr = """Triplet SG — THINK (≤3w) then CAPTION (10-15w, concise Flickr GT style) on separate lines:

• PRIMARY FOCUS: man [medium]
  SECONDARY: none
  SETTING: sidewalk, building
  ACTIONS:
    [man] wearing [white shirt]  <ATTR>
    [man] walking on [sidewalk]  <COCO★★★|cnt=500>
  THINK: merge clothing
  CAPTION: A man in a white shirt walking down the street in front of a building .

• PRIMARY FOCUS: dog [medium]
  SECONDARY: frisbee [small]
  SETTING: park, grass
  ACTIONS:
    [dog] catching [frisbee]  <COCO★★★|cnt=500>
    [frisbee] is [purple]  <ATTR>
  THINK: mention object color
  CAPTION: A dog is jumping to catch a purple frisbee in the park .

• PRIMARY FOCUS: dog [medium]
  SECONDARY: building [large]
  SETTING: street, building
  ACTIONS:
    [dog] is [white]  <ATTR>
    [dog] looking out of [window]
    [building] is [turquoise]  <ATTR>
  THINK: two colors
  CAPTION: A fluffy white dog hangs out of the window of a turquoise building .

• PRIMARY FOCUS: girl [medium]
  SECONDARY: dog [small]
  SETTING: grass, fence
  ACTIONS:
    [girl] wearing [red jacket]  <ATTR>
    [girl] playing with [dog]  <COCO★★★|cnt=200>
  THINK: merge clothing
  CAPTION: A girl in a red jacket playing with a dog in the grass .

Now generate for this triplet scene graph:
{sg_text}

THINK:"""
    return sys_flickr, few_flickr, sys_flickr, few_flickr


def _postprocess(raw: str) -> str:
    raw = raw.strip()
    caption = ""
    m = re.search(r"(?:^|\n|\.|\s)CAPTION:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if m:
        caption = m.group(1).strip()
    if not caption:
        for line in raw.split("\n"):
            line = line.strip()
            if line.lower().startswith("caption:"):
                caption = line[len("caption:") :].strip()
                break
    if not caption:
        lines = [l.strip() for l in raw.split("\n") if l.strip() and not l.strip().lower().startswith("think:")]
        caption = lines[0] if lines else ""

    caption = caption.lstrip("→•-").strip()
    for pfx in (
        "Caption:", "Answer:", "Output:", "Here is", "Here's", "Corrected:",
        "A photo of ", "The image shows ", "This image shows ", "One can see ",
        "In this image,", "The photo shows ", "Based on", "SG:", "Scene:", "Example:", "Generated:",
    ):
        if caption.lower().startswith(pfx.lower()):
            caption = caption[len(pfx) :].strip()

    caption = caption.strip("\"'").strip()
    caption = re.sub(r"^\d+\.\s*", "", caption)
    caption = re.sub(r"\bpersons\b", "people", caption)
    caption = re.sub(r"\bindividuals?\b", "people", caption)

    if caption and not caption.endswith("."):
        caption += " ."
    elif caption.endswith(".") and not caption.endswith(" ."):
        caption = caption[:-1] + " ."
    if caption:
        caption = caption[0].upper() + caption[1:]
    return caption


_LEAK_KEYWORDS = re.compile(
    r"\b(COCO★|reliable|implausible|context suggest|The relation|The presence|cannot be on|"
    r"simultaneously|suggests a scene|but the|but both|but being|but sitting|but standing|"
    r"replace with|is unreliable|is incoherent)\b"
)


def _is_bad_caption(caption: str) -> bool:
    if not caption or len(caption.split()) < 4:
        return True
    low = caption.lower()
    if any(low.startswith(p) for p in ("there is", "there are", "the image", "this image", "this photo", "one can see", "the scene", "we can see", "it appears", "it seems", "i cannot", "i can't", "here is")):
        return True
    return bool(_LEAK_KEYWORDS.search(caption))


# ==============================================================================
# BACKEND 1: OPENAI ASYNC GENERATOR (DEFAULT)
# ==============================================================================
class OpenAIKeyPool:
    def __init__(self, keys: List[str], timeout: float):
        from openai import AsyncOpenAI
        self._keys = keys
        self._clients: Dict[int, AsyncOpenAI] = {}
        self._idx = 0
        self._lock = asyncio.Lock()
        self._backoff = [0.0] * len(keys)
        self._timeout = timeout
        self._cls = AsyncOpenAI

    async def next_client(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                n = len(self._keys)
                for _ in range(n):
                    idx = self._idx % n
                    self._idx += 1
                    if self._backoff[idx] <= now:
                        if idx not in self._clients:
                            self._clients[idx] = self._cls(api_key=self._keys[idx], timeout=self._timeout, max_retries=0)
                        return self._clients[idx], idx
                earliest = min(range(n), key=lambda i: self._backoff[i])
                wait = max(0.0, self._backoff[earliest] - now)
            if wait > 0:
                await asyncio.sleep(wait)

    def mark_rate_limited(self, idx: int, retry_after: float = 60.0) -> None:
        self._backoff[idx] = time.monotonic() + retry_after


async def run_openai_backend(
    jobs: List[dict],
    dataset: str,
    output_dir: Path,
) -> Dict[str, str]:
    from openai import RateLimitError
    from tqdm.asyncio import tqdm as async_tqdm

    raw_keys = os.getenv("OPENAI_API_KEYS", "") or os.getenv("OPENAI_API_KEY", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("Missing OPENAI_API_KEY (or OPENAI_API_KEYS) in environment / .env file.")

    pool = OpenAIKeyPool(keys, timeout=CFG.request_timeout)
    max_tokens = CFG.max_tokens_coco if dataset == "coco" else CFG.max_tokens_flickr
    pred_path = output_dir / "predictions.json"
    results: Dict[str, str] = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.is_file() else {}

    pending = [j for j in jobs if j["image_id"] not in results]
    if not pending:
        return results

    sem = asyncio.Semaphore(CFG.max_concurrent)
    lock = asyncio.Lock()

    async def _call_messages(messages: List[dict]) -> str:
        for _ in range(len(keys)):
            client, idx = await pool.next_client()
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=CFG.openai_model,
                        messages=messages,
                        temperature=CFG.temperature,
                        max_tokens=max_tokens,
                        n=1,
                    ),
                    timeout=CFG.request_timeout,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if isinstance(exc, RateLimitError) or "429" in str(exc):
                    pool.mark_rate_limited(idx, 60.0)
                    continue
                return ""
        return ""

    async def _worker(job: dict) -> None:
        async with sem:
            caption = ""
            for attempt in range(1, CFG.max_retries + 1):
                raw = await _call_messages(job["messages"])
                cand = _postprocess(raw)
                if cand and not _is_bad_caption(cand):
                    caption = cand
                    break
                if attempt < CFG.max_retries:
                    await asyncio.sleep(CFG.retry_base_delay * (2 ** (attempt - 1)))

            if not caption and job.get("fallback_messages"):
                raw = await _call_messages(job["fallback_messages"])
                cand = _postprocess(raw)
                if cand and not _is_bad_caption(cand):
                    caption = cand

            async with lock:
                results[job["image_id"]] = caption
                if len(results) % 25 == 0:
                    pred_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = [_worker(j) for j in pending]
    async for fut in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[OpenAI {CFG.openai_model}]"):
        await fut

    pred_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


# ==============================================================================
# BACKEND 2: LOCAL OPEN-WEIGHT QWEN GENERATOR (--backend qwen)
# ==============================================================================
def run_qwen_backend(
    jobs: List[dict],
    dataset: str,
    output_dir: Path,
) -> Dict[str, str]:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pred_path = output_dir / "predictions.json"
    results: Dict[str, str] = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.is_file() else {}
    pending = [j for j in jobs if j["image_id"] not in results]
    if not pending:
        return results

    log.info("Loading Qwen model %s (4-bit=%s)...", CFG.qwen_model, CFG.load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(CFG.qwen_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"device_map": "auto"}
    if CFG.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(CFG.qwen_model, **model_kwargs).eval()
    max_new_tokens = CFG.max_tokens_coco if dataset == "coco" else CFG.max_tokens_flickr

    def _call_qwen(msg_batch: List[List[dict]], temp: float = 0.0) -> List[str]:
        rendered = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msg_batch]
        inputs = tokenizer(rendered, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
        width = inputs.input_ids.shape[-1]
        gen_opts = {"do_sample": temp > 0.0, "temperature": temp if temp > 0.0 else 1.0, "top_p": 1.0, "top_k": 0}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                **gen_opts,
            )
        return tokenizer.batch_decode(out[:, width:], skip_special_tokens=True)

    pending.sort(key=lambda j: len(j["messages"][1]["content"]))
    batch_size = CFG.qwen_batch_size
    offset = 0
    pbar = tqdm(total=len(pending), desc=f"[Qwen {CFG.qwen_model}]")

    while offset < len(pending):
        batch = pending[offset : offset + batch_size]
        try:
            raw_outs = _call_qwen([b["messages"] for b in batch], temp=CFG.temperature)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            continue

        caps = [_postprocess(t) for t in raw_outs]
        for retry in (1, 2, 3):
            bad_idxs = [i for i, c in enumerate(caps) if _is_bad_caption(c)]
            if retry == 3:
                bad_idxs = [i for i in bad_idxs if batch[i].get("fallback_messages") is not None]
            if not bad_idxs:
                break
            temp_r = 0.0 if dataset == "coco" else 0.3 + retry * 0.2
            msg_key = "fallback_messages" if retry == 3 else "messages"
            retry_outs = _call_qwen([batch[i][msg_key] for i in bad_idxs], temp=temp_r)
            for idx_local, text_r in zip(bad_idxs, retry_outs):
                caps[idx_local] = _postprocess(text_r)

        for job, cap in zip(batch, caps):
            results[job["image_id"]] = cap if not _is_bad_caption(cap) else ""
        pred_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        offset += len(batch)
        pbar.update(len(batch))

    pbar.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SPARKG-Cap Phase 2: Caption Generation on Official Karpathy Splits.")
    parser.add_argument("--backend", choices=["openai", "qwen"], default="openai", help="Primary 'openai' backend or local open-weight 'qwen' backend")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--qwen-model", type=str, default=None, help="Hugging Face model ID or local path for Qwen backend (default: Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable NF4 4-bit quantization for Qwen backend")
    parser.add_argument("--qwen-batch-size", type=int, default=16)
    parser.add_argument("--dataset", choices=["coco", "flickr30k"], default="coco")
    parser.add_argument("--karpathy-json", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--split", choices=["val", "test", "train"], default="test")
    parser.add_argument("--sg-root", type=Path, default=Path("outputs/coco_karpathy_test"))
    parser.add_argument("--sg-type", choices=["sg_with_triplets", "sg"], default="sg_with_triplets")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coco_karpathy_test/captions_openai"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print serialized prompts without calling OpenAI or loading Qwen")
    args = parser.parse_args()

    CFG.backend = args.backend
    if args.model:
        CFG.openai_model = args.model
    if args.qwen_model:
        CFG.qwen_model = args.qwen_model
    CFG.load_in_4bit = args.load_in_4bit
    CFG.qwen_batch_size = args.qwen_batch_size
    CFG.temperature = args.temperature
    CFG.max_concurrent = args.max_concurrent

    items = load_karpathy_manifest(args.karpathy_json, dataset=args.dataset, split=args.split)
    if args.limit:
        items = items[: args.limit]

    sg_dir = args.sg_root / args.sg_type
    sys_prompt, user_tpl, fb_sys, fb_tpl = build_prompts_for_dataset(args.dataset)

    jobs: List[dict] = []
    for it in items:
        sg_path = sg_dir / f"{it.image_id}_sg.json"
        if not sg_path.is_file():
            continue
        sg_data = json.loads(sg_path.read_text(encoding="utf-8"))
        ctx = build_context_triplets(sg_data, dataset=args.dataset) if args.sg_type == "sg_with_triplets" else build_context_raw(sg_data)
        if not ctx:
            continue
        fb_msgs = None
        if args.dataset == "coco" and args.sg_type == "sg_with_triplets":
            fb_ctx = build_context_raw(sg_data)
            if fb_ctx:
                fb_msgs = [{"role": "system", "content": fb_sys}, {"role": "user", "content": fb_tpl.format(sg_text=fb_ctx)}]
        jobs.append({
            "image_id": it.image_id,
            "numeric_id": it.numeric_id,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_tpl.format(sg_text=ctx)}],
            "fallback_messages": fb_msgs,
            "context": ctx,
        })

    if args.dry_run:
        print("=" * 72)
        print(f"DRY RUN | backend={CFG.backend} | dataset={args.dataset} | split={args.split} | prompts={len(jobs)}")
        print("=" * 72)
        for j in jobs[: min(3, len(jobs))]:
            print(f"\n--- Image ID: {j['image_id']} ---")
            print(j["context"])
        return

    if not jobs:
        raise FileNotFoundError(f"No scene graph files (*_sg.json) matched Karpathy {args.split} IDs in {sg_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if CFG.backend == "openai":
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        results = asyncio.run(run_openai_backend(jobs, args.dataset, args.output_dir))
    else:
        results = run_qwen_backend(jobs, args.dataset, args.output_dir)

    # Export COCO-format submission JSON and CSV summary
    sub_list = [{"image_id": j["numeric_id"] if j["numeric_id"] is not None else j["image_id"], "caption": results.get(j["image_id"], "")} for j in jobs]
    (args.output_dir / "predictions_coco_format.json").write_text(json.dumps(sub_list, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "caption"])
        w.writeheader()
        for j in jobs:
            w.writerow({"image_id": j["image_id"], "caption": results.get(j["image_id"], "")})

    log.info("Saved %d generated captions -> %s", len(results), args.output_dir)


if __name__ == "__main__":
    main()

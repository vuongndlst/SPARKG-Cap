"""
Controlled Component & Leave-One-Out Ablation Runner on Karpathy Test Splits.

Evaluates additive structured-hint variants `(a)-(g)` and leave-one-out graph
ablations (`no_role`, `thing_stuff`, `area_centrality`, `minus_fg`, `minus_bg`,
`minus_aux`, `no_geometry`, `objects_only`, `spatial_only`, `grounded_only`,
`minus_aliases`, `minus_attributes`, `flat_context`) using either `--backend openai`
(default `gpt-4o-mini`) or `--backend qwen` (`Qwen/Qwen2.5-14B-Instruct`).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import load_karpathy_manifest
from src.phase2_caption_generation.generate_captions import (
    ANIMATE_CLASSES,
    Config,
    _apply_coco_alias,
    _apply_flickr_alias,
    build_context_triplets,
)


ABLATION_ARMS = (
    "full",
    "no_role",
    "thing_stuff",
    "area_centrality",
    "minus_fg",
    "minus_bg",
    "minus_aux",
    "objects_only",
    "spatial_only",
    "grounded_only",
    "minus_aliases",
    "minus_attributes",
    "flat_context",
)


def transform_graph_for_arm(sg_data: dict, arm: str) -> dict:
    """Apply deterministic leave-one-out or role-replacement ablation to an enriched scene graph."""
    g = copy.deepcopy(sg_data)
    objs = g.get("objects", [])

    if arm == "full":
        return g
    if arm == "no_role":
        for o in objs:
            if not o.get("is_clothing_node"):
                o["label"] = "objects"
                o["FGS"] = 0.0
                o["BGS"] = 0.0
        return g
    if arm == "thing_stuff":
        for o in objs:
            if not o.get("is_clothing_node"):
                o["label"] = "fg" if o.get("type") == "thing" else "bg"
                o["FGS"] = float(o.get("score") or o.get("det_conf") or 0.0)
                o["BGS"] = 0.0
        return g
    if arm == "area_centrality":
        for o in objs:
            if o.get("is_clothing_node"):
                continue
            area = float(o.get("area_ratio", o.get("s_size", 0.0)) or 0.0)
            s_ctr = float(o.get("s_center", 0.5) or 0.5)
            sal = 0.5 * area + 0.5 * s_ctr
            o["label"] = ("fg" if (sal >= 0.35 or area >= 0.08) else "objects") if o.get("type") == "thing" else "bg"
            o["FGS"] = round(sal, 4)
            o["BGS"] = round(max(0.0, 1.0 - sal), 4)
        return g
    if arm in {"minus_fg", "minus_bg", "minus_aux"}:
        drop_lbl = {"minus_fg": "fg", "minus_bg": "bg", "minus_aux": "objects"}[arm]
        kept = [o for o in objs if o.get("label") != drop_lbl]
        kept_ids = {o["id"] for o in kept}
        g["objects"] = kept
        g["relations"] = [r for r in g.get("relations", []) if r.get("source_id") in kept_ids and r.get("target_id") in kept_ids]
        return g
    if arm == "objects_only":
        g["objects"] = [o for o in objs if not o.get("is_clothing_node")]
        g["relations"] = []
        return g
    if arm == "spatial_only":
        g["objects"] = [o for o in objs if not o.get("is_clothing_node")]
        spatial_rels = []
        for r in g.get("relations", []):
            sgeo = str(r.get("spatial_geo_rel") or "").replace("_", " ").strip()
            if sgeo and sgeo != "unknown":
                rc = dict(r)
                rc["predicate"] = sgeo
                rc["rel_type"] = "spatial"
                rc["triplet_level"] = "KB_v2"
                spatial_rels.append(rc)
        g["relations"] = spatial_rels
        return g
    if arm == "grounded_only":
        g["relations"] = [r for r in g.get("relations", []) if r.get("triplet_level") in {"KB_v2", "attribute"}]
        return g
    if arm == "minus_attributes":
        g["objects"] = [o for o in objs if not o.get("is_clothing_node")]
        for o in g["objects"]:
            o["attributes"] = {}
        g["relations"] = [r for r in g.get("relations", []) if r.get("triplet_level") != "attribute"]
        return g
    if arm == "minus_aliases":
        for o in objs:
            if not o.get("is_clothing_node"):
                o.pop("alias", None)
                o.pop("fine_label", None)
        for r in g.get("relations", []):
            r.pop("source_alias", None)
            r.pop("target_alias", None)
        return g
    return g


def build_flat_context(sg_data: dict, dataset: str = "coco", cfg: Config = Config()) -> str:
    """Unstratified flat serialization (`OBJECTS` + `RELATIONS`) preserving identical nodes and edges."""
    objects = sg_data.get("objects", []) + sg_data.get("orphan_objects", [])
    relations = sg_data.get("relations", [])
    all_names = {o.get("class", "").lower() for o in objects} | {(o.get("alias") or "").lower() for o in objects}
    alias_fn = _apply_flickr_alias if dataset == "flickr30k" else _apply_coco_alias

    obj_dict: Dict[int, dict] = {}
    descs: List[str] = []
    for o in objects:
        raw_cls = o.get("class", "object")
        alias = o.get("alias") or o.get("fine_label", "")
        color = (o.get("attributes") or {}).get("color", "")
        disp = alias_fn(alias if alias else raw_cls, all_names)
        cpfx = color if (color and raw_cls not in ANIMATE_CLASSES) else ""
        full = f"{cpfx} {disp}".strip() if cpfx else disp
        obj_dict[o["id"]] = {"desc": disp, "color": cpfx}
        if not o.get("is_clothing_node"):
            descs.append(full)

    rel_lines: List[str] = []
    seen = set()
    for r in relations:
        if r.get("confidence", r.get("edge_weight", 1.0)) < cfg.rel_conf_threshold:
            continue
        sid, tid = r.get("source_id"), r.get("target_id")
        if sid not in obj_dict or tid not in obj_dict:
            continue
        s, t = obj_dict[sid], obj_dict[tid]
        sd = r.get("source_alias") or s["desc"]
        td = r.get("target_alias") or t["desc"]
        if s["color"]:
            sd = f"{s['color']} {sd}"
        if t["color"]:
            td = f"{t['color']} {td}"
        line = f"[{sd}] {r['predicate']} [{td}]"
        if line not in seen:
            seen.add(line)
            rel_lines.append(f"  {line}")
        if len(rel_lines) >= cfg.max_rels_trip + 4:
            break

    out = ["OBJECTS: " + (", ".join(descs) if descs else "none")]
    if rel_lines:
        out.append("RELATIONS:")
        out.extend(rel_lines)
    else:
        out.append("RELATIONS: none")
    return "\n".join(out)


def render_arm_context(sg_data: dict, arm: str, dataset: str = "coco") -> str:
    transformed = transform_graph_for_arm(sg_data, arm)
    if arm == "flat_context":
        return build_flat_context(transformed, dataset=dataset)
    return build_context_triplets(transformed, dataset=dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or export SPARKG-Cap ablation contexts on Karpathy splits.")
    parser.add_argument("--dataset", choices=["coco", "flickr30k"], default="coco")
    parser.add_argument("--karpathy-json", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--sg-dir", type=Path, default=Path("outputs/coco_karpathy_test/sg_with_triplets"))
    parser.add_argument("--arms", type=str, default=",".join(ABLATION_ARMS))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    items = load_karpathy_manifest(args.karpathy_json, dataset=args.dataset, split=args.split)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    shown = 0
    for it in items:
        sg_path = args.sg_dir / f"{it.image_id}_sg.json"
        if not sg_path.is_file():
            continue
        sg = json.loads(sg_path.read_text(encoding="utf-8"))
        print("=" * 72)
        print(f"Image ID: {it.image_id} ({args.dataset} Karpathy {args.split})")
        for arm in arms:
            print(f"\n--- Arm: {arm} ---")
            print(render_arm_context(sg, arm, dataset=args.dataset))
        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()

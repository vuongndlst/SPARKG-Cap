"""
Phase 3 (Optional): Upload Enriched Image Scene Graphs to Neo4j Property Graph.

Indexes image-specific scene graphs (`sg_with_triplets`) under the SPARKG-Cap
hierarchical ontology (`Root -> Domain -> Category -> Class`, `Role`, `Image`,
`Object`, `Attribute`, and typed relational edges).

Credentials are read strictly from environment variables (`NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD`, `NEO4J_DATABASE`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _finite(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _label_to_role(label: str) -> str:
    lbl = (label or "").strip().lower()
    if lbl == "fg":
        return "foreground"
    if lbl == "bg":
        return "background"
    return "objects"


def _safe_reltype(pred: str) -> str:
    safe = pred.upper().replace(" ", "_").replace("-", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_") or "RELATED_TO"
    return ("R_" + safe) if safe[0].isdigit() else safe


def load_category_map(path: Path) -> Dict[str, Tuple[str, str]]:
    if not path.is_file():
        return {}
    mapping: Dict[str, Tuple[str, str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        domain, category, classes_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
        for cls in classes_str.split(","):
            c = _norm(cls)
            if c:
                mapping[c] = (domain, category)
    return mapping


def parse_sg_json(data: dict, cat_map: Dict[str, Tuple[str, str]]) -> dict:
    img_name = data.get("image_name", "unknown")
    split = data.get("split", "karpathy_test")
    img_hw = data.get("img_hw", [None, None])
    img_h = _finite(img_hw[0]) if img_hw and len(img_hw) > 0 else None
    img_w = _finite(img_hw[1]) if img_hw and len(img_hw) > 1 else None

    img_row = {"name": img_name, "split": split, "img_h": img_h, "img_w": img_w}
    rels_raw = data.get("relations", [])
    connected_ids: Set = {r["source_id"] for r in rels_raw} | {r["target_id"] for r in rels_raw}

    all_objs = list(data.get("objects", [])) + list(data.get("orphan_objects", []))
    obj_rows: List[dict] = []
    attr_rows: List[dict] = []

    for obj in all_objs:
        oid = obj.get("id")
        if oid is None:
            continue
        obj_name = f"{img_name}_{oid}"
        cls = _norm(obj.get("class", "unknown"))
        domain = "thing" if obj.get("type", "thing") == "thing" else "stuff"
        label = obj.get("label", "objects")
        role = _label_to_role(label)
        _, category = cat_map.get(cls, (domain, "other"))

        box = obj.get("box") or obj.get("bbox") or []
        bx1 = _finite(box[0]) if len(box) > 0 else None
        by1 = _finite(box[1]) if len(box) > 1 else None
        bx2 = _finite(box[2]) if len(box) > 2 else None
        by2 = _finite(box[3]) if len(box) > 3 else None

        alias = obj.get("alias", "") or ""
        alias_conf = _finite(obj.get("alias_conf"), 0.0) or 0.0

        obj_rows.append({
            "img_name": img_name,
            "obj_name": obj_name,
            "class": cls,
            "domain": domain,
            "category": category,
            "label": label,
            "role": role,
            "FGS": _finite(obj.get("FGS")),
            "BGS": _finite(obj.get("BGS")),
            "area_ratio": _finite(obj.get("area_ratio")),
            "bbox_x1": bx1,
            "bbox_y1": by1,
            "bbox_x2": bx2,
            "bbox_y2": by2,
            "orphan": oid not in connected_ids,
            "alias": alias,
            "alias_conf": alias_conf,
        })

        for attr_type, attr_value in (obj.get("attributes") or {}).items():
            if attr_value:
                attr_rows.append({
                    "obj_name": obj_name,
                    "attr_name": f"{attr_type}::{attr_value}",
                    "attr_type": str(attr_type),
                    "attr_value": str(attr_value),
                })
        if alias:
            attr_rows.append({
                "obj_name": obj_name,
                "attr_name": f"alias::{alias}",
                "attr_type": "alias",
                "attr_value": alias,
            })

    rel_rows: List[dict] = []
    for rel in rels_raw:
        sid, tid = rel.get("source_id"), rel.get("target_id")
        if sid is None or tid is None:
            continue
        pred = str(rel.get("predicate", "related_to"))
        rel_rows.append({
            "src_name": f"{img_name}_{sid}",
            "tgt_name": f"{img_name}_{tid}",
            "reltype": _safe_reltype(pred),
            "predicate": pred,
            "spatial_geo_rel": str(rel.get("spatial_geo_rel", "")),
            "edge_weight": _finite(rel.get("edge_weight"), 1.0),
            "triplet_level": str(rel.get("triplet_level", "")),
            "triplet_count": int(rel.get("triplet_count", 0) or 0),
        })

    return {"img_row": img_row, "obj_rows": obj_rows, "rel_rows": rel_rows, "attr_rows": attr_rows}


def upload_directory(sg_dir: Path, category_map_path: Path, dry_run: bool = False, batch_size: int = 200) -> None:
    cat_map = load_category_map(category_map_path)
    sg_files = sorted(sg_dir.glob("*_sg.json"))
    if not sg_files:
        raise FileNotFoundError(f"No *_sg.json files found in {sg_dir}")

    parsed = [parse_sg_json(json.loads(p.read_text(encoding="utf-8")), cat_map) for p in sg_files]
    log.info("Parsed %d scene graphs (%d objects, %d relations).",
             len(parsed), sum(len(x["obj_rows"]) for x in parsed), sum(len(x["rel_rows"]) for x in parsed))

    if dry_run:
        log.info("[Dry-Run] Skipping Neo4j database connection.")
        return

    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD environment variable must be set before uploading to Neo4j.")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for label in ("Root", "Domain", "Category", "Role", "Class", "Image", "Object", "Attribute"):
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
            session.run("""
                MERGE (root:Root {name:'root'})
                MERGE (dt:Domain {name:'thing'})
                MERGE (ds:Domain {name:'stuff'})
                MERGE (rf:Role {name:'foreground'})
                MERGE (rb:Role {name:'background'})
                MERGE (ro:Role {name:'objects'})
                MERGE (root)-[:HAS_DOMAIN]->(dt)
                MERGE (root)-[:HAS_DOMAIN]->(ds)
                MERGE (root)-[:HAS_ROLE]->(rf)
                MERGE (root)-[:HAS_ROLE]->(rb)
                MERGE (root)-[:HAS_ROLE]->(ro)
            """)

            for start in range(0, len(parsed), batch_size):
                chunk = parsed[start : start + batch_size]
                imgs = [x["img_row"] for x in chunk]
                objs = [r for x in chunk for r in x["obj_rows"]]
                attrs = [r for x in chunk for r in x["attr_rows"]]
                rels_by_type: Dict[str, List[dict]] = defaultdict(list)
                for x in chunk:
                    for r in x["rel_rows"]:
                        rels_by_type[r["reltype"]].append(r)

                session.run(
                    "UNWIND $rows AS row MERGE (i:Image {name: row.name}) SET i.split=row.split, i.img_h=row.img_h, i.img_w=row.img_w",
                    rows=imgs,
                )
                if objs:
                    session.run("""
                        UNWIND $rows AS row
                        MERGE (o:Object {name: row.obj_name})
                        SET o.class=row.class, o.domain=row.domain, o.category=row.category,
                            o.label=row.label, o.FGS=row.FGS, o.BGS=row.BGS,
                            o.area_ratio=row.area_ratio, o.alias=row.alias, o.alias_conf=row.alias_conf
                        WITH o, row
                        MATCH (i:Image {name: row.img_name}) MERGE (i)-[:HAS_OBJECT]->(o)
                        WITH o, row
                        MERGE (c:Class {name: row.class}) MERGE (o)-[:INSTANCE_OF]->(c)
                        WITH o, row
                        MATCH (r:Role {name: row.role}) MERGE (o)-[:ASSIGNED_TO]->(r)
                    """, rows=objs)
                if attrs:
                    session.run("""
                        UNWIND $rows AS row
                        MERGE (a:Attribute {name: row.attr_name})
                        SET a.type=row.attr_type, a.value=row.attr_value
                        WITH a, row
                        MATCH (o:Object {name: row.obj_name}) MERGE (o)-[:HAS_ATTRIBUTE]->(a)
                    """, rows=attrs)
                for rtype, rlist in rels_by_type.items():
                    session.run(f"""
                        UNWIND $rows AS row
                        MATCH (s:Object {{name: row.src_name}}), (t:Object {{name: row.tgt_name}})
                        MERGE (s)-[rel:{rtype} {{predicate: row.predicate}}]->(t)
                        SET rel.spatial_geo_rel=row.spatial_geo_rel,
                            rel.edge_weight=row.edge_weight,
                            rel.triplet_level=row.triplet_level,
                            rel.triplet_count=row.triplet_count
                    """, rows=rlist)
        log.info("Uploaded %d scene graphs to Neo4j (%s).", len(parsed), uri)
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload SPARKG-Cap scene graphs to Neo4j.")
    parser.add_argument("--sg-dir", type=Path, default=Path("outputs/coco_karpathy_test/sg_with_triplets"))
    parser.add_argument("--category-map", type=Path, default=ROOT / "statistics" / "coco_category_map.txt")
    parser.add_argument("--dry-run", action="store_true", help="Parse scene graphs without connecting to Neo4j")
    args = parser.parse_args()
    upload_directory(args.sg_dir, args.category_map, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

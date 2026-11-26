"""
Phase 0 (Step 4): Classify Surface Predicates into Action vs. 2D Spatial Geometry.

Provides both a deterministic rule-based classifier (`classify_predicate_rule`) backed by
`knowledge_graph/geo_rel_alias.txt` and an optional batch OpenAI classifier for unseen predicates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]

_ACTION_KEYWORDS = frozenset({
    "holding", "wearing", "riding", "sitting", "standing", "walking", "running", "playing",
    "eating", "drinking", "looking", "watching", "carrying", "driving", "flying", "jumping",
    "catching", "throwing", "hitting", "kicking", "using", "operating", "parked", "lying",
    "laying", "hanging", "swimming", "surfing", "skiing", "skateboarding", "snowboarding",
    "feeding", "petting", "pulling", "pushing", "talking", "smiling", "leaning", "cutting",
})


def load_geo_alias_map(path: Path) -> Dict[str, str]:
    """Load `surface_predicate -> canonical_spatial_geo_rel` from `geo_rel_alias.txt`."""
    mapping: Dict[str, str] = {}
    if not path.is_file():
        return mapping
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        canon, aliases = line.split(":", 1)
        canon_key = canon.strip().lower().replace(" ", "_")
        mapping[canon.strip().lower()] = canon_key
        for a in aliases.split(","):
            al = a.strip().lower()
            if al:
                mapping[al] = canon_key
    return mapping


def classify_predicate_rule(predicate: str, geo_map: Dict[str, str]) -> Tuple[str, str]:
    """Deterministic fallback classifier returning `(pred_type, spatial_geo_rel)`."""
    p = predicate.strip().lower()
    if p in geo_map:
        geo = geo_map[p]
        is_act = any(tok in _ACTION_KEYWORDS for tok in p.split())
        return ("action" if is_act else "spatial"), geo
    tokens = set(p.split())
    if tokens & _ACTION_KEYWORDS:
        for prep, canon in (("on", "on"), ("in", "in"), ("under", "under"), ("over", "above"), ("near", "next_to"), ("next to", "next_to")):
            if prep in tokens or p.endswith(f" {prep}"):
                return "action", canon
        return "action", "unknown"
    return "spatial", "unknown"


def classify_batch_openai(predicates: List[str], model: str = "gpt-4o-mini") -> List[dict]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when running --use-llm.")
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "Classify each image scene-graph predicate into JSON key 'results' containing objects with:\n"
        "1. 'predicate': exact input string\n"
        "2. 'type': 'action' | 'geo_relation' | 'other'\n"
        "3. 'spatial_mapping': one of ['in', 'on', 'under', 'above', 'next_to', 'overlap', 'contains'] if geo_relation else null."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(predicates)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content).get("results", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify predicates into action vs. canonical 2D spatial geometry.")
    parser.add_argument("--input", type=Path, default=Path("outputs/kg_build/predicate_frequency.csv"))
    parser.add_argument("--geo-alias", type=Path, default=ROOT / "knowledge_graph" / "geo_rel_alias.txt")
    parser.add_argument("--output", type=Path, default=Path("outputs/kg_build/predicate_classified.csv"))
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI API for predicates not covered by geo_rel_alias.txt")
    args = parser.parse_args()

    geo_map = load_geo_alias_map(args.geo_alias)
    df = pd.read_csv(args.input)
    pred_col = "predicate" if "predicate" in df.columns else "Predicate"
    preds = df[pred_col].dropna().astype(str).tolist()

    rows = []
    if args.use_llm:
        for i in range(0, len(preds), 50):
            batch = preds[i : i + 50]
            rows.extend(classify_batch_openai(batch))
            time.sleep(0.5)
    else:
        for p in preds:
            ptype, sgeo = classify_predicate_rule(p, geo_map)
            rows.append({"predicate": p, "type": ptype, "spatial_mapping": sgeo})

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[Predicate Geo] Classified {len(out_df):,} predicates -> {args.output}")


if __name__ == "__main__":
    main()

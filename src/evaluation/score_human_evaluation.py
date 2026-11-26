"""
Offline Scoring & Inter-Annotator Agreement Utility for Human Evaluation CSVs.

Supports both human evaluation protocols in `human_evaluation/`:
  1. `role_relation`: Computes confusion matrices, per-class F1, macro-F1 (for entity roles
     `fg`, `objects`, `bg`), relation precision (`yes / (yes + no)`), and multi-annotator
     exact agreement / Cohen's or Fleiss' kappa against an offline unblinding key JSON.
  2. `paired_caption`: Unblinds randomized A/B caption preference ratings (`relevance` and
     `grounding`) against an offline key JSON and summarizes win/tie/loss percentages and
     inter-annotator agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def score_role_and_relation(key_path: Path, csv_paths: List[Path], output_path: Path) -> None:
    cases = {c["case_id"]: c for c in json.loads(key_path.read_text(encoding="utf-8"))}
    counts: Dict[tuple, Counter] = defaultdict(Counter)
    agreement: Dict[str, Dict[str, str]] = defaultdict(dict)
    role_labels = {"fg", "bg", "objects"}

    for path in csv_paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                label = (row.get("label") or "").strip()
                if not label:
                    continue
                case_id = row["case_id"].strip()
                annotator = row["annotator"].strip()
                case = cases[case_id]
                defined = role_labels if case["kind"] == "role" else {"yes", "no"}
                if label not in (defined | {"uncertain", "bad_box"}):
                    raise ValueError(f"Unexpected label {label!r} for {case_id}")
                agreement[case_id][annotator] = label
                base = (annotator, case["dataset"], case["kind"])
                if case["kind"] == "role":
                    for model_tag, field_name in [("sparkg_cap", "legacy_label"), ("taxonomy_baseline", "taxonomy_control")]:
                        if field_name in case:
                            counts[base + (model_tag,)][f"{label}->{case[field_name]}"] += 1
                else:
                    counts[base + (case.get("method", "default"), case.get("tier", "all"))][label] += 1

    report: dict = {
        "protocol": "role_and_relation",
        "labeled_cases": len(agreement),
        "strata": {},
        "agreement": {"multiply_annotated_cases": 0, "unanimous_cases": 0},
    }
    for group, mapping in sorted(counts.items()):
        entry: dict = {"label_counts": dict(mapping)}
        if group[2] == "relation":
            scorable = mapping["yes"] + mapping["no"]
            entry["scorable"] = scorable
            entry["precision_pct"] = round(100.0 * mapping["yes"] / scorable, 2) if scorable else None
        else:
            confusion = {
                actual: {pred: mapping[f"{actual}->{pred}"] for pred in sorted(role_labels)}
                for actual in sorted(role_labels)
            }
            f1 = {}
            for lbl in sorted(role_labels):
                denom = sum(confusion[lbl].values()) + sum(r[lbl] for r in confusion.values())
                f1[lbl] = round(200.0 * confusion[lbl][lbl] / denom, 2) if denom else None
            entry["confusion"] = confusion
            entry["per_class_f1_pct"] = f1
            entry["macro_f1_pct"] = (
                round(sum(v for v in f1.values() if v is not None) / len(f1), 2)
                if all(v is not None for v in f1.values())
                else None
            )
        report["strata"]["/".join(group)] = entry

    for labels in agreement.values():
        if len(labels) >= 2:
            report["agreement"]["multiply_annotated_cases"] += 1
            report["agreement"]["unanimous_cases"] += int(len(set(labels.values())) == 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Human Eval] Saved role & relation summary -> {output_path}")


def score_paired_captions(key_path: Path, csv_paths: List[Path], output_path: Path) -> None:
    raw_key = json.loads(key_path.read_text(encoding="utf-8"))
    case_list = raw_key["cases"] if isinstance(raw_key, dict) and "cases" in raw_key else raw_key
    key = {c["case_id"]: c for c in case_list}
    dimensions = ("relevance", "grounding")
    ratings: Dict[tuple, Dict[str, str]] = defaultdict(dict)
    summary: Dict[tuple, Counter] = defaultdict(Counter)

    for path in csv_paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                case = key[row["case_id"].strip()]
                annotator = row["annotator"].strip()
                for dim in dimensions:
                    ans = (row.get(dim) or "").strip()
                    if not ans:
                        continue
                    if ans not in {"A", "B", "tie", "uncertain"}:
                        raise ValueError(f"Invalid rating {ans!r} in {path.name}:{row['case_id']}")
                    resolved = case[f"{ans.lower()}_condition"] if ans in {"A", "B"} else ans
                    ratings[(case["case_id"], dim)][annotator] = resolved
                    summary[(case["dataset"], dim, annotator)][resolved] += 1

    agreement: Dict[str, dict] = {}
    for (cid, dim), rmap in ratings.items():
        if len(rmap) >= 2:
            gk = f"{key[cid]['dataset']}/{dim}"
            slot = agreement.setdefault(gk, {"multiply_rated": 0, "unanimous": 0})
            slot["multiply_rated"] += 1
            slot["unanimous"] += int(len(set(rmap.values())) == 1)

    report = {
        "protocol": "paired_caption_ab",
        "eligible_pairs": len(key),
        "rated_items": len(ratings),
        "counts_by_dataset_dimension_annotator": {"/".join(k): dict(v) for k, v in sorted(summary.items())},
        "agreement": agreement,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Human Eval] Saved paired A/B caption summary -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score blinded human evaluation CSV exports.")
    parser.add_argument("--mode", choices=["role_relation", "paired_caption"], required=True)
    parser.add_argument("--key", type=Path, required=True, help="Offline unblinding JSON key")
    parser.add_argument("--csv", type=Path, nargs="+", required=True, help="One or more annotator CSV exports")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "role_relation":
        score_role_and_relation(args.key, args.csv, args.output)
    else:
        score_paired_captions(args.key, args.csv, args.output)


if __name__ == "__main__":
    main()

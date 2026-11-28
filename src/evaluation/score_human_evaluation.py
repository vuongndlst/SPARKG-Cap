"""
Offline Scoring & Inter-Annotator Agreement Utility for Human Evaluation CSVs.

Supports both human evaluation protocols in `human_evaluation/`:
  1. `role_relation`: Computes confusion matrices, per-class precision / recall / F1,
     macro-F1 with bootstrap 95% CIs (for entity roles `fg`, `objects`, `bg`),
     tier-stratified relation precision with Wilson 95% CIs, population-reweighted
     estimates, and multi-annotator exact agreement with Cohen's (two raters) or
     Fleiss' (three or more raters) kappa, against an offline unblinding key JSON.
  2. `paired_caption`: Unblinds randomized A/B caption preference ratings (`relevance`
     and `grounding`) against an offline key JSON and summarizes win / tie / loss
     percentages with Wilson 95% CIs, an exact two-sided sign test on decided pairs,
     and inter-annotator agreement with Cohen's / Fleiss' kappa.

Both modes keep the original CLI; the 2026-11-28 revision only adds report fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20261128
ROLE_LABELS = ("bg", "fg", "objects")
NON_CLASS_LABELS = ("uncertain", "bad_box")


# --------------------------------------------------------------------- statistics

def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> Dict[str, float] | None:
    """Wilson score 95% CI for a binomial proportion (percentage points)."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    halfwidth = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return {"point_pct": round(100.0 * p, 2),
            "lo_pct": round(100.0 * max(0.0, centre - halfwidth), 2),
            "hi_pct": round(100.0 * min(1.0, centre + halfwidth), 2),
            "n": total}


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test on decided (non-tied) pairs, p0 = 0.5."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return round(min(1.0, 2.0 * tail), 6)


def _f1_from_pairs(pairs: Sequence[tuple], labels: Sequence[str]) -> Dict[str, float | None]:
    """pairs: (human_label, system_label). Returns per-class F1 plus macro-F1 (pct)."""
    tp = Counter(); fp = Counter(); fn = Counter()
    for human, system in pairs:
        if human == system:
            tp[human] += 1
        else:
            fn[human] += 1
            fp[system] += 1
    out: Dict[str, float | None] = {}
    for label in labels:
        denom = 2 * tp[label] + fp[label] + fn[label]
        out[label] = round(200.0 * tp[label] / denom, 2) if denom else None
    present = [v for v in out.values() if v is not None]
    out["macro"] = round(sum(present) / len(present), 2) if len(present) == len(labels) else None
    return out


def bootstrap_macro_f1(pairs: Sequence[tuple], labels: Sequence[str]) -> Dict[str, float] | None:
    """Percentile bootstrap 95% CI for macro-F1 over annotated cases."""
    if len(pairs) < 10:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    n = len(pairs)
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        value = _f1_from_pairs(sample, labels)["macro"]
        if value is not None:
            draws.append(value)
    if len(draws) < BOOTSTRAP_ITERATIONS // 2:
        return None
    draws.sort()
    return {"lo_pct": round(draws[int(0.025 * len(draws))], 2),
            "hi_pct": round(draws[int(0.975 * len(draws)) - 1], 2),
            "iterations": len(draws)}


def cohens_kappa(pairs: Sequence[tuple]) -> float | None:
    """Cohen's kappa for two raters over commonly rated items."""
    n = len(pairs)
    if n == 0:
        return None
    categories = sorted({c for pair in pairs for c in pair})
    observed = sum(1 for a, b in pairs if a == b) / n
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    expected = sum((first[c] / n) * (second[c] / n) for c in categories)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return round((observed - expected) / (1.0 - expected), 4)


def fleiss_kappa(items: Sequence[Sequence[str]]) -> float | None:
    """Fleiss' kappa; every item must carry the same number of ratings (>= 2)."""
    items = [list(r) for r in items if len(r) >= 2]
    if not items:
        return None
    n = len(items[0])
    if any(len(r) != n for r in items) or n < 2:
        return None
    categories = sorted({c for r in items for c in r})
    total = len(items)
    p_i = []
    cat_totals = Counter()
    for ratings in items:
        counts = Counter(ratings)
        cat_totals.update(counts)
        p_i.append((sum(v * v for v in counts.values()) - n) / (n * (n - 1)))
    p_bar = sum(p_i) / total
    p_e = sum((cat_totals[c] / (total * n)) ** 2 for c in categories)
    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else None
    return round((p_bar - p_e) / (1.0 - p_e), 4)


def agreement_block(per_case: Dict[str, Dict[str, str]]) -> dict:
    """Exact agreement plus Cohen's / Fleiss' kappa over multiply annotated items."""
    annotators = sorted({a for labels in per_case.values() for a in labels})
    multi = {cid: labels for cid, labels in per_case.items() if len(labels) >= 2}
    block: dict = {
        "annotators": annotators,
        "multiply_annotated_cases": len(multi),
        "unanimous_cases": sum(1 for labels in multi.values() if len(set(labels.values())) == 1),
        "percent_agreement_pct": None,
        "cohens_kappa": None,
        "fleiss_kappa": None,
        "pairwise_cohens_kappa": {},
    }
    if not multi:
        return block
    block["percent_agreement_pct"] = round(100.0 * block["unanimous_cases"] / len(multi), 2)
    for i, a in enumerate(annotators):
        for b in annotators[i + 1:]:
            pairs = [(labels[a], labels[b]) for labels in multi.values() if a in labels and b in labels]
            if pairs:
                block["pairwise_cohens_kappa"][f"{a}|{b}"] = {"kappa": cohens_kappa(pairs),
                                                              "n": len(pairs)}
    if len(annotators) == 2:
        a, b = annotators
        entry = block["pairwise_cohens_kappa"].get(f"{a}|{b}")
        block["cohens_kappa"] = entry["kappa"] if entry else None
    elif len(annotators) > 2:
        complete = [[labels[a] for a in annotators] for labels in multi.values()
                    if all(a in labels for a in annotators)]
        block["fleiss_kappa"] = fleiss_kappa(complete)
    return block


# ------------------------------------------------------------- role & relation CSV

def _read_rows(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                yield row


def score_role_and_relation(key_path: Path, csv_paths: List[Path], output_path: Path) -> None:
    key_cases = json.loads(key_path.read_text(encoding="utf-8"))
    if isinstance(key_cases, dict) and "cases" in key_cases:
        key_cases = key_cases["cases"]
    cases = {c["case_id"]: c for c in key_cases}
    for case in cases.values():
        if case["kind"] == "role" and case.get("legacy_label") not in ROLE_LABELS:
            raise ValueError(f"{case['case_id']}: gold legacy_label {case.get('legacy_label')!r} "
                             f"is outside the annotatable role set {ROLE_LABELS}")

    counts: Dict[tuple, Counter] = defaultdict(Counter)
    agreement: Dict[str, Dict[str, str]] = defaultdict(dict)
    role_labels = set(ROLE_LABELS)

    # observations[(scope, system)] -> list of (human_label, system_label, weight)
    role_obs: Dict[tuple, list] = defaultdict(list)
    # relation observations[(scope, tier)] -> list of (label, weight)
    rel_obs: Dict[tuple, list] = defaultdict(list)

    for row in _read_rows(csv_paths):
        label = (row.get("label") or "").strip()
        if not label:
            continue
        case_id = row["case_id"].strip()
        annotator = row["annotator"].strip()
        case = cases[case_id]
        defined = role_labels if case["kind"] == "role" else {"yes", "no"}
        if label not in (defined | set(NON_CLASS_LABELS)):
            raise ValueError(f"Unexpected label {label!r} for {case_id}")
        agreement[case_id][annotator] = label
        base = (annotator, case["dataset"], case["kind"])
        weight = float(case.get("sampling_weight") or 1.0)
        if case["kind"] == "role":
            for model_tag, field_name in [("sparkg_cap", "legacy_label"),
                                          ("taxonomy_baseline", "taxonomy_control")]:
                if field_name in case:
                    counts[base + (model_tag,)][f"{label}->{case[field_name]}"] += 1
                    for scope in ((annotator, case["dataset"]), ("pooled", case["dataset"]),
                                  ("pooled", "all")):
                        role_obs[scope + (model_tag,)].append((label, case[field_name], weight))
        else:
            tier = case.get("tier", "all")
            counts[base + (case.get("method", "default"), tier)][label] += 1
            for scope in ((annotator, case["dataset"]), ("pooled", case["dataset"]),
                          ("pooled", "all")):
                rel_obs[scope + (tier,)].append((label, weight))
                rel_obs[scope + (f"FAMILY:{case.get('tier_family', 'unknown')}",)].append((label, weight))
                rel_obs[scope + ("ALL_TIERS",)].append((label, weight))

    report: dict = {
        "protocol": "role_and_relation",
        "labeled_cases": len(agreement),
        "key_cases": len(cases),
        "strata": {},
        "agreement": {"multiply_annotated_cases": 0, "unanimous_cases": 0},
    }

    # ---- original per-annotator strata block (kept for backward compatibility)
    for group, mapping in sorted(counts.items()):
        entry: dict = {"label_counts": dict(mapping)}
        if group[2] == "relation":
            scorable = mapping["yes"] + mapping["no"]
            entry["scorable"] = scorable
            entry["precision_pct"] = round(100.0 * mapping["yes"] / scorable, 2) if scorable else None
            entry["precision_ci95"] = wilson_interval(mapping["yes"], scorable)
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

    # ---- role: confusion / P / R / F1 / macro-F1 + CI, per annotator, pooled, weighted
    role_block: dict = {}
    for (scope, dataset, system), observations in sorted(role_obs.items()):
        decided = [(h, s, w) for h, s, w in observations if h in role_labels]
        skipped = Counter(h for h, _, _ in observations if h not in role_labels)
        confusion = {h: {s: 0 for s in ROLE_LABELS} for h in ROLE_LABELS}
        weighted = {h: {s: 0.0 for s in ROLE_LABELS} for h in ROLE_LABELS}
        for human, system_label, weight in decided:
            confusion[human][system_label] += 1
            weighted[human][system_label] += weight
        pairs = [(h, s) for h, s, _ in decided]
        f1 = _f1_from_pairs(pairs, ROLE_LABELS)
        per_class = {}
        for label in ROLE_LABELS:
            tp = confusion[label][label]
            support = sum(confusion[label].values())
            predicted = sum(confusion[h][label] for h in ROLE_LABELS)
            per_class[label] = {
                "support_human": support,
                "predicted_by_system": predicted,
                "recall_pct": round(100.0 * tp / support, 2) if support else None,
                "precision_pct": round(100.0 * tp / predicted, 2) if predicted else None,
                "f1_pct": f1[label],
            }
        total_w = sum(w for _, _, w in decided)
        acc_w = sum(w for h, s, w in decided if h == s)
        role_block[f"{scope}/{dataset}/{system}"] = {
            "annotations": len(observations),
            "decided": len(decided),
            "excluded_by_annotator": dict(skipped),
            "confusion_human_rows_x_system_cols": confusion,
            "per_class": per_class,
            "macro_f1_pct": f1["macro"],
            "macro_f1_ci95": bootstrap_macro_f1(pairs, ROLE_LABELS),
            "accuracy_pct": round(100.0 * sum(1 for h, s in pairs if h == s) / len(pairs), 2) if pairs else None,
            "accuracy_ci95": wilson_interval(sum(1 for h, s in pairs if h == s), len(pairs)),
            "population_weighted_accuracy_pct": round(100.0 * acc_w / total_w, 2) if total_w else None,
            "population_weighted_confusion": {h: {s: round(v, 2) for s, v in row.items()}
                                              for h, row in weighted.items()},
        }
    report["role_validation"] = role_block

    # ---- relation: tier-stratified precision with CIs
    rel_block: dict = {}
    for (scope, dataset, tier), observations in sorted(rel_obs.items()):
        label_counts = Counter(lbl for lbl, _ in observations)
        yes, no = label_counts["yes"], label_counts["no"]
        scorable = yes + no
        weight_yes = sum(w for lbl, w in observations if lbl == "yes")
        weight_scorable = sum(w for lbl, w in observations if lbl in ("yes", "no"))
        rel_block[f"{scope}/{dataset}/{tier}"] = {
            "annotations": len(observations),
            "label_counts": dict(label_counts),
            "scorable": scorable,
            "precision_pct": round(100.0 * yes / scorable, 2) if scorable else None,
            "precision_ci95": wilson_interval(yes, scorable),
            "population_weighted_precision_pct": (round(100.0 * weight_yes / weight_scorable, 2)
                                                  if weight_scorable else None),
            "uncertain_rate_pct": (round(100.0 * label_counts["uncertain"] / len(observations), 2)
                                   if observations else None),
            "bad_box_rate_pct": (round(100.0 * label_counts["bad_box"] / len(observations), 2)
                                 if observations else None),
            "underpowered": scorable < 30,
        }
    report["relation_validation"] = rel_block

    # ---- agreement (legacy counters kept, kappa added)
    for labels in agreement.values():
        if len(labels) >= 2:
            report["agreement"]["multiply_annotated_cases"] += 1
            report["agreement"]["unanimous_cases"] += int(len(set(labels.values())) == 1)
    overall = agreement_block(agreement)
    report["agreement"].update({k: v for k, v in overall.items()
                                if k not in ("multiply_annotated_cases", "unanimous_cases")})
    report["agreement_by_kind"] = {
        kind: agreement_block({cid: labels for cid, labels in agreement.items()
                               if cases[cid]["kind"] == kind})
        for kind in ("role", "relation")
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Human Eval] Saved role & relation summary -> {output_path}")
    _print_role_relation_console(report)


def _print_role_relation_console(report: dict) -> None:
    print(f"  labeled cases: {report['labeled_cases']} / {report['key_cases']}")
    pooled_role = {k: v for k, v in report["role_validation"].items() if k.startswith("pooled/all/")}
    for name, block in sorted(pooled_role.items()):
        ci = block["macro_f1_ci95"]
        ci_txt = f" [95% CI {ci['lo_pct']}-{ci['hi_pct']}]" if ci else ""
        print(f"  ROLE {name}: macro-F1 {block['macro_f1_pct']}%{ci_txt} "
              f"(n={block['decided']}, acc {block['accuracy_pct']}%)")
        for label, stats in block["per_class"].items():
            print(f"      {label:<8} P={stats['precision_pct']} R={stats['recall_pct']} "
                  f"F1={stats['f1_pct']} (support {stats['support_human']})")
    for name, block in sorted(report["relation_validation"].items()):
        if not name.startswith("pooled/all/"):
            continue
        ci = block["precision_ci95"]
        ci_txt = f" [95% CI {ci['lo_pct']}-{ci['hi_pct']}]" if ci else ""
        flag = "  <-- UNDERPOWERED" if block["underpowered"] else ""
        print(f"  REL  {name.split('/')[-1]:<16} precision {block['precision_pct']}%{ci_txt} "
              f"(n={block['scorable']}){flag}")
    agr = report["agreement"]
    print(f"  agreement: {agr['percent_agreement_pct']}% exact, Cohen's kappa "
          f"{agr['cohens_kappa']}, Fleiss' kappa {agr['fleiss_kappa']} "
          f"(n={agr['multiply_annotated_cases']})")
    for kind, block in report["agreement_by_kind"].items():
        print(f"      {kind}: kappa {block['cohens_kappa'] or block['fleiss_kappa']} "
              f"(n={block['multiply_annotated_cases']})")


# ------------------------------------------------------------------ paired A/B CSV

def score_paired_captions(key_path: Path, csv_paths: List[Path], output_path: Path) -> None:
    raw_key = json.loads(key_path.read_text(encoding="utf-8"))
    case_list = raw_key["cases"] if isinstance(raw_key, dict) and "cases" in raw_key else raw_key
    key = {c["case_id"]: c for c in case_list}
    dimensions = ("relevance", "grounding")
    ratings: Dict[tuple, Dict[str, str]] = defaultdict(dict)
    summary: Dict[tuple, Counter] = defaultdict(Counter)
    buckets: Dict[tuple, Counter] = defaultdict(Counter)

    for row in _read_rows(csv_paths):
        case = key[row["case_id"].strip()]
        annotator = row["annotator"].strip()
        for dim in dimensions:
            ans = (row.get(dim) or "").strip()
            if not ans:
                continue
            if ans not in {"A", "B", "tie", "uncertain"}:
                raise ValueError(f"Invalid rating {ans!r} for {row['case_id']} / {dim}")
            resolved = case[f"{ans.lower()}_condition"] if ans in {"A", "B"} else ans
            ratings[(case["case_id"], dim)][annotator] = resolved
            summary[(case["dataset"], dim, annotator)][resolved] += 1
            for scope in (annotator, "pooled"):
                buckets[(scope, case["dataset"], dim)][resolved] += 1
                buckets[(scope, "all", dim)][resolved] += 1

    agreement: Dict[str, dict] = {}
    for (cid, dim), rmap in ratings.items():
        if len(rmap) >= 2:
            gk = f"{key[cid]['dataset']}/{dim}"
            slot = agreement.setdefault(gk, {"multiply_rated": 0, "unanimous": 0})
            slot["multiply_rated"] += 1
            slot["unanimous"] += int(len(set(rmap.values())) == 1)

    # win / tie / loss, from the proposed system's point of view
    wtl: Dict[str, dict] = {}
    for (scope, dataset, dim), counter in sorted(buckets.items()):
        wins = counter["graph_context"]
        losses = counter["direct_image"]
        ties = counter["tie"]
        uncertain = counter["uncertain"]
        scorable = wins + losses + ties
        decided = wins + losses
        wtl[f"{scope}/{dataset}/{dim}"] = {
            "n_ratings": sum(counter.values()),
            "wins_graph_context": wins,
            "losses_direct_image": losses,
            "ties": ties,
            "uncertain": uncertain,
            "scorable": scorable,
            "win_pct": round(100.0 * wins / scorable, 2) if scorable else None,
            "tie_pct": round(100.0 * ties / scorable, 2) if scorable else None,
            "loss_pct": round(100.0 * losses / scorable, 2) if scorable else None,
            "win_rate_ci95": wilson_interval(wins, scorable),
            "win_among_decided_pct": round(100.0 * wins / decided, 2) if decided else None,
            "win_among_decided_ci95": wilson_interval(wins, decided),
            "sign_test_p_two_sided": sign_test_p(wins, losses),
        }

    # per dimension agreement with kappa
    kappa_block = {}
    for dim in dimensions:
        per_case = {cid: rmap for (cid, d), rmap in ratings.items() if d == dim}
        kappa_block[dim] = agreement_block(per_case)
    kappa_block["overall"] = agreement_block(
        {f"{cid}|{dim}": rmap for (cid, dim), rmap in ratings.items()})

    report = {
        "protocol": "paired_caption_ab",
        "eligible_pairs": len(key),
        "rated_items": len(ratings),
        "conditions": {"graph_context": "SPARKG-Cap (proposed)", "direct_image": "baseline"},
        "counts_by_dataset_dimension_annotator": {"/".join(k): dict(v) for k, v in sorted(summary.items())},
        "win_tie_loss": wtl,
        "agreement": agreement,
        "agreement_kappa": kappa_block,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Human Eval] Saved paired A/B caption summary -> {output_path}")
    for name, block in sorted(report["win_tie_loss"].items()):
        if not name.startswith("pooled/"):
            continue
        ci = block["win_rate_ci95"]
        ci_txt = f" [95% CI {ci['lo_pct']}-{ci['hi_pct']}]" if ci else ""
        print(f"  A/B {name:<24} win {block['win_pct']}%{ci_txt} / tie {block['tie_pct']}% "
              f"/ loss {block['loss_pct']}% (n={block['scorable']}, sign-test p="
              f"{block['sign_test_p_two_sided']})")
    for dim, block in report["agreement_kappa"].items():
        print(f"  agreement {dim:<10} {block['percent_agreement_pct']}% exact, Cohen's kappa "
              f"{block['cohens_kappa']}, Fleiss' kappa {block['fleiss_kappa']} "
              f"(n={block['multiply_annotated_cases']})")


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

"""
Unified Karpathy Split Benchmark Evaluation & Paired Statistical Significance Testing.

Evaluates generated captions against official Karpathy `test` (or `val`) split references
using Stanford `PTBTokenizer` preprocessing across:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - METEOR (with newline-sanitized stdio pipe for modern Java 11+)
  - ROUGE-L
  - CIDEr-D
  - SPICE (semantic propositional F1)
  - Caption length statistics (mean, SD, median, IQR)
  - Paired bootstrap 95% confidence intervals (B = 10,000) and Holm-Bonferroni correction
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karpathy_splits import format_coco_stem, load_karpathy_references


JAVA_OPENS_MODERN = [
    "--add-opens", "java.base/java.lang=ALL-UNNAMED",
    "--add-opens", "java.base/java.util=ALL-UNNAMED",
    "--add-opens", "java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens", "java.base/java.text=ALL-UNNAMED",
    "--add-opens", "java.base/java.io=ALL-UNNAMED",
    "--add-opens", "java.base/java.math=ALL-UNNAMED",
    "--add-opens", "java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens", "java.base/java.net=ALL-UNNAMED",
]


def _get_java_major() -> int:
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True)
        first = out.splitlines()[0]
        import re
        m = re.search(r'version "(\d+)(?:\.(\d+))?', first)
        if m:
            return int(m.group(2) if m.group(1) == "1" else m.group(1))
    except Exception:
        pass
    return 11


class SafeMeteorScorer:
    """Thread-safe METEOR 1.5 wrapper that sanitizes embedded newlines in reference captions."""

    def __init__(self) -> None:
        import pycocoevalcap.meteor.meteor as mm
        orig_dir = os.path.dirname(os.path.abspath(mm.__file__))
        if " " in orig_dir:
            safe_dir = os.path.join(tempfile.gettempdir(), "sparkg_meteor_eval")
            if not os.path.exists(safe_dir):
                shutil.copytree(orig_dir, safe_dir)
            self.dir = safe_dir
        else:
            self.dir = orig_dir
        self.jar = os.path.join(self.dir, "meteor-1.5.jar")
        self.available = os.path.isfile(self.jar)
        flags = JAVA_OPENS_MODERN if _get_java_major() >= 9 else []
        self.cmd = ["java", "-jar", "-Xmx2G"] + flags + [self.jar, "-", "-", "-stdio", "-l", "en", "-norm"]
        self._lock = threading.Lock()

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).replace("|||", " ").split())

    def compute_score(self, gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Tuple[float, np.ndarray]:
        if not self.available:
            return 0.0, np.zeros(len(gts))
        keys = sorted(gts.keys())
        with self._lock:
            proc = subprocess.Popen(
                self.cmd, cwd=self.dir,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            try:
                time.sleep(0.3)
                eval_line = "EVAL"
                for k in keys:
                    hyp = self._clean(res[k][0])
                    refs = [self._clean(r) for r in gts[k]]
                    assert proc.stdin and proc.stdout
                    proc.stdin.write(f"SCORE ||| {' ||| '.join(refs)} ||| {hyp}\n")
                    proc.stdin.flush()
                    stat = proc.stdout.readline().strip()
                    eval_line += f" ||| {stat}"
                assert proc.stdin and proc.stdout
                proc.stdin.write(eval_line + "\n")
                proc.stdin.flush()
                per_img = [float(proc.stdout.readline().strip() or 0.0) for _ in keys]
                corpus = float(proc.stdout.readline().strip() or 0.0)
                return corpus, np.asarray(per_img, dtype=float)
            finally:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    pass


def load_predictions(path: Path, dataset: str = "coco") -> Dict[str, str]:
    """Load predictions from `.json`, `.jsonl`, or `.csv` keyed by canonical Karpathy `image_id`."""
    preds: Dict[str, str] = {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, v in data.items():
                key = format_coco_stem(int(k)) if dataset == "coco" and str(k).isdigit() else Path(str(k)).stem
                preds[key] = str(v).strip()
        elif isinstance(data, list):
            for row in data:
                raw_id = row.get("image_id", row.get("cocoid", ""))
                key = format_coco_stem(int(raw_id)) if dataset == "coco" and str(raw_id).isdigit() else Path(str(raw_id)).stem
                preds[key] = str(row.get("caption", "")).strip()
    elif suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw_id = row.get("image_id", "")
            key = format_coco_stem(int(raw_id)) if dataset == "coco" and str(raw_id).isdigit() else Path(str(raw_id)).stem
            preds[key] = str(row.get("caption") or "").strip()
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                raw_id = row.get("image_id") or row.get("file_name") or ""
                digits = "".join(ch for ch in Path(str(raw_id)).stem if ch.isdigit())
                key = format_coco_stem(int(digits)) if dataset == "coco" and digits else Path(str(raw_id)).stem
                preds[key] = str(row.get("caption") or "").strip()
    else:
        raise ValueError(f"Unsupported prediction format: {path}")
    return preds


def evaluate_predictions(
    gt_refs: Dict[str, List[str]],
    preds: Dict[str, str],
    use_spice: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    keys = sorted(gt_refs.keys())
    ptb = PTBTokenizer()
    tok_gts = ptb.tokenize({k: [{"caption": c} for c in gt_refs[k]] for k in keys})
    tok_res = ptb.tokenize({k: [{"caption": preds.get(k, "")}] for k in keys})

    bleu_s, bleu_img = Bleu(4).compute_score(tok_gts, tok_res)
    rouge_s, rouge_img = Rouge().compute_score(tok_gts, tok_res)
    cider_s, cider_img = Cider().compute_score(tok_gts, tok_res)
    meteor_s, meteor_img = SafeMeteorScorer().compute_score(tok_gts, tok_res)

    scores = {
        "BLEU_1": float(bleu_s[0]) * 100.0,
        "BLEU_2": float(bleu_s[1]) * 100.0,
        "BLEU_3": float(bleu_s[2]) * 100.0,
        "BLEU_4": float(bleu_s[3]) * 100.0,
        "METEOR": float(meteor_s) * 100.0,
        "ROUGE_L": float(rouge_s) * 100.0,
        "CIDEr": float(cider_s) * 100.0,
    }
    per_image = {
        "BLEU_4": np.asarray(bleu_img[3], dtype=float) * 100.0,
        "METEOR": np.asarray(meteor_img, dtype=float) * 100.0,
        "ROUGE_L": np.asarray(rouge_img, dtype=float) * 100.0,
        "CIDEr": np.asarray(cider_img, dtype=float) * 100.0,
    }

    if use_spice:
        try:
            from pycocoevalcap.spice.spice import Spice
            spice_s, spice_img = Spice().compute_score(tok_gts, tok_res)
            scores["SPICE"] = float(spice_s) * 100.0
            per_image["SPICE"] = np.asarray([float(x["All"]["f"]) for x in spice_img], dtype=float) * 100.0
        except Exception as exc:
            print(f"[Warn] SPICE unavailable ({type(exc).__name__}); continuing with remaining metrics.")

    lengths = np.asarray([len(tok_res[k][0].split()) if tok_res[k][0].strip() else 0 for k in keys], dtype=float)
    len_stats = {
        "mean_words": float(lengths.mean()),
        "sd_words": float(lengths.std(ddof=0)),
        "p25_words": float(np.quantile(lengths, 0.25)),
        "median_words": float(np.median(lengths)),
        "p75_words": float(np.quantile(lengths, 0.75)),
    }
    return scores, per_image, len_stats


def paired_bootstrap_ci(
    sys_a: np.ndarray,
    sys_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    diff = sys_a - sys_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tail = min(int(np.sum(boots <= 0)), int(np.sum(boots >= 0)))
    p_val = min(1.0, 2.0 * (tail + 1) / (n_boot + 1))
    return {
        "delta": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(p_val),
    }


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(ordered):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate captions on official Karpathy splits with PTBTokenizer.")
    parser.add_argument("--dataset", choices=["coco", "flickr30k"], default="coco")
    parser.add_argument("--karpathy-json", type=Path, default=Path("data/karpathy/dataset_coco.json"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--predictions", type=Path, required=True, help="System predictions (.json, .jsonl, or .csv)")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline predictions for paired bootstrap CI")
    parser.add_argument("--no-spice", action="store_true", help="Skip Java SPICE evaluation")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_report.json"))
    args = parser.parse_args()

    gt_refs, _ = load_karpathy_references(args.karpathy_json, dataset=args.dataset, split=args.split)
    preds = load_predictions(args.predictions, dataset=args.dataset)

    scores, per_img, len_stats = evaluate_predictions(gt_refs, preds, use_spice=not args.no_spice)
    report: dict = {
        "dataset": args.dataset,
        "karpathy_split": args.split,
        "n_images": len(gt_refs),
        "tokenization": "Stanford PTBTokenizer",
        "scores_x100": scores,
        "length_stats": len_stats,
    }

    if args.baseline and args.baseline.is_file():
        base_preds = load_predictions(args.baseline, dataset=args.dataset)
        base_scores, base_img, base_len = evaluate_predictions(gt_refs, base_preds, use_spice=not args.no_spice)
        paired = {}
        raw_p = {}
        for m in per_img:
            if m in base_img:
                paired[m] = paired_bootstrap_ci(per_img[m], base_img[m])
                raw_p[m] = paired[m]["p_two_sided"]
        adj_p = holm_bonferroni(raw_p)
        for m, ap in adj_p.items():
            paired[m]["p_holm_bonferroni"] = ap
        report["baseline_scores_x100"] = base_scores
        report["baseline_length_stats"] = base_len
        report["paired_bootstrap_vs_baseline"] = paired

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

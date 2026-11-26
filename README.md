# SPARKG-Cap: Knowledge-Graph-Guided Image Captioning via Foreground, Background, and Auxiliary Entity Relationships

Official implementation, corpus-level knowledge graph ($G_K$), statistical priors, prompt specifications, and human evaluation suite for **SPARKG-Cap**:

> **A Knowledge Graph Based on the Semantic Relationships Among Foreground, Background, and Auxiliary Entities for Image Captioning**  
> *Nguyen Do Thai Nguyen, Nguyen Dinh Vuong, and Thanh The Van*

---

## Overview

**SPARKG-Cap** is a modular, training-free, knowledge-graph-guided image captioning framework that decouples visual perception from linguistic realization through an explicit, inspectable scene-graph intermediate representation:

$$I \;\longrightarrow\; O_I \;\longrightarrow\; E = E_{\text{fg}} \cup E_{\text{bg}} \cup E_{\text{aux}} \;\longrightarrow\; G_I = (E, R_I, A_I) \;\longrightarrow\; C_I \;\longrightarrow\; Y_I$$

1. **Panoptic Entity Extraction ($O_I$)**: Combines instance detection (**Mask R-CNN R50-FPN** over 80 `thing` classes) and semantic segmentation (**DeepLabV3 R50-D8** over 91 `stuff` classes) to capture both discrete actors/objects and amorphous environmental regions.
2. **Tri-Partite Salience & Role Partitioning ($E_{\text{fg}}, E_{\text{bg}}, E_{\text{aux}}$)**: Computes closed-form Foreground ($FGS_i$) and Background ($BGS_i$) salience scores from relative area, spatial centrality, interaction degree, and corpus-derived semantic priors, followed by deterministic consistency rules ($R_0\text{--}R_5$).
3. **Geometry-Grounded Knowledge Graph Lookup ($G_K \rightarrow R_I$)**: Maps pairwise 2D bounding-box geometry (`on`, `in`, `contains`, `above`, `under`, `next_to`, `overlap`) through four materialized lookup views ($K_1\text{--}K_4$) and a hierarchical cascade ($L_1\text{--}L_5$), pruned by physical/semantic compatibility constraints and degree caps.
4. **Attribute & Triplet Enrichment ($A_I, T_1\text{--}T_2$)**: Refines visual crops via **GrabCut** and zero-shot **CLIP ViT-B/32** for fine-grained lexical aliases, object colors, and human clothing attributes, then upgrades generic geometric edges with corpus-backed human predicates ($T_1, T_2$).
5. **Stratified Context Serialization & Caption Generation ($C_I \rightarrow Y_I$)**: Serializes $G_I$ into a compact, salience-ranked textual prompt (`PRIMARY FOCUS`, `SECONDARY`, `SETTING`, `ACTIONS`, `SPATIAL`) for deterministic decoding (`temperature = 0`) using **OpenAI (`gpt-4o-mini`, default)** or an open-weight local **Qwen (`Qwen/Qwen2.5-14B-Instruct`)** backend.

---

## Repository Structure

```text
SPARKG-Cap/
├── docs/
│   ├── PROMPTS.md                        # System prompts, 4-gram boosters, and few-shot exemplars
│   ├── KNOWLEDGE_GRAPH_SCHEMA.md         # Schema of G_K, materialized views K1-K4, L1-L5 & T1-T2 cascades
│   └── PARAMETERS_AND_WEIGHTS.md         # Closed-form salience weights, thresholds, and compatibility rules
├── knowledge_graph/
│   ├── knowledge_base_v2.csv             # Operational corpus-level KG G_K (156,905 canonical triplets)
│   ├── raw_merged_triplets.csv.gz        # Merged multi-source triplet inventory for T1/T2 enrichment
│   ├── knowledge_base_clean_v2.csv       # PMI-filtered canonical triplet table
│   ├── coco_alias_clip.txt               # Zero-shot CLIP fine-grained alias inventory
│   ├── coco_alias_v2.txt                 # Canonical entity alias resolution dictionary
│   ├── geo_rel_alias.txt                 # Predicate-to-2D-geometry mapping
│   ├── predicate_stats.txt               # Corpus predicate frequency summary
│   └── spatial_geo_rel_full_list.txt     # Canonical spatial relation inventory
├── statistics/
│   ├── derived_weights_coco_things_stuff_penalty_gamma.csv
│   ├── derived_bg_weights_farcenter_penalty_gamma.csv
│   ├── coco_caption_foreground_stats_trainval.csv
│   ├── cocostuff_caption_foreground_stats_trainval.csv
│   ├── foreground_things_coco.txt
│   ├── background_stuff_coco.txt
│   ├── coco_category_map.txt
│   ├── tmp_4gram_coco_v8.txt             # Training-split 4-gram phrase boosters (MS-COCO Karpathy train)
│   └── tmp_4gram_flickr_v3.txt           # Training-split 4-gram phrase boosters (Flickr30K Karpathy train)
├── src/
│   ├── karpathy_splits.py                # Official Karpathy split resolver (MS-COCO & Flickr30K)
│   ├── phase0_build_kg/                  # Offline construction of G_K with strict Karpathy val/test exclusion
│   ├── phase1_scene_graph/               # Stage 1-3: Detection, segmentation, FGS/BGS, K1-K4 & CLIP enrichment
│   ├── phase2_caption_generation/        # Stage 4: Caption generation (--backend openai | --backend qwen)
│   ├── phase3_neo4j/                     # Optional Neo4j property-graph indexing
│   └── evaluation/                       # Karpathy PTBTokenizer + SPICE + bootstrap CI + human eval scoring
└── human_evaluation/                     # Self-contained browser suite for blinded human evaluation
```

---

## Installation

1. **Clone the repository and install Python dependencies:**

```bash
git clone https://github.com/vuongndlst/SPARKG-Cap.git
cd SPARKG-Cap
pip install -r requirements.txt
```

2. **(For Stage 1–2 Scene Graph Extraction from Raw Images) Install OpenMMLab and CLIP:**

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet mmsegmentation
pip install git+https://github.com/openai/CLIP.git
```

3. **Configure environment variables:**

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=... (if using the default OpenAI backend)
```

> **Java Requirement for Evaluation**: Official `METEOR`, `SPICE`, and `PTBTokenizer` scoring in `pycocoevalcap` requires Java (JDK 8–11 recommended for `SPICE`).

---

## Dataset Setup (Standard Karpathy Splits)

All pipeline stages and evaluation scripts follow the standard **Karpathy splits** for both **MS-COCO** and **Flickr30K**. Arrange the datasets under `data/` as follows:

```text
data/
├── karpathy/
│   ├── dataset_coco.json                 # Official Karpathy split manifest for MS-COCO
│   └── dataset_flickr30k.json            # Official Karpathy split manifest for Flickr30K
├── coco/
│   ├── train2014/                        # (or unified images/ directory; resolved automatically)
│   └── val2014/
└── flickr30k/
    └── flickr30k-images/
```

The helper module `src/karpathy_splits.py` automatically resolves image filenames across standard MS-COCO (`COCO_val2014_000000xxxxxx.jpg` or `000000xxxxxx.jpg`) and Flickr30K layouts while strictly enforcing `train` / `val` / `test` split boundaries.

---

## Quick Start: End-to-End Pipeline

### Phase 0 (Optional): Rebuild the Corpus-Level Knowledge Graph $G_K$
Pre-built operational knowledge graph tables are already included in `knowledge_graph/` and `statistics/`. To rebuild $G_K$ from raw MS-COCO Karpathy `train` captions and Visual Genome region graphs (automatically excluding all Karpathy `val` and `test` images, including cross-dataset Visual Genome `coco_id` links):

```bash
python src/phase0_build_kg/build_knowledge_base.py \
  --karpathy-coco data/karpathy/dataset_coco.json \
  --vg-zip data/visual_genome/region_graphs.json.zip \
  --vg-image-data data/visual_genome/image_data.json \
  --output knowledge_graph/knowledge_base_v2.csv
```

### Phase 1: Scene Graph Construction & Multi-Modal Enrichment ($G_I$)

**Step 1A — Panoptic Detection, FGS/BGS Role Assignment, and $K_1\text{--}K_4$ Lookup:**
```bash
# MS-COCO Karpathy test split
python src/phase1_scene_graph/generate_scene_graph.py \
  --dataset coco \
  --karpathy-json data/karpathy/dataset_coco.json \
  --split test \
  --img-root data/coco \
  --output-dir outputs/coco_karpathy_test

# Flickr30K Karpathy test split
python src/phase1_scene_graph/generate_scene_graph.py \
  --dataset flickr30k \
  --karpathy-json data/karpathy/dataset_flickr30k.json \
  --split test \
  --img-root data/flickr30k/flickr30k-images \
  --output-dir outputs/flickr30k_karpathy_test
```

**Step 1B — GrabCut + CLIP ViT-B/32 Alias, Attribute, and $T_1/T_2$ Triplet Enrichment:**
```bash
# MS-COCO Karpathy test split
python src/phase1_scene_graph/enrich_alias_triplets.py \
  --dataset coco \
  --karpathy-json data/karpathy/dataset_coco.json \
  --split test \
  --img-root data/coco \
  --sg-root outputs/coco_karpathy_test

# Flickr30K Karpathy test split
python src/phase1_scene_graph/enrich_alias_triplets.py \
  --dataset flickr30k \
  --karpathy-json data/karpathy/dataset_flickr30k.json \
  --split test \
  --img-root data/flickr30k/flickr30k-images \
  --sg-root outputs/flickr30k_karpathy_test
```

### Phase 2: Knowledge-Graph-Guided Caption Generation ($Y_I$)

#### Option A: Primary OpenAI Backend (Default — `gpt-4o-mini`, `temperature=0`)
```bash
# MS-COCO Karpathy test split
python src/phase2_caption_generation/generate_captions.py \
  --backend openai \
  --model gpt-4o-mini \
  --dataset coco \
  --karpathy-json data/karpathy/dataset_coco.json \
  --split test \
  --sg-root outputs/coco_karpathy_test \
  --output-dir outputs/coco_karpathy_test/captions_openai

# Flickr30K Karpathy test split
python src/phase2_caption_generation/generate_captions.py \
  --backend openai \
  --model gpt-4o-mini \
  --dataset flickr30k \
  --karpathy-json data/karpathy/dataset_flickr30k.json \
  --split test \
  --sg-root outputs/flickr30k_karpathy_test \
  --output-dir outputs/flickr30k_karpathy_test/captions_openai
```

#### Option B: Open-Weight Local Qwen Backend (`Qwen/Qwen2.5-14B-Instruct`)
```bash
# MS-COCO Karpathy test split (supports 4-bit NF4 quantization via --load-in-4bit)
python src/phase2_caption_generation/generate_captions.py \
  --backend qwen \
  --qwen-model Qwen/Qwen2.5-14B-Instruct \
  --load-in-4bit \
  --dataset coco \
  --karpathy-json data/karpathy/dataset_coco.json \
  --split test \
  --sg-root outputs/coco_karpathy_test \
  --output-dir outputs/coco_karpathy_test/captions_qwen
```

To inspect serialized prompts without invoking an LLM API or loading GPU weights, pass `--dry-run --limit 5`.

---

## Evaluation & Statistical Testing

Evaluate generated captions on the official Karpathy `test` split using standard **Stanford `PTBTokenizer`** preprocessing across `BLEU-1..4`, `METEOR`, `ROUGE-L`, `CIDEr`, `SPICE`, caption length distributions, paired bootstrap 95% confidence intervals, and Holm–Bonferroni multiple-comparison correction:

```bash
python src/evaluation/evaluate_karpathy.py \
  --dataset coco \
  --karpathy-json data/karpathy/dataset_coco.json \
  --split test \
  --predictions outputs/coco_karpathy_test/captions_openai/predictions.json \
  --baseline outputs/coco_karpathy_test/baseline/predictions.json \
  --output outputs/coco_karpathy_test/eval_report.json
```

---

## Human Evaluation Suite

A self-contained, blinded browser-based evaluation portal is provided in [`human_evaluation/`](human_evaluation/index.html) (compatible with GitHub Pages or any static browser session):

1. **Calibration Pilot (`calibration_pilot.html`)**: Annotator onboarding set for aligning role (`Foreground`, `Auxiliary`, `Background`) and relation verification criteria prior to main scoring.
2. **Intrinsic Role & Relation Verification (`role_and_relation_eval.html`)**: Blinded image-grounded verification of entity role assignments and extracted relation edges with bounding-box overlays.
3. **Extrinsic Blinded Paired Caption Evaluation (`paired_caption_eval.html`)**: Randomized A/B comparison of caption **Relevance** and **Visual Grounding** (penalizing hallucinated objects, attributes, or actions).

Each interface saves progress locally in the browser and exports standard UTF-8 `.csv` files for offline aggregation via `src/evaluation/score_human_evaluation.py`.

---

## Documentation

- [**Prompt & Serialization Specification (`docs/PROMPTS.md`)**](docs/PROMPTS.md)
- [**Knowledge Graph Schema & Lookup Hierarchy (`docs/KNOWLEDGE_GRAPH_SCHEMA.md`)**](docs/KNOWLEDGE_GRAPH_SCHEMA.md)
- [**Salience Weights, Thresholds & Ontology Rules (`docs/PARAMETERS_AND_WEIGHTS.md`)**](docs/PARAMETERS_AND_WEIGHTS.md)

---

## License

This project is released under the [MIT License](LICENSE).

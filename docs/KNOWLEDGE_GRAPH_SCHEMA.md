# Knowledge Graph Schema & Lookup Hierarchy

This document describes the offline corpus-level knowledge graph $G_K$, the four materialized lookup views ($K_1\text{--}K_4$), the hierarchical geometric-semantic relation cascades ($L_1\text{--}L_5$ and $T_1\text{--}T_2$), and the optional Neo4j property-graph schema.

---

## 1. Corpus-Level Knowledge Graph ($G_K$)

$G_K$ aggregates canonicalized human-annotated relational triplets from **MS-COCO training captions** and **Visual Genome region graphs** (strictly excluding all Karpathy `val` and `test` split images, including Visual Genome images cross-linked via `coco_id` to held-out Karpathy splits).

### `knowledge_graph/knowledge_base_v2.csv` (156,905 rows)

| Column | Type | Description |
| :--- | :--- | :--- |
| `coco_class` | `string` | Canonical MS-COCO / COCO-Stuff entity class (80 `thing` + 91 `stuff` classes) |
| `class_type` | `string` | Domain category (`thing` or `stuff`) |
| `sources` | `string` | Provenance (`coco`, `vg`, or `coco+vg`) |
| `role` | `string` | Syntactic role of `coco_class` in the triplet (`subject` or `object`) |
| `predicate` | `string` | Surface natural-language predicate (e.g., `riding`, `sitting on`, `holding`) |
| `rel_type` | `string` | Functional predicate category (`action` or `spatial`) |
| `geo_rel` | `string` | Raw spatial phrase prior to canonicalization |
| `spatial_geo_rel` | `string` | Canonical 2D bounding-box geometry (`on`, `in`, `above`, `under`, `next_to`, `overlap`, `unknown`) |
| `partner_canonical` | `string` | Canonical class of the partner entity |
| `partner_coco_type` | `string` | Domain category of the partner entity (`thing` or `stuff`) |
| `frequency` | `int` | Empirical corpus count across training annotations (retained only if $\text{PMI}(s, o) \ge 0$) |

---

## 2. Materialized Lookup Views ($K_1\text{--}K_4$) & Cascade ($L_1\text{--}L_5$)

During Stage 1 (`generate_scene_graph.py`), rows with valid 2D spatial geometry (`in`, `on`, `above`, `under`, `next_to`, `overlap`) are normalized to `(subj, obj, spatial_geo_rel)` and sorted by `(rel_type == 'action' first, frequency descending)` to materialize four O(1) hash tables:

| View | Key Tuple | Value | Cascade Level | Base Edge Weight |
| :--- | :--- | :--- | :---: | :---: |
| **$K_1$** (`kb_exact`) | `(subj, obj, spatial_geo_rel)` | Top action/spatial `predicate` | **$L_1$** | `1.00` |
| **$K_2$** (`kb_subj_geo`) | `(subj, spatial_geo_rel)` | Top action/spatial `predicate` | **$L_2$** | `0.85` |
| **$K_3$** (`kb_obj_geo`) | `(obj, spatial_geo_rel)` | Top action/spatial `predicate` | **$L_3$** | `0.70` |
| **$K_4$** (`kb_geo_label`) | `(spatial_geo_rel,)` | Most frequent raw `geo_rel` | **$L_4$** | `0.50` |
| Canonical Fallback | `spatial_geo_rel` | Literal spatial preposition | **$L_5$** | `0.30` |

In addition, high-frequency non-geometric action triplets (`spatial_geo_rel == 'unknown'`, `rel_type == 'action'`) populate semantic rescue tables (`Lsem`, weight `0.65`) for distal visual interactions such as `looking at`, `feeding`, or `watching`.

---

## 3. Fine-Grained Alias & Triplet Enrichment ($T_1\text{--}T_2$)

During Stage 2 (`enrich_alias_triplets.py`), **CLIP ViT-B/32** resolves fine-grained entity aliases (`coco_alias_clip.txt`, e.g., `person` $\rightarrow$ `man`, `woman`, `tennis player`, `skateboarder`), dominant object colors (on **GrabCut**-segmented foreground masks), and torso clothing attributes (`shirt`, `suit`, `pants`, `jeans`, `dress`, `bikini` $\times$ 12 colors).

Using `knowledge_graph/raw_merged_triplets.csv.gz`, pairwise relations are upgraded through a two-tier corpus lookup subject to physical/geometric compatibility checks (`_is_forbidden`):

- **$T_1$ (`wt = 0.92`)**: Exact canonical match `(subj_class, obj_class)` in the merged triplet inventory (`count >= 2`), boosted by `+0.05` when consistent with the Stage-1 `spatial_geo_rel`.
- **$T_2$ (`wt = 0.80`)**: Alias-expanded match `(subj_alias, obj_class)`, `(subj_class, obj_alias)`, or `(subj_alias, obj_alias)`.
- **`KB_v2` Fallback**: Retains the Stage-1 geometric-semantic relation if no $T_1/T_2$ predicate passes physical/semantic verification.
- **`attribute` (`<ATTR>`)**: Attaches verified person-clothing nodes via `[person] wearing [color clothing]`.

---

## 4. Optional Neo4j Property-Graph Schema (`upload_to_neo4j.py`)

For interactive graph exploration and Cypher querying, `src/phase3_neo4j/upload_to_neo4j.py` indexes image-level scene graphs into Neo4j under the following hierarchical ontology:

```text
(:Root {name: 'root'})
  ├──[:HAS_DOMAIN]──> (:Domain {name: 'things' | 'stuff'})
  │                     └──[:HAS_CATEGORY]──> (:Category)
  │                                             └──[:HAS_CLASS]──> (:Class)
  └──[:HAS_ROLE]────> (:Role {name: 'foreground' | 'background' | 'objects'})

(:Image {name, split, img_h, img_w})
  └──[:HAS_OBJECT]──> (:Object {name, class, alias, label, FGS, BGS, area_ratio, ...})
                        ├──[:INSTANCE_OF]──> (:Class)
                        ├──[:ASSIGNED_TO]──> (:Role)
                        ├──[:HAS_ATTRIBUTE]─> (:Attribute {name: 'color::red' | 'alias::man'})
                        └──[:<PREDICATE> {edge_weight, triplet_level, triplet_count, spatial_geo_rel}]──> (:Object)
```

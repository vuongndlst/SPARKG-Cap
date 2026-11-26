# Salience Weights, Fixed Thresholds, and Compatibility Rules

All scalar parameters in **SPARKG-Cap** are derived from training-split corpus statistics (`statistics/`) or fixed deterministically prior to evaluation without test-split tuning.

---

## 1. Closed-Form Salience Criterion Weights

Loaded automatically from `statistics/derived_weights_coco_things_stuff_penalty_gamma.csv` ($w^{\text{fg}}$) and `statistics/derived_bg_weights_farcenter_penalty_gamma.csv` ($w^{\text{bg}}$):

$$\text{FGS}_i = w^{\text{fg}}_{\text{size}}\,s_{\text{size},i} + w^{\text{fg}}_{\text{center}}\,s_{\text{center},i} + w^{\text{fg}}_{\text{sem}}\,p^{\text{fg}}(c_i) + w^{\text{fg}}_{\text{rel}}\,s_{\text{rel},i}$$

$$\text{BGS}_i = w^{\text{bg}}_{\text{size}}\,s_{\text{size},i} + w^{\text{bg}}_{\text{center}}\,(1 - s_{\text{center},i}) + w^{\text{bg}}_{\text{sem}}\,p^{\text{bg}}(c_i) + w^{\text{bg}}_{\text{rel}}\,s_{\text{rel},i}$$

| Criterion | Raw Separation Weight ($w^{\text{fg}}_{\text{csv}}$) | Normalized Foreground ($w^{\text{fg}}$) | Raw Separation Weight ($w^{\text{bg}}_{\text{csv}}$) | Normalized Background ($w^{\text{bg}}$) | Definition |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Relative Area (`size`)** | `0.1670` | `0.1590` | `0.4070` | `0.3876` | Bounding-box area divided by image area $H \times W$ |
| **Spatial Centrality (`center`)** | `0.2018` | `0.1922` | `0.3917` | `0.3730` | Normalized Euclidean proximity to image center (inverted $1 - s_{\text{center}}$ for `BGS`) |
| **Semantic Prior (`semantic`)** | `0.6312` | `0.6012` | `0.2013` | `0.1917` | Training-caption mention probability $p^{\text{fg}}(c_i)$ / $p^{\text{bg}}(c_i)$ |
| **Interaction Degree (`relation`)** | `0.0500` (prior) | `0.0476` | `0.0500` (prior) | `0.0476` | Saturated bbox-overlap degree $\deg(i) / (\deg(i) + 5)$ |

---

## 2. Role Partitioning & Consistency Rules ($R_0\text{--}R_5$)

### Primary Threshold Rule
- **`fg` (Foreground)**: $\text{FGS}_i \ge \tau_{\text{fg}} = 0.55$ and $\text{FGS}_i \ge \text{BGS}_i$
- **`bg` (Background)**: $\text{BGS}_i \ge \tau_{\text{bg}} = 0.15$ and $\text{BGS}_i > \text{FGS}_i$
- **`objects` (Auxiliary)**: All remaining detected entities

### Deterministic Domain-Consistency Recheck ($R_0\text{--}R_5$)
- **$R_0$**: Demote `fg` $\rightarrow$ `objects` if detector confidence $\text{det\_conf} < 0.45$.
- **$R_1$ / $R_{1b}$**: Demote `fg` $\rightarrow$ `objects` or `bg` if class belongs exclusively to `stuff`.
- **$R_2$**: Promote `bg` $\rightarrow$ `objects` if class belongs exclusively to discrete `thing` instances.
- **$R_3$ / $R_4$**: Assign `objects` $\rightarrow$ `bg` if class belongs exclusively to amorphous `stuff`.
- **$R_5$**: Assign `stuff` region with $\text{area\_ratio} \ge 0.40$ unconditionally to `bg`.

---

## 3. Detection, Geometry, and Sparsification Thresholds

| Stage | Parameter | Value | Purpose |
| :--- | :--- | :---: | :--- |
| **Detection** | `BBOX_SCORE_THR` | `0.55` | Minimum Mask R-CNN confidence for `thing` instances |
| **Segmentation** | `STUFF_AREA_THR` | `500 px` | Minimum connected pixel area for DeepLabV3 `stuff` regions |
| **Deduplication** | `DUP_IOU_THR` / `DUP_IOMIN_THR` | `0.75` / `0.85` | Same-class duplicate suppression |
| **2D Geometry** | `_IN_COVER` / `_OVERLAP_IOU` | `0.95` / `0.10` | Enclosure (`in`/`contains`) and `overlap` IoU thresholds |
| **2D Geometry** | `_ON_VERT_GAP` / `_ON_HORIZ_OVER` | `0.25` / `0.25` | Vertical contact gap and horizontal overlap for `on` |
| **2D Geometry** | `_DIST_FACTOR` | `1.5` | Maximum normalized centroid distance before `unknown` |
| **Graph Density** | `MIN_EDGE_WEIGHT` | `0.70` | Retains $L_1\text{--}L_3$ edges in primary pass |
| **Graph Density** | `NEXT_TO_MIN_LEVEL` | `'L1'` | Restricts generic `next_to` edges to exact class-pair matches |
| **Graph Density** | `DROP_BG_BG_EDGES` | `True` | Eliminates `bg` $\leftrightarrow$ `bg` clutter edges |
| **Graph Density** | `MAX_DEGREE_PER_NODE` / `MAX_RELS_PER_IMG` | `3` / `30` | Per-node degree cap and per-image relation cap |
| **CLIP Enrichment** | `ALIAS_CONF_THR` / `CLIP_CHUNK_CONF_THR` | `0.45` / `0.27` | Minimum softmax confidence for zero-shot CLIP aliases |
| **CLIP Attributes** | `COLOR_CONF_THR` / `CLOTHING_CONF_THR` | `0.35` / `0.45` | Minimum confidence for GrabCut object color and torso clothing |

---

## 4. Physical & Semantic Compatibility Verification (`_is_forbidden`)

Before admitting any $T_1/T_2$ triplet edge `(subj, pred, obj)`:
1. **Contact Intersection**: Contact predicates (`wearing`, `holding`, `carrying`, `riding`, `eating`, `sitting on`, `lying on`) require strictly intersecting 2D bounding boxes (`IoU > 0`).
2. **Relative Scale**: `holding` rejects targets with $\text{area}(\text{obj}) > 2.5 \times \text{area}(\text{subj})$; `riding` rejects targets with $\text{area}(\text{obj}) < 0.10 \times \text{area}(\text{subj})$.
3. **Agent & Affordance Typing**:
   - `wearing`, `holding`, `driving`, `riding` require `subj == person`.
   - `flying` restricts subjects to `{bird, airplane, kite, frisbee, sports ball}`.
   - `swimming` / `floating` restricts subjects to `{boat, bird, person, dog, bear, elephant, surfboard}`.
   - `parked` restricts subjects to vehicles `{bicycle, car, motorcycle, bus, train, truck, boat, airplane}`.
   - `sitting on` / `lying on` restricts subjects to animate classes and rejects small handheld objects (`cup`, `fork`, `cell phone`, etc.).

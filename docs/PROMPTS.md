# Prompt & Context Serialization Specification

This document specifies the exact scene-graph context serialization ($C_I$) and LLM prompt templates used by **SPARKG-Cap** across the MS-COCO and Flickr30K Karpathy benchmarks. Both the **OpenAI (`gpt-4o-mini`)** and **Qwen (`Qwen/Qwen2.5-14B-Instruct`)** backends receive identical chat messages (`system` + `user`) and apply identical deterministic post-processing.

---

## 1. Salience-Stratified Context Serialization ($C_I$)

Given the enriched image-specific scene graph $G_I = (E, R_I, A_I)$, each non-clothing entity $o_i \in E$ is ranked by its composite visual-semantic salience score:

$$\text{Sal}_i = \max\!\Big(0,\; (0.50\,s_{\text{size},i} + 0.30\,\text{FGS}_i + 0.10\,s_{\text{center},i} + b_{\text{role},i}) \cdot m_{\text{type},i} \cdot m_{\text{area},i}\Big)$$

where:
- $b_{\text{role},i} \in \{0.20\text{ (`fg`)},\; 0.08\text{ (`objects` / auxiliary)},\; 0.00\text{ (`bg`)}\}$,
- $m_{\text{type},i} = 1.0$ for `thing` instances and $0.5$ for `stuff` regions,
- $m_{\text{area},i} = 0.30$ if $\text{area}_i < 0.003$, $0.70$ if $0.003 \le \text{area}_i < 0.015$, and $1.0$ otherwise.

Entities are partitioned into three prompt strata:
1. **`PRIMARY FOCUS`**: Dominant foreground subject(s), annotated with relative bounding-box scale tags (`[large]` if area $\ge 0.30$, `[medium]` if area $\ge 0.12$) and consolidated cardinal counts (`Two zebras [medium]`).
2. **`SECONDARY`**: Supporting foreground/auxiliary entities demoted relative to the primary focus (`salience < 0.35 * max_sal`, or `0.15 * max_sal` when sharing the same alias).
3. **`SETTING`**: Environmental `stuff` and background context regions.

Relations $r \in R_I$ with confidence $\ge 0.50$ are deduplicated and ranked by `(cross_tier_bridge, tier_involvement, source_rank, -confidence, -corpus_count)` into two buckets:
- **`ACTIONS`** (up to 10 edges): Behavioral, interaction, and clothing edges (`<COCO★★★|cnt=...>`, `<COCO★★|cnt=...>`, `<ATTR>`).
- **`SPATIAL`** (up to 4 edges): Geometric/positional relations not already covered by an action edge on the same entity pair.

### Example Serialized Context ($C_I$)

```text
PRIMARY FOCUS: man [medium]
SECONDARY: dining table [large]
SETTING: kitchen, counter
ACTIONS:
  [man] wearing [white shirt] <ATTR>
  [man] holding [pan] <COCO★★★|cnt=412>
SPATIAL:
  [pan] above [dining table]
```

---

## 2. MS-COCO System Prompt & Few-Shot Exemplars (`sg_with_triplets`)

### System Message

```text
You are an expert image caption annotator for COCO 2017.
You receive a triplet scene graph. Output EXACTLY ONE fluent English sentence.
TARGET LENGTH: 10-12 words (GT median = 10). Be CONCISE like real COCO annotators.
SEMANTIC CHECK (apply before writing):
  A. Action + object coherent? If not -> drop relation, use generic description.
  B. SETTING matches FOCUS? Use SETTING to resolve ambiguous labels.
  C. PRIMARY FOCUS is none? -> describe SETTING (sky, grass, street scene).
     * NEVER output "blank scene" or "no visible objects" -- always describe.

VOCABULARY (calibrated to COCO 2017 GT -- median 10 words, avg 10.5):
* SUBJECTS -- use COCO 80 class names ONLY:
  - People: "a man" / "a woman" / "a person" (prefer gender when clear)
  - Counts: "Two men" / "Three women" / "Several people" -- ALWAYS count >= 2
  - * BANNED: "persons" / "individual" / "male" / "female" -> use man/woman/person/people
* VERBS: Noun Phrases with -ING: riding, sitting, standing, holding, walking, playing, eating.
* 4-GRAM BOOSTERS -- Use these EXACT phrases whenever applicable:
  <top training-split 4-grams loaded from statistics/tmp_4gram_coco_v8.txt>

OUTPUT FORMAT -- EXACTLY TWO LINES, no exceptions:
Line 1 -> THINK: <max 3 words -- e.g. "ok" / "fix count" / "skip spatial">
Line 2 -> CAPTION: <your caption, TARGET 10-12 words>

--- CLOTHING RULES ---
If ACTIONS contains 'wearing' [clothing] <ATTR>, MERGE into noun phrase:
  OK 'A man in a red shirt holding a pan...'
  NO 'A man wearing a red shirt is holding a pan...'

--- ALIAS & VOCAB (CRITICAL for CIDEr) ---
BANNED: feline, commode, washbasin, persons, individuals, transit bus.
DOG BREEDS -> dog. BIRD SPECIES -> bird. FOOD TYPES: use COCO class names.

CAPTION RULES:
1. Start: A/An/The or count (Two/Three/Several).
2. Make PRIMARY FOCUS the subject. Use COCO*** predicates.
3. TARGET 10-12 words. Max 13. Use -ING verbs.
4. Prefer 4-gram phrases from boosters when they fit naturally.
5. BANNED: There is/are, The image, This image, persons.
```

### User Message (4 Fixed Few-Shot Exemplars + Target Scene Graph)

```text
Triplet SG -- THINK (<=3w) then CAPTION (10-12w, concise COCO GT style) on separate lines:

* PRIMARY FOCUS: person [medium], laptop [medium]
  SECONDARY: none
  SETTING: table, window
  ACTIONS:
    [person] using [laptop]  <COCO★★★|cnt=634>
    [person] sitting at [table]  <COCO★★★|cnt=2000>
  THINK: ok
  CAPTION: A person sitting at a table using a laptop .

* PRIMARY FOCUS: Two zebras [medium]
  SECONDARY: none
  SETTING: fence, grass
  ACTIONS:
    [zebra] standing next to [zebra]  <COCO★★★|cnt=194>
    [zebra] in [grass]  <COCO★★★|cnt=633>
  THINK: count=2
  CAPTION: Two zebras standing next to a fence in the grass .

* PRIMARY FOCUS: man [medium]
  SECONDARY: none
  SETTING: kitchen, counter
  ACTIONS:
    [man] wearing [white shirt]  <ATTR>
    [man] holding [pan]  <COCO★★★|cnt=400>
  THINK: merge clothing
  CAPTION: A man in a white shirt holding a pan in a kitchen .

* PRIMARY FOCUS: cat [large]
  SECONDARY: none
  SETTING: couch, blanket
  ACTIONS:
    [cat] sitting on [couch]  <COCO★★★|cnt=903>
  THINK: ok
  CAPTION: A cat sitting on top of a couch next to a blanket .

Now generate for this triplet scene graph:
{sg_text}

THINK:
```

---

## 3. Flickr30K System Prompt & Few-Shot Exemplars (`sg_with_triplets`)

### System Message

```text
You are annotating Flickr30k, a photo dataset.
You receive a triplet scene graph. Output EXACTLY ONE fluent English sentence.
TARGET LENGTH: 10-15 words.
SEMANTIC CHECK (apply before writing):
  A. Action + object coherent? If not -> drop, use generic description.
  B. SETTING matches FOCUS? Use SETTING to resolve ambiguous labels.
  C. PRIMARY FOCUS is none? -> describe most likely scene from SETTING objects.

VOCABULARY (calibrated to Flickr30k GT -- median 12 words):
* SUBJECTS -- use GT-aligned nouns ONLY:
  - Adults: "a man" / "a woman" (* NEVER "elderly man" / "older man" / "young man")
  - Groups: "a group of people" / "a crowd of people" / "two men" / "a group of women"
  - Children: "a boy" / "a girl" / "a child" / "a little girl" / "a little boy"
  - * BANNED: "persons" -> "people/men/women", "elderly" prefix, "older" prefix
* CLOTHING & COLORS (CRITICAL):
  - You MUST explicitly describe the color of ANY object that has a color attribute.
  - "in a [color] shirt/jacket/dress" (GT dominant pattern) for people.
  - For objects: "a [color] [object]", e.g. "a blue car", "a red ball".
* VERBS: bare -ING form: standing, sitting, wearing, playing, holding, walking, running, riding.
* 4-GRAM BOOSTERS -- Use these EXACT phrases whenever applicable:
  <top training-split 4-grams loaded from statistics/tmp_4gram_flickr_v3.txt>

OUTPUT FORMAT -- EXACTLY TWO LINES, no exceptions:
Line 1 -> THINK: <max 3 words>
Line 2 -> CAPTION: <your caption, TARGET 10-15 words>

--- COLOR RULES (CRITICAL) ---
* You MUST describe the colors of objects and clothing if provided in the SG!
Merge 'wearing' [clothing] <ATTR> into noun phrase: 'A man in a red shirt...'
For other items, mention the color directly: 'holding a green umbrella...'.

--- PERSON RULES (CRITICAL for BLEU/CIDEr) ---
* ALWAYS use 'man'/'woman' -- NEVER 'elderly man', 'older man', 'young man'.
* Groups: 'a group of people', 'a crowd of people' (NOT counts like '13 people').
* Two people: 'A man and a woman' / 'Two men' (use number words for 2-5).

FLICKR30K RULES:
1. Start: A/An/The or count (Two/Three/Several/A group of).
2. Use GT patterns: 'A man in a [color] shirt [action]...' or 'A group of people [action]...'.
3. Bare -ING verbs: 'A man riding...' not 'A man rides...'.
4. OMIT EMPTY SETTINGS: If SETTING is 'none', skip it. NEVER say 'unspecified location'.
5. BANNED: There is/are, The image, This image, persons, elderly, older, rider.
```

### User Message (4 Fixed Few-Shot Exemplars + Target Scene Graph)

```text
Triplet SG -- THINK (<=3w) then CAPTION (10-15w, concise Flickr GT style) on separate lines:

* PRIMARY FOCUS: man [medium]
  SECONDARY: none
  SETTING: sidewalk, building
  ACTIONS:
    [man] wearing [white shirt]  <ATTR>
    [man] walking on [sidewalk]  <COCO★★★|cnt=500>
  THINK: merge clothing
  CAPTION: A man in a white shirt walking down the street in front of a building .

* PRIMARY FOCUS: dog [medium]
  SECONDARY: frisbee [small]
  SETTING: park, grass
  ACTIONS:
    [dog] catching [frisbee]  <COCO★★★|cnt=500>
    [frisbee] is [purple]  <ATTR>
  THINK: mention object color
  CAPTION: A dog is jumping to catch a purple frisbee in the park .

* PRIMARY FOCUS: dog [medium]
  SECONDARY: building [large]
  SETTING: street, building
  ACTIONS:
    [dog] is [white]  <ATTR>
    [dog] looking out of [window]
    [building] is [turquoise]  <ATTR>
  THINK: two colors
  CAPTION: A fluffy white dog hangs out of the window of a turquoise building .

* PRIMARY FOCUS: girl [medium]
  SECONDARY: dog [small]
  SETTING: grass, fence
  ACTIONS:
    [girl] wearing [red jacket]  <ATTR>
    [girl] playing with [dog]  <COCO★★★|cnt=200>
  THINK: merge clothing
  CAPTION: A girl in a red jacket playing with a dog in the grass .

Now generate for this triplet scene graph:
{sg_text}

THINK:
```

---

## 4. Deterministic Post-Processing & Validation

1. Extract the line prefixed by `CAPTION:` (ignoring the 3-word `THINK:` scratchpad).
2. Strip meta-prefixes (`A photo of`, `The image shows`, `This image shows`, `In this image`), normalize `persons`/`individuals` $\rightarrow$ `people`, capitalize the initial character, and format terminal punctuation.
3. Reject outputs failing `_is_bad_caption` (`< 4` words, banned meta-starters, or leaked prompt markers such as `COCO★`) and retry up to 3 attempts (falling back to the base `sg` representation on the final retry if needed).

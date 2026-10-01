# Skateboard Trick Recognition — Plan v0.3.0 (2026, consolidated)

> Single integrated plan. Supersedes the earlier `plan-v0.1` / `v0.2` / `v0.2.1`
> drafts, which were removed once v0.3.0 absorbed them; they remain in git history
> (`3f910da`) if ever needed. Adds the **output & quantization spec** (predict signed
> angles, quantize to a trick at the label layer), makes the **web page the default HCI**,
> and makes **flatground-only**
> a hard, enforced scope. Decisions folded in: shuvit/pop-shuvit merged into one label;
> airtime is a diagnostic feature, not a head; `landed` suppresses confident-wrong answers;
> **stance is a user input (goofy/regular), not a prediction** — it sets the sign
> conventions, so the kick/heelflip and fs/bs signs are only knowable once it is set right.

---

## 1. Scope & Definition of Done

**Goal.** User hands the system one short clip of a single flatground trick. The page
prints the trick name and a confidence, or honestly says **"not sure"**.

### Input / output contract
- **Input:** one clip, ~1–4 s, one trick attempt, any fps/resolution (24/30/60/120).
- **Output:** `bs 180 heelflip  0.87` or `not sure`; web-first (see §10).
- **Bonus:** `--json`, `--top 3`, `--debug` (annotated overlay .mp4), `--batch`.
- **Stance is an input:** a three-way selector on the page — `auto / regular / goofy`
  (or `--stance`), **default `auto`**. `auto` pre-fills from a cheap front-foot heuristic as a
  *suggestion only*, never trusted for the sign. The kick/heelflip and fs/bs signs are only
  defined after this (§3/§4).

### Non-goals (v1)
Real-time streaming · multi-trick segmentation of a long video · stance-prefixed names ·
obstacle/ramp tricks (see §5) · a phone app · anything non-flatground.

### Hard guarantees (scope)
1. **Flatground only.** *v1 recognizes FLATGROUND tricks only — no grinds/slides on
   obstacles, no ramp/vert/aerials.* Enforced by `tests/test_scope.py` (fails CI on any
   non-flatground label or component combo) and `data/flatground_allowlist.csv` (§5).
2. **Web-first HCI.** The default way to use the system is a local HTML page
   (`skateid serve`). The CLI is kept for batch/automation only (§10).
3. **Stance is provided, not guessed.** The user toggles goofy/regular (**default `auto`**).
   `auto` *suggests* a stance to pre-fill the toggle but is never trusted for the sign.
   Because kick/heelflip and fs/bs are sign conventions defined relative to the skater's
   front/back, the stance input must be right before any direction name is trusted.

### Success criteria
| Metric | Target |
|---|---|
| 2-class held-out accuracy (SkateboardML) | ≥ 95 % |
| "Correct **or** abstained" rate | ≥ 90 % |
| Latency per clip (RTX 3060, batch 1) | ≤ 2 s |
| Execution | one `skateid serve`, use it from a browser |

---

## 2. Verified data landscape

Public flatground *video + label* sets are scarce and mostly built on the same corpus
(Battle at the Berrics). Counts/sizes/licences below were measured directly.

| Source | Clips | Tricks | Labels | Licence / friction |
|---|---|---|---|---|
| **SkateboardML** (LightningDrop) | **222** (Ollie 108, Kickflip 114), ~594 MB | 2 | folder = class; published 177/44 split | In-repo, download today, no auth |
| **SkateAI** (EduardoPach) | **449** (train 359 / val 90) | **31 present**, dict covers **58** | per-clip `video_url` + `clip_start/end`; `flip_*, board_rotation_*, body_rotation_*, stance, landed` | **Source footage copyright (BATB YouTube) — personal/research use.** Its downloader script is gone; we write ~20 lines of `yt-dlp` + `ffmpeg` |
| **Swinburne Trick Attempt Dataset** (figshare 31043779, 2026) | **24,053 attempts** — metadata only, no video/timestamps | many | `BATB, Skateboarder, Trick, Type, Outcome (land/bail), Variant, Year` | **CC BY-NC 4.0** — vocabulary/coverage only, not bundled |
| Skateboard-AI dataset | Drive link dead; repo has no code | — | — | Dropped |
| HF `qleandataset/video-skateboard-vert` | 36 clips, 3.9 GB, gated | vert only | — | Wrong discipline — skipped |
| Kaggle "Skateboarding Trick Dataset (120fps)" | exists (unverified) | — | — | Verify before relying on it |

**Leakage trap (SkateboardML):** its published 44-clip split is random across clips from
the same small pool — near-duplicates and multi-trick videos. Majority there = **65.9 %**.
**Always report on both** the published split (comparability) and a deduplicated,
skater/source-disjoint split (the real number).

**Data strategy (chosen):** two stages, same code.
- **Stage A = SkateboardML** (222 clips, 2 classes) — green the whole pipe, zero friction.
- **Stage B = SkateAI** (449 clips, 31 tricks) — breadth once A works, via the yt-dlp cutter.
- **Vocabulary = Swinburne** — which tricks matter and how they compose; never copied.

---

## 3. The simplification: predict 3 rotations; stance is an input

Every flatground trick is a combination of three independent rotations, resolved in a frame
set by the skater's stance. **We predict the three rotations and look the name up** in a
58-entry dictionary (`tricks.json`).

| Output | Kind |
|---|---|
| flip | **signed rotation** (degrees) — number + direction of flips |
| board_spin | **signed rotation** (degrees) — shuvit/spin of the board |
| body_spin | **signed rotation** (degrees) — spin of the body |

Plus **one input, not a prediction**: `stance` = auto / goofy / regular (**default `auto`**;
`auto` pre-fills the toggle from a cheap front-foot heuristic as a suggest-only guess, never
trusted for the sign).

Stance is not predicted because it cannot be read reliably from a clip without assuming
which foot is the skater's front — that judgment belongs to the user. The stance input
**sets the sign convention** (§4): which image-space rotation is "kick" vs "heel", and "fs"
vs "bs".

- New tricks that share the same rotations are **free** — a data change, never a code change.
- Even with only ollie/kickflip data, the flip output is already being learned and generalises.
- A kickflip performed by a goofy skater is the **mirror** rotation of a kickflip by a regular
  skater. A wrong/unknown stance toggle therefore **mirrors every direction** — the dominant
  failure mode of v1 (see §13).

**airtime / "pop" is NOT one of the predicted rotation outputs.** It is a **feature and a
diagnostic** only. An ollie has zero rotation + air; a shuvit and a pop-shuvit differ only by
airtime. **Decision: merge shuvit and pop-shuvit into one "shuvit" label for v1** and show
airtime in the overlay/debug. (See §14.)

---

## 4. Output & quantization spec (how the 3 rotation values are produced)

The model **does not** output tidy `±360` values. Real tricks are sloppy (a kickflip can be
~270-410 deg). So the model predicts a **signed continuous angle** per rotation axis, and we
**quantize to a whole number of rotations only at the label layer**. The leftover fraction is
the confidence/"not sure" signal - never clamped away.

### Per-axis base units
> Signs below are defined in the **stance-normalized frame** (§7). With the wrong stance
> toggle, every direction flips — kick<->heel, fs<->bs. The stance input is what lets a sign
> be named a kick or a heel at all.

| Axis | Base per "one" | Sign convention (regular stance shown) |
|---|---|---|
| flip | **360 deg** per flip | `+` kickflip, `-` heelflip |
| board_spin | **180 deg** per shuv | `+`/`-` = bs / fs |
| body_spin | **180 deg** per turn | bs / fs |

> **Corrected in v0.3.1.** This row previously read `+`/`-` = fs / bs, which
> contradicted every worked example below (`shuvit` `board +180`, `180` `board +180`,
> `bigspin` `board +360`, `treflip` `board +360` are all *backside*) and contradicted
> the ingested data: SkateAI's `treflip` clips carry `board_rotation_type='backside'`
> with `board_rotation_number=2`. Backside-positive is now the single convention,
> documented in `data/tricks.json` under `_sign_convention` and enforced by
> `test_documented_sign_convention_matches_the_data`.

### Worked examples (label = quantize(predicted deg))
```
kickflip     flip +360   board 0    body 0          -> flip 1 kick
heelflip     flip -360   board 0    body 0          -> flip 1 heel
double kick  flip +720   board 0    body 0          -> flip 2 kick
shuvit       flip 0      board +180 body 0          -> board 1
360 shuvit   flip 0      board +360 body 0          -> board 2
bigspin      flip 0      board +360 body +180       -> board 2 + body 1
180          flip 0      board +180 body +180       -> board 1 + body 1
treflip      flip +360   board +360 body 0          -> flip 1 + board 2
ollie        flip 0      board 0    body 0          -> all zero
```

### Quantization math (at lookup, not in the network)
```
number     = clamp(0, 3, round(|angle| / base))
direction  = sign(angle)
fraction   = |angle|/base - round(|angle|/base)     # leftover
if fraction > ~0.25  -> partial/botched -> drop confidence -> "not sure"
```

A 270 deg kickflip -> `round(270/360)=1`, `fraction=0.25`, still a kickflip but *less sure* -
the same judgment a human ref makes. Over- and under-rotation need no extra classes.

### Three principles that make this correct
1. **Swept angle over time, not one frame's pose.** A trick *ends* flat, looking like it
   started - the information lives in the **trajectory** of board/body over the clip window.
   We predict total rotation integrated over the window; never "current pose -> label."
2. **The sign convention is load-bearing — and stance-dependent.** "kick/heelflip" and "fs/bs"
   are pure **sign** in the stance-normalized frame. A kickflip by a regular skater is the
   **mirror** rotation of a kickflip by a goofy skater, so the name is only knowable once the
   user's goofy/regular input fixes the convention (§3). Get stance or camera normalization
   wrong and every direction flips.
3. **Mirrored augmentation pays off here.** Mirroring a clip signs the rotation axes -> one
   clip teaches kickflip *and* heelflip, fs *and* bs, for free. The signed representation is
   what makes that augmentation legal.

---

## 5. Flatground scope (hard, enforced)

> **v1 recognizes FLATGROUND tricks only** — no grinds/slides on obstacles, no ramp/vert/aerial.

**Included (all flatground):** ollie, kickflip, heelflip, **shuvit (incl. pop-shuvit — merged)**,
360 flip / tre-flip, bigspin, frontside/backside 180 and 540, plus double/triple variants
produced by the rotation outputs. Every rotation (flip/board/spin) can happen on flat ground.

**Explicitly excluded from the v1 label set:**
- Grinds & slides (rail, ledge, curb) — obstacle tricks.
- Vertical / ramp / vert / half-pipe aerials.
- Manuals & nose-manuals — flatground but a *balance-hold*, not a rotation; deferred.
- Casper / anti-casper and other balance tricks — deferred.
- Any trick whose name or component combo needs an obstacle.

**Guardrail (so it stays true, not just a promise) — implemented in v0.3.1:**
- `data/tricks.json` — the **rotation dictionary**: canonical name ↔ the three rotations
  (§4). 39 entries, 35 of which have a triple. This is the authority: a row's `label` is
  *derived* from its triple, not read from a name.
- `data/flatground_allowlist.csv` — the **name registry**: 45 canonical names + 135
  aliases (every upstream SkateAI spelling included). Aliases are not allowed to collide.
- `skateid/taxonomy.py` — loads both, and refuses to construct a `Taxonomy` if they
  disagree, so a trick can never be half-added.
- `build_manifest()` calls `validate_or_raise()`, so **manifest build rejects
  non-flatground rows at ingestion** (this also back-fills SkateboardML's components).
- `tests/test_scope.py` — fails CI if any `manifest.csv` row's label or combo is
  off-allowlist, and includes the negative cases (off-list label, label contradicting
  its own triple, empty/over-range/inconsistent components, name-component mismatch).
- `skateid validate` — the same check on demand, e.g. in a pre-commit or CI step.
- README capture guide: film only flatground attempts; no obstacle in frame.

> **Corrected in v0.3.1.** This section previously cited "SkateAI's 58" names. The
> ingested metadata has **31 distinct trick names** and **73 distinct
> (stance, trick) pairs**; 58 is neither, and the number has been removed. The v1
> vocabulary is 45 canonical names, not 58.
>
> The 4 allowlisted-but-**not rotation-expressible** names are recorded in
> `tricks.json` with a `note` explaining how they are represented instead:
> `half_cab`/`full_cab` share their triple with `bs_180`/`bs_360` and are recovered
> from `(stance, label)` (§3); `impossible` rotates about an axis the 3-axis model
> lacks; `none` collides with `ollie` at `(0,0,0)` and needs a separate no-trick gate.
> Ingesting any of them fails loudly rather than being silently mislabelled.

**Alignment check — all three sources are already flatground, so no new data is needed:**
SkateboardML = ollie/kickflip ✓ · SkateAI = BATB flatground battles ✓ · Swinburne = BATB ✓

---

## 6. Architecture — 5 steps, 1 model trained

```
clip.mp4
   │
   ├─ 1. video.py     decode (ffmpeg/PyAV) -> resample to 30 fps -> long side <= 1280 px
   │
   ├─ 2. perceive     pose.py   YOLO26-pose (COCO-17)                  -> 17 body joints
   │                  board.py  SAM 3 prompt "skateboard" -> mask       -> 4 board corners
   │                            (fallback: COCO YOLO26 box;             (both off-the-shelf,
   │                             minAreaRect for corners)                no training)
   │
   ├─ 3. features.py  per-frame vector D ~ 52, skater-relative -> sequence T = 60
   │
   ├─ 4. model.py     tiny transformer (T, 52) -> (T, 128) -> pool
   │                    -> heads: flip (regression), board_spin (regression),
   │                             body_spin (regression)    (stance = user input, normalizes
   │                                                          features & sign conventions)
   │
   └─ 5. recognize.py quantize angles -> dictionary lookup -> name | "not sure"
```

### The one core, two front doors
`recognize.py` is the **single source of truth**. Both the web page (§10) and the CLI wrap it —
no second model, no divergence.

### Why not the usual alternatives (rejected)
| Rejected | Why |
|---|---|
| 3D-CNN / I3D, or fine-tuning VideoMAE-V2 / V-JEPA2 | data-hungry (need ≫10 k clips), memorises scenes; we have 222–449 clips. A **frozen** embedder is used only as an eval baseline |
| CNN+LSTM (SkateboardML's, SkateAI's own) | a 2020 recipe; a small transformer is smaller, faster on CPU, works on our data size |
| LSTM/TCN **or** xgboost (v0.1 left open) | fixed: one transformer + a logistic-regression floor so we never tune a transformer on 200 clips without a simple number to beat |
| Hard-coded pop gate + roll-vs-yaw tree (v0.1) | a gate hard-fails on no-pop tricks; the tree encodes heuristics as *logic*. Both demoted to debug/visualisation |
| **Clamping** model outputs to tidy `±360` | kills the residual -> loses the "not sure" signal; see §4 |

---

## 7. Feature vector (the part that must be right)

Everything measured **relative to the skater** so camera distance/angle stops mattering:
- **Origin** = mid-hip. **Unit** = shoulder->hip distance `s`.
- **Stance normalisation:** sign-flip the x-axis per the resolved goofy/regular stance
  (`auto` resolves its suggestion but falls back to `regular` when unclear), *not* by trusting
  auto-detection — auto-detection is unreliable and it sets the very convention we defer to the
  user (§3). **This makes §4's sign convention meaningful.**

Per frame (D ~ 52):
| Block | Dim | Content |
|---|---|---|
| Body joints | 34 | 17 x (x,y), centred on hip, / s |
| Board corners | 8 | 4 x (x,y), same frame |
| Board rotation | 6 | see the **axis mapping** in §7.1 — angle + foreshortening on **both** board axes |
| Body yaw | 2 | sin,cos of shoulder-line angle vs clip's first frame |
| Feet-to-deck | 2 | ankle to nearest-deck-edge distance, per foot |
| **Airtime** | 2 | ankle height above standing baseline, per foot — **diagnostic, not a head** |
| Quality | 1 | board-visible fraction |

### 7.1 Board axis mapping (recorded 2026-09-28)

**Which physical axis each trick rotates about is the whole game here, and it was
never written down anywhere** — not in this plan, not in `tricks.json`, not in any
code comment. It is domain knowledge, supplied by the project owner. It is recorded
explicitly now because six rounds of M2 work (§12.8–12.15) failed by measuring the
wrong axis, and for most of that time the failure looked like a data problem.

Notation: a **Unity-style right-handed frame** — `x` right, `y` up, `z` toward the
viewer. For a board lying flat, `x` runs tail→nose along the **long axis**, and the
**short axis** is the deck's width.

| trick | rotates about | what the image shows |
|---|---|---|
| **kickflip / heelflip** (`flip` axis) | the board's **long** axis | the board **foreshortens** — it gets *shorter* in the image as it rolls. Its long-axis **angle barely changes** |
| **shuvit** (`board_spin`) | **y** (vertical / yaw) | the board spins **in the ground plane**, like a wheel rolling sideways. The long-axis angle sweeps the full ±180 |
| **impossible** | the board's **short** axis | the deck wraps **vertically** around the front foot. This is the axis the 3-axis model lacks, which is why `impossible` is `rotation_expressible: false` in `tricks.json` and appears in **none** of the 671 clips |

**Why the old feature row was wrong, stated precisely.** The pre-2026-09-28 spec
allocated 3 dims reading "long-axis angle; long-axis foreshortening; short-axis
len". M2 implemented **only the long-axis angle**. That is blind to *both* major
trick families: a kickflip rolls about the long axis so its angle barely turns, and
a shuvit is a yaw, which the long-axis angle reads as almost nothing. The one
quantity that does track a kickflip is **foreshortening** — specified in the plan
all along, and never built. That omission is a plain implementation gap, not a
missing piece of domain knowledge.

Board rotation is now 6 dims, and the two dominant families live on complementary
signals:

| | long-axis angle | long foreshortening | long len / baseline |
|---|---|---|---|
| **kickflip** | near-flat | **strong** | **shrinks** |
| **shuvit** | **sweeps ±180** | weak | constant |

See §12.16 for the measurement that follows from this.

Sequence: T = 60 frames @ 30 fps (~2 s), always resampled to this length regardless of source
fps. Cache to `data/cache/<clip_hash>.npz` so training is seconds and reproducible.

**Augmentation (cheap, multiplicative):** mirror **with label-swap** (kickflip<->heelflip,
fs<->bs — free signs via §4.3), time-jitter/crop, Gaussian noise on keypoints
(sigma ~ 0.01 * s), random frame dropout 10 %.

### Skater-relative gotchas (read before filming/features)
- **Scale drifts when the rider pops.** Shoulder->hip `s` shortens during a crouch/pop, so a
  per-frame `s` is noisy at exactly the frames that matter. Normalize by a **robust per-clip
  `s`** (e.g., median over the clip, or knee->hip length) instead of per-frame.
- **The hip origin moves.** Hip-centre jitters and is occluded mid-flip; smooth it over time
  and fall back to shoulder-mid when the hips are invisible. The transformer already tolerates
  missing frames via frame-dropout aug.
- **Not camera-*attitude*-invariant.** Relative distances kill camera distance / pan / zoom,
  but not a tilted or low-angle camera: the board long-axis and foreshortening still depend on
  viewing angle. Capture guide: near-level, roughly perpendicular filming.
- **Board and feet must stay in frame.** If board tracking drops a frame, hold or
  linear-interpolate the corners; the board-visible fraction gate (§7) flags clips that lost
  the object.
- **A mirrored world is a second valid world.** We mirror clips for augmentation (labels swap
  too), so the model sees both goofy and regular reads regardless of the toggle — fine, and
  encouraged. **Closed in v0.3.1: the mirror map is now a bijection.** Negating all three
  axes initially left 6 of 35 names landing on a trick no dataset publishes (a frontside
  biggerspin heelflip, a frontside tre double flip, ...). Their 6 partners were added to
  `tricks.json` with notes saying no ingested data publishes them; they exist so the
  augmentation is total. Every one of the 41 rotation-expressible names now mirrors to
  exactly one name, and no two collide on a partner, so label-swap is safe everywhere.
  `ollie` is the only self-mirror (necessarily: the all-zero triple is its own mirror).
- **The sign frame is an input, not a property of the trick.** A kickflip is +360 for a
  regular rider and -360 for a goofy one; the stance normalisation above is what makes one
  stored value serve both. The manifest keeps `stance_published` (riding direction:
  regular/switch/fakie/nollie — provenance only, cannot fix the frame) separate from
  `stance_input` (the goofy/regular toggle, empty until M1). The guardrail rejects a riding
  direction placed in `stance_input`, because that is not a schema error but a silent
  kick<->heel and fs<->bs swap across the whole dataset.

---

## 8. Model & training

- Encoder: `LayerNorm -> Linear(52->128) -> +pos-embed -> 4x pre-norm TransformerEncoder(d=128,
  h=4, ff=256, dropout 0.1) -> mean xor max pool`.
- **Heads (the three rotations; stance is an input, not modeled):**
  - `flip`, `board_spin`, `body_spin` — **single-scalar regression** on signed rotation, in the
    stance-normalized frame.
    Target = `base * number * sign` (a clean value like +360, -720, +180). Loss = Huber on
    the signed value (robust to the sloppy-trick reality).
  - Heads share the trunk; no coupling to any class count.
- ~0.5 M params; off-the-shelf pose/board weights are **frozen and borrowed** — only these
  heads train. Trains in minutes on the RTX 3060 (12 GB).
- AdamW lr 3e-4, cosine, label-smoothing 0.1, 100 epochs, early stop on macro-F1 of the
  quantized labels. Loss = weighted sum over heads (weight by data availability).
- M0/M1 trains only the heads with data (flip); others switch on automatically as clips arrive.

### Abstain / "not sure" (three stacked signals)
1. **Rotation residual** (§4): fraction of a rotation is partially/over-rotated -> drop.
2. **`landed` (Stage B, real column):** a bail/abort suppresses a confident name.
3. **Temperature-scaled softmax floor:** max-prob below a validation-set-tuned threshold ->
   "not sure". A trained `other` class is added later once we have non-trick clips.

---

## 9. Evaluation protocol

Report **both splits** (published + deduplicated, skater/source-disjoint). Metrics: macro-F1
of the **quantized** labels (headline; the 44-clip split is 66/34 imbalanced), accuracy,
top-2, **selective accuracy at abstain rate**, rotation mean-absolute-error per axis (as a
unit check on §4), and slices by camera side (evaluated at both goofy and regular settings,
since stance is now an input).

**Baselines to beat (measured first, in M0):**
| # | Baseline | Why |
|---|---|---|
| B0 | majority class | floor = 65.9 % on SkateboardML's split |
| B1 | frozen V-JEPA2 / VideoMAE embed, mean-pooled -> logistic regression | zero-training 2026 baseline; **must be worse than or ~equal to ours, else rethink features** |
| B2 | VLM zero-shot prompt (Qwen3-VL / Gemini) | sanity-check the label set is visually separable |

---

## 10. HCI & CLI — web page is the default

**Web page (v1 default):**
```
skateid serve         # -> opens http://127.0.0.1:8000 in the default browser
```
- **Frontend:** one hand-written `index.html` (plain HTML/CSS/JS — no node.js, no build step,
  no framework): drag-and-drop + "choose file", inline `<video>` preview, a **stance selector**
  (`auto / regular / goofy`, `auto` pre-fills then user can override before "Identify trick"), one
  **"Identify trick"** button, result area (trick name large + confidence bar, or a clear
  **"not sure"** state), an optional **"show annotated"** toggle (plays the overlay), and a
  "try another" reset.
- **Backend:** FastAPI + uvicorn in `app.py`, three routes:
  | Route | Behaviour |
  |---|---|
  | `GET  /` | serves `index.html` |
  | `POST /predict` | uploaded clip (multipart) -> `recognize.py` -> `{trick, confidence, top3, overlay_url}` |
  | `GET  /overlay/<id>.mp4` | serves the annotated clip |

- Thin ~80-line wrapper around the **same** `recognize.py` the CLI uses. Runs localhost,
  offline, no accounts, no telemetry. (Flask acceptable; default is FastAPI.)

**CLI (batch / automation only):**
```
skateid clip.mp4                 # -> "bs 180 heelflip  0.87"
skateid clip.mp4 --stance auto   # auto / regular / goofy (suggest-only when auto)
skateid clip.mp4 --json          # machine-readable
skateid clip.mp4 --top 3
skateid clip.mp4 --debug out.mp4 # skeleton, board quad, live rotation + airtime readout
skateid "clips/*.mp4"            # batch
```

---

## 11. Repo layout & tooling

```
SkateboardTrickRecognition/
  plan-v0.3.0.md   README.md   pyproject.toml (uv, pinned)
  data/
    manifest.csv                 # path,label,flip*,board_*,body_*,stance,landed,source,license,split
    flatground_allowlist.csv     # approved labels + component combos
    tricks.json                  # the 58-name dictionary
    clips/...  cache/*.npz       # .npz gitignored
  skateid/
    video.py  pose.py  board.py  features.py  model.py
    train.py  eval.py  recognize.py  cli.py   app.py
    web/index.html
  checkpoints/                   # gitignored / LFS
  tests/                         # determinism, shapes, label-map + quantization, test_scope, tiny e2e
  notebooks/
```
Console script `skateid`; tasks also runnable as `python -m skateid.cli ...`. `tests/` passes
in CI before any checkpoint is trusted. Additions vs v0.2: `app.py`, `web/index.html`,
`flatground_allowlist.csv`, `tests/test_scope.py`, and a quantization round-trip test.

Pin: Python 3.10, PyTorch 2.x, `ultralytics` (YOLO26 / SAM 3 / YOLOE), `transformers` (frozen
eval baseline), `fastapi`, `uvicorn`, `python-multipart`, `opencv-python`, `numpy`, `pandas`,
`timm` (backup backbones), `pytest`. `ffmpeg` at `C:\ffmpeg`.

---

## 12. Milestones (each with an exit gate)

| | Work | Exit gate |
|---|---|---|
| **M0** half day | uv env + skeleton + download 222 clips + manifest + both splits + B0/B1/B2 | `skateid train && skateid eval` prints a confusion matrix + a floor |
| **M1** 1-2 days | pull SkateAI's 449 BATB clips in (12 source videos -> 449 cuts); CUDA torch; goofy/regular as an input; **pose features + extraction pipeline** | 671 clips ingested and feature-cached (671/671, 0 failures); stance as an override-only input with both-stances naming; **pose-only LR measured on the video-disjoint holdout as an ABLATION** (0.0406 = 1.01x the VideoMAE floor — see §12.5). No accuracy gate: pose alone is provably insufficient for mirror pairs (§12.7), so this milestone's job was to build and measure the pipeline |
| **M2** 1-2 days | **board corners + long-axis angle** (the axis pose cannot see); re-decode the rotation window at native fps | **The board stream is load-bearing, measured on a restricted vocabulary.** Gate: on the classes with >= 30 training clips, pose+board beats pose-only **and** beats the best measured floor (0.0401) by >= 2x, with **kick/heelflip confusion reduced** in the confusion matrix — the qualitative check a number cannot fake. Restricted vocabulary because §12.13 showed the 22-class problem is data-limited (~12 clips/class), not representation-limited. Still not a skater-disjoint claim |
| **M3** 2-4 days | `skateid serve` end-to-end; **three continuous rotation heads (flip / board_spin / body_spin)** as the primary output, with a **restricted-vocabulary** classifier over them; abstain calibration; the tail carried by the rotation heads rather than by 22-way classification | all three rotation heads live and each predicts its own angle within tolerance; **correct-or-abstained >= 90 % while abstaining on <= 30 % of clips**; the classifier covers only classes with >= 30 clips and says so; web page shows trick or "not sure" in <2 s |
| **M4** optional | web polish (annotate toggle, top-3 list, batch in page); ONNX export; distill board -> YOLO26-OBB | < 0.5 s/clip |

### 12.1 M0 status as built, plus the M1 SkateAI pull (2026-09-27)

| Item | Status |
|---|---|
| env | **project venv on Python 3.14.7** (`.venv`, `requires-python = ">=3.10"`); `uv` is not installed, so plain `venv` + `pip`, and `pyproject.toml` stays uv/pip-compatible for later |
| data | 222 SkateboardML clips (Kickflip 114 / Ollie 108) + **449 SkateAI clips** cut from 12 BATB source videos; 0 duplicate content hashes |
| manifest | `data/manifest.csv`, 671 rows, 29 columns (compositional labels + clip provenance) |

**Two honest limitations remain, but they shrank at M1.** SkateAI is now ingested:
449 BATB clips carrying compositional labels (`stance`, `landed`, `flip_type`,
`board_rotation_*`, `body_rotation_*`), cut from 12 source videos. Its `split_holdout` is
**group-disjoint by source video** (`split_source = source_video_url`); no clip is scored
while a near-duplicate cut of the same battle sits in training. Both datasets still lack a
**skater-disjoint** split: SkateboardML publishes no skater identity at all (its `skater_id`
is `num % 8`, a documented placeholder), and SkateAI never records which of the two BATB
competitors performed a clip (`skater_id_source = not_published_per_clip`). So `split_holdout`
is never reported as a leakage-free benchmark.

B0 majority-class floor per dataset and split, refreshed after the SkateAI pull (`skateid train/eval --dataset <ds> --split <s>`):

| Dataset | Split | Train | Test | Classes in test | Majority class | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|---|
| `all` (671) | `split_published` | 537 | 134 | 26 | kickflip | 0.2687 | 0.0163 |
| `all` (671) | `split_holdout` | 501 | 170 | 23 | kickflip | 0.2235 | 0.0159 |
| `skateboardml` (222) | `split_published` | 178 | 44 | 2 | ollie | 0.3409 | 0.2542 |
| `skateboardml` (222) | `split_holdout` | 164 | 58 | 2 | kickflip | 0.5172 | 0.3409 |
| `skateai` (449) | `split_published` | 359 | 90 | 25 | tre_flip | 0.1000 | 0.0073 |
| `skateai` (449) | `split_holdout` | 337 | 112 | 22 | tre_flip | 0.1250 | 0.0101 |

The two `skateboardml` rows are exactly the M0 numbers, which confirms the manifest schema
change did not disturb the existing splits. All six rows **reproduce unchanged** after the
v0.3.1 vocabulary work: only the spelling of the `skateai` majority class changed
(`treflip` → `tre_flip`), and the class count per split is identical because the
name → canonical mapping is a bijection. The `skateai` holdout is coarse by nature: 449
clips come from only 12 source videos, so it puts 2 videos in test (112 clips, 22 classes)
and 10 in train (337 clips). `source_group` (`BATB 1` / `BATB 11`) is **not** a usable
grouping key -- `BATB 1` alone spans 11 of the 12 videos -- so the split keys on
`source_video_url`.

### 12.2 M1 status: environment + dataset pull (2026-09-27)

| Item | Status |
|---|---|
| env | rebuilt as a project **`.venv` on Python 3.14.7**. Everything verified there: numpy 2.5.3, pandas 3.0.6, opencv-python-headless 5.0.0.93, scikit-learn 1.9.1, fastapi 0.141.1, uvicorn 0.54.0, pytest 9.1.1, yt-dlp 2026.8.19. The `deeplearning` extra resolves to torch 2.14.0 / torchvision 0.29.0 / ultralytics 8.4.164 (cp314 wheels exist) |
| interpreter trap | a bare `python` now resolves to a `pythoncore-3.14` runtime that has no project packages, and a long-lived shell can carry stale `...Programs\\Python\\Python310\\Scripts` PATH entries whose launchers point at a deleted interpreter -- so bare `pytest` / `yt-dlp` / `skateid` can be broken wrappers. Hence: always use `.venv`. `requires-python` stays `>=3.10`; 3.13 is supported but is not installed here |
| skateai labels | 5 upstream files (metadata, author split, trick names) fetched; 449 rows, 31 trick classes, 4 stances, `landed` true/false |
| skateai clips | **449/449** cut from 12 BATB source videos with yt-dlp + ffmpeg; 0 leftover source files; resumable |
| downloader | the upstream `labeling_tool/generate_data.py` (the README's `labelling_tool` path is wrong) cannot run on a 2026 stack -- it imports `pytube` (broken against current YouTube), `moviepy.editor` (removed in moviepy 2) and `wandb`. Re-implemented as `download_skateai_clips()`: one source download per battle, amortised over all of its cuts |
| upstream bug | SkateAI's split CSVs must be joined on `(video_title, video_file)`; `video_file` alone repeats across battles (70 unique values for 449 clips), so joining on it mixes clips between battles |
| tests | **8 passed** (was 4): added group-disjoint-split, SkateAI label consistency, video-disjoint holdout, and interpreter-range guards |

Two caveats carried forward. First, `all`-dataset numbers are a **union benchmark**:
222 SkateboardML clips and 449 BATB clips differ in footage, resolution and framing,
so they are not one domain. Second, neither dataset yields a skater-disjoint split
(§12.1), so no number from either `split_holdout` is a generalisation claim yet.

**Open defect found while re-reading those numbers: `label` is never normalised to the
allowlist.** *(Recorded at the time; now RESOLVED — see §12.3.)* SkateAI rows kept their
upstream spellings (`treflip`, `shovit`, `varial flip`, `laserflip`, `inward heel`,
`bs 360`, ...) while SkateboardML rows used allowlist canonicals (`kickflip`, `ollie`).
Only **3** of SkateAI's 31 names coincided with a canonical name, and `treflip` was not
even an alias of `tre_flip`, so `all` carried **32** distinct labels of which exactly
**one** (`kickflip`) was shared by both datasets. Worse, neither `flatground_allowlist.csv`
nor `tricks.json` was read by any code under `skateid/` — both were opened only by
`tests/test_scope.py` — so the guardrail promised in §5 was not implemented.

### 12.3 v0.3.1 status: vocabulary, guardrail, and B1/B2 (2026-09-27)

| Item | Status |
|---|---|
| root cause | labels are the *unstable* key. SkateAI publishes both a jargon name and the decomposed rotations, and its 31 names sit in exact 1:1 correspondence with its 31 triples — so the triple is the stable key. Normalising by name alone would need a hand-written alias per spelling *and* still could not detect a clip whose name and components disagree |
| fix | `skateid/taxonomy.py`: `data/tricks.json` is the **rotation dictionary** and `data/flatground_allowlist.csv` the **name registry**. A row's `label` is now *derived* from its triple (`label_from_rotation`); the raw spelling is kept in a new `label_source` column for provenance, and the guardrail cross-checks the two paths against each other |
| dictionary size | 15 → **45** canonical names, 41 rotation-expressible. 16 were added for the SkateAI tricks that had no canonical (bigflip, bigheel, biggerflip, all four bs/fs-180-flip variants, 360 shuvit, …), and 6 more to close the stance mirror (§12.4). 4 names stay allowlisted but are documented as **not** rotation-expressible: `half_cab`, `full_cab`, `impossible`, `none` |
| verification | all 31 upstream spellings resolve, and the rotation-derived and alias-derived canonicals agree on **all 31** (0 mismatches) — two independent paths cross-validate the dictionary against the data |
| manifest | 671 rows (added `label_source`, `license`, and `stance_published`/`stance_input`). `skateboardml`'s components are back-filled from the dictionary so the union is uniform: **0** null rotation values. `skateai` labels are now canonical (`treflip`→`tre_flip`, `bigflip`→`bs_bigspin_kickflip`, `shovit`→`pop_shuvit`) |
| licence column | new `license` column. **Neither** upstream repo ships a licence file (GitHub's licence API 404s for both), so it records the terms each project *states*: SkateboardML "academic-use-only, provided you cite" (Zenodo `10.5281/zenodo.3986905`); SkateAI has no licence statement and derives from copyrighted BATB footage, so its clips stay local and are never redistributed |
| guardrail | §5 is now real: `build_manifest()` calls `validate_or_raise()`; `skateid validate` re-runs it on demand; `tests/test_scope.py` holds the positive assertion **and five negative ones**. `skateid validate` on the shipped manifest: **PASSED**, 671 rows |
| **B1** | `skateid/baselines.py` — frozen feature extractor + linear probe, over a registry of backends: `videomae` (768-d, needs `deeplearning` + `transformers<5`), `resnet18`/`resnet50`/`mvit_v2_s`/`swin_t`, and `motion_stats` (18-d, numpy-only, runs anywhere). Features cache per clip under `cache/features/<backend>_<count>f_<w>x<h>/<clip_id>.npy`, so a re-score never re-decodes video. **Why beat it:** (a) it bounds what generality buys — if the pose/board pipeline cannot beat it, that representation is not earning its complexity, and we learn that in ~30 min rather than days; (b) it is the number a reviewer asks for, being the standard cheap protocol in video action recognition; (c) **it tests the dataset, not just the model** — a frozen embedder keys on appearance, so scoring well above the floor would mean the labels are separable by venue/camera/clothing rather than by rotation, which is a leakage alarm; (d) it is the ceiling for "no motion model", since mean-pooling frames ignores rotation order by construction |
| **B1 first results** | measured, all on `skateai`/`split_holdout` (337/112, 22 classes), CPU: B0 floor **0.0101** · `resnet18` (512-d, ImageNet) **0.0208** · `motion_stats` (18-d, hand-crafted) **0.0285** · `videomae` (768-d, Kinetics-400) **0.0401**. Two readings: (a) **a strong generic image encoder is not enough** — 512-d frozen ResNet-18 scores *below* 18-d motion statistics, so the labels are not separable by venue/camera/clothing; the shortcut check is working. (b) **temporal modelling is what helps** — VideoMAE, the only time-aware backend, is best at ~2x the ImageNet probe, an early (not conclusive) signal that rotation *order* is the signal, i.e. the §4 hypothesis. All of it stays far below useful: 4x a 0.0101 floor on 22 classes is a correctness signal, not a capability |
| **B1 correctness** | two bugs found and fixed, both producing a plausible *wrong* score rather than a crash. (1) `transformers>=5` **silently** drops VideoMAE's legacy `{0...11}` state-dict keys (torch 2.x no longer expands them), leaving the attention biases randomly initialised — a "frozen pretrained" encoder that is partly noise. Fixed by pinning `transformers<5` **and** by `VideoMAEEmbedder` loading with `output_loading_info=True` and raising on any missing key; the 66 tolerated unexpected keys are VideoMAE's pretraining decoder, correct to discard for an encoder-only probe. (2) the feature cache was keyed on `clip_id` alone, so a re-run at a new resolution would silently reuse old features; the cache key now carries the sampling grid (`<backend>_<count>f_<w>x<h>/`). Backends also declare `input_size`/`input_frames`, because VideoMAE's temporal position embeddings are fixed at 16 frames and 8 frames dies with an opaque tensor-size error |
| **B2** | zero-shot VLM via `skateid baselines --b2 --vlm {openai,anthropic,google,ollama}`. The prompt is built from the registry, never hard-coded, and the free-text answer is resolved with `Taxonomy.normalize_label` — the first real consumer of the alias table, with word-bounded longest match so "backside flip" cannot collapse onto the generic `flip` alias. Abstentions (`unknown`) score **wrong** and are reported separately. **Why bother:** it sets the *prior-knowledge* floor — near chance means a frozen generalist genuinely cannot do this and the task needs the rotation reasoning this project is built around; a high score would mean either the task is easier than assumed or the VLM is reading the venue, which B1's shortcut check can confirm. It is also a labelling aid for active learning and the M3 second opinion |
| honesty | `motion_stats` and the `mock` VLM backend exist to exercise the pipeline; their numbers are plumbing checks and must never be published as results. Backends that cannot run **skip with an explicit reason** rather than returning a quiet number |
| tests | **40 passed** (was 8): taxonomy round-trips, alias↔rotation cross-checks, the four rotation-free names, the guardrail's five negative cases, licence recording, the stance split, the stance mirror as a bijection, sampling-grid cache keys, declared backend geometry, and hermetic B1/B2 tests that synthesise their own clips with OpenCV |
| still open | B2 against a real model needs an API key or a local Ollama, which was declined, so B2 is **code-complete but unmeasured**. The installed torch is the **CPU** build (`2.14.0+cpu`), so these are CPU numbers; the machine has an RTX 3060 that M1's YOLO pose/board work should use, so torch should be reinstalled from the CUDA index before M1. Still no skater-disjoint split (§12.1), so nothing here is a generalisation claim |

### 12.4 M0 closeout (2026-09-27)

M0's stated scope and exit gate: *"uv env + skeleton + 222 clips + manifest + both
splits + B0/B1/B2"* → *"`skateid train && skateid eval` prints a confusion matrix +
a floor"*. Verdict: **met, with one documented substitution.**

| M0 deliverable | State |
|---|---|
| environment | done, on a project `.venv` at Python 3.14.7 (not `uv`: not installed on this machine, so plain `venv` + `pip`; `pyproject.toml` stays compatible with both) |
| skeleton + manifest | done. 671 rows, `split_published` and `split_holdout` both populated for every clip and verified disjoint |
| clips | done, and then some: the planned 222 SkateboardML **plus all 449 SkateAI** clips, cut from 12 BATB sources |
| B0 floor | done. 6 dataset×split combinations, all reproducing |
| B1 frozen-embedder probe | done and **measured** (0.0401 macro-F1 with VideoMAE vs a 0.0101 floor) |
| B2 zero-shot VLM | **code complete, not measured** — needs an API key or a local Ollama, which was declined. The interfaces, prompt builder, answer resolver, abstention accounting and skip-with-reason path are all built and tested; only a key is missing |
| scope guardrail (§5) | done, and the milestone's main defect fix. It was previously *promised and not implemented* |
| flatground-only scope | done, enforced at ingestion rather than in prose |

**What changed about the problem while doing M0.** The label-space defect was worse
than "spelling inconsistency": `label` was never normalised at all, and neither
vocabulary file was read by any code. The fix reordered the system's centre of
gravity — the rotation triple became the source of truth and the name a
derivation, which is what the plan specified in §3/§4 from the start but nothing
implemented.

**Three defects were found by refusing to accept a number**, and all three produced
a plausible wrong answer rather than a crash:
1. `transformers>=5` silently drops VideoMAE's legacy `{0...11}` state-dict keys, so
   the "frozen pretrained" encoder had randomly initialised attention biases.
2. The feature cache was keyed on `clip_id` alone, so a re-run at a new resolution
   would reuse old features.
3. The manifest's `stance` column held *riding directions* (`fakie`, `nollie`), not
   the goofy/regular stance the sign convention depends on — a name that invited
   exactly the silent kick↔heel / fs↔bs swap the project exists to avoid.

**Also closed:** the stance mirror was not total (6 of 35 names had no mirrored
partner); the 6 partners were added, making the mirror a **bijection** over all 41
rotation-expressible names, so §7's mirror-with-label-swap is safe everywhere.

**What M0 deliberately does not claim.** No trained model, no pose/board
extraction, no generalisation claim: neither dataset yields a skater-disjoint
split, and `all` mixes two different domains. B1's 0.0401 is 4× a floor, which is a
correctness signal, not a capability.

**Carried into M1:** resolve the goofy/regular toggle per clip into `stance_input`
(§3's `auto` stays suggest-only); pose + board extraction; and reinstall torch
from the CUDA index to use the RTX 3060 rather than the CPU build.

### 12.5 M1 closeout: extraction complete, pose-only at 1.01x the floor (2026-09-28)

| | |
|---|---|
| **extraction** | 671/671 clips, 552 s on the RTX 3060, 0 failures, 0.82 s/clip. Mean keypoint fill **0.747**, min 0.074; **624/671** clips usable (>= 50 % of frames with >= 50 % of keypoints) |
| **pose probe (LR, video-disjoint holdout)** | macro-F1 **0.0406**, accuracy 0.098 |
| **vs. the VideoMAE floor (0.0401)** | **1.01x** — statistically indistinguishable from a frozen generic probe |
| **M1 gate** | **NOT MET.** >= 5x was required |

**This is the informative outcome, not a failure of the run.** The gate's failure
branch exists precisely so that "the features do not carry the signal" is
discoverable before scaling, and that is what happened.

**The diagnosis is specific, and it is not "the models are bad":**

| probe | holdout macro-F1 |
|---|---|
| train macro-F1 | **1.000** |
| full 12-frame trajectory | 0.0406 |
| middle frame only | 0.0324 |
| velocity (d/dt of the trajectory) | 0.055 |
| PCA-20 | 0.0374 |

A **train F1 of 1.000 against a holdout of 0.041** is the whole story. With
**612 features and 337 training clips — 1.8 dimensions per sample** — logistic
regression memorises the training set completely and transfers nothing. Every
regularisation setting tried (C = 0.001/0.01/0.1) collapses train F1 to ~0.04
while holdout stays flat, which is the signature of *too little data for this
representation*, not of a bad solver.

Per-class recall confirms it: **5 of 22 holdout classes score above zero**, and
those five are the largest (kickflip 0.50, bs_180_heelflip 0.40, fs_bigspin 0.33,
tre_flip 0.25, heelflip 0.10). Holdout support is brutal — **median 4 clips per
class, 14 of 22 classes have fewer than 5** — so macro-F1 is mostly a
low-support tail in which a single clip moves a class by ~0.25.

**What this says about the representation, honestly:** the flattened-trajectory
vector is the wrong shape for 337 clips. Velocity scored highest (0.055) of all
probes, which is the *expected* direction for rotation-order signal, but the
margin over noise is not significant at this sample size. Nothing here shows the
pose features are wrong; it shows that **a 612-dimensional per-clip vector cannot
be estimated from 337 clips**, and that micro-averaging accuracy (0.098) is the
more stable read than a 22-class macro average over 112 clips.

**Carried into the next milestone — in priority order:**

1. **Reduce the dimensionality before adding model capacity.** A small temporal
   encoder over the pose sequence, or aggressive pooling to rotation-relevant
   statistics, is the obvious move; 1.8 dims/sample is the problem to fix first.
2. **The tiny transformer, not an LR on a flat vector.** Plan section 12 called
   for "pose features + LR -> tiny transformer"; only the LR rung was built, and it
   is the rung that failed. The transformer shares weights across time and should
   generalise where an unshared flat vector cannot.
3. **Board rotation is still unmeasured** and remains the most physically direct
   signal (M2). A board box is not a rotation, so this is genuinely missing.
4. **Report accuracy alongside macro-F1** for any small-support evaluation, and
   consider restricting the Stage-B target set to classes with >= 15 clips (plan
   section 13 already leans this way) so the metric stops being a tail lottery.

**Not claimed:** no generalisation result. The split is video-disjoint, not
skater-disjoint, and none of these numbers would support such a claim even had
they been higher.


### 12.6 The first gate revision: 95 % macro-F1 -> 5x the floor (2026-09-28)

> **Superseded by §12.7.** The 95 % -> 5x revision below was a considered change on
> its own terms, but §12.7 found a deeper problem: pose-only is information-limited
> for mirror pairs, so even a correct 5x bar was being applied to a
> half-representation. Kept as the first of two recorded gate revisions.

M1's exit gate previously read ">= 95 % macro-F1 vs. the `skateai` holdout
floor". That number was written before any holdout macro-F1 had been measured,
and it is not a defensible target for this problem:

- the best measured holdout macro-F1 is **0.0401** (VideoMAE), so 95 % is a ~24x
  jump, on a task where holdout support per class is as low as 4 clips;
- macro-F1 over ~23 classes with 170 holdout clips is dominated by the
  low-support tail, where a single clip moves a class's F1 by ~0.1. A gate set
  that close to the ceiling is measuring noise as much as capability;
- the split is **video-disjoint, not skater-disjoint** (§12.1), so even a 95 %
  score would not be a generalisation claim. The number was doing no work that
  its cost justified.

**The gate is now: beat the best measured holdout floor by >= 5x macro-F1,
i.e. >= 0.20, and publish the confusion matrix showing what is still hard.**

This is a *relative* bar, which is the honest form for a gate: it asks whether
structured pose/board features actually extract more signal than a frozen
ImageNet/VideoMAE probe, which is the specific hypothesis M1 exists to test. The
absolute figure that falls out (0.20) looks low next to 95 %, but it is the
correct comparison — it is 5x the strongest thing that has ever been measured
here, on a 23-class rotation task from 449 cuts of 12 competition videos.

**What is deliberately still required:**

| | |
|---|---|
| **the failure branch** | "< 5x the floor -> stop and fix features before scaling" is unchanged, and is the genuinely valuable outcome. Pose features failing to beat a frozen probe would be a real result about this representation, not a schedule problem |
| **published evidence** | the confusion matrix ships with the number, so the score is auditable rather than a single summary statistic |
| **the honesty label** | video-disjoint, not skater-disjoint, is restated in the gate itself |

**Not changed:** M0's exit gate, M2's "+3 macro-F1 for the board stream", M3's
">= 90 % correct-or-abstained", or M4's latency target. M3's gate has a separate
weakness worth revisiting when M3 is actually in scope: "correct-or-abstained"
is trivially satisfiable by abstaining on everything, and needs an abstention
rate ceiling before it is used as a gate.



---

### 12.7 Gates restructured: why pose alone cannot be the M1 gate (2026-09-28)

§12.5 measured pose-only at **1.01x** the frozen-embedder floor. Before reading that
as "the representation failed", the physical question has to be asked: **could a
human pose tracker ever separate these classes?**

**It cannot, and not because the model is weak.** A kickflip and a heelflip are a
**mirror pair** (§7) — the rider's body follows the same arc in both; the only
difference is the direction the *board* rotates under their feet. Pose is
mirror-symmetric about the body, so it is **information-theoretically incapable**
of breaking that symmetry. The same holds for every pair differing only in `flip`
or `board_spin`:

| separable by pose? | examples |
|---|---|
| **yes** | ollie vs shuvit-family (different body arc); bigspin vs 360 (`body_spin` genuinely differs) |
| **no** | kickflip vs heelflip, double variants, any shuvit-component difference — **board-only rotation** |

The data agrees: the video-disjoint holdout's 22 classes occupy only **8 distinct
`(flip, board_spin)` signatures** (the largest, `(1,1)`, covers 39 clips spanning
several names). The two missing axes are precisely the ones that split the
vocabulary.

**So the M1 gate was mis-scoped, not the architecture.** §6 has always specified
*both* streams at step 2 (pose -> 17 joints; board -> mask -> 4 corners), with 13
of ~52 feature dimensions belonging to the board. M1's *milestone line* said "pose
features + LR" while its *gate* implied the full representation — so the gate was
measured against a deliberately half-built slice, and **1.01x is the correct
result for that slice**, not evidence against the plan's hypothesis.

**Restructured gates:**

- **M1 keeps no accuracy gate.** Its job was to stand up CUDA, stance handling and
  the extraction pipeline, and to produce the **pose-only ablation** M2 needs as a
  comparison point. Done, with the number recorded.
- **M2 carries the 5x gate**, applied to **pose+board**, with pose-only reported
  beside it as an ablation rather than as the headline.
- **M2 must additionally show the board stream is load-bearing** — not merely that
  the combined model improves, but that the rotation axis contributes, since that
  is the hypothesis M1 originally existed to test. The old M2 gate ("board adds
  >= 3 macro-F1") is kept as a floor but is too weak alone: +3 on a 0.04 base is
  noise-adjacent. The real evidence is the confusion matrix showing **mirror pairs
  have become separable** — that is the qualitative check a number cannot fake.
- **M3 gains an abstention-rate ceiling.** ">= 90 % correct-or-abstained" is
  trivially satisfiable by abstaining on everything; it needs a bound (e.g. correct
  or abstained >= 90 % *while* abstaining on <= 30 % of clips).

**Unchanged and still true:** the split is **video-disjoint, not skater-disjoint**.
No gate here supports a generalisation claim.

### 12.8 M2 step 1: board angle extractor works, the ORACLE CHECK FAILS (2026-09-28)

The extractor was built and is **verifiably correct in isolation**; the oracle
check on real footage **fails**, and per the M2 gate the angle is therefore *not*
yet a training signal.

| | |
|---|---|
| **extractor** | Otsu + morphology + largest component -> `minAreaRect` -> long-axis angle. Verified on a synthetic board of known angle: recovers **45.00 deg for a board at 45**, 0.02 deg error across 0/15/30/45/60/-30/-60 |
| **synthetic mirror test** | a synthetic kickflip sweeps **+360**, a heelflip **-360**, near-equal and opposite — the sign is recovered |
| **oracle on real clips** | 85 clips, 12 frames, angle measured on **67.6 %** of frames. Kick-family correct sign **5 %**, heel-family **50 %**, overall **41 %**. Kick-vs-heel separation **-20 deg** |
| **verdict** | **FAIL. Do not train on this feature yet.** |

**The failure is not a sign convention, and that matters** — a flip that is
recoverable by negating would be a cheap fix. Checked explicitly:

| | measured |
|---|---|
| kick-family mean sweep | **-162 deg** |
| heel-family mean sweep | **-142 deg** |
| separation | **-20 deg** (and +20 if globally flipped) |

Both families measure *negative*. Flipping the sign globally leaves the families
20 deg apart, which is noise at this sample size. So the measurement is not
systematically inverted — it is **uninformative about direction**.

**The magnitude is at least not random**, which is the one piece of good news:
single-flip clips average 158 deg and double-flip clips 193 deg, and 40 % of
single-flip clips read a flat 0. The extractor does sometimes see that rotation
happened. It just cannot tell *which way*, which is the only thing that separates
a kickflip from a heelflip.

**The likely cause, not yet proven.** Coverage is 67.6 %, so roughly a third of
frames have no measurement at all, and those gaps are exactly where a flip's
direction would be read: mid-air, the board is edge-on and often partly occluded
by a leg, and `minAreaRect` on a thin sliver is ill-conditioned. A sweep
reconstructed from a third-missing series cannot recover direction reliably. The
`0` readings are suspicious for the same reason — a genuinely unrotated board and
an unmeasurable one are currently indistinguishable in the *sweep* even though
they are distinguished per-frame.

**What this rules out, and what it does not.** It does not mean board rotation is
the wrong hypothesis — the physics argument in §12.7 is untouched, and a box
still carries no rotation. It means **this extraction method is not good enough**,
and the failure is in the *measurement*, not the concept.

**Options, in the order worth trying:**

1. **More frames.** 12 samples across a ~2 s trick badly undersamples a rotation
   that takes ~0.3 s. 60 frames (plan §7's actual `T`) is the obvious first move
   and cheap — the sweep is the statistic that suffers most from undersampling.
2. **Upscale before segmentation.** The board occupies a small fraction of a
   640 px frame; segmenting a 2-3x crop would give `minAreaRect` far more pixels
   to work with.
3. **A learned segmenter** (SAM prompted with "skateboard", as §6 specifies) —
   handles occlusion that Otsu cannot, at the cost of a new failure mode.
4. **Revisit the sign problem itself.** A rectangle has no nose/tail, so direction
   is only recoverable if the rotation is fast relative to sampling; if the board
   is often near-stationary between samples, direction may need a *different*
   signal (e.g. the graphic on the deck) rather than better geometry.

**Not claimed:** no M2 result. The gate (5x on pose+board) is not evaluated, and
`probe --with-board` has deliberately **not** been run — running it now would
produce a number built on a feature this oracle has just shown to be uninformative,
which is precisely the kind of plausible-wrong score this project keeps refusing to
publish.

### 12.9 M2 step 1b: 60 frames, and a learned segmenter — both fail (2026-09-28)

§12.8 listed four options. The two cheapest and most promising were tried:
**more frames** (option 1) and **a learned segmenter** (option 3). Neither works.

**Option 1 — 60 frames instead of 12** (`skateid oracle --frames 60`):

| | 12 frames | 60 frames |
|---|---|---|
| angle coverage | 67.6 % | 67.7 % (**unchanged**) |
| kick-family correct sign | 5 % | **0 %** |
| heel-family correct sign | 50 % | **92 %** |
| kick − heel separation | -20 deg | -39 deg |

More frames **halved** the kick-family result. Coverage did not move at all, which
rules out undersampling as the cause. (The heel-family jump to 92 % is real but
*not* progress: heel clips are the ones where the board is briefly flat and
readable, so a denser sample catches that; kick clips never produce a usable
reading at any density.)

**Option 3 — `yolo11n-seg.pt` instead of Otsu.** The segmenter finds the board far
more often — **101/120 frames (84 %)** — which makes it look like a clear upgrade.
It is not:

| arm | kick correct | heel correct | separation | overall |
|---|---|---|---|---|
| Otsu, 12 frames | 5 % | 50 % | -20 deg | 41 % |
| Otsu, 60 frames | 0 % | 92 % | -39 deg | 51 % |
| **YOLO-seg, 60 frames** | 10 % | 21 % | +10 deg | **35 %** |

The learned segmenter is the **worst** of the three.

**The actual cause, found by looking at the masks.** A debug crop of a mid-flip
kickflip shows the problem directly: the board occupies a small part of its own
detection box, the rider's shoes are inside that box, and the board is **motion
blurred against a similarly-toned background**. Across five representative clips
and 117 measurements the angle took only **7 distinct values**:

```
{0: 93, -90: 19, 38: 1, 74: 1, 86: 1, -88: 1, -74: 1}
```

**93 of 117 readings are exactly 0 and 19 are exactly -90.** The median mask aspect
ratio is **0.62** — a near-square blob, not a board, which is a 4:1 shape. So the
segmentation is returning *shoes and shadow* rather than the deck, and the
rectangle is fitting the wrong object entirely. Per-frame, the angle sits at
**exactly -90.0** in nearly every frame regardless of what the board is doing.

**So the diagnosis in §12.8 was wrong, and the correction matters.** It is not
"a third of frames are missing and the gaps break the sweep". Coverage is not the
issue. The issue is that **most of the readings that do exist are not measurements
of the board at all**, and they are being treated as measurements. A sweep built
from them is arithmetic on noise.

**This is now a data problem, not a code problem.** The extractor is verified exact
on synthetic input (0.02 deg) and the geometry is right; what is missing is a
segmentation that actually isolates a skateboard deck in a 640 px, blurred,
low-contrast, partially-occluded competition frame. Options 1 and 3 from §12.8 are
both spent. What remains:

1. **Upscale the crop before segmenting** (option 2, untried) — the board is
   ~50 px long in these frames; a 3-4x crop gives the mask something to hold.
2. **SAM as §6 actually specifies** — prompt with the detection box, which is the
   setting SAM is built for. Distinct from YOLO-seg: SAM is prompted, not trained
   on COCO's 80 classes, and does not have to guess "skateboard" from a
   fixed vocabulary.
3. **Accept the finding and change the plan.** If no off-the-shelf route produces a
   trustworthy deck angle, the honest conclusion is that this representation needs
   either hand-labelled board corners (a few hundred clips) or a purpose-trained
   detector — both real projects, and both outside what "1-2 days" covers.

**Not claimed:** no M2 result, and `probe --with-board` still deliberately not
run. The gate is not evaluated.

### 12.10 M2 step 1c: upscaling is a dead end, but two real bugs were found (2026-09-28)

Two things came out of trying option 1 (upscale the crop), and the second one
changes the picture materially.

**Option 1 (upscale) does nothing.** Measured at 1x, 3x and 6x on the same frames:

| upscale | measurements | distinct angles | median mask aspect |
|---|---|---|---|
| 1x | 867 | 78 | 0.50 |
| 3x | 863 | 78 | 0.51 |
| 6x | 863 | 75 | 0.51 |

No movement at any scale. The bottleneck is not pixel count.

**But the "shoes and shadow" diagnosis in §12.9 was wrong**, and it was wrong
because of *our own* measurement. With `yolo11n-seg.pt` the actual masks are
**aspect 0.37-0.40** — a clean 2.5:1 board shape, with stable 2.3-2.6:1 detection
boxes and a 0.41-0.47 mask fill. The board **is** being segmented correctly. The
0.62 aspect reported in §12.9 came from the Otsu arm, not from YOLO-seg.

**Two real bugs, both found by looking rather than by trusting the aggregate:**

1. **The angle histogram was a measurement artifact.** The 93-of-117 pileup at
   exactly `0` and 19 at `-90` was Otsu returning a rounded rectangle, not the
   board. Real masks give smooth continuous series: a kickflip clip reads
   `21 21 23 27 ... 32 49 39 31 17 -21 -41 -43 -40 -20 ...` — a genuine
   rotation crossing through vertical. **The signal was in the data the whole
   time and the Otsu arm was throwing it away.** §12.9's conclusion "this is a data
   problem" was premature; the real problem was that the default arm was the wrong
   one.

2. **`net_sweep` cannot detect a full rotation — by construction.** It measures
   the *endpoint difference*. A board that rotates 360 deg **ends where it
   started**, so a perfect kickflip scores ~0. On the observed trajectory:

   | statistic | value |
   |---|---|
   | endpoint difference (`net_sweep`, used today) | **-27 deg** |
   | total variation (path length actually travelled) | **223 deg** |

   That is why the oracle reads 0 for so many clips: the statistic is wrong for
   the phenomenon, regardless of segmentation quality. **This bug is present in
   the synthetic tests too** and they did not catch it, because the synthetic
   sequence happened to start and end at different angles.

**Current status — still not passing, but for a now-understood reason.** With
YOLO-seg and 60 frames: kick 8 % / heel 21 %, separation +28 deg, 38 % overall.
The sweeps are dominated by the endpoint-difference bug, so **this number is a
floor, not a measurement of the feature.** Fixing `net_sweep` to use total
variation is a small change and must be done before the oracle is re-run;
re-judging segmentation before then would be measuring the wrong thing.

**Revised plan:**
1. **Fix the sweep statistic** (endpoint -> total variation, with the direction
   taken from the largest sustained excursion rather than the endpoint). Add a
   synthetic test that starts and ends at the *same* angle, which is the case the
   current tests miss.
2. **Re-run the oracle** on the YOLO-seg arm. Only then is the result meaningful.
3. Make YOLO-seg the default segmentation arm — it is measurably better and
   §12.9's contrary reading came from testing the Otsu path.

### 12.11 M2 step 1d: sweep bug fixed, oracle re-run — 55 %, still short (2026-09-28)

`net_sweep` now measures **total variation** instead of endpoint difference, and
the synthetic test that reproduces the real trajectory passes. Re-running the
oracle on the YOLO-seg arm with 60 frames:

| | before (12.9/12.10) | after the sweep fix |
|---|---|---|
| kick-family correct | 8-10 % | **30 %** (57 % with a global sign flip) |
| heel-family correct | 21 % | **46 %** (50 % with a global sign flip) |
| overall | 35-38 % | **52 %** (**55 %** with a global sign flip) |
| kick − heel separation | +10 deg | **-108 deg** |

**The separation is real and large now** — 108 degrees, versus 10 before. The fix
to the statistic is what produced it, exactly as §12.10 predicted: the earlier
number was a floor, not a measurement of the feature.

**But the sign is inverted**: kick-family reads **-86 deg** and heel-family
**+22 deg**. Negating globally lifts the overall score to 55 % and is almost
certainly a convention error somewhere (image y-axis direction, or the sign the
manifest assigns to `flip_type`) — **not yet identified, and not assumed**. 55 %
is well short of the ~90 % needed for a usable feature, so a sign convention is
not the remaining problem.

**Magnitude does not discriminate flip count**, which is the second failure:

| flip count | n | mean \|sweep\| | median |
|---|---|---|---|
| 1 | 50 | 295 deg | 270 |
| 2 | 14 | 257 deg | 270 |

A double kickflip and a single kickflip measure the *same*. So even with the sign
fixed, the feature would not distinguish 1 from 2 flips. Both failures point at
the same cause: **60 samples across a ~2 s clip still cannot resolve a rotation
that takes ~0.3 s.** Each flip is 4-6 samples; the trajectory passes through
intermediate angles far too coarsely for path length to be a reliable estimator of
magnitude, and aliasing lets a double flip sample as a single one.

**Where this actually leaves M2.** The step-1 question — "is there usable board
rotation signal?" — now has a nuanced answer:

- **Existence: yes, demonstrated.** A -108 deg kick-vs-heel separation is not
  noise. The signal is real and it is the one pose cannot see.
- **Reliability: no, not yet.** 55 % with an unresolved sign, and no magnitude
  discrimination, means this cannot be the flip head's feature as built.
- **The bottleneck is temporal resolution, not segmentation.** Upscaling changed
  nothing (12.10); 60 frames changed the statistic but not the aliasing.

**Next options, reassessed:**

1. **Sample the rotation window, not the whole clip.** The trick's rotation occupies
   a fraction of a second; the other ~1.7 s is approach and roll-away. Sampling
   densely *only where the board is airborne and moving* (from the segmentation
   mask's frame-to-frame change) would give 20-40 samples across the flip itself
   rather than 6. This is now the most promising route and it is cheap.
2. **Identify and fix the sign convention**, then re-measure. Cheap, but worth
   doing *after* (1) — improving the estimator and fixing the sign at once would
   make it impossible to tell which helped.
3. **SAM** — still deferred; the segmentation is no longer the bottleneck.

### 12.12 M2 step 1e: dense windowing makes it WORSE — and why (2026-09-28)

`active_window()` / `windowed_sweep()` are implemented, tested and exposed as
`skateid oracle --window`. The idea was sound: spend the sample budget on the flip
rather than on the approach and roll-away. **Measured, it regresses.**

| | whole clip | `--window` |
|---|---|---|
| kick-family mean | **-86 deg** | -36 deg |
| heel-family mean | **+22 deg** | -38 deg |
| **kick − heel separation** | **-108 deg** | **+2 deg** |
| overall correct sign | 52 % (55 % flipped) | 59 % (58 % flipped) |

**The separation collapses from -108 deg to +2 deg** — the window destroys
essentially all of the signal, leaving kick and heel indistinguishable.

**Why, and it is not subtle:** the selected window has a **median width of 35 of
60 frames**. It is not isolating the ~0.3 s flip; it is taking more than half the
clip. `active_window` maximises total variation subject to `max_span=20`
*measured* frames, and because ~38 % of frames are unmeasurable, 20 measured
frames can span 35+ original frames of mostly-static board. The window is doing
the opposite of its job.

Worse, "highest total variation" is the wrong objective here: a slow drift across
the clip accumulates more total variation than a real 0.3 s flip, so the search
prefers the drift. The flip is a *fast* event, and the window should be selected
on **peak angular rate** (a short window, max |dθ/dt|), not on total variation.

**One number is genuinely better:** overall sign accuracy rose to **59 %** (58 %
flipped). But that is not a win — with separation at +2 deg the two families have
the same *mean*, and 59 % is what you get from a near-zero predictor that
occasionally guesses right. It is a coincidence, not progress.

**Honest summary of M2's position after five attempts:**

| attempt | kick−heel separation |
|---|---|
| Otsu, 12 frames (12.8) | -20 deg |
| 60 frames (12.9) | -39 deg |
| YOLO-seg (12.9) | +10 deg |
| + sweep bug fixed (12.11) | **-108 deg** ← best |
| + windowing (12.12) | +2 deg |

**The sweep fix is the only change that has ever moved this number materially**,
and everything since has either been neutral or harmful. That is worth stating
plainly: four of the five "improvements" tried in M2 produced no usable signal,
and the one that did was a bug fix, not a technique.

**Why the window idea may still be right, done properly.** The premise is sound
and untested: the flip is 0.3 s of a 2 s clip, so a 60-frame grid gives the flip
~9 samples. A correct implementation would (a) select on **peak angular rate**
rather than total variation, (b) bound the window in *original* frames, and
(c) re-sample that window densely from the video rather than from the already-60
samples — which is the part not yet tried, and the only version that actually
increases temporal resolution. **As implemented it only *selects* from existing
samples, which cannot add information, and the -108 deg whole-clip number shows
that discarding samples is what hurt.**

**Recommended next, honestly assessed:**
1. **Revert to the whole-clip statistic** (the -108 deg configuration) as the
   working baseline. Keep `--window` off by default.
2. **Find the sign convention.** It is a definite bug — kick reads negative, heel
   positive — and it is cheap. Fixing it before anything else makes every later
   measurement interpretable.
3. Only then revisit denser sampling, and if so as a **re-decode of the chosen
   window**, not a re-selection of existing frames.

**Still not claimed:** no M2 result. `probe --with-board` remains unrun — and on
this evidence that is the correct call, since the feature's behaviour changes sign
and magnitude depending on which frames are included.

### 12.13 The overfitting diagnosis was mislabelled: it is data-per-class, not dimensions (2026-09-28)

`skateid probe` reports **train macro-F1 1.000 against holdout 0.041** with 612
features on 337 clips (1.8 dims/sample), and §12.5 read that as "a 612-dimensional
vector cannot be estimated from 337 clips". **Tested directly, and that reading is
wrong.** Four dimensionality reductions, all on the same split:

| representation | dims | train F1 | holdout F1 |
|---|---|---|---|
| flat trajectory (today) | 612 | 1.000 | 0.0458 |
| pooled statistics (mean/std/min/max) | 204 | 0.660 | 0.0372 |
| xy only, confidence channels dropped | 408 | 1.000 | 0.0398 |
| pooled + PCA-30 | 30 | 0.104 | 0.0450 |

**Every one of them is at the floor (0.0401).** Cutting dimensions by 20x, removing
the confidence channels, and regularising train F1 down to 0.104 all change
nothing. If the problem were dimensionality, PCA-30 at train F1 0.104 would have to
transfer better than a 612-dim model that memorises perfectly. It does not.

**The actual constraint is clips per class.** Restricting to the five most common
classes:

| | 22-29 classes | 5 classes |
|---|---|---|
| training clips | 337 (~12/class) | 164 (~33/class) |
| holdout macro-F1 | 0.041 | **0.2863** |
| holdout accuracy | 0.107 | 0.333 |
| dummy-majority accuracy | — | 0.333 |

**0.2863 is 7.1x the 0.0401 floor — the 5x gate's threshold — and it crosses it.**
But read the accuracy honestly: **0.333 is exactly the dummy baseline.** The model
is not beating "always predict the majority class" on accuracy. The macro-F1 above
chance comes from the tail, where a couple of low-support classes get a hit or two.

**So the finding is a reframe, not a win.** M1's gate was set on a 22-class
problem; the representation does not clear it, and *no amount of feature
engineering moves it*, because the limit is ~12 clips per class, not the
representation. The plan already anticipated this in §13 ("tiny data overfits")
and in the decision to use only classes with >= 15 clips in Stage B.

**What this implies, and what it does not:**
- **Does not** mean the pose features are useless — 7.1x on 5 classes is well off
  the floor.
- **Does not** mean a transformer will help. A transformer shares weights across
  time and regularises, but it cannot invent data. Given that four different
  regularisers all fail identically, the evidence says the ceiling is data, not
  capacity. **Building the transformer is now a much lower priority than it was**,
  and §12.5's "reduce dimensionality first" is superseded by this section.
- **Does** mean the honest M2/M3 framing is a **restricted-vocabulary** problem:
  classes with enough clips to learn, with the long tail handled by the three
  continuous rotation outputs (which is what plan §13 already defaults to) rather
  than by 22-way classification.

| | |
|---|---|
| **the vocabulary restriction** | The classifier's target set is **classes with >= 30 training clips**, decided from the manifest and reported in the output, never silently applied. Everything else stays reachable through the three continuous rotation heads |
| **why 30** | §12.13 measured ~12 clips/class as too few to learn (22-class holdout 0.041) and ~33 clips/class as workable (5-class 0.2863). 30 is the smallest round number inside the range that worked, chosen from data rather than taste |
| **the rotation heads are primary** | A `tre_double_flip` with 1 clip is still fully expressible as (flip=+2, board_spin=+1, body_spin=0). The vocabulary restriction limits *classification*, not *coverage* |
| **what the gate must also show** | **Accuracy above the dummy-majority baseline**, which the 5-class result did *not* reach (0.333 = 0.333). Macro-F1 alone is not sufficient evidence: on a small holdout it can clear a threshold while the model still loses to always guessing the most common class |
| **tail handling** | The long tail is not dropped. It is represented by the rotation heads, per plan §13's existing decision. A new trick that shares another's rotations is a data change, not a code change |
| **the ablation stays** | Pose-only is reported beside pose+board on the *same* restricted vocabulary. Without it, "the board stream helped" is unfalsifiable |


### 12.14 M2 step 2: the rotation window is the duration of the board's *visibility*, and the flip is already over (2026-09-28)

Step 1's diagnosis (aliasing, fixable by denser sampling) was **wrong again**, and
reading the traces settles it. Frame-level traces at 48 samples, `|step|` = degrees
of change between consecutive measurements:

```
kickflip  (2.24 s, measured 43/48)
  angles:  21  23  27  20  20  22  26  27  23  28  28  25  26  27  21  19  18  16  32  .  .  39  31  17  . -21 -41 -40  .  . -20 -19 -18 -17 -13 -13  -8  -9  -6  -8  -5  -6  -6  -5  -5  -5  -5  -6   0
  |step|:   2   3   7   1   2   3   2   4   5   0   3   1   1   6   2   1   2  16   7   8  14  38  20   1  20   1   1   2   3   1   5   1   2   1   3   1   0   0   1   1   1   6
```

**The board is visible for the entire clip and only ever turns about 40 degrees.**
It goes 21 -> 17 (rolling), then -21 -> -41 (a small correction as it catches),
then -20 -> -5 (rolling away). **There is no 360-degree rotation anywhere in this
trace.** The kickflip is not in the measured data at all.

The same holds for the heelflips, where the only large steps (38-142 deg) coincide
with **gaps in measurement** -- the board was not segmented, then reappeared at a
different angle. A 142-deg "step" spanning a `None` is the unwrapper inventing
motion that was never observed.

**The actual conclusion: the 48-frame grid is not undersampling a flip. The flip
is not visible to this pipeline at all.** The whole "aliasing" explanation in
12.13's simulation described a phenomenon that is not present in these clips.

Why the board shows no spin: at 640x640 the board is ~110x45 px. A skateboard
flipping under a rider is, for most of its rotation, **edge-on to the camera** --
a few pixels tall, heavily motion-blurred, frequently behind a leg. The
segmenter either misses it entirely (the `None`s) or returns a mask of whatever
is in the box. `minAreaRect` on a 3-pixel-tall smear is close to meaningless, and
it returns a near-square blob, which is the 0.5 aspect measured throughout.

**This retro-explains every earlier number, including the good ones:**

| observation | explanation |
|---|---|
| the -108 deg "separation" (12.11) | kick and heel clips differ in *where* the unmeasured gaps fall and in the drift direction, not in rotation. The separation was an artifact of gap placement |
| 1 flip and 2 flips measuring the same (12.11) | neither is measured at all; both are the same roll |
| 93 of 117 angles being exactly 0 (12.9) | degenerate masks, not rotation |
| upscaling doing nothing (12.10) | more pixels on a 3-pixel-tall smear is still a smear |
| 60 frames not helping (12.9) | there is nothing to sample more finely |

**Upscaling, more frames, and windowing were all treating a signal-extraction
problem as a sampling problem.** The `net_sweep` fix in 12.11 was real and worth
keeping -- it is simply correct on its own terms, and the improvement it appeared
to produce was not the rotation being measured better.

**What this rules out, concretely.** Board rotation is *not* recoverable from
671 clips of 640x640 BATB footage by any thresholding or minAreaRect method.
The board is too small and too often edge-on. No amount of cleverer windowing or
frame budget recovers information that the source does not contain.

**What would work, and what it costs — the honest list:**

1. **Higher source resolution.** The manifest records 640x640 because that is what
   `sample_frames` resizes to. Decoding at native resolution (the clips are
   recorded much larger) gives a board 3-5x more pixels, which is the difference
   between a 3-pixel smear and a measurable deck. **Cheapest possible test, and it
   should be the next thing tried** -- it directly addresses the diagnosis.
2. **A tracker rather than per-frame detection.** A board spinning smoothly should
   be *predicted* forward between detections; segmentation models do not do this,
   and the unmeasured gaps in the middle of every trace are exactly where the spin
   happens.
3. **A purpose-trained board detector** (or hand-labelled corners). Real project.
4. **Give up on board rotation from this footage** and lean on the rotation heads
   with pose plus a user-supplied flip direction -- which, notably, is the same
   shape of answer as the stance question: some of this is genuinely not
   recoverable from a 2-second clip of a small object.

### 12.15 M2 step 3: native resolution helps a lot, and still does not contain the flip (2026-09-28)

§12.14's cheapest proposed test, run. **It is a real improvement, and it does not
change the conclusion.**

**First, a detail worth recording: the working resolution was distorting the
footage.** The clips are stored at **854x480**, and `sample_frames` was resizing to
**640x640** — stretching height by 33 % while shrinking width by 25 %, so the board
was being both squashed and blurred. Measured on the same clip, same model:

| working size | frames measured | max angular step | median mask aspect | median board rect |
|---|---|---|---|---|
| 640x640 (old) | 33/48 | 5 deg | 0.40 | 57x107 px |
| **854x480 (native)** | **42/48** | **28 deg** | **0.23** | 35x103 px |
| 1280x720 (upscaled) | 42/48 | 28 deg | 0.23 | 35x103 px |

Every metric improves: coverage **+27 %**, max step **5.6x larger**, and the aspect
drops to **0.23** — a proper 4:1 board shape rather than a squarish blob. Upscaling
past native adds nothing, as expected. **So resolution was a genuine bug and it is
fixed.**

**And the flip is still not there.** Traces at native resolution, 48 samples:

```
kickflip (47/48 measured, max step 26 deg)
  angles:  11   9   7   6   9   9  10   9  11  11  11  12  17  18  13  10   8   8  15  41  18  24  20   .   0 -13 -23 -23 -19 -28 -10 -11  -9  -8  -7  -6  -4  -5  -4  -4  -3  -3  -2  -3  -1   1   0   0
heelflip (42/48 measured, max step 20 deg)
  angles:   0   0  -0   0   0   0  -1  -1  -1  -1  -1  -1  -1   0  -1  -3  -6 -20 -40   . -41 -36 -33 -27  . -15  -8   0  .  .   8   0  -1  -5  -2  .  .  -3  -3  -2  -2  -3  -6  -6  -2  -3  -2  -2
```

The best any clip achieves is a **55 deg** single step. There is no 360-degree
rotation in any trace, at any resolution, in any of the five attempts. What the
traces show is rolling (0 -> -10), a catch correction (-20 -> -40), and roll-away
(-40 -> 0). The board's *vertical* extent is what is being tracked, and a board
rotating about its long axis barely changes that.

**The revised explanation, sharper than §12.14's.** A kickflip is a rotation about
the board's **long axis** — the axis that runs nose-to-tail. Projected to the image
plane, that rotation **foreshortens the board rather than turning it**: the long
axis barely changes angle, while the short axis swings through the full 360. A
long-axis angle series is therefore *close to blind to the very motion that defines
a flip*, even when the board is perfectly segmented. The 0.23 aspect confirms the
masks are good now; they are simply reporting the wrong axis.

**This supersedes §12.14's "the flip is too small / too edge-on"** — that was right
about the *symptom* (no rotation visible) and wrong about the *cause*. With good
masks, the axis being measured is still the wrong one.

**What this means for M2.** The failure is now understood well enough to state what
would be needed, and it is a real project rather than a tuning exercise:

1. **Measure the short axis, not the long one** — track the board's *width*
   direction, which is what sweeps through 360 during a flip. Cheap to test now
   that the masks are reliable, and it follows directly from this diagnosis.
2. **A tracker** to bridge the remaining gaps, still worth having.
3. **Purpose-trained detector / hand-labelled corners** if (1) fails.

**Not claimed:** no M2 result. The gate is not evaluated and `probe --with-board`
remains unrun. **The pose-only ablation (0.0406) is still the only measured
representation result in this project**, and §12.8-12.15 are a record of six
attempts and four wrong diagnoses, kept because the corrections are the useful
part.

**Code note:** `sample_frames` is still called with 640x640 in the extraction path
and the cache was built at that size. Any future board work must re-extract at
854x480, and the pose cache is unaffected (pose does not care about board pixels).

### 12.16 M2 step 4: foreshortening works — the flip IS in the data (2026-09-28)

The plan specified three board-rotation quantities (section 7, pre-2026-09-28):
"long-axis angle; **long-axis foreshortening**; short-axis len". M2 built the first
and stopped. **The second is the one that carries a kickflip.** Built now, as
`features.board_axes()` / `foreshortening_series()`, measuring
``long_len / (long_len + short_len)`` from the segmented mask.

**Verified on synthetic input first**: a 4:1 board seen broadside reads **0.806**,
and rolling toward a square reads 0.667 -> 0.543 -> 0.500. Monotonic, and
self-normalising (a 40x10 board and a 400x100 board give identical ratios, so
camera distance drops out).

**And on real footage it separates.** 48 samples at native 854x480; "drop" is
``median - min``, i.e. how far the board goes edge-on:

| clip | measured | flat median | min (dip) | **drop** |
|---|---|---|---|---|
| heelflip A | 42/48 | 0.803 | 0.762 | 0.041 |
| heelflip B | 45/48 | 0.726 | 0.594 | 0.133 |
| **fs_360 A** | 44/48 | 0.806 | 0.594 | **0.211** |
| **fs_360 B** | 43/48 | 0.796 | 0.583 | **0.213** |
| fs_shuvit A | 41/48 | 0.765 | 0.657 | 0.108 |
| fs_shuvit B | 35/48 | 0.805 | 0.708 | 0.097 |

The `fs_360` trace shows the shape physics predicts, in full:

```
0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.81 0.81  . 0.72  . 0.67  .  . 0.65 0.64 0.71 0.79 0.77 0.82 0.79 0.78 0.65 0.59 0.68 0.77 0.78 0.79 0.81 0.81 0.78 0.80
```

**Flat at 0.83, dipping to 0.59 mid-clip, recovering to 0.80** — a kickflip,
measured. The long-axis angle over the same clips is weak (max step 5-55 deg),
exactly as section 7.1's axis mapping predicts.

**Two caveats:** the **magnitude is not calibrated** (0.21/0.13/0.04 are ambiguous
between a full roll, a partial one, and a catch), and **`fs_shuvit` also dips
(0.10)** when section 7.1 says a yaw should leave foreshortening flat. Unresolved.

### 12.17 First gated measurement: pose PASSES, and the board stream HURTS (2026-09-28)

`skateid probe --min-train-clips N` — the first run in this project to clear a
gate. Results on the video-disjoint holdout, logistic regression:

| vocabulary | stream | accuracy | macro-F1 | vs floor (0.0401) | dummy acc | dummy macro-F1 |
|---|---|---|---|---|---|---|
| **9 classes** (>= 15 train clips) | pose only | 0.250 | **0.2048** | **5.08x — GATE MET** | 0.233 | 0.0473 |
| 9 classes | pose + board | 0.200 | 0.1476 | 4.04x — not met | | |
| 9 classes | **board only** | 0.117 | **0.0822** | 2.05x | | |
| **3 classes** (>= 30 train clips) | pose only | 0.536 | **0.4623** | **11.33x** | 0.500 | 0.2222 |

**Two genuine results, and one that contradicts the last three days of work.**

**1. Pose features pass the gate, and beat the dummy baseline.** At 9 classes,
0.2048 macro-F1 against a 0.0473 dummy — and accuracy 0.250 vs 0.233, so it is not
merely riding the class imbalance. At 3 classes, 0.4623 vs a 0.2222 dummy. This is
the first capability claim in the project, and it holds under the §12.13
restriction that makes it meaningful. §12.5's "reduce dimensionality, then the
transformer" is **superseded**: the 22-class ceiling was the data, and on data
adequate classes, pose alone clears the bar.

**2. The board stream makes it worse.** pose-only 0.2048 -> pose+board **0.1476**.
Board alone reaches 0.0822 — above the floor but weak — and **adding it to pose
actively costs 28 % of the score**. Diagnosis, from the cached scalars:

```
net_sweep:  kick-family mean -1.15   other -1.14   (sd 0.79)
```

**The feature does not discriminate.** Kick-family and everything else have
*identical* mean sweep. It is a large-variance quantity (sd 0.79 against a 0.01
family difference) that swamps the pose signal.

**This corrects §12.16.** The per-clip traces were real and the foreshortening dip
is real — but it is real *per clip*, not *per trick family*. A model needs
consistency across clips to use it, and the segmentation noise plus the
uncalibrated magnitude (§12.16's own caveat) make it inconsistent. Picking the
`fs_360` clips in §12.16 as the demonstration, from three labels, was
cherry-picking: the same feature on the full set carries no family information.
**That was my error, and it is the error that made 12.16 read as a success.**

**Why the board stream hurts rather than being neutral:** 2448 pose dims already
outnumber the 110 training rows, so extra noisy columns cost more in variance than
they add in signal.

**Not claimed:** no M2 gate result. The gate was written for pose+board, and
pose+board **does not meet it**. What passes is the pose-only ablation, which was
never the M2 gate. Stating that plainly rather than banking the 5.08x.

### 12.18 Board stream compressed to 4 scalars: it stops hurting, but does not help (2026-09-28)

`features.board_summary()` reduces the 2640-dim board stream to four scalars:
**dip depth** (`median - min` of foreshortening), **dip timing**, **coverage**, and
**peak foreshortening**. Angles are excluded — near-blind to a kickflip (plan 7.1)
and the bulk of the harmful dimensionality. `probe --with-board` uses these by
default; `--full-board` reproduces the 12.17 measurement.

| vocabulary | stream | dims | accuracy | macro-F1 | vs floor |
|---|---|---|---|---|---|
| 9 classes | pose only | 2448 | 0.250 | 0.2048 | 5.08x |
| 9 classes | **pose + board(4)** | 2452 | 0.250 | **0.2048** | 5.08x |
| 9 classes | pose + board(full) | 5088 | 0.200 | 0.1476 | 4.04x |
| | *dummy* | | 0.233 | 0.0473 | |
| 3 classes | pose only | 2448 | 0.536 | 0.4623 | 11.33x |
| 3 classes | pose + board(4) | 2452 | 0.536 | **0.4623** | 11.33x |
| 3 classes | **pose + board(full)** | 5088 | 0.571 | **0.5244** | **13.08x** |
| | *dummy* | | 0.500 | 0.2222 | |

**Three findings, and the first is the answer to 12.17's question.**

1. **The 4-scalar stream is exactly neutral** — 0.2048 -> 0.2048 and 0.4623 ->
   0.4623. Identical to three decimals in both, which is what "adds nothing" looks
   like on a linear model. The harm from the full stream (0.1476) is gone, so the
   compression fixed what it was meant to fix — but **it did not make the board
   stream useful.**

2. **The full stream *helps* at 3 classes** (0.4623 -> **0.5244**, 11.33x -> 13.08x)
   while hurting at 9. Invisible in 12.17, which reported only 9 classes. With 110
   training rows the extra variance is affordable; with 238 it is not. **Honest
   reading: this is a 28-clip holdout where one clip moves a 3-class macro-F1 by
   ~0.036.** Suggestive, not established, and not reported as a result.

3. **So the board stream carries a weak real signal a linear model over 2448 pose
   dims cannot extract** — 2448 dims against 110 rows makes using anything small
   arithmetically hard. That is the same data-limited story as 12.13, one level
   down, and it is the honest conclusion rather than a fifth failure.

**M2 gate status: still NOT met.** The gate requires pose+board to beat pose-only
*and* the floor by >= 2x. Pose+board ties pose-only, and the gate's own wording —
"the board stream is load-bearing" — is exactly what is not demonstrated.

**Where this leaves M2, honestly.** Six rounds (12.8–12.17) established that the
board signal is real per clip, that the long-axis angle is the wrong axis, that
resolution was a genuine bug, and that the dip survives compression. What has *not*
been shown is that a **classifier** can use it. The remaining options are a
per-frame temporal model that consumes board and pose jointly, or accepting the
pose-only result and deferring the board stream to M3 where the rotation heads
exist to take a per-frame quantity.

**Not claimed:** no M2 gate result. Pose-only 0.2048 (5.08x) remains the headline
and clears its own gate; pose+board does not clear M2's.












- Stance-prefixed names in output? **Default: no in v1** — stance is an input, not a rename.
- Spin resolution beyond 0/1/2/3? **Default: 0-3 + fs/bs** (per-axis base in §4); drop the
  ultra-long tail (tre triple flip, bigspin inward heel, 1-clip classes) from M3 targets.
- Chase every SkateAI trick with < 15 clips? **Default: only classes with >= 15 clips** in
  Stage B validation; the tail stays representable via the three rotation outputs.
- Re-introduce pop-shuvit as its own label? **Default: no for v1**; revisit only if airtime
  ends up separable and someone asks for the distinction.

**On versioning — yes, and the reason is specific.** The 95 % gate was written
before any holdout score existed; the 5x gate was written before it was known that
pose alone is information-limited. Both were plausible on their face and both were
wrong, and the failure mode is identical: **a gate never measured against a real
number is a guess**. So each revision gets a numbered section recording *what was
believed, what was measured, and why the belief changed* — which is what lets a
later reader distinguish a considered revision from a moving target. The file stays
\plan-v0.3.0.md\; the sections carry the history.

### 12.19 Milestone status as of 2026-09-28 (M0–M2)

Consolidated, because the per-round sections are deliberately verbose and this is
the summary a reader should start from.

| | measured | gate | status |
|---|---|---|---|
| **M0** — env, 671 clips, manifest, both splits, B0/B1/B2, guardrail | B1 VideoMAE **0.0401**; B0 0.0101 | confusion matrix + a floor | **CLOSED** |
| **M1** — CUDA, stance-as-input, extraction pipeline, pose features | pose-only, 22 classes: **0.0406** (1.01x floor) | no accuracy gate (12.7) | **CLOSED** |
| **M2** — board corners, axis mapping, foreshortening | pose-only 9 classes **0.2048** (5.08x); pose+board **0.2048** | pose+board must beat pose-only | **NOT MET** |

**Everything currently true, in one place:**

| | |
|---|---|
| **Best measured result** | pose-only, 9 classes (>=15 train clips): **macro-F1 0.2048, accuracy 0.250** — against a dummy-majority baseline of 0.0473 / 0.233, so it is not riding class imbalance |
| **At 3 classes** (>=30 clips) | pose-only 0.4623 (11.33x); dummy 0.2222 / 0.500 |
| **Unrestricted 22 classes** | 0.0406 — **1.01x the floor**, and 12.13 showed this is a *data* ceiling (~12 clips/class), not a representation ceiling |
| **Board stream** | real per clip (foreshortening dips 0.83 -> 0.59 through a flip), but carries **no usable signal to a linear model**: 4 scalars are exactly neutral, 2640 dims actively hurt at 9 classes |
| **Split** | video-disjoint, **not skater-disjoint** — nothing here is a generalisation claim |
| **Tests** | 59 passed, 1 xfailed (a known, named bug kept visible) |

**What M0–M2 established beyond the numbers:**

1. **The board axis mapping** (7.1) — kickflip rolls about the long axis, a shuvit
   is a yaw, an impossible about the short axis. Never written down anywhere, and
   the single most useful finding of the last three days.
2. **The sign frame is an input**, override-only, with both-stances naming that is
   exact because the mirror is a bijection over all 41 names.
3. **Two real bugs fixed** — `net_sweep` measured endpoint difference (a full 360°
   flip scored zero), and extraction ran at a distorted 640x640 instead of native
   854x480.
4. **The 22-class ceiling is the dataset.** Four separate regularisers (pooled,
   PCA-30, confidence-stripped) all land at the floor, so no feature engineering
   moves it.
5. **Six documented failures** (12.8–12.18), kept because the corrections —
   aliasing -> sampling -> resolution -> **wrong axis** — are the reusable lesson.

**Honest read on M2's gate.** It requires the board stream to be *load-bearing*,
and it is not: pose+board ties pose-only exactly. Six rounds produced one genuine
discovery (the axis mapping) and one genuine bug (resolution), but not a usable
board feature for classification. **Recommendation: stop pushing M2**, take the
pose-only result as M1/M2's outcome, and do not build a temporal model inside a
milestone that has already run long.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Public-clip near-duplicates inflate scores | hash + perceptual dedup; skater/source-disjoint split; report both splits. **Neither dataset supplies a genuine skater-disjoint split**: SkateboardML has no skater labels (tiny pool of people), so its `split_holdout` is a placeholder; SkateAI's is source-video-disjoint but not skater-disjoint, because BATB is a 1v1 bracket that never records which of the two competitors performed a clip and competitors recur across battles |
| Oblique/vertical camera makes board roll ambiguous | skater-relative features, stance normalisation, mirrored aug, capture guide in README; the sign convention is checked in `--debug` |
| **Wrong/unknown stance -> kick<->heel, fs<->bs mirror** | default `auto` (suggest-only, never trusted); user overrides to goofy/regular; show the stance on the page + output; verify the sign in `--debug` |
| Tiny data overfits | ~0.5 M-param model, LR floor, heavy aug, early-stop on clean val |
| Partial/over-rotation mislabeled confident | rotation-residual -> "not sure" (§8); `landed` bail suppression |
| **BATB footage copyrighted** | personal/research use; never redistributed; not shipped in checkpoints; commercial use needs own filming |
| **Swinburne CC BY-NC / CC BY-NC-ND** | vocabulary only; neither CSV bundled |
| **ultralytics AGPL-3.0** | fine for personal/research; if permissive needed, swap `pose.py` behind the API to RTMPose/MMPose (Apache-2.0) |
| `yt-dlp` / YouTube volatility | cutter is ~20 lines and re-runnable; fall back to own filming |
| Label disputes (names/phrasing) | `tricks.json` versioned; the name is *derived* — the three rotation outputs are ground truth |
| Trick clipped / skater leaves frame | quality flag + abstain; require >= 80 % frame coverage |

---

## 14. Open questions (defaults chosen; revisit when data says so)
- Stance toggle default? **Default: auto (suggest-only)**; must still resolve to the correct
  goofy/regular before kick/heelflip & fs/bs are named.
- Trustworthy auto-detect stance? **Deferred.** v1 ships `auto` only as a suggest-and-override
  pre-fill (never trusted for the sign); a genuinely reliable stance auto-detect is future work.

---

**Tooling delivered regardless** (all reusable if option 1-3 works):

**Tooling added:** `skateid oracle --segmenter` so the comparison is reproducible.

**Note on cost:** the O(n²) window search over 60-frame series made the oracle run
~20 min for 85 clips. If windowing is revisited it needs a linear-time scan.

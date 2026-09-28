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
- `data/flatground_allowlist.csv` — the **name registry**: 39 canonical names + 121
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
> vocabulary is 39 canonical names, not 58.
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
| Board rotation | 3 | long-axis angle; **long-axis foreshortening** (long-axis len / baseline); short-axis len |
| Body yaw | 2 | sin,cos of shoulder-line angle vs clip's first frame |
| Feet-to-deck | 2 | ankle to nearest-deck-edge distance, per foot |
| **Airtime** | 2 | ankle height above standing baseline, per foot — **diagnostic, not a head** |
| Quality | 1 | board-visible fraction |

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
  encouraged.

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
| **M1** 1-2 days | pull SkateAI's 449 BATB clips in (12 source videos -> 449 cuts); pose features + LR -> tiny transformer, flip head (regression + quantization) | 449 `skateai` clips in the manifest with a **video-disjoint** `split_holdout`; then >= 95 % macro-F1 vs. the `skateai` holdout floor (still not a skater-disjoint claim); **else stop and fix features before scaling** |
| **M2** 1-2 days | board corners merged (still zero labels); oracle plots for board features | board stream adds >= 3 macro-F1 over pose-only; flip visible in `--debug` |
| **M3** 2-4 days | `skateid serve` end-to-end; more heads (flip / board / body); abstain calibration; exploit SkateAI's 31-class compositional labels | all three rotation heads live; >= 90 % correct-or-abstained; web page shows trick or "not sure" in <2 s |
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
| dictionary size | 15 → **39** canonical names, 35 rotation-expressible. The 16 added are exactly the SkateAI tricks that had no canonical (bigflip, bigheel, biggerflip, all four bs/fs-180-flip variants, 360 shuvit, …). 4 names stay allowlisted but are documented as **not** rotation-expressible: `half_cab`, `full_cab`, `impossible`, `none` |
| verification | all 31 upstream spellings resolve, and the rotation-derived and alias-derived canonicals agree on **all 31** (0 mismatches) — two independent paths cross-validate the dictionary against the data |
| manifest | 671 rows × **31** columns (added `label_source`, `license`). `skateboardml`'s components are back-filled from the dictionary so the union is uniform: **0** null rotation values. `skateai` labels are now canonical (`treflip`→`tre_flip`, `bigflip`→`bs_bigspin_kickflip`, `shovit`→`pop_shuvit`) |
| licence column | new `license` column. **Neither** upstream repo ships a licence file (GitHub's licence API 404s for both), so it records the terms each project *states*: SkateboardML "academic-use-only, provided you cite" (Zenodo `10.5281/zenodo.3986905`); SkateAI has no licence statement and derives from copyrighted BATB footage, so its clips stay local and are never redistributed |
| guardrail | §5 is now real: `build_manifest()` calls `validate_or_raise()`; `skateid validate` re-runs it on demand; `tests/test_scope.py` holds the positive assertion **and five negative ones**. `skateid validate` on the shipped manifest: **PASSED**, 671 rows |
| **B1** | `skateid/baselines.py` — frozen feature extractor + linear probe, over a registry of backends: `videomae` (768-d, needs `deeplearning` + `transformers<5`), `resnet18`/`resnet50`/`mvit_v2_s`/`swin_t`, and `motion_stats` (18-d, numpy-only, runs anywhere). Features cache per clip under `cache/features/<backend>_<count>f_<w>x<h>/<clip_id>.npy`, so a re-score never re-decodes video. **Why beat it:** (a) it bounds what generality buys — if the pose/board pipeline cannot beat it, that representation is not earning its complexity, and we learn that in ~30 min rather than days; (b) it is the number a reviewer asks for, being the standard cheap protocol in video action recognition; (c) **it tests the dataset, not just the model** — a frozen embedder keys on appearance, so scoring well above the floor would mean the labels are separable by venue/camera/clothing rather than by rotation, which is a leakage alarm; (d) it is the ceiling for "no motion model", since mean-pooling frames ignores rotation order by construction |
| **B1 first results** | measured, all on `skateai`/`split_holdout` (337/112, 22 classes), CPU: B0 floor **0.0101** · `resnet18` (512-d, ImageNet) **0.0208** · `motion_stats` (18-d, hand-crafted) **0.0285** · `videomae` (768-d, Kinetics-400) **0.0401**. Two readings: (a) **a strong generic image encoder is not enough** — 512-d frozen ResNet-18 scores *below* 18-d motion statistics, so the labels are not separable by venue/camera/clothing; the shortcut check is working. (b) **temporal modelling is what helps** — VideoMAE, the only time-aware backend, is best at ~2x the ImageNet probe, an early (not conclusive) signal that rotation *order* is the signal, i.e. the §4 hypothesis. All of it stays far below useful: 4x a 0.0101 floor on 22 classes is a correctness signal, not a capability |
| **B1 correctness** | two bugs found and fixed, both producing a plausible *wrong* score rather than a crash. (1) `transformers>=5` **silently** drops VideoMAE's legacy `{0...11}` state-dict keys (torch 2.x no longer expands them), leaving the attention biases randomly initialised — a "frozen pretrained" encoder that is partly noise. Fixed by pinning `transformers<5` **and** by `VideoMAEEmbedder` loading with `output_loading_info=True` and raising on any missing key; the 66 tolerated unexpected keys are VideoMAE's pretraining decoder, correct to discard for an encoder-only probe. (2) the feature cache was keyed on `clip_id` alone, so a re-run at a new resolution would silently reuse old features; the cache key now carries the sampling grid (`<backend>_<count>f_<w>x<h>/`). Backends also declare `input_size`/`input_frames`, because VideoMAE's temporal position embeddings are fixed at 16 frames and 8 frames dies with an opaque tensor-size error |
| **B2** | zero-shot VLM via `skateid baselines --b2 --vlm {openai,anthropic,google,ollama}`. The prompt is built from the registry, never hard-coded, and the free-text answer is resolved with `Taxonomy.normalize_label` — the first real consumer of the alias table, with word-bounded longest match so "backside flip" cannot collapse onto the generic `flip` alias. Abstentions (`unknown`) score **wrong** and are reported separately. **Why bother:** it sets the *prior-knowledge* floor — near chance means a frozen generalist genuinely cannot do this and the task needs the rotation reasoning this project is built around; a high score would mean either the task is easier than assumed or the VLM is reading the venue, which B1's shortcut check can confirm. It is also a labelling aid for active learning and the M3 second opinion |
| honesty | `motion_stats` and the `mock` VLM backend exist to exercise the pipeline; their numbers are plumbing checks and must never be published as results. Backends that cannot run **skip with an explicit reason** rather than returning a quiet number |
| tests | **36 passed** (was 8): taxonomy round-trips, alias↔rotation cross-checks, the four rotation-free names, the guardrail's five negative cases, licence recording, sampling-grid cache keys, declared backend geometry, and hermetic B1/B2 tests that synthesise their own clips with OpenCV |
| still open | B2 against a real model needs an API key or a local Ollama. The installed torch is the **CPU** build (`2.14.0+cpu`), so these are CPU numbers; the machine has an RTX 3060 that M1's YOLO pose/board work should use, so torch should be reinstalled from the CUDA index before M1. Still no skater-disjoint split (§12.1), so nothing here is a generalisation claim |



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
- Stance-prefixed names in output? **Default: no in v1** — stance is an input, not a rename.
- Spin resolution beyond 0/1/2/3? **Default: 0-3 + fs/bs** (per-axis base in §4); drop the
  ultra-long tail (tre triple flip, bigspin inward heel, 1-clip classes) from M3 targets.
- Chase every SkateAI trick with < 15 clips? **Default: only classes with >= 15 clips** in
  Stage B validation; the tail stays representable via the three rotation outputs.
- Re-introduce pop-shuvit as its own label? **Default: no for v1**; revisit only if airtime
  ends up separable and someone asks for the distinction.
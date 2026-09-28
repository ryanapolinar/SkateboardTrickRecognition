# Skateboard Trick Recognition — Plan v0.3.0 (2026, consolidated)

> Single integrated plan. **Supersedes `plan-v0.2.md` and `plan-v0.2.1.md`.**
> Adds the **output & quantization spec** (predict signed angles, quantize to a trick at
> the label layer), makes the **web page the default HCI**, and makes **flatground-only**
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
| board_spin | **180 deg** per shuv | `+`/`-` = fs / bs |
| body_spin | **180 deg** per turn | fs / bs |

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

**Guardrail (so it stays true, not just a promise):**
- `data/flatground_allowlist.csv` — approved labels + allowed component combos.
- `tests/test_scope.py` — fails CI if any `manifest.csv` row's label or combo is off-allowlist.
- Manifest build rejects non-flatground rows at ingestion.
- `tricks.json` is restricted to flatground names (SkateAI's 58 are all flatground; drop any
  that need an obstacle).
- README capture guide: film only flatground attempts; no obstacle in frame.

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
  plan-v0.1.md  plan-v0.2.md  plan-v0.2.1.md  plan-v0.3.0.md   README.md   pyproject.toml (uv, pinned)
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
| **M1** 1 day | pose features + LR -> tiny transformer, flip head (regression + quantization) | >= 95 % macro-F1 vs. `split_holdout` sanity floor (not a generalisation claim until M3); **else stop and fix features before scaling** |
| **M2** 1-2 days | board corners merged (still zero labels); oracle plots for board features | board stream adds >= 3 macro-F1 over pose-only; flip visible in `--debug` |
| **M3** 2-4 days | `skateid serve` end-to-end on 2 classes; then the BATB cutter + SkateAI's 449 clips; more heads; abstain calibration | all three rotation heads live; >= 90 % correct-or-abstained; web page shows trick or "not sure" in <2 s |
| **M4** optional | web polish (annotate toggle, top-3 list, batch in page); ONNX export; distill board -> YOLO26-OBB | < 0.5 s/clip |

### 12.1 M0 status as built (2026-09-27)

| Item | Status |
|---|---|
| env | **system Python 3.10** — `uv` is not installed on this machine; `pyproject.toml` stays uv/pip-compatible for later |
| data | 222 SkateboardML clips (Kickflip 114 / Ollie 108), 0 duplicate content hashes |
| manifest | `data/manifest.csv`, 222 rows |

**Honest limitation — there is no clean split yet.** SkateboardML publishes no skater or
session identity, so `skater_id` in the manifest is a synthetic bucket over the clip number
(`num % 8`, recorded in the `skater_id_source` column as `synthetic_clip_number`). The column
is deliberately named `split_holdout`, **not** `split_clean`, because it is a placeholder that
only proves the split plumbing works. It must never be reported as a leakage-free benchmark,
and the dataset is drawn from such a small pool of people that a genuine person-disjoint split
is not achievable from SkateboardML alone. A real clean split becomes possible at **M3**, once
SkateAI's per-clip skater labels are in the manifest; until then the M1 exit gate below reads
"macro-F1 on `split_holdout`" as a sanity floor, not as a generalisation claim.

M0 baseline floor (B0 majority class, no learning):

| Split | Test clips | Majority class | Accuracy | Macro F1 |
|---|---|---|---|---|
| `split_published` (author's own list) | 44 | ollie | 0.3409 | 0.2542 |
| `split_holdout` (placeholder) | 58 | kickflip | 0.5172 | 0.3409 |


---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Public-clip near-duplicates inflate scores | hash + perceptual dedup; skater/source-disjoint split; report both splits. **SkateboardML cannot supply a real skater-disjoint split** (no skater labels, tiny pool of people), so its `split_holdout` is a placeholder and the first genuine clean split arrives with SkateAI at M3 |
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
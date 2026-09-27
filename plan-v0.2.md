# Skateboard Trick Recognition — Plan v0.2 (2026)

> This is a rewrite of `plan-v0.1.md`, corrected for 2026 tooling and re-scoped to
> the simplest thing that works: **one clip in → one trick name out (or "not sure")**.
>
> Updated with two verified data sources beyond SkateboardML: **SkateAI** (449 labelled
> clips, 31 tricks, compositional labels) and the **Swinburne Trick Attempt Dataset**
> (24,053 labelled attempts, metadata only). The original Skateboard-AI Google Drive
> dataset link is dead and that repo contains no code — drop it as a source.

---

## 1. Scope & Definition of Done

### Goal
User hands the system a short video clip of one skateboarding trick. The system prints the
trick name and a confidence, or honestly says **"not sure"**.

### Input / output contract
- **Input:** one clip, ~1–4 s, a single trick attempt, any fps/resolution (24/30/60/120).
- **Output:** `bs 180 heelflip  0.87` or `not sure`.
- **Bonus outputs:** `--json`, `--top 3`, `--debug` (writes an overlay .mp4: skeleton +
  board quad + live rotation readout), `--batch clips/*.mp4`.

### Non-goals (explicitly out of scope for v1)
- Real-time streaming.
- Auto-segmenting *multiple* tricks out of one long video (clip must be pre-trimmed; a naive
  auto-trim fallback is listed as an optional later nicety).
- Stance-qualified names as separate outputs (nollie/fakie/switch are predicted as one of the
  four heads, but v1 does not rename the trick around them).
- Contests/occlusion-wrecked footage, and heavy slow-motion edits.
- A web app / phone app (a Gradio drag-drop is an optional M4 add-on).

### Success criteria
| Metric | Target |
|---|---|
| 2-class held-out accuracy (SkateboardML) | ≥ 95 % |
| Overall "correct **or** abstained" rate | ≥ 90 % |
| Latency per clip (RTX 3060, batch of 1) | ≤ 2 s |
| Execution | one command, no config files |
---

## 2. The verified data landscape

> Public skateboarding trick *video + label* sets are scarce and mostly built on the same
> corpus (Battle at the Berrics). Numbers below were measured directly (sizes, counts,
> licence) rather than guessed.

| Source | Clips | Tricks | Labels | Licence / friction |
|---|---|---|---|---|
| **SkateboardML** (github.com/LightningDrop/SkateboardML) | **222** (Ollie 108, Kickflip 114), ~594 MB | 2 | folder = class + published 177/44 split | In-repo, downloadable today, no auth; video credits in repo |
| **SkateAI** (github.com/EduardoPach/SkateAI) | **449** (train 359 / val 90 in-repo) | **31 present**, dict covers **58** | Per-clip `video_url` + `clip_start`/`clip_end`; compositional `flip_type, flip_number, board_rotation_*, body_rotation_*, stance, landed` | **Source footage is copyright (BATB YouTube) — personal/research use only, not redistributable, not commercial.** The downloader script its README references is gone (folder 404s); we write ~20 lines of `yt-dlp` + `ffmpeg` ourselves |
| **Swinburne "Skateboarding Trick Attempt Dataset"** (figshare 31043779, 2026) | **24,053 attempts** — **metadata only, no video, no timestamps** | many | `BATB, Skateboarder, Trick, Type (offence/defence), Outcome (land/bail), Variant, Year_Posted` | **CC BY-NC 4.0** — quote for vocabulary/coverage, do not bundle the file |
| Skateboard-AI dataset | Drive link dead; repo has **no code** | — | — | Dropped |
| HF `qleandataset/video-skateboard-vert` | 36 clips, 3.9 GB, gated | vert only | — | Wrong discipline — skipped |
| Kaggle "Skateboarding Trick Dataset (120fps)" | exists (unverified contents) | — | — | Verify before relying on it |

**Leakage trap (SkateboardML):** its published 44-clip split is random across clips drawn from
the same small pool — it contains near-duplicates and multi-trick videos. Naively reporting
accuracy on it overstates real-world performance (majority baseline there = 65.9 %, so any
claim must clear it, but that number is a *lower* bar than a clean split). **We report both**
(the published split for comparability, and a deduplicated, skater/source-disjoint split as
the real number).

**Data strategy (chosen):** two stages, same code.
- **Stage A = SkateboardML** (222 clips, 2 classes): get the whole pipe green, zero legal
  friction, today.
- **Stage B = SkateAI** (449 clips, 31 tricks): expand breadth once the pipe works, using a
  ~20-line `yt-dlp --download-sections` + `ffmpeg` cutter driven by its CSV.
- **Vocabulary reference = Swinburne** (which tricks matter, how they compose) — never copied,
  only learned-from.
---

## 3. The big simplification: predict components, not names

Every skateboard trick is a combination of four independent numbers. **We predict the four
numbers and look the name up.**

| Head | Answers | Example outputs |
|---|---|---|
| flip | none / kickflip / heelflip × count 0–3 | heelflip × 1 |
| board_spin | fs / bs × 0–3 | bs × 2 |
| body_spin | fs / bs × 0–3 | none × 0 |
| stance | regular / fakie / nollie / switch | regular |

```
treflip  = flip kickflip 1, board bs 2, body 0, stance regular
kickflip = flip kickflip 1, board 0,        body 0, stance regular
bigspin  = flip none,      board fs 2,      body fs 1
```

- New tricks that share a component combination are **free** (no new class to collect).
- Even with only ollie/kickflip data, the model is learning the *flip* axis that generalises.
- `TRICK_NAMES.json` (a copy of SkateAI's 58-entry dictionary) maps components → display name.
- Abstention = any head output that maps to nothing, low confidence, or `landed=false` on the
  attempted subset — see §7.

---

## 4. Architecture — 5 steps, 1 model trained

```
clip.mp4
   │
   ├─ 1. video.py     decode (ffmpeg/PyAV) → resample to 30 fps → long side ≤ 1280 px
   │
   ├─ 2. perceive     pose.py   YOLO26-pose (COCO-17)                 → 17 body joints
   │                  board.py  SAM 3 text-prompt "skateboard" → mask  → 4 board corners
   │                            (fallback: COCO YOLO26 detector box;   (both off-the-shelf,
   │                             minAreaRect for corners)               no training)
   │
   ├─ 3. features.py  per-frame vector D ≈ 52, skater-relative → sequence T = 60
   │
   ├─ 4. model.py     tiny transformer (T, 52) → (T, 128) → pool → 4 heads
   │
   └─ 5. recognize.py dictionary lookup → "bs 180 heelflip 0.87" | "not sure"
```

### Why not the usual 2026 alternatives (rejected with reasons)
| Rejected | Why |
|---|---|
| 3D-CNN / I3D, or fine-tuning VideoMAE-V2 / V-JEPA2 | Data-hungry (≫10 k clips), memorises scenes, expensive. We have 222–449 labelled clips. We do use a **frozen** embedder only as an eval baseline |
| CNN+LSTM (SkateboardML's and SkateAI's own approach) | Inherits the 2020 recipe; a small transformer is smaller, faster on CPU, and handles variable length |
| LSTM/TCN **or** xgboost (v0.1 left it open) | Fixed: one transformer; a logistic-regression-on-pooled-features run is the mandatory floor so we never "tune a transformer on 200 clips" without a simple number to beat |
| Hand-written pop-detector gate + roll-vs-yaw decision tree (v0.1 §4) | A gate hard-fails on no-pop tricks and on mis-detections; the tree encodes fragile heuristics as *logic*. Both demoted to debug/visualisation only (§7) |

---

## 5. Feature vector (the part that must be right)

Everything measured **relative to the skater** so camera distance/angle stops mattering:
- **Origin** = mid-hip. **Unit** = shoulder->hip distance `s`.
- **Stance normalisation:** sign-flip the x-axis if the front foot is the right foot (goofy),
  auto-detected from the first frames — so the model sees one world, not two.

Per frame (D ~ 52):
| Block | Dim | Content |
|---|---|---|
| Body joints | 34 | 17 x (x,y), centred on hip, / s |
| Board corners | 8 | 4 x (x,y), same frame |
| Board rotation | 3 | long-axis angle; **long-axis foreshortening** (long-axis len / baseline); short-axis len |
| Body yaw | 2 | sin,cos of shoulder-line angle vs clip's first frame |
| Feet-to-deck | 2 | ankle to nearest-deck-edge distance, per foot |
| Airtime | 2 | ankle height above standing baseline, per foot |
| Quality | 1 | board-visible fraction |

Sequence: T = 60 frames @ 30 fps (~2 s), always resampled to this length **regardless of source
fps** (fixes v0.1's ambiguous "30-45 frames"). Cache to `data/cache/<clip_hash>.npz` so training
is seconds and reproducible.

**Augmentation (cheap, multiplicative):** mirror **with label-swap** (kickflip<->heelflip,
fs<->bs) — turns every clip into two; time-jitter/crop; Gaussian noise on keypoints
(sigma ~ 0.01 * s); random frame dropout 10 %.

---

## 6. Model & training

- Encoder: `LayerNorm -> Linear(52->128) -> +pos-embed -> 4x pre-norm TransformerEncoder(d=128,
  h=4, ff=256, dropout 0.1) -> mean xor max pool` -> per-head `Linear(128->n)`.
- Heads share the trunk, no class-count coupling -> adding data/classes never changes the code.
- ~0.5 M params; **we do not touch any off-the-shelf weights** (pose/board are frozen and
  borrowed). Trains in minutes on the RTX 3060 (12 GB).
- Optimiser AdamW lr 3e-4, cosine, label-smoothing 0.1, class weights proportional to 1/sqrt(freq),
  100 epochs, early stop on macro-F1. Loss = sum over the four heads (weighted by data availability).
- M0/M1 uses only the heads that have data (flip, effectively); inactive heads switch on
  automatically as clips arrive — **same code both stages**.

### Abstain / "not sure"
- Temperature-scale the softmax on the validation set, then abstain below a max-prob threshold.
- In Stage B, `landed` is a real column -> a bail/abort pattern can suppress a confident name.
- A proper trained `other` class is added only once we have non-trick clips (b-roll, fails).

---

## 7. Evaluation protocol

Report on **both** splits (their published split and our deduplicated, skater/source-disjoint
split). Metrics: clip **macro-F1** (headline — the 44-clip split is 66/34 imbalanced), accuracy,
top-2, **selective accuracy at abstain rate**, confusion matrix per head, and slices by stance
and by camera side.

**Baselines to beat (measured first, in M0):**
| # | Baseline | Why |
|---|---|---|
| B0 | majority class | floor = 65.9 % on SkateboardML's split |
| B1 | frozen V-JEPA2 / VideoMAE embed, mean-pooled -> logistic regression | the zero-training 2026 baseline; **required to be worse than or roughly equal to ours, else rethink features** |
| B2 | VLM zero-shot prompt (Qwen3-VL / Gemini) | sanity-check that the label set is even visually separable |

---

## 8. UX & CLI contract

```
skateid clip.mp4                 # -> "bs 180 heelflip  0.87"
skateid clip.mp4 --json          # machine-readable
skateid clip.mp4 --top 3         # confidences
skateid clip.mp4 --debug out.mp4 # overlay: skeleton, board quad, live rotation
skateid "clips/*.mp4"            # batch
```
Optional M4: `skateid --serve` -> Gradio drag-drop (~40 lines). No config files, no server in
the default path.

---

## 9. Repo layout & tooling

```
SkateboardTrickRecognition/
  plan-v0.1.md  plan-v0.2.md  README.md  pyproject.toml (uv, pinned)
  data/
    manifest.csv                 # path,label,flip*,board_*,body_*,stance,landed,source,license,split
    tricks.json                  # the 58-name dictionary
    clips/...  cache/*.npz       # .npz gitignored
  skateid/
    video.py  pose.py  board.py  features.py  model.py
    train.py  eval.py  recognize.py  cli.py
  checkpoints/                   # gitignored / LFS
  tests/                         # feature determinism, shapes, label-map round-trip, tiny e2e
  notebooks/                     # optional: inspect one clip's features
```
Console script `skateid`; all tasks also runnable as `python -m skateid.cli ...`.
`tests/` must pass in CI before any checkpoint is trusted.

Pin: Python 3.10 (system), PyTorch 2.x, `ultralytics` (YOLO26 / SAM 3 / YOLOE), `transformers`
(frozen embedder baseline), `opencv-python`, `numpy`, `pandas`, `timm` (backup backbones),
`gradio` (optional), `pytest`. `ffmpeg` present at `C:\ffmpeg`.

---

## 10. Milestones (each with an exit gate)

| | Work | Exit gate |
|---|---|---|
| **M0** half day | `uv` env + skeleton + download 222 clips + manifest + both splits + B0/B1/B2 | `skateid train && skateid eval` prints a confusion matrix with a floor |
| **M1** 1 day | pose features + LR -> tiny transformer, flip head | >= 95 % macro-F1 on the clean split; **else stop and fix features before scaling** |
| **M2** 1-2 days | board corners merged (still zero labels); oracle plots for board features | board stream adds >= 3 macro-F1 over pose-only; flip visible in `--debug` |
| **M3** 2-4 days | write the ~20-line BATB cutter, add SkateAI's 449 clips, enable more heads, calibrate abstain | all four heads live; >= 90 % correct-or-abstained; per-stance slices sane |
| **M4** optional | ONNX export, Gradio `--serve`, distill board segmenter -> YOLO26-OBB for speed | < 0.5 s/clip |

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Public-clip near-duplicates inflate scores | hash + perceptual dedup; skater/source-disjoint split; report both splits |
| Oblique/vertical camera makes board roll ambiguous | skater-relative features, stance normalisation, mirrored aug, capture guide in README |
| Tiny data overfits | ~0.5 M-param model, LR floor, heavy aug, early-stop on clean val |
| **BATB footage is copyrighted** | personal/research use only; never redistributed; watermark/credit; do not ship in checkpoints; commercial use requires new data (own filming) |
| **Swinburne CC BY-NC / CC BY-NC-ND** | quoted for vocabulary only; neither CSV bundled into the repo |
| **ultralytics AGPL-3.0** | fine for a personal/research repo; if it must become permissive, swap `pose.py` behind the API to RTMPose/MMPose (Apache-2.0) |
| `yt-dlp` / YouTube volatility | cutter is ~20 lines and re-runnable; if BATB videos vanish, fall back to own filming |
| Label disputes (names/phrasing) | `tricks.json` is versioned and arguable; name is a *derived* string, the four numbers are the ground truth |
| Trick clipped / skater leaves frame | quality flag + abstain; require >= 80 % frame coverage |

---

## 12. Corrections to plan v0.1 (header list)

1. **YOLOv8 / YOLOv11 -> YOLO26 / YOLO26-pose** (Jan 2026, NMS-free, ~43 % faster CPU-ONNX);
   kept behind a swappable `pose.py`.
2. **`board_aspect_ratio` as a flip cue -> removed.** Aspect ratio conflates camera pitch,
   perspective and roll; replaced by board **corners + long-axis foreshortening**.
3. **`board_angle` -> angle *and* foreshortening** — a flipping board's long axis shrinks in the
   image; angle alone is blind to a flip.
4. **Pop detector as a pipeline gate -> demoted** to a feature/debug overlay.
5. **Roll-vs-yaw decision tree -> one learned model, four heads, plus abstain.**
6. **Raw-pixel features -> skater-relative** (origin mid-hip, scale shoulder->hip, stance sign-flip).
7. **Fixed "30-45 frames at 30 fps" -> resample to T=60 at 30 fps** regardless of source fps.
8. **"LSTM, TCN or xgboost" -> one transformer + an LR floor.**
9. **No reject class -> abstain** via temperature-scaled threshold (and `landed` in Stage B).
10. **No data plan -> manifests, dedup, source-disjoint splits, two verified public sources.**
11. **No eval protocol -> macro-F1, top-2, selective accuracy, per-stance slices, B0-B2 baselines.**
12. **No tooling/licence/UX spec -> uv pins, CLI contract, AGPL/BATB/Swinburne caveats.**
13. **Skateboard-AI as a source -> dropped** (Drive dead, repo has no code).
14. **MediaPipe alternative -> dropped** (one stack: YOLO26 + SAM 3); **ByteTrack/supervision for
    the rider -> dropped** (identity tracking is complexity a 2 s clip does not need; SAM 3's own
    video tracking is used only for the board).

**Kept from v0.1:** pose-first (no scene-memorising 3D CNN); compact per-frame feature vectors
instead of raw pixels; one-second-scale temporal window; the taxonomy idea (now as labels, not
as a pipeline).

---

## 13. Open questions (default chosen; revisit when data says so)
- Stance-prefixed names in output? **Default: no in v1** — stance is a head, not a rename.
- Spin resolution beyond 0/1/2/3? **Default: 0-3 + fs/bs**, matching SkateAI's dictionary; drop
  the ultra-long tail (tre triple flip, bigspin inward heel, 1-clip classes) from M3 targets.
- Chase every SkateAI trick with < 15 clips? **Default: only classes with >= 15 clips** in the
  Stage B validation; the tail stays representable via components.
# SkateID - Flatground Skateboard Trick Recognition

SkateID predicts flatground skateboard trick names with confidence (or abstains with "not sure") from video clips.

## Features
- Continuous 3-axis rotation prediction (`flip`, `board_spin`, `body_spin`)
- Dictionary lookup over canonical tricks
- Stance toggle (`auto` / `regular` / `goofy`, default `auto`)
- Web UI (FastAPI + single vanilla `index.html`) and CLI
- A flatground **scope guardrail** enforced at manifest build time
- Reference baselines: **B1** frozen-embedder probe and **B2** zero-shot VLM

## Environment

Developed and verified on **Python 3.14.7**. The code and dependency floors
support 3.10-3.14 (`requires-python = ">=3.10"`). Create the project venv once:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

Interpreter notes for this machine:

- `uv` is not installed, so the project uses plain `venv` + `pip`; `pyproject.toml`
  stays compatible with both.
- **Do not rely on a bare `python` / `pip` / `pytest`.** A `pythoncore-3.14`
  runtime installed through the Python Install Manager supplies `python` and
  `pip`, but it has no project packages; and a long-lived shell can still carry
  stale `...\Programs\Python\Python310\Scripts` entries whose launchers reference
  a deleted interpreter, so bare `pytest`, `yt-dlp` and `skateid` may resolve to
  broken wrappers. Always go through the venv (`.\.venv\Scripts\python.exe -m ...`
  or `.\.venv\Scripts\Activate.ps1`), and restart terminals/VS Code to drop the
  stale PATH entries.
- Verified on 3.14: numpy 2.5.3, pandas 3.0.6, opencv-python-headless 5.0.0.93,
  scikit-learn 1.9.1, fastapi 0.141.1, uvicorn 0.54.0, pytest 9.1.1,
  yt-dlp 2026.8.19. The `deeplearning` extra also has cp314 wheels
  (torch 2.14.0, torchvision 0.29.0, ultralytics 8.4.164).

## Quickstart

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\skateid.exe fetch                          # all datasets -> data/manifest.csv
.\.venv\Scripts\skateid.exe validate                       # flatground scope guardrail
.\.venv\Scripts\skateid.exe train --split published         # fit the B0 majority-class baseline
.\.venv\Scripts\skateid.exe eval  --split published         # accuracy / macro-F1 / confusion matrix
.\.venv\Scripts\python.exe -m pytest
```

`skateid fetch` is a no-op for anything already on disk, so it is safe to re-run
to rebuild the manifest. `--dataset skateboardml|skateai` narrows it, and
`--force` re-downloads. The build refuses to write a manifest that falls outside
the flatground vocabulary — see [Label vocabulary](#label-vocabulary-and-the-scope-guardrail).

Reference baselines (see [B1 / B2](#reference-baselines-b1--b2)):

```powershell
.\.venv\Scripts\skateid.exe baselines                      # list backends and what is runnable
.\.venv\Scripts\skateid.exe baselines --b1 --dataset skateai --split holdout
.\.venv\Scripts\skateid.exe baselines --b2 --vlm ollama --dataset skateai --split holdout
```

## Datasets

| `dataset` | Clips | Classes | Labels | `split_published` |
|---|---|---|---|---|
| `skateboardml` | 222 | 2 (kickflip, ollie) | class folder only | author's `trainlist02` / `testlist02` |
| `skateai` | 449 | 31 | per-clip trick decomposition + `stance` + `landed` | author's `train_split` / `validation_split` |

### SkateAI (BATB)

SkateAI publishes **labels, not video**. `data/metadata/metadata.csv` describes
each of the 449 cuts as a source video URL, an in/out interval and the trick
breakdown (`flip_type`/`flip_number`, `board_rotation_*`, `body_rotation_*`,
`stance`, `landed`), and you are expected to re-cut the clips yourself. Its own
downloader (`labeling_tool/generate_data.py`) does not run on a 2026 stack: it
imports `pytube` (broken against current YouTube) and `moviepy.editor` (removed
in moviepy 2), plus `wandb`. `skateid` re-implements that job with yt-dlp +
ffmpeg:

```powershell
skateid fetch --dataset skateai              # labels only (5 JSON/CSV files)
skateid fetch --dataset skateai --with-clips # + 12 BATB downloads -> 449 cut clips
```

One source video is downloaded per battle and every clip belonging to it is cut
from that single download, so 449 clips cost 12 downloads instead of 449. The job
is resumable (existing clips are skipped) and needs `ffmpeg` on PATH.
Smoke-test it with `--limit 3 --max-sources 1`.

The footage is copyrighted (Battle at the Berrics): keep it local for research.
`data/raw/` is gitignored and the clips must not be redistributed.

## Splits, and two honest caveats

`data/manifest.csv` carries two split columns plus `split_source`, which records
how the holdout grouping was formed:

| Column | `skateboardml` | `skateai` |
|---|---|---|
| `split_published` | author's `trainlist02.txt` / `testlist02.txt`, 178 / 44 | author's `train_split.csv` / `validation_split.csv`, 359 / 90 |
| `split_holdout` | placeholder partition, 164 / 58 | source-video-disjoint partition |
| `split_source` | `synthetic_clip_number` | `source_video_url` |

**For `skateboardml`, `split_holdout` is a placeholder.** SkateboardML publishes
no skater or session identity, so `skater_id` is synthesised from the clip
number (`num % 8`) and `skater_id_source` says so explicitly
(`synthetic_clip_number`). The leakage risk is real: if the same person, spot or
session appears in both train and test, a model can score well by recognising
the setting rather than the trick. That check is impossible with SkateboardML
alone, and its pool of people is too small for a person-disjoint split to be
meaningful anyway. So this column only proves the split machinery runs end to
end -- do not quote its numbers as evidence of generalisation.

**For `skateai`, `split_holdout` is group-disjoint but still not
skater-disjoint.** BATB is a 1v1 bracket and competitors recur across battles,
and the labels never record which of the two skaters performed a given clip
(`skater_id_source = not_published_per_clip`). Grouping by source video
nevertheless removes the worst leakage: no clip is scored while a near-duplicate
cut of the same battle sits in training. `skateai`'s *published* split is leaked
in exactly that way -- its author stratified on `stance`/`landed` at clip level,
so one battle contributes to both sides. Comparing `split_published` against
`split_holdout` on `skateai` therefore measures how much of a score is leakage
rather than skill.

One upstream bug worth recording: SkateAI's split CSVs must be joined on
`(video_title, video_file)`. `video_file` alone repeats across battles (every
battle folder has its own `00001.mp4`), so joining on it mixes clips up.
`skateid` keys on the pair, which is unique and non-overlapping across the files.

See `plan-v0.3.0.md` §12.1.

## Label vocabulary and the scope guardrail

Two data files define what this project is allowed to recognise, and one module
binds them.

| File | Role |
|---|---|
| `data/tricks.json` | **rotation dictionary** — canonical name ↔ the three rotations the model predicts |
| `data/flatground_allowlist.csv` | **name registry** — 39 canonical names + 121 aliases (every upstream spelling) |

`skateid/taxonomy.py` loads both and refuses to start if they disagree, so a
trick can never be half-added. Rotations live only in the dictionary and aliases
only in the registry, so there is exactly one place to change either.

**A row's `label` is derived from its rotations, not read from a name.** That
matters because names are the unreliable part. SkateAI ships 31 jargon names
alongside the decomposition of each trick, and those 31 names sit in exact 1:1
correspondence with their 31 rotation triples — so the triple is the stable key.
The manifest keeps the upstream spelling in `label_source` for provenance, and the
guardrail cross-checks the two paths against each other, which is what catches a
clip whose name and components disagree.

```
treflip        -> tre_flip              (flip +1, board +2 backside, body 0)
bigflip        -> bs_bigspin_kickflip   (flip +1, board +2, body +1)
shovit         -> pop_shuvit            (flip  0, board +1, body 0)
varial flip    -> varial_kickflip       (flip +1, board +1, body 0)
```

SkateboardML publishes only a class folder, so its components are **back-filled
from the dictionary**. That is what makes the union of the two datasets uniform
instead of leaving one side with empty rotations.

**The guardrail runs at ingestion.** `skateid fetch` → `build_manifest()` calls
`validate_or_raise()` and will not write a manifest containing a row whose label
is off-allowlist, whose components are empty, over-range (`> 3`) or internally
inconsistent (`flip_type='none'` with `flip_number=1`), or whose label disagrees
with its own triple. Re-run it any time with:

```powershell
.\.venv\Scripts\skateid.exe validate
```

Four names are allowlisted but deliberately **not rotation-expressible**, each
carrying a `note` in `tricks.json` explaining how it is represented instead:

| Name | Why there is no triple |
|---|---|
| `half_cab` | a fakie backside 180 — same triple as `bs_180`, recovered from `(stance_published, label)` |
| `full_cab` | a fakie backside 360 — same triple as `bs_360`, recovered from `(stance_published, label)` |
| `impossible` | the board wraps vertically around the front foot, about an axis the 3-axis model lacks |
| `none` | absence of a trick; `(0,0,0)` is already taken by `ollie`, so it needs a separate no-trick gate |

Asking for one of these from the model raises rather than guessing. None appears in
either ingested dataset, so nothing is lost today.

### Licensing

`license` is a manifest column. **Neither** upstream repository ships a licence
file (GitHub's licence API returns 404 for both), so the column records the terms
each project actually *states* rather than a legal determination:

| Dataset | Recorded terms |
|---|---|
| `skateboardml` | academic-use-only, provided you cite (Zenodo `10.5281/zenodo.3986905`) |
| `skateai` | no licence statement; derived from copyrighted BATB footage — research use, clips stay local, never redistributed |

## Reference baselines (B1 / B2)

Neither baseline uses this project's representation. They exist so that every
claim about pose/board features has something to be measured against.

### B1 — frozen embedder + linear probe

A pretrained video model, frozen, with only a logistic regression on top. It has
never seen a skateboard trick and uses none of our pose/board machinery.

```powershell
.\.venv\Scripts\skateid.exe baselines --b1 --dataset skateai --split holdout --embedder videomae
```

**Why bother beating it:**

- **It bounds what generality buys.** If the pose/board pipeline cannot beat a
  frozen embedder, that hand-crafted representation is not earning its complexity
  — and we find out in ~30 minutes of compute rather than days of training.
- **It is the number a reviewer asks for.** "Linear probe on frozen features" is
  the standard cheap protocol in video action recognition, so the result is
  comparable to published work rather than a bespoke metric.
- **It tests the dataset, not just the model.** A frozen embedder keys on
  appearance. If it scores well above the majority floor, the labels are probably
  separable by shortcut cues — venue, camera, clothing, skater — rather than by
  the rotation. That is a leakage alarm worth having early.
- **It is the ceiling for "no motion model."** Mean-pooling a clip's frames is a
  bag of frames and ignores rotation order by construction, so whatever it cannot
  do is what the trajectory representation is *for*.

| Backend | Dim | Needs |
|---|---|---|
| `videomae` | 768 | `deeplearning` extra (`transformers<5`, see below) |
| `resnet18` / `resnet50` | 512 / 2048 | `deeplearning` extra |
| `mvit_v2_s` / `swin_t` | 768 | `deeplearning` extra |
| `motion_stats` | 18 | nothing — a weak hand-crafted floor |

Features cache per clip under `cache/features/<backend>_<count>f_<w>x<h>/<clip_id>.npy`,
so re-scoring with a different probe never re-decodes video, and the sampling grid is
part of the cache key so a run at a new resolution cannot silently reuse old features.
A backend that cannot run **skips with the reason printed**, rather than quietly
returning a number.

Two correctness guards, because both underlying bugs produce a plausible-looking
*wrong* score rather than a crash:

- **`transformers` is pinned `<5`.** VideoMAE's published checkpoint stores its state
  dict with the legacy `{0...11}` layer-broadcast keys, which torch 2.x no longer
  expands. Under transformers 5.x those keys are dropped, the attention biases stay
  **randomly initialised**, and the encoder looks frozen while being partly noise.
  `VideoMAEEmbedder` therefore loads with `output_loading_info=True` and **raises** on
  any missing key, turning a broken pin into a loud error instead of a quiet low score.
  The 66 unexpected keys it tolerates are VideoMAE's pretraining decoder, which an
  encoder-only probe correctly discards.
- **Backends declare their input geometry.** VideoMAE's temporal position embeddings
  are fixed at 16 frames (8 tubelets x 196 patches = 1568); 8 frames yields 784 and
  dies with an opaque tensor-size error. `input_size` / `input_frames` are properties
  of the backend and `extract_features` honours them.

### B2 — vision-language model, zero-shot

```powershell
.\.venv\Scripts\skateid.exe baselines --b2 --vlm ollama --dataset skateai --split holdout
```

Backends: `ollama` (local, no key), `openai`, `anthropic`, `google`. The prompt is
built from the registry rather than hard-coded, and the free-text answer is
resolved through the same alias table the manifest uses, with word-bounded
longest match so `backside flip` cannot collapse onto the generic `flip` alias. A
model that answers off-vocabulary **abstains**: scored wrong, reported separately.

This sets the *prior-knowledge* floor. Near chance means a frozen generalist
genuinely cannot do this and the task needs the rotation reasoning this project is
built around; a high score would mean either the task is easier than assumed, or
the VLM is reading the venue instead of the trick — which B1's shortcut check can
then confirm. It doubles as a labelling aid for active learning and the M3 second
opinion.

### Measured numbers, and what they are not

All on `skateai` / `split_holdout` (337 train / 112 test / 22 classes), identical split, CPU.

| Run | Dim | Macro F1 | vs floor |
|---|---|---|---|
| B0 majority-class | — | **0.0101** | 1.0x — the floor |
| B1 `resnet18` (ImageNet, frozen) | 512 | 0.0208 | 2.1x |
| B1 `motion_stats` (hand-crafted) | 18 | 0.0285 | 2.8x |
| B1 `videomae` (Kinetics-400, frozen) | 768 | **0.0401** | **4.0x** — best |
| B2 VLM zero-shot | — | not run | needs an API key or a local Ollama |

Two things worth reading off this table:

- **A strong generic image encoder is not enough.** Frozen ResNet-18 features — 512-d,
  trained on natural images — score *below* the 18-d hand-crafted motion statistics.
  Appearance alone does not carry this task, which is the shortcut check working as
  intended: the labels are not separable by venue, camera or clothing.
- **Temporal modelling is what helps.** VideoMAE, the only backend that models time,
  is the best of the three, roughly 2x the ImageNet probe. That is a small but real
  early signal that rotation *order* is the signal — the hypothesis of plan §4, and
  what the M1 pose/board trajectory model is meant to exploit properly.

All of it is still far below useful: 4x a 0.0101 floor on a 22-class problem is a
correctness signal, not a capability. `motion_stats` and the `mock` VLM backend exist
to exercise the pipeline; their numbers are plumbing checks, never results.

## Stance: the sign frame is a separate input

A kickflip is **+360 for a regular rider and −360 for a goofy one.** Pop shuvit is
+180 regular, −180 goofy. That dependency is real, and it is why the sign
convention cannot be stated without a stance.

It does **not** belong in the dictionary, because of *where* it lives in the
pipeline. The stored triple is already in the **stance-normalized frame**: the
feature extractor (plan §7) sign-flips the x-axis per the resolved goofy/regular
toggle, and only then is the triple looked up. One stored value therefore serves
both stances, which keeps `label_from_rotation()` a pure function of the triple
and makes mirror-with-label-swap a free augmentation.

The manifest keeps the two concepts in **separate columns**:

| Column | Values | Meaning |
|---|---|---|
| `stance_published` | `regular`, `switch`, `fakie`, `nollie` | riding direction / pop type, exactly as the dataset published it. **Provenance only** — none of these can fix the sign frame, and `fakie` is a direction, not a foot forward |
| `stance_input` | `regular`, `goofy`, or empty | the resolved goofy/regular toggle that selects the sign frame |

`stance_input` is **empty on every row today**, because there is no feature
extractor yet (M1). That is the honest state: guessing it wrong silently mirrors
every sign, swapping kick↔heel and fs↔bs across the whole dataset, so an empty
value is strictly better than a plausible-looking wrong one.

The guardrail enforces this. `stance_input` must be `regular`/`goofy`/empty, and a
**riding direction copied into it is a separate, named violation** — because that
mistake is not a schema error, it is a silent sign flip on the data.

`Rotation.mirrored()` exposes the operation (negating all three axes *is* the
geometric mirror), and `Taxonomy.name_for_mirrored()` names the result or returns
`None` rather than inventing a name.

### Mirroring does not close — 6 of 35 names

Worth knowing before relying on mirror augmentation: the mirror of a trick need
not be a *named* trick.

| | |
|---|---|
| Named (29) | kickflip↔heelflip, pop_shuvit↔fs_shuvit, tre_flip↔laser_flip, bs_180↔fs_180, bigflip↔bigheel, … |
| Unnamed (6) | `bs_biggerspin_kickflip`, `bs_bigspin_inward_heelflip`, `tre_double_flip`, `hard_double_flip`, `bs_180_double_kickflip`, `fs_180_double_kickflip` |

Their mirrors are real tricks — a frontside biggerspin heelflip, a frontside tre
double flip — that **no dataset publishes**, so the dictionary cannot name them.
Mirroring such a clip is still valid *input* augmentation; there is just no label
to swap to. `ollie` is the one self-mirror, which is correct.

## Baseline floor (B0)

B0 always predicts the most common training label, so these numbers are the floor
any real model must clear. Reproduce with `skateid train --dataset <ds> --split <s>`
followed by `skateid eval --dataset <ds> --split <s>`.

| Dataset | Split | Train | Test | Classes | Majority class | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|---|
| `all` (671) | `split_published` | 537 | 134 | 26 | kickflip | 0.2687 | 0.0163 |
| `all` | `split_holdout` | 501 | 170 | 23 | kickflip | 0.2235 | 0.0159 |
| `skateboardml` (222) | `split_published` | 178 | 44 | 2 | ollie | 0.3409 | 0.2542 |
| `skateboardml` | `split_holdout` | 164 | 58 | 2 | kickflip | 0.5172 | 0.3409 |
| `skateai` (449) | `split_published` | 359 | 90 | 25 | tre_flip | 0.1000 | 0.0073 |
| `skateai` | `split_holdout` | 337 | 112 | 22 | tre_flip | 0.1250 | 0.0101 |

The two `skateboardml` rows are byte-for-byte the M0 numbers, which confirms the
manifest schema change did not disturb the existing splits. All six rows reproduce
after the label-normalisation work; only the `skateai` majority class is now spelled
`tre_flip` instead of upstream's `treflip`, and the class count per split is unchanged.

### What `skateai`'s holdout actually tests

There are only **12 source videos** for 449 clips, so a video-disjoint holdout is
coarse: it puts 2 videos in test (112 clips across 22 classes) and 10 in train
(337 clips). Note that `source_group` (`BATB 1` / `BATB 11`) is *not* a usable
grouping key -- `BATB 1` alone spans 11 of the 12 videos -- so the split is keyed
on `source_video_url`. A future season-level split would need to group by
`source_group` and accept a very small test set.


# SkateID - Flatground Skateboard Trick Recognition

SkateID predicts flatground skateboard trick names with confidence (or abstains with "not sure") from video clips.

## Features
- Continuous 3-axis rotation prediction (`flip`, `board_spin`, `body_spin`)
- Dictionary lookup over canonical tricks
- Stance toggle (`auto` / `regular` / `goofy`, default `auto`)
- Web UI (FastAPI + single vanilla `index.html`) and CLI

## Quickstart

```powershell
pip install -e .
skateid fetch                                   # download clips + build data/manifest.csv
skateid train --split published                 # fit the B0 majority-class baseline
skateid eval  --split published                 # accuracy / macro-F1 / confusion matrix
pytest
```

`skateid fetch` skips the download when clips are already in `data/raw/`, so it is safe to
re-run to rebuild the manifest.

## Splits, and one honest caveat

`data/manifest.csv` carries two split columns:

| Column | What it is |
|---|---|
| `split_published` | The split the SkateboardML authors shipped (`trainlist02.txt` / `testlist02.txt`), 178 train / 44 test. |
| `split_holdout` | A placeholder partition, 164 train / 58 test. |

**`split_holdout` is not a clean split.** SkateboardML publishes no skater or session
identity, so `skater_id` is synthesised from the clip number (`num % 8`) and the
`skater_id_source` column says so explicitly (`synthetic_clip_number`). The real risk this
guards against is leakage: if the same person, spot or session appears in both train and
test, the model can score well by recognising the setting rather than the trick. That check
is genuinely impossible with SkateboardML alone, and its pool of people is too small for a
person-disjoint split to be meaningful anyway.

`split_holdout` therefore only proves the split machinery runs end to end. Do not quote its
numbers as evidence the model generalises. The first trustworthy clean split arrives at M3,
once datasets that ship per-skater labels (SkateAI) are ingested. See `plan-v0.3.0.md` §12.1.

## M0 baseline floor

B0 always predicts the most common training label, so these numbers are the floor any real
model must clear.

| Split | Test clips | Majority class | Accuracy | Macro F1 |
|---|---|---|---|---|
| `split_published` | 44 | ollie | 0.3409 | 0.2542 |
| `split_holdout` | 58 | kickflip | 0.5172 | 0.3409 |


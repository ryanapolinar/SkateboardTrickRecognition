# SkateID — findings

**Status: shelved, 2026-10-01.** This is a research project that answered its
question. It did not ship a working product, and this document says why.

SkateID set out to name flatground skateboard tricks from video. It cannot. The
interesting part is *why*, and the reason turned out to be a clean, provable
statement about 2D video rather than a shortage of effort.

Every number below is on a **video-disjoint holdout** (no source video appears on
both sides), measured with plain scikit-learn logistic regression over cached
features. `plan-v0.3.0.md` is the full lab notebook; this file is the summary.

---

## The headline result

| quantity | status | number |
|---|---|---|
| **rider pose / kinematics** | **works** | **0.2943** macro-F1, 7.5× a degenerate baseline |
| **body-rotation family** (none / backside-half-turn) | **works** | **0.7087** accuracy vs 0.6019 dummy |
| board *detection* | works, but only ever confirmed presence | mask coverage 0.30–0.41 |
| board **rotation** (kickflip/heelflip sign) | **fails** | see below |
| board **rotation** (shuvit yaw) | **fails** | 0.97× its own dummy — i.e. baseline |

**The rider works. The board does not.** A kickflip and a heelflip are the same
board rotated about its own long axis in opposite directions, and that motion is
**provably invisible in 2D**: the silhouette is identical either way. This is not a
tuning failure, it is geometry.

---

## What pose actually gives us

Measured against a degenerate predictor that feeds every clip the *identical mean
pose vector*:

| representation | macro-F1 |
|---|---|
| **all 2448 dims, temporal (what we ship)** | **0.2943** |
| temporal mean only (sequence discarded) | 0.1130 |
| mean + std per joint (order discarded) | 0.1598 |
| every 3rd frame only (sparser sampling) | 0.2261 |
| **degenerate: same vector for every clip** | **0.0392** |

The ladder is the finding, not the top number:

- Collapsing the 48-frame sequence to a mean **costs 61 %** of the score.
  **The trajectory is the signal** — the project's core hypothesis, confirmed.
- Keeping frame *order* is worth another 0.135 over order-free statistics. That is
  the part only a temporal model can use, and it is the most transferable lesson
  here.
- Much of the signal survives 16 frames instead of 48, if extraction cost matters.

This is the whythetrick-style insight — *body kinematics are the teachable signal* —
established by measurement rather than assumed. It is also the project's most
useful result, because it is the one part that survived.

## Where the board work went

Two bugs found late, both worth more than the feature:

1. **The "foreshortening" feature was never in the data.** `board_summary` read a
   column labelled as a foreshortening ratio and computed a "dip depth" of 33.98
   from it — but the cached column holds a long-axis **angle in degrees**, and
   `foreshortening_series` was never wired into extraction. The four scalars that
   had been reported as "the board stream, compressed" were angle statistics
   mislabelled as shape statistics.
2. **The polarity was backwards anyway.** A deck rolling edge-on *narrows*, so the
   ratio **rises** toward 1.0; the code measured a dip (`median − min`) instead of
   a peak (`max − median`).

Fixed, with the reasoning recorded in the docstrings so it cannot be "corrected"
back. But fixing them did not rescue the result, because of the geometry above.

## The grip-tape probe

Since silhouette can't carry the sign, we built what should: split the board mask's
**interior** into dark (grip tape) and bright (graphic), and read the dark→bright
centroid displacement along the deck. Its sign *does* flip on a synthetic mirror
pair, and it is invariant to in-plane rotation — so the mechanism is sound.

On real footage it does not separate kickflip from heelflip: **agreement 0.36,
below the 0.50 coin-flip baseline.** On a 118×27 px board with the rider's feet on
the deck, the dark pixels are trucks, wheels, shadow and motion blur, so "dark" is
not reliably grip tape.

That closes both routes to the flip sign at this resolution. Including the board
stream makes the classifier *worse*:

| features | dims | macro-F1 |
|---|---|---|
| **pose only** | 2448 | **0.2943** |
| pose + board (4 summaries) | 2452 | 0.2837 |
| pose + board (full stream) | 5088 | 0.2567 |

## What did improve things: merging mirror pairs

Since the flip sign is unreadable, the mirror pairs are unanswerable *questions*.
Collapsing all 41 names into 19 mirror classes (`kickflip`+`heelflip`,
`varial_kickflip`+`varial_heelflip`, …):

| vocabulary | macro-F1 | accuracy | dummy |
|---|---|---|---|
| original, 9 classes | 0.2021 | 0.233 | 0.233 |
| **mirror-merged, 8 classes** | **0.2943** | **0.302** | 0.209 |

Note the original vocabulary scored **exactly its own dummy** — it was never
meaningfully above baseline. Merging doubles clips per class (the binding
constraint throughout) and deletes the questions the data cannot answer.

**The cost is permanent and must be surfaced, not hidden:** the recognizer can no
longer tell a kickflip from a heelflip. A shipping version should say
`varial_flip (kickflip or heelflip)`, not guess one.

---

## Honest status

**Not a working product, and not close.** 0.2943 macro-F1 / 30 % accuracy is above
baseline and far from usable naming.

The abstention layer — the thing this project was built around — **has no working
signal**. Two independent candidates both measured flat: residual distance in
rotation space, and the product of per-axis classification confidences. Nothing
measured so far predicts when the recognizer is wrong. That is the most important
open problem, and it is unsolved.

A defensible product *would* be coach-facing — score body rotation, flag
under-rotation, refuse to name a trick — because that is what the numbers support.
Not attempted.

## If you pick this up

Ranked by what the evidence says, not by what is easiest:

1. **Fix abstention.** Everything else is secondary to a tool that knows when it
   doesn't know. Neither rotation residual nor classifier confidence works; try
   something structurally different (test-time augmentation disagreement, say).
2. **Get more data.** ~12 clips/class is the wall behind nearly every failure
   here, including the yaw result. Nothing clever fixes that.
3. **Higher-resolution board crops** through a dedicated board model, if you want
   the flip sign back. It is a resolution problem, not an idea problem.
4. **Don't** re-tune the rotation regressors or the quantiser — both were measured
   and are at their ceiling.

## Repository map

| path | what it is |
|---|---|
| `plan-v0.3.0.md` | full lab notebook: every hypothesis, gate, and dead end, in order |
| `skateid/features.py` | pose + board extraction; `face_contrast` is the sign probe |
| `skateid/recognize.py` | the abstaining recognizer and the rotation heads |
| `skateid/taxonomy.py` | rotation dictionary and the flatground scope guardrail |
| `data/tricks.json` | 45 canonical names, 41 rotation-expressible |
| `tests/test_scope.py` | 79 tests, incl. the synthetic mirrors that caught the bugs |

## A note on the silent bug

The single most useful debugging moment was a classifier that scored 0.398 where a
hand-derived replica of the same pipeline scored 0.709. No error, no warning — just
a plausible number. The cause: scikit-learn returns `coef_` of shape `(1, D)` for a
two-class target, and a softmax over *rows* silently produces one wrong score per
class.

What found it was not review. It was **re-deriving the same pipeline by hand and
noticing the two disagreed by 31 points**. For anyone learning CV: check your
baseline before you check your model, and when a number looks reasonable, verify
the path that produced it.

---

*Built as a first CV project with no prior computer-vision experience. The negative
results are the point: two are provable (silhouette invariance; mirror pairs
unanswerable) and each closed a direction rather than merely disappointing it.*

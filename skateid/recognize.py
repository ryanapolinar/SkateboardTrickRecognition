"""Recognition: one model, one source of truth, honest about uncertainty.

Plan sections 6 and 10 make ``recognize`` the single implementation both the web
page and the CLI wrap, so there is no second model to drift out of sync.

**The design constraint that matters.** The best measured holdout score in this
project is macro-F1 **0.2048** on a 9-class vocabulary (plan 12.19) — roughly 25 %
accuracy against a dummy-majority baseline of 23 %. A recogniser that always
returns its best guess is therefore confidently wrong three times out of four. So
this module is built around **abstention**: it names a trick only when the margin
between the top two classes clears a threshold, and otherwise says "not sure".

That threshold is the product. It is not tuned for accuracy, because at this score
level maximising accuracy maximises the number of confident mistakes. It is tuned
so the tool is *usable*: a wrong answer costs the user more than a refusal, so the
default is deliberately slow to speak.

Stance is an **input, never inferred** (plan 3). ``stance="auto"`` returns both
readings of the same prediction rather than picking one, because the sign frame
cannot be recovered from the footage and a wrong guess mirrors every label.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .stance import STANCE_VALUES, trick_for_both_stances
from .taxonomy import Rotation, Taxonomy

#: Minimum probability gap between the top two classes to name a trick. Tuned by
#: ``skateid calibrate`` rather than guessed.
DEFAULT_MARGIN = 0.15

#: Below this top-class probability, abstain regardless of margin. A model that is
#: 40/30/30 is not confident under any reading, and its top-two margin would
#: otherwise look decisive.
DEFAULT_FLOOR = 0.35


@dataclass
class Prediction:
    """One clip's answer, with everything needed to judge how much to trust it."""

    clip_id: str
    label: Optional[str]
    confidence: float
    runner_up: str
    runner_up_confidence: float
    abstained: bool
    reason: str
    stance: Optional[str] = None
    readings: Dict[str, str] = field(default_factory=dict)
    top: List[Tuple[str, float]] = field(default_factory=list)
    vocabulary: List[str] = field(default_factory=list)

    @property
    def display(self) -> str:
        """What a user should read: the name, or an explicit refusal."""
        return "not sure" if self.abstained or self.label is None else self.label

    def as_dict(self) -> dict:
        """JSON-serialisable form, for `--json` and the web API."""
        out = asdict(self)
        out["top"] = [{"label": name, "confidence": score} for name, score in self.top]
        out["display"] = self.display
        return out
def _class_probs(raw, classes):
    """Class probabilities from linear scores, handling sklearn's binary form.

    sklearn returns ``coef_`` of shape **(1, D)** for a two-class target, with
    that row's sigmoid being the probability of ``classes_[1]`` (verified with a
    controlled fit -- plan 12.22). Iterating such a matrix row-wise yields one
    wrong score per class, which silently produced 0.398 where the correct
    mapping gives 0.709. Every model in this module scores through here so that
    case cannot be handled once and forgotten elsewhere.
    """
    raw = np.asarray(raw, dtype=np.float64)
    classes = list(classes)
    if raw.size not in (1, len(classes)):
        raise ValueError(
            f"model has {raw.size} score(s) for {len(classes)} classes; a fitted model "
            "has one score per class, or exactly one for a two-class head"
        )
    if raw.size == 1 and len(classes) == 2:
        positive = 1.0 / (1.0 + np.exp(-float(raw[0])))
        return np.array([1.0 - positive, positive])
    logits = raw - np.max(raw)
    exp = np.exp(logits)
    total = exp.sum()
    return exp / total if total > 0 else np.full_like(exp, 1.0 / exp.size)


def rotation_family(body_rotation_number) -> str:
    """Bucket a trick by how far the **rider's body** rotates.

    This is the axis a pose model can actually see, and it is why it is the
    primary M3 target (plan 12.21): grouping the training vocabulary by
    ``body_rotation`` gives 17 of 29 classes (59 %) with no body rotation, so
    within a bucket every remaining distinction is board-borne and pose is
    structurally blind to it.

    Values are half-turns (``1`` = 180 deg), matching the manifest's
    ``body_rotation_number``, which counts 180 deg steps. Signed, because a
    frontside 180 is a different visible motion from a backside one.
    """
    try:
        number = int(float(body_rotation_number))
    except (TypeError, ValueError):
        return "none"
    if number == 0:
        return "none"
    return f"bs{number}" if number > 0 else f"fs{-number}"


def rotation_targets(frame) -> "pd.DataFrame":
    """Attach the three continuous rotation targets, plus the body-rotation family.

    ``flip`` is in **whole 360 deg flips** (+1 kickflip, -1 heelflip), matching the
    manifest's ``flip_number``. ``board_spin`` and ``body_spin`` are in
    **half-turns** (1 = 180 deg), matching their manifest columns. Mixing those units
    would be a silent unit bug, so both are stated here and pinned by tests.

    Targets derive from the manifest's rotation columns, which come from the trick
    *name*. A head's angular error therefore measures agreement with **our own
    decomposition** of the name, not an independent annotation -- a mistake in that
    decomposition would be invisible to the metric. Recorded rather than glossed.
    """
    out = frame[["clip_id", "label"]].copy()
    out["flip"] = pd.to_numeric(frame["flip_number"], errors="coerce").fillna(0).astype(int)
    out["board_spin"] = (
        pd.to_numeric(frame["board_rotation_number"], errors="coerce").fillna(0).astype(int)
    )
    out["body_spin"] = (
        pd.to_numeric(frame["body_rotation_number"], errors="coerce").fillna(0).astype(int)
    )
    out["body_family"] = [rotation_family(value) for value in out["body_spin"]]
    return out


class RotationHeads:
    """Three continuous rotation regressors plus a body-rotation family classifier.

    **Why regress rather than classify.** The trick name is the composition of
    three signed integers, and the vocabulary's long tail is far too thin to learn
    as 29-way classes (plan 12.13 — ~12 clips/class). Predicting the *rotations* and
    deriving the name by dictionary lookup (plan section 6) covers every trick in
    the dictionary from the same three numbers, including the 1-clip ones. That is
    the whole point of the rotation representation, and it is why this is a
    regression problem rather than a classification one.

    **Targets and their units** (mixing these would be a silent unit bug):

    | head | unit | +1 means |
    |---|---|---|
    | ``flip`` | whole 360 deg flips | kickflip |
    | ``board_spin`` | half-turns (180 deg) | backside |
    | ``body_spin`` | half-turns (180 deg) | backside |

    Each regressor is a standardised ridge regression, which at 238 training rows
    and 2448 features is far better behaved than anything with capacity to fit the
    training set exactly — the failure mode measured in plan 12.5. The family
    classifier is a logistic regression over the *same* features.

    **Honest limitation, stated once here:** the targets are derived from the trick
    *name* via the rotation dictionary, so angular error measures agreement with
    our own decomposition rather than an independent annotation. A mistake in that
    decomposition would be invisible to every number this class reports.
    """

    AXES = ("flip", "board_spin", "body_spin")

    def __init__(self, axes, family, families, taxonomy, include_board=False, metrics=None,
                 axis_scales=None, levels=None):
        self.axes = axes
        self.family = family
        self.families = list(families)
        self.taxonomy = taxonomy
        self.include_board = include_board
        self.metrics = metrics or {}
        # Per-axis residual SD, used to make the three axes comparable when
        # ranking dictionary triples. None until fitted; ranking then falls back
        # to unit weights, which is measurably worse but never crashes.
        self.axis_scales = dict(axis_scales) if axis_scales else None
        # Option 3: per-axis integer-level classifiers, keyed by axis. An axis
        # absent from this dict has no usable classifier and falls back to the
        # regressor (see predict_levels).
        self.levels = dict(levels) if levels else {}
        self._candidates = None

    def predict_rotations(self, features):
        """Predicted (flip, board_spin, body_spin), **unrounded**.

        Raw rather than quantised: plan section 4 is explicit that the model should
        not output tidy multiples, because the residual *is* the "not sure" signal
        (plan section 8). Quantising here would destroy it.
        """
        out = {}
        for axis in self.AXES:
            model = self.axes[axis]
            scaled = (np.asarray(features, dtype=np.float64) - model["means"]) / model["scales"]
            out[axis] = float(model["coef"] @ scaled + model["intercept"])
        return out

    def predict_family(self, features):
        """Predicted body-rotation family and its probability.

        **Binary targets need handling.** sklearn represents a 2-class logistic
        regression as ``coef_`` of shape ``(1, D)`` with the *second* class as the
        implicit negative, not as ``(2, D)``. Feeding a ``(1, D)`` matrix to the
        softmax loop below iterates over rows and yields one wrong score per class,
        which silently produced **0.398 instead of 0.709** — a plausible number
        rather than an error. The (1, D) case is now scored explicitly.
        """
        model = self.family
        scaled = (np.asarray(features, dtype=np.float64) - model["means"]) / model["scales"]
        raw = model["coef"] @ scaled + model["intercept"]
        classes = model["classes"]
        probs = _class_probs(raw, classes)
        index = int(np.argmax(probs))
        return classes[index], float(probs[index])

    def predict_levels(self, features):
        """Per-axis integer level and its probability. (Option 3.)

        Classification over the integer levels instead of regression. The
        regressors shrink every prediction toward the training target mean
        (plan 12.23), which pinned ``body_spin`` into 0.36-0.60 for *all* clips
        and collapsed ``board_spin`` onto 1. A classifier has no such failure
        mode: each level is a separate hypothesis, so a clip either argues for
        "two half-turns" or it does not, and the probability that the level is
        *absent* is available as a real number.

        Returns ``{axis: (level, probability)}``.
        """
        out = {}
        for axis in self.AXES:
            model = self.levels.get(axis)
            if model is None:
                # No classifier fitted for this axis (too few clips per level).
                # Fall back to the regressor rather than inventing a level, and
                # tolerate that being absent too -- predict_levels must never raise
                # just because an axis was left unfitted.
                regressor = self.axes.get(axis)
                level = 0
                if regressor is not None:
                    scaled_r = (
                        (np.asarray(features, dtype=np.float64) - regressor["means"])
                        / regressor["scales"]
                    )
                    level = int(round(float(regressor["coef"] @ scaled_r + regressor["intercept"])))
                out[axis] = (level, 0.0)
                continue
            scaled = (np.asarray(features, dtype=np.float64) - model["means"]) / model["scales"]
            raw = model["coef"] @ scaled + model["intercept"]
            probs = _class_probs(raw, model["classes"])
            index = int(np.argmax(probs))
            out[axis] = (int(model["classes"][index]), float(probs[index]))
        return out

    def rotations_from_levels(self, levels):
        """Build a continuous-shaped triple from classified levels.

        Levels are exact integers, so this is a pass-through — but it keeps
        :meth:`rank_names` and the residual maths usable with classified input,
        which is what lets the two head types share one scoring path.
        """
        return {axis: float(levels[axis][0]) for axis in self.AXES}

    def level_confidence(self, features):
        """Joint confidence in a triple built from the level classifiers.

        The product of the per-axis probabilities, which is the probability the
        three classifications are *simultaneously* right under an independence
        assumption. Multiplying (rather than taking the mean) is deliberate: one
        axis being unsure should sink the whole triple, since a trick whose
        rotation is unknown is not a known trick at 60 %.
        """
        levels = self.predict_levels(features)
        joint = 1.0
        for axis in self.AXES:
            joint *= max(0.0, float(levels[axis][1]))
        return joint, levels

    def candidates(self):
        """Every rotation triple in the dictionary, cached.

        Ranking happens against this list instead of rounding, so a prediction
        never has to land exactly on an integer triple.
        """
        if self._candidates is None:
            seen, triples = set(), []
            for name in self.taxonomy.dictionary.expressible_names():
                rot = self.taxonomy.dictionary.rotation_for_label(name)
                if rot is not None and rot not in seen:
                    seen.add(rot)
                    triples.append(rot)
            self._candidates = triples
        return self._candidates

    def rank_names(self, rotations, top_k: int = 3):
        """Rank dictionary triples by axis-scaled distance. (Option 1.)

        Each axis is divided by its training residual SD, so "off by 0.3 flips"
        and "off by 0.3 half-turns" weigh the same. Plain Euclidean distance on
        raw axes is not comparable across them: board/body spin live in
        half-turns and can run to +-2 while flip rarely exceeds +-1, so an
        unscaled metric quietly under-weights board rotation.

        Returns ``[(name, distance), ...]`` sorted nearest-first. This replaces
        per-axis ``round()``: rounding a triple like (0.44, 0.72, 0.45) to
        (0, 1, 0) reports ``fs_180`` with total confidence even though the
        prediction was nowhere near it.
        """
        weights = self.axis_scales or {axis: 1.0 for axis in self.AXES}
        prediction = np.array(
            [float(rotations[axis]) / float(weights.get(axis, 1.0)) or 0.0
             for axis in self.AXES],
            dtype=np.float64,
        )
        ranked = []
        for rot in self.candidates():
            other = np.array(
                [float(getattr(rot, axis)) / float(weights.get(axis, 1.0)) or 0.0
                 for axis in self.AXES],
                dtype=np.float64,
            )
            distance = float(np.sqrt(((prediction - other) ** 2).sum()))
            ranked.append((self.taxonomy.dictionary.label_from_rotation(rot), distance))
        ranked.sort(key=lambda pair: pair[1])
        return ranked[: max(top_k, 1)]

    def name_for(self, rotations, max_distance: Optional[float] = None):
        """Nearest dictionary name, or ``None`` if nothing is close. (Options 1+2.)

        The residual distance is the "not sure" signal plan section 4/8 asks for,
        so it is *kept* rather than discarded by rounding, and returned to the
        caller as a distance instead of being buried in a boolean.

        ``max_distance`` is in scaled units: above it, the honest answer is "no
        known trick", which is preferable to a confident wrong name.
        """
        ranked = self.rank_names(rotations, top_k=1)
        if not ranked:
            return None
        name, distance = ranked[0]
        if max_distance is not None and distance > max_distance:
            return None
        return name

    def rotation_confidence(self, rotations):
        """Confidence in the name, from the residual distance. (Option 2.)

        Monotone in distance and independent of how many dictionary entries
        happen to be nearby, which a softmax over names would not be. 1.0 at an
        exact hit, decaying with distance; callers compare against a tuned floor.
        """
        ranked = self.rank_names(rotations, top_k=1)
        if not ranked:
            return 0.0, None, None
        name, distance = ranked[0]
        return float(np.exp(-distance)), name, distance


class Recognizer:
    """Loads a trained classifier and applies the stance + abstention rules.

    The classifier is deliberately simple (standardised logistic regression over
    pose features): at 337 training clips the evidence is that model *capacity* is
    not the binding constraint (plan 12.13 — four regularisers all land at the
    floor), so a small model that trains in seconds and can be audited line by
    line is worth more than a larger one that cannot be.

    Persisted as JSON with the scaler stored as means/scales rather than a pickle,
    so the artefact is inspectable and does not depend on a sklearn version.
    """

    def __init__(
        self,
        classes: List[str],
        means: np.ndarray,
        scales: np.ndarray,
        coefficients: np.ndarray,
        intercepts: np.ndarray,
        taxonomy: Taxonomy,
        margin: float = DEFAULT_MARGIN,
        floor: float = DEFAULT_FLOOR,
        min_train_clips: int = 0,
        metrics: Optional[dict] = None,
    ) -> None:
        self.classes = list(classes)
        self.means = np.asarray(means, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.intercepts = np.asarray(intercepts, dtype=np.float64)
        self.taxonomy = taxonomy
        self.margin = margin
        self.floor = floor
        self.min_train_clips = min_train_clips
        self.metrics = metrics or {}
        # StandardScaler leaves a 0/1 for a constant column; dividing by it would
        # produce inf and poison every prediction via one dead feature.
        if self.scales.size:
            self.scales = np.where(self.scales == 0, 1.0, self.scales)

    def probabilities(self, features: np.ndarray) -> Dict[str, float]:
        """Class probabilities for one feature vector, via softmax."""
        if features.size != self.means.size:
            raise ValueError(
                f"feature vector has {features.size} dims but the model was fitted on "
                f"{self.means.size}; re-extract or re-fit"
            )
        scaled = (np.asarray(features, dtype=np.float64) - self.means) / self.scales
        logits = self.coefficients @ scaled + self.intercepts
        logits = logits - logits.max()  # stability; cancels in the softmax
        exp = np.exp(logits)
        total = exp.sum()
        probs = exp / total if total > 0 else np.full_like(exp, 1.0 / exp.size)
        return {name: float(p) for name, p in zip(self.classes, probs)}

    def predict(
        self,
        features: np.ndarray,
        clip_id: str = "clip",
        stance: str = "auto",
        top_k: int = 3,
    ) -> Prediction:
        """Name a clip, or abstain — and report which rule fired.

        ``stance="auto"`` does **not** guess. It returns the same prediction under
        both readings so the caller picks. Anything else must be a real stance: a
        riding direction here would mirror every label silently, which is the
        failure this whole project is built to avoid.
        """
        stance_value = (stance or "auto").strip().lower()
        if stance_value not in STANCE_VALUES and stance_value != "auto":
            raise ValueError(
                f"stance must be 'auto', {list(STANCE_VALUES)}, or empty; got {stance!r}. "
                "It is the goofy/regular toggle, not a riding direction: fakie/switch/nollie "
                "cannot fix the sign frame."
            )

        probs = self.probabilities(features)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: max(top_k, 1)]
        best_label, best_prob = ranked[0]
        runner_label, runner_prob = ranked[1] if len(ranked) > 1 else ("", 0.0)
        stated_stance = None if stance_value == "auto" else stance_value
        vocabulary = list(self.classes)

        # Both abstention rules fire *before* any naming, so a clip that fails
        # either never gets a confident label attached.
        if best_prob < self.floor:
            return Prediction(
                clip_id, None, best_prob, runner_label, runner_prob, True,
                f"top class {best_label} only {best_prob:.2f} (floor {self.floor:.2f})",
                stated_stance, {}, top, vocabulary,
            )
        if best_prob - runner_prob < self.margin:
            return Prediction(
                clip_id, None, best_prob, runner_label, runner_prob, True,
                f"{best_label} {best_prob:.2f} vs {runner_label} {runner_prob:.2f} "
                f"(margin {best_prob - runner_prob:.2f} < {self.margin:.2f})",
                stated_stance, {}, top, vocabulary,
            )

        readings: Dict[str, str] = {}
        try:
            pair = trick_for_both_stances(best_label, self.taxonomy)
            readings = {"regular": pair["regular"] or "", "goofy": pair["goofy"] or ""}
        except Exception:
            # A label outside the dictionary must not break serving; it just means
            # there are no named readings to show.
            readings = {}

        if stance_value == "auto":
            return Prediction(
                clip_id, best_label, best_prob, runner_label, runner_prob, False,
                "stance not given; both readings shown", None, readings, top, vocabulary,
            )

        chosen = readings.get(stance_value) or None
        if chosen is None:
            # The mirror has no name for this stance. Saying so beats naming the
            # other reading, which would be a confidently wrong label.
            return Prediction(
                clip_id, None, best_prob, runner_label, runner_prob, True,
                f"no named trick for a {stance_value} rider (unnamed mirror)",
                stance_value, readings, top, vocabulary,
            )
        return Prediction(
            clip_id, chosen, best_prob, runner_label, runner_prob, False,
            f"stance {stance_value}", stance_value, readings, top, vocabulary,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(
                {
                    "classes": self.classes,
                    "means": self.means.tolist(),
                    "scales": self.scales.tolist(),
                    "coefficients": self.coefficients.tolist(),
                    "intercepts": self.intercepts.tolist(),
                    "margin": self.margin,
                    "floor": self.floor,
                    "min_train_clips": self.min_train_clips,
                    "metrics": self.metrics,
                },
                handle,
            )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        taxonomy: Taxonomy,
        margin: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> "Recognizer":
        with open(path) as handle:
            blob = json.load(handle)
        return cls(
            blob["classes"], np.array(blob["means"]), np.array(blob["scales"]),
            np.array(blob["coefficients"]), np.array(blob["intercepts"]), taxonomy,
            margin=blob["margin"] if margin is None else margin,
            floor=blob["floor"] if floor is None else floor,
            min_train_clips=blob.get("min_train_clips", 0),
            metrics=blob.get("metrics", {}),
        )


def fit_rotation_heads(
    manifest,
    cache_dir: Path,
    taxonomy: Taxonomy,
    include_board: bool = False,
    min_family_clips: int = 10,
    min_level_clips: int = 15,
    alpha: float = 10.0,
    C: float = 0.1,
):
    """Fit the three regressors and the family classifier; return heads plus metrics.

    ``min_family_clips`` drops the rarest body-rotation families. The plan's
    families have support 235/199/15, so a one-clip family is unlearnable and
    would only add a class nobody can score. Dropped families are **reported**,
    never silently discarded.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression, Ridge

    from .features import EXTRACTOR_VERSION, load_feature_table

    frame = manifest[manifest["dataset"] == "skateai"].copy()
    frame = frame.merge(rotation_targets(frame), on=["clip_id", "label"], how="left")

    train_all = frame[frame["split_holdout"] != "test"]
    counts = train_all["body_family"].value_counts()
    keep = [name for name, count in counts.items() if count >= min_family_clips]
    dropped = sorted(set(counts.index) - set(keep))

    train = train_all[train_all["body_family"].isin(keep)]
    holdout = frame[(frame["split_holdout"] == "test") & (frame["body_family"].isin(keep))]

    X_train, y_train, ids_train = load_feature_table(
        train, cache_dir, EXTRACTOR_VERSION, include_board=include_board
    )
    if X_train.size == 0:
        raise ValueError("no cached features; run `skateid extract`")

    # Align targets to the features by clip_id: load_feature_table drops clips with
    # no cached features, so positional indexing against `train` would silently
    # misalign the labels.
    aligned = train[train["clip_id"].isin(set(ids_train))].set_index("clip_id").loc[ids_train]

    means = X_train.mean(axis=0)
    scales = X_train.std(axis=0)
    scales = np.where(scales == 0, 1.0, scales)
    scaled = (X_train - means) / scales

    axes = {}
    for axis in RotationHeads.AXES:
        ridge = Ridge(alpha=alpha)
        ridge.fit(scaled, aligned[axis].to_numpy(dtype=float))
        axes[axis] = {
            "coef": ridge.coef_, "intercept": float(ridge.intercept_),
            "means": means, "scales": scales,
        }

    # NO class_weight here, and that is deliberate. The families are 173/158 in
    # training and 62/41 on the holdout -- near-balanced -- so `balanced` reweights
    # a two-class problem toward the smaller class and *loses* accuracy (measured:
    # 0.398 balanced vs 0.709 unweighted at C=0.1). It is the standard reflex for
    # an imbalanced target set and it is wrong for this one. Regularisation is
    # tuned by the caller's C instead.
    family_model = LogisticRegression(max_iter=3000, C=C)
    family_model.fit(scaled, aligned["body_family"].to_numpy())

    # Residual SD per axis, used to make the three axes comparable when ranking
    # dictionary triples. Plan 12.23.
    #
    # These MUST be out-of-fold. With 2448 features and 331 rows, ridge
    # interpolates the training set almost exactly, so in-sample residuals come
    # out around 0.02 -- three orders of magnitude below the real error. Dividing
    # by them inflated every distance to ~19 and made the ranking metric
    # meaningless (and the abstention sweep unable to keep a single clip).
    # K-fold gives residuals the model has not seen, at 5 fits per axis.
    from sklearn.model_selection import KFold

    axis_scales = {}
    folds = min(5, max(2, X_train.shape[0] // 20))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=0)
    for axis in RotationHeads.AXES:
        target = aligned[axis].to_numpy(float)
        oof = np.zeros_like(target)
        for train_idx, test_idx in splitter.split(scaled):
            fold = Ridge(alpha=alpha).fit(scaled[train_idx], target[train_idx])
            oof[test_idx] = fold.predict(scaled[test_idx])
        residual = target - oof
        # Floor the scale: a near-zero value would again make distances explode.
        axis_scales[axis] = max(float(np.sqrt((residual ** 2).mean())), 1e-3)

    # Option 3: per-axis integer-level classifiers. Levels below
    # min_level_clips are dropped per axis and reported, because a level with a
    # handful of clips is unlearnable and only adds a class nobody can score.
    # Note this is a *different* threshold from min_family_clips: a level can be
    # thin overall and still be the dominant value for its axis (board_spin=0 is
    # the majority of every trick class), so the two gates are not merged.
    levels = {}
    level_report = {}
    for axis in RotationHeads.AXES:
        target = aligned[axis].to_numpy(int)
        level_counts = pd.Series(target).value_counts()
        usable = [int(v) for v, c in level_counts.items() if c >= min_level_clips]
        dropped_levels = sorted(int(v) for v in level_counts.index if level_counts[v] < min_level_clips)
        mask = np.isin(target, usable)
        if len(usable) < 2 or len(set(target[mask])) < 2:
            level_report[axis] = {
                "fitted": False,
                "reason": "fewer than two usable levels",
                "levels": sorted(usable),
                "dropped_levels": dropped_levels,
            }
            continue
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(scaled[mask], target[mask])
        levels[axis] = {
            "coef": clf.coef_, "intercept": clf.intercept_,
            "means": means, "scales": scales,
            "classes": list(clf.classes_),
        }
        level_report[axis] = {
            "fitted": True,
            "levels": [int(v) for v in clf.classes_],
            "dropped_levels": dropped_levels,
            "train_counts": {int(k): int(v) for k, v in level_counts.items()},
        }

    heads = RotationHeads(
        axes,
        {"coef": family_model.coef_, "intercept": family_model.intercept_,
         "means": means, "scales": scales,
         # The class order coef_ rows correspond to. For a 2-class problem sklearn
         # emits a single row and this is what makes the ordering unambiguous.
         "classes": list(family_model.classes_)
         if family_model.coef_.shape[0] == len(family_model.classes_)
         else [family_model.classes_[0], family_model.classes_[1]]},
        list(family_model.classes_),
        taxonomy,
        include_board=include_board,
        axis_scales=axis_scales,
        levels=levels,
    )

    metrics = {
        "families": list(family_model.classes_),
        "dropped_families": dropped,
        "min_family_clips": min_family_clips,
        "min_level_clips": min_level_clips,
        "alpha": alpha,
        "axis_scales": axis_scales,
        "level_report": level_report,
        "train_clips": int(X_train.shape[0]),
        "dims": int(X_train.shape[1]),
        "holdout_clips": 0,
        "family_accuracy": None,
        "family_dummy_accuracy": None,
        "axis_mae": {},
        "name_accuracy": None,
        "name_hit_rate": None,
        # Ranking metrics (option 1) and residual confidence (option 2).
        "ranked_name_accuracy": None,
        "ranked_top3_accuracy": None,
        "mean_residual_distance": None,
    }

    X_test, y_test, ids_test = load_feature_table(
        holdout, cache_dir, EXTRACTOR_VERSION, include_board=include_board
    )
    if X_test.size:
        truth = holdout[holdout["clip_id"].isin(set(ids_test))].set_index("clip_id").loc[ids_test]
        predicted_families = [heads.predict_family(row)[0] for row in X_test]
        metrics["holdout_clips"] = int(len(y_test))
        metrics["family_accuracy"] = float(
            np.mean([g == t for g, t in zip(predicted_families, truth["body_family"])])
        )
        dummy = DummyClassifier(strategy="most_frequent").fit(
            scaled, aligned["body_family"].to_numpy()
        )
        metrics["family_dummy_accuracy"] = float(
            dummy.score((X_test - means) / scales, truth["body_family"])
        )
        metrics["family_counts_holdout"] = truth["body_family"].value_counts().to_dict()
        metrics["family_counts_train"] = aligned["body_family"].value_counts().to_dict()

        for axis in RotationHeads.AXES:
            predicted = np.array([heads.predict_rotations(row)[axis] for row in X_test])
            metrics["axis_mae"][axis] = float(
                np.abs(predicted - truth[axis].to_numpy(float)).mean()
            )
        named = [heads.name_for(heads.predict_rotations(row)) for row in X_test]
        metrics["name_hit_rate"] = float(np.mean([n is not None for n in named]))
        metrics["name_accuracy"] = float(np.mean([n == t for n, t in zip(named, truth["label"])]))

        # Ranking (option 1) is now always defined: every prediction lands on some
        # dictionary triple, so the old hit-rate question is obsolete. What matters
        # is top-1 / top-3 accuracy and the residual that drives abstention (option 2).
        labels = truth["label"].to_numpy()
        top1 = [heads.rank_names(heads.predict_rotations(row), top_k=3) for row in X_test]
        metrics["ranked_name_accuracy"] = float(
            np.mean([bool(r) and r[0][0] == t for r, t in zip(top1, labels)])
        )
        metrics["ranked_top3_accuracy"] = float(
            np.mean([bool(r) and t in [n for n, _ in r] for r, t in zip(top1, labels)])
        )
        distances = [r[0][1] for r in top1 if r]
        metrics["mean_residual_distance"] = float(np.mean(distances)) if distances else None
        # Sanity check that ranking and rounding can now disagree; if they ever
        # match exactly the scaling is a no-op and one of the two paths is dead code.
        metrics["ranked_vs_rounded_disagreements"] = int(
            sum(1 for r, n in zip(top1, named) if r and r[0][0] != n)
        )

        # Option 3 evaluation: does per-axis classification beat regression? Judged
        # on the per-axis level itself (the honest unit) and on the resulting name.
        metrics["level_accuracy"] = {}
        metrics["level_dummy_accuracy"] = {}
        for axis in RotationHeads.AXES:
            if axis not in heads.levels:
                metrics["level_accuracy"][axis] = None
                metrics["level_dummy_accuracy"][axis] = None
                continue
            predicted = [heads.predict_levels(row)[axis][0] for row in X_test]
            actual = truth[axis].to_numpy(int)
            # Only score levels the classifier was actually fitted to predict;
            # scoring a dropped level would measure "can it guess the unseen".
            known = np.isin(actual, np.array(heads.levels[axis]["classes"]))
            metrics["level_accuracy"][axis] = (
                float(np.mean([p == a for p, a in zip(predicted, actual)]))
                if known.any() else None
            )
            train_mask = np.isin(aligned[axis].to_numpy(int),
                                 np.array(heads.levels[axis]["classes"]))
            majority = (
                pd.Series(aligned[axis].to_numpy(int)[train_mask])
                .value_counts().idxmax()
            )
            metrics["level_dummy_accuracy"][axis] = (
                float(np.mean(actual[known] == majority)) if known.any() else None
            )

        level_names, level_confs = [], []
        for row, lbl in zip(X_test, truth["label"].to_numpy()):
            levels_pred = heads.predict_levels(row)
            triple = heads.rotations_from_levels(levels_pred)
            name = heads.name_for(triple)
            joint, _ = heads.level_confidence(row)
            level_names.append(name)
            level_confs.append(joint)
        metrics["level_name_accuracy"] = float(
            np.mean([n == t for n, t in zip(level_names, truth["label"].to_numpy())])
        )
        metrics["level_name_hit_rate"] = float(np.mean([n is not None for n in level_names]))
        # This is the first configuration where confidence is expected to carry
        # information: it is a product of per-axis classification probabilities,
        # not a distance in rotation space (which measured flat in 12.23).
        finite = [(c, n == t) for c, n, t in zip(level_confs, level_names, truth["label"].to_numpy())]
        for cut in (0.1, 0.2, 0.3, 0.4, 0.5):
            kept = [(c, ok) for c, ok in finite if c >= cut]
            metrics.setdefault("level_abstention_sweep", {})[str(cut)] = {
                "kept": len(kept),
                "accuracy": float(np.mean([ok for _, ok in kept])) if kept else None,
                "coverage": len(kept) / len(finite) if finite else 0.0,
            }

    heads.metrics = metrics
    return heads, metrics


def fit(
    manifest,
    cache_dir: Path,
    taxonomy: Taxonomy,
    include_board: bool = False,
    min_train_clips: int = 15,
    C: float = 0.1,
    margin: float = DEFAULT_MARGIN,
    floor: float = DEFAULT_FLOOR,
):
    """Fit the pose classifier on the training split and report what it scored.

    ``min_train_clips`` implements the restricted vocabulary (plan 12.13): classes
    with fewer training clips are not learnable, and including them measures the
    dataset rather than the representation. They are **not lost** — each is still
    expressible through the rotation heads — and the excluded names are returned so
    the caller reports them rather than dropping them silently.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from .features import EXTRACTOR_VERSION, load_feature_table

    frame = manifest[manifest["dataset"] == "skateai"]
    train_all = frame[frame["split_holdout"] != "test"]
    counts = train_all["label"].value_counts()
    keep = set(counts[counts >= min_train_clips].index) if min_train_clips else set(counts.index)
    excluded = sorted(set(frame["label"]) - keep)

    train = train_all[train_all["label"].isin(keep)]
    holdout = frame[(frame["split_holdout"] == "test") & (frame["label"].isin(keep))]

    X_train, y_train, _ = load_feature_table(
        train, cache_dir, EXTRACTOR_VERSION, include_board=include_board
    )
    if X_train.size == 0:
        raise ValueError("no cached features for the training split; run `skateid extract`")

    means = X_train.mean(axis=0)
    scales = X_train.std(axis=0)
    scales = np.where(scales == 0, 1.0, scales)

    model = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
    model.fit((X_train - means) / scales, y_train)

    recognizer = Recognizer(
        classes=list(model.classes_),
        means=means,
        scales=scales,
        # sklearn gives (1, D) for binary and (K, D) otherwise; normalise to (K, D).
        coefficients=model.coef_ if model.coef_.shape[0] > 1 else model.coef_.T,
        intercepts=model.intercept_ if model.intercept_.size > 1 else model.intercept_,
        taxonomy=taxonomy,
        margin=margin,
        floor=floor,
        min_train_clips=min_train_clips,
    )

    metrics = {
        "classes": len(keep),
        "excluded_classes": excluded,
        "train_clips": int(X_train.shape[0]),
        "dims": int(X_train.shape[1]),
        "holdout_clips": 0,
        "macro_f1_all_named": None,
        "abstain_rate": None,
        "macro_f1_when_named": None,
        "accuracy_when_named": None,
    }

    X_test, y_test, _ = load_feature_table(
        holdout, cache_dir, EXTRACTOR_VERSION, include_board=include_board
    )
    if X_test.size:
        # Score the *abstaining* predictor, not the raw argmax, so the number
        # reported is the one the user will actually experience.
        raw = [recognizer.predict(row, top_k=1).label or "__abstain__" for row in X_test]
        metrics["holdout_clips"] = int(X_test.shape[0])
        metrics["macro_f1_all_named"] = float(f1_score(y_test, raw, average="macro"))
        named = [(t, p) for t, p in zip(y_test, raw) if p != "__abstain__"]
        metrics["abstain_rate"] = float(1 - len(named) / max(len(y_test), 1))
        if named:
            metrics["macro_f1_when_named"] = float(
                f1_score([t for t, _ in named], [p for _, p in named], average="macro")
            )
            metrics["accuracy_when_named"] = float(
                sum(1 for t, p in named if t == p) / len(named)
            )
    recognizer.metrics = metrics
    return recognizer, metrics


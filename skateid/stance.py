"""The goofy/regular toggle that selects the sign frame (plan section 3).

Plan section 3's rule is that **the sign frame is an input, not a property of
the trick**. A kickflip is ``+360`` read by a regular rider and ``-360`` read by
a goofy one, so one stored rotation can serve both only if the reader knows
which frame to read it in. That knowledge is what this module carries, and it
arrives from the *input* side: a toggle the user sets, with an optional
suggestion attached.

Why this is not derived from the data:

- ``stance_published`` holds SkateAI's riding direction (fakie/switch/nollie,
  plus ``regular`` meaning *natural stance relative to the skater*). None of
  those say whether the skater is goofy or regular, so the sign frame cannot be
  recovered from them. A ``regular`` value and a ``stance_input`` of ``regular``
  are different facts that happen to share a word.
- A wrong answer here is the worst kind of quiet failure in this project: it
  does not crash, it just negates every sign, swapping kick<->heel and fs<->bs
  on every clip in the dataset. So the default is **no stance at all**, and a
  value is only ever written when a human or a reviewable heuristic commits it.

The suggestion path exists because making the user set a toggle for all 671
clips is not a real product, but it deliberately produces a *suggestion*, never a
value: :func:`apply_to_manifest` writes only what was explicitly confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional

import pandas as pd

from .taxonomy import STANCE_VALUES, Taxonomy

#: Column holding the confirmed goofy/regular toggle. Empty means "unknown", and
#: unknown is a supported, meaningful state.
STANCE_COLUMN = "stance_input"

#: Column holding the per-clip suggestion's confidence, kept separate so a
#: suggestion can never be mistaken for a confirmed value by a reader, or by a
#: later code path that has forgotten which is which.
SUGGESTION_COLUMN = "stance_suggested_confidence"

#: A suggestion weaker than this is recorded but never auto-applied. Low by
#: design: the cost of a wrong stance is a systematic sign flip, so the heuristic
#: has to be clearly right before anyone stops checking it.
CONFIDENCE_FLOOR = 0.8


@dataclass(frozen=True)
class StanceSuggestion:
    """A *proposed* stance plus why it was proposed and how confident it is.

    Deliberately not a bare string. The confidence exists so the caller can
    refuse to act on a weak suggestion, and carrying it in the same object stops
    a bare string from being passed somewhere it would be acted on.
    """

    stance: str
    confidence: float
    #: Free-form note about what the decision rested on, shown in the CLI.
    reason: str

    def __post_init__(self) -> None:
        if self.stance not in STANCE_VALUES:
            raise ValueError(f"stance must be one of {list(STANCE_VALUES)}, got {self.stance!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")

    @property
    def confident(self) -> bool:
        return self.confidence >= CONFIDENCE_FLOOR

    def as_dict(self) -> dict:
        return {"stance": self.stance, "confidence": self.confidence, "reason": self.reason}

    def __repr__(self) -> str:
        return f"StanceSuggestion({self.stance!r}, {self.confidence:.2f})"


def suggest_from_land_foot(left_foot_forward: Optional[bool]) -> StanceSuggestion:
    """Propose a stance from which foot leads, the cue a person actually uses.

    A regular rider leads with the left foot, a goofy rider with the right. This
    is a *suggestion* from a single frame's worth of evidence, so it is returned
    with low confidence and never written without confirmation.

    ``None`` (no skater detected, or a close-up of the board) returns a
    0.0-confidence proposal so the caller still has something to display;
    :attr:`StanceSuggestion.confident` is False and :func:`apply_to_manifest`
    will not write it.
    """
    if left_foot_forward is None:
        return StanceSuggestion("regular", 0.0, "no skater pose detected")
    stance = "regular" if left_foot_forward else "goofy"
    return StanceSuggestion(
        stance,
        0.5,
        "left foot leads (regular)" if left_foot_forward else "right foot leads (goofy)",
    )


def suggest_from_sequence(left_foot_forward_by_frame: List[Optional[bool]]) -> StanceSuggestion:
    """Vote across a clip's frames rather than trusting one.

    Agreement across frames is the only thing that separates a stable lead foot
    from one frame of occlusion or a bail, so this votes and reports the winning
    share as the confidence. A unanimous vote is still not proof, which is why
    the confidence is capped below :data:`CONFIDENCE_FLOOR` and this path is
    always reviewed by a human. That is intentional, not a bug to tune away.
    """
    votes = [value for value in left_foot_forward_by_frame if value is not None]
    if not votes:
        return StanceSuggestion("regular", 0.0, "no skater pose detected in any frame")

    agree = sum(1 for value in votes if value)
    other = len(votes) - agree
    if agree == other:  # an exact tie, no majority to act on
        return StanceSuggestion("regular", 0.0, f"lead foot tied {agree}/{len(votes)}")

    share = agree / len(votes)
    stance = "regular" if agree > other else "goofy"
    # Reaches 0.75 at unanimity, which is below CONFIDENCE_FLOOR (0.8): even a
    # clean vote is reviewed rather than trusted.
    confidence = min(0.75, 0.5 + abs(share - 0.5))
    return StanceSuggestion(stance, confidence, f"left foot leads in {agree}/{len(votes)} frames")


def apply_to_manifest(
    frame: pd.DataFrame,
    overrides: Optional[Mapping[str, str]] = None,
    suggestions: Optional[Mapping[str, StanceSuggestion]] = None,
) -> pd.DataFrame:
    """Return ``frame`` with the stance columns filled in from *confirmed* input.

    Precedence, and the whole point of the function:

    1. a clip in ``overrides`` is written, whatever else is true of it;
    2. a clip with a ``confident`` suggestion is written;
    3. everything else is left **empty**.

    A weak suggestion is recorded in the suggestion column but never promoted, so
    the manifest's meaning is unchanged for any clip a human has not looked at.
    :meth:`~skateid.taxonomy.Taxonomy.validate_frame` then re-checks the result,
    and a value outside {regular, goofy} still fails loudly.
    """
    if "clip_id" not in frame.columns:
        raise ValueError("frame must have a clip_id column to key stance overrides on")

    out = frame.copy()
    out[STANCE_COLUMN] = out[STANCE_COLUMN].fillna("").astype(object) if STANCE_COLUMN in out.columns else ""
    clip_ids = out["clip_id"].astype(str)

    normalized = {str(k): str(v).strip().lower() for k, v in (overrides or {}).items()}
    for clip_id, value in normalized.items():
        if value not in STANCE_VALUES:
            raise ValueError(
                f"override for {clip_id!r} is {value!r}, which must be one of "
                f"{list(STANCE_VALUES)}. It is the goofy/regular toggle, not a riding "
                "direction: fakie/switch/nollie cannot fix the sign frame."
            )

    if suggestions:
        out[SUGGESTION_COLUMN] = [
            suggestions[clip_id].confidence if clip_id in suggestions else ""
            for clip_id in clip_ids
        ]

    confirmed = clip_ids.map(normalized)
    if suggestions:
        # Only *confident* suggestions are promoted; the rest stay empty.
        auto = clip_ids.map(
            {
                clip_id: suggestion.stance
                for clip_id, suggestion in suggestions.items()
                if suggestion.confident
            }
        )
        confirmed = confirmed.fillna(auto)

    existing = out[STANCE_COLUMN].astype(str).str.strip()
    out[STANCE_COLUMN] = confirmed.where(confirmed.notna(), existing)
    out[STANCE_COLUMN] = out[STANCE_COLUMN].replace("", pd.NA)
    return out


def trick_for_both_stances(label: object, taxonomy: Taxonomy) -> dict:
    """Return the trick name as read by a regular and a goofy rider.

    The ambiguity is a clean binary, and this project already made it tractable:
    the stance mirror is a **bijection** over all 41 rotation-expressible names
    (verified: 41 named, 0 unmirrored), so every trick has an exact partner name
    to give. That makes "which trick is it?" answerable *up to* the stance
    question, which is a far better failure mode than guessing.

    This is deliberately **not** a guess. A clip of one rider doing a kickflip is
    evidence about ``{kickflip, heelflip}`` and cannot distinguish the two without
    knowing which foot leads. Returning both, labelled, is the honest answer and
    costs nothing extra to compute -- it is one dictionary lookup on a map the
    taxonomy already builds.

    ``unresolved`` is returned when the mirror has no name (which cannot happen
    for the current dictionary, but is checked rather than assumed, because the
    whole point of this project is not assuming).
    """
    canonical = taxonomy.normalize_label(label)
    rotation = taxonomy.rotation_for_label(canonical)
    mirrored = taxonomy.name_for_mirrored(rotation)

    by_stance = {"regular": canonical, "goofy": None}
    if mirrored is not None:
        by_stance["goofy"] = mirrored
    return {
        "label": canonical,
        "regular": canonical,
        "goofy": mirrored,
        # True when the two stances are actually distinguishable by name, i.e.
        # the mirror is a different trick and not the same one relabelled.
        "stance_dependent": mirrored is not None and mirrored != canonical,
        "unresolved": mirrored is None,
    }


def stance_summary(frame: pd.DataFrame) -> dict:
    """Counts of confirmed/suggested/empty stances, for the CLI and the docs."""
    filled = frame[STANCE_COLUMN].fillna("").astype(str).str.strip().str.lower()
    populated = filled[filled != ""]
    return {
        "rows": int(len(frame)),
        "confirmed": int(len(populated)),
        "empty": int((filled == "").sum()),
        "goofy": int((populated == "goofy").sum()),
        "regular": int((populated == "regular").sum()),
    }

"""Flatground trick taxonomy: the rotation dictionary and the scope guardrail.

Two artefacts define what this project is allowed to recognise (plan section 5):

* ``data/tricks.json`` - the **rotation dictionary**. Maps a canonical trick name
  to the three rotations the model predicts (plan section 4). It is the
  authority: a row's ``label`` is *derived* from its rotation triple, not from
  whatever the upstream dataset happened to call the trick.
* ``data/flatground_allowlist.csv`` - the **name registry**. Canonical names plus
  the human and upstream spellings (aliases) that normalise onto them.

Only the dictionary stores rotations and only the registry stores aliases, so
there is exactly one place to change either. ``Taxonomy`` ties them together and
provides the two operations the rest of the package needs:
``label_from_rotation`` (the derivation used at ingestion, and later at
inference: rotations in, canonical name out) and ``validate_frame`` (the
guardrail - a row outside this vocabulary is an error, not a warning).

Why derive labels instead of matching names
-------------------------------------------
SkateAI publishes labels as raw jargon *plus* decomposed rotations, and its 31
names sit in exact 1:1 correspondence with its 31 triples. That makes the triple
the stable key and the spelling the unreliable one: upstream writes ``treflip``
where this project writes ``tre_flip``, ``shovit`` for ``pop_shuvit``, and so
on. Name matching would need a hand-maintained alias per spelling and still
could not detect a clip whose name and components disagree. Deriving from
rotations cannot disagree with itself, and the registry aliases are
cross-checked against it.

Sign convention (plan section 4)
--------------------------------
Rotations are signed integers in the stance-normalized frame: ``flip`` counts
whole 360 deg flips (``+`` kickflip, ``-`` heelflip); ``board_spin`` and
``body_spin`` count whole 180 deg rotations (``+`` backside, ``-`` frontside).
``backside`` is positive because that is what the ingested data uses: SkateAI's
``treflip`` clips carry ``board_rotation_type='backside'`` with number ``2``,
and ``tre_flip`` is ``board_spin=2``. Magnitudes are clamped to ``0..3``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

DEFAULT_TRICKS_PATH = Path("data/tricks.json")
DEFAULT_ALLOWLIST_PATH = Path("data/flatground_allowlist.csv")

#: Vocabulary of the manifest's component columns.
FLIP_TYPES: Tuple[str, ...] = ("none", "kickflip", "heelflip")
SPIN_TYPES: Tuple[str, ...] = ("none", "backside", "frontside")

#: Plan section 4 clamps a quantized magnitude to 0..3.
MAX_ROTATION_NUMBER = 3

#: The six component columns, in manifest order.
ROTATION_COLUMNS: Tuple[str, ...] = (
    "flip_type",
    "flip_number",
    "board_rotation_type",
    "board_rotation_number",
    "body_rotation_type",
    "body_rotation_number",
)

#: Maps a component column pair onto the axis name used by ``tricks.json``.
_AXES: Tuple[Tuple[str, str, str], ...] = (
    ("flip", "flip_type", "flip_number"),
    ("board_spin", "board_rotation_type", "board_rotation_number"),
    ("body_spin", "body_rotation_type", "body_rotation_number"),
)

#: Stance is the goofy/regular toggle that fixes the sign convention (plan section 3).
#: It is *not* the riding direction a dataset publishes, which is why the manifest
#: keeps the two in separate columns.
STANCE_VALUES: Tuple[str, ...] = ("regular", "goofy")

#: What a dataset may publish instead of the goofy/regular toggle: riding
#: directions and pop types. None of these can be converted into a stance, which
#: is exactly why `stance_published` must never feed the sign convention.
PUBLISHED_RIDING_VALUES: Tuple[str, ...] = ("regular", "switch", "fakie", "nollie")

_NEGATIVE_TYPE: Mapping[str, str] = {
    "flip": "heelflip",
    "board_spin": "frontside",
    "body_spin": "frontside",
}
_POSITIVE_TYPE: Mapping[str, str] = {
    "flip": "kickflip",
    "board_spin": "backside",
    "body_spin": "backside",
}
_AXIS_TYPES: Mapping[str, Tuple[str, ...]] = {
    "flip": FLIP_TYPES,
    "board_spin": SPIN_TYPES,
    "body_spin": SPIN_TYPES,
}


class ScopeError(ValueError):
    """Raised when data violates the flatground scope guardrail (plan section 5)."""


def split_axis(axis: str, signed: int) -> Tuple[str, int]:
    """Decompose a signed rotation into its manifest ``(type, number)`` form."""
    if isinstance(signed, bool) or not isinstance(signed, int):
        raise ScopeError(f"{axis}={signed!r} must be a signed integer")
    if abs(signed) > MAX_ROTATION_NUMBER:
        raise ScopeError(
            f"{axis}={signed} exceeds the +/-{MAX_ROTATION_NUMBER} the quantizer allows "
            "(plan section 4); clamp before building the dictionary"
        )
    if signed == 0:
        return "none", 0
    return (_POSITIVE_TYPE[axis] if signed > 0 else _NEGATIVE_TYPE[axis]), abs(signed)


def join_axis(axis: str, type_value: str, number: object) -> int:
    """Recompose a manifest ``(type, number)`` pair into a signed rotation."""
    if type_value not in _AXIS_TYPES[axis]:
        raise ScopeError(f"{axis} type {type_value!r} not one of {_AXIS_TYPES[axis]}")
    try:
        numeric = float(number)
    except (TypeError, ValueError):
        raise ScopeError(f"{axis} number {number!r} is not numeric") from None
    if numeric != int(numeric) or numeric < 0:
        raise ScopeError(f"{axis} number {number!r} must be a non-negative whole number")
    number = int(numeric)
    if number > MAX_ROTATION_NUMBER:
        raise ScopeError(f"{axis} number {number} exceeds {MAX_ROTATION_NUMBER}")
    if type_value == "none":
        if number != 0:
            raise ScopeError(f"{axis} type 'none' cannot carry number {number}")
        return 0
    if number == 0:
        raise ScopeError(f"{axis} type {type_value!r} cannot carry number 0")
    return number if type_value == _POSITIVE_TYPE[axis] else -number


@dataclass(frozen=True, order=True)
class Rotation:
    """A quantized ``(flip, board_spin, body_spin)`` triple in the stance frame.

    Stance is deliberately **not** a field of the triple. It is a separate input
    that selects the frame (plan sections 3 and 7): a kickflip is +360 for a
    regular rider and -360 for a goofy one, and the stance normalisation in the
    feature extractor is what makes a single stored value serve both. Keeping
    stance out of the key is what lets one clip's mirror cover the other stance
    for free, and what keeps :meth:`name_for` a pure function of the triple.

    Use :meth:`mirrored` to move between the two stances' raw readings.
    """

    flip: int = 0
    board_spin: int = 0
    body_spin: int = 0

    def mirrored(self) -> "Rotation":
        """The same trick as read by a rider in the opposite stance.

        Negating all three axes is exactly the geometric mirror (plan section 7
        sign-flips the x-axis), and it is what turns a regular rider's kickflip
        (+360) into a goofy rider's (-360). Note that mirroring need not land on
        a *named* trick: the mirror of ``bs_biggerspin_kickflip`` is a frontside
        biggerspin heelflip, which no dataset publishes. :func:`Taxonomy.
        name_for_mirrored` reports that rather than inventing a name.
        """
        return Rotation(-self.flip, -self.board_spin, -self.body_spin)

    def components(self) -> Dict[str, object]:
        """The six manifest columns implied by this triple."""
        out: Dict[str, object] = {}
        for axis, type_col, number_col in _AXES:
            type_value, number = split_axis(axis, getattr(self, axis))
            out[type_col] = type_value
            out[number_col] = number
        return out

    def as_dict(self) -> Dict[str, int]:
        return {axis: getattr(self, axis) for axis, _, _ in _AXES}

    @classmethod
    def from_components(
        cls,
        flip_type: str,
        flip_number: object,
        board_rotation_type: str,
        board_rotation_number: object,
        body_rotation_type: str,
        body_rotation_number: object,
    ) -> "Rotation":
        """Build a triple from the manifest's ``(type, number)`` columns."""
        return cls(
            flip=join_axis("flip", flip_type, flip_number),
            board_spin=join_axis("board_spin", board_rotation_type, board_rotation_number),
            body_spin=join_axis("body_spin", body_rotation_type, body_rotation_number),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, int]) -> "Rotation":
        """Build a triple from a ``tricks.json`` entry (missing axes default to 0)."""
        unknown = set(values) - {axis for axis, _, _ in _AXES}
        if unknown:
            raise ScopeError(f"unknown rotation axes {sorted(unknown)}")
        return cls(**{axis: int(values.get(axis, 0)) for axis, _, _ in _AXES})

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "Rotation":
        return cls.from_components(
            str(row["flip_type"]),
            row["flip_number"],
            str(row["board_rotation_type"]),
            row["board_rotation_number"],
            str(row["body_rotation_type"]),
            row["body_rotation_number"],
        )

    def __str__(self) -> str:
        return f"flip={self.flip:+d} board={self.board_spin:+d} body={self.body_spin:+d}"


AXIS_NAMES: Tuple[str, ...] = tuple(axis for axis, _, _ in _AXES)


@dataclass(frozen=True)
class Trick:
    """One entry of the rotation dictionary."""

    name: str
    display: str
    rotation: Optional[Rotation]
    note: str = ""

    @property
    def rotation_expressible(self) -> bool:
        return self.rotation is not None


class TrickDictionary:
    """Canonical name <-> rotation triple, loaded from ``tricks.json``."""

    def __init__(self, tricks: Mapping[str, Trick]) -> None:
        self.tricks: Dict[str, Trick] = dict(tricks)
        self._by_rotation: Dict[Rotation, str] = {}
        for name, trick in self.tricks.items():
            if trick.rotation is None:
                continue
            clash = self._by_rotation.get(trick.rotation)
            if clash is not None:
                raise ScopeError(
                    f"{name!r} and {clash!r} both claim rotation {trick.rotation}. A "
                    "triple must name exactly one trick or the label cannot be derived; "
                    "if two jargon names share a triple (half cab vs backside 180), keep "
                    "one here and make the other an alias in flatground_allowlist.csv."
                )
            self._by_rotation[trick.rotation] = name

    @classmethod
    def load(cls, path: Path | str = DEFAULT_TRICKS_PATH) -> "TrickDictionary":
        with open(Path(path), encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ScopeError("tricks.json must contain a JSON object")

        tricks: Dict[str, Trick] = {}
        for name, entry in raw.items():
            if name.startswith("_"):  # documentation keys such as _sign_convention
                continue
            if not isinstance(entry, dict):
                raise ScopeError(f"tricks.json entry {name!r} must be an object")
            expressible = bool(entry.get("rotation_expressible", True))
            rotation: Optional[Rotation] = None
            if expressible:
                missing = [axis for axis in AXIS_NAMES if axis not in entry]
                if missing:
                    raise ScopeError(
                        f"tricks.json entry {name!r} is missing rotation axes {missing}. "
                        "Set \"rotation_expressible\": false with a \"note\" instead if the "
                        "trick cannot be expressed in the flip/board/body model."
                    )
                rotation = Rotation.from_dict({axis: entry[axis] for axis in AXIS_NAMES})
            tricks[name] = Trick(
                name=name,
                display=str(entry.get("display", name)),
                rotation=rotation,
                note=str(entry.get("note", "")),
            )
        if not tricks:
            raise ScopeError("tricks.json defines no tricks")
        return cls(tricks)

    def names(self) -> List[str]:
        """Every canonical name, sorted."""
        return sorted(self.tricks)

    def expressible_names(self) -> List[str]:
        """Canonical names that have a rotation triple, sorted."""
        return sorted(name for name, trick in self.tricks.items() if trick.rotation_expressible)

    def label_from_rotation(self, rotation: Rotation) -> Optional[str]:
        """Canonical name for a triple, or ``None`` if the triple is unknown."""
        return self._by_rotation.get(rotation)

    def rotation_for_label(self, label: str) -> Optional[Rotation]:
        """Triple for a canonical name, or ``None`` if it has no rotation form."""
        trick = self.tricks.get(label)
        return trick.rotation if trick else None

    def is_expressible(self, label: str) -> bool:
        trick = self.tricks.get(label)
        return bool(trick and trick.rotation_expressible)

    def display(self, label: str) -> str:
        trick = self.tricks.get(label)
        return trick.display if trick else label


def normalize_name(value: object) -> str:
    """Lowercase and collapse whitespace, so aliases match the way people write them."""
    return " ".join(str(value).strip().lower().split())


@dataclass(frozen=True)
class AllowlistEntry:
    """A canonical name, its category, and every spelling that normalises onto it."""

    canonical_name: str
    category: str
    aliases: Tuple[str, ...]


class FlatgroundAllowlist:
    """The name registry: canonical names plus aliases, from ``flatground_allowlist.csv``."""

    def __init__(self, entries: Mapping[str, AllowlistEntry]) -> None:
        self.entries: Dict[str, AllowlistEntry] = dict(entries)
        self._lookup: Dict[str, str] = {}
        for name, entry in self.entries.items():
            for form in (name, *entry.aliases):
                clash = self._lookup.get(form)
                if clash is not None and clash != name:
                    raise ScopeError(
                        f"spelling {form!r} resolves to both {clash!r} and {name!r}; "
                        "an alias must be unambiguous"
                    )
                self._lookup[form] = name

    @classmethod
    def load(cls, path: Path | str = DEFAULT_ALLOWLIST_PATH) -> "FlatgroundAllowlist":
        frame = pd.read_csv(Path(path), dtype=str).fillna("")
        entries: Dict[str, AllowlistEntry] = {}
        for row in frame.to_dict("records"):
            name = normalize_name(row.get("canonical_name", ""))
            if not name:
                raise ScopeError(f"{path}: a row has no canonical_name")
            if name in entries:
                raise ScopeError(f"{path}: duplicate canonical_name {name!r}")
            aliases = tuple(
                alias
                for alias in (normalize_name(item) for item in str(row.get("aliases", "")).split(";"))
                if alias and alias != name
            )
            entries[name] = AllowlistEntry(
                canonical_name=name,
                category=str(row.get("category", "")).strip().lower(),
                aliases=aliases,
            )
        if not entries:
            raise ScopeError(f"{path} defines no canonical names")
        return cls(entries)

    def canonical_names(self) -> List[str]:
        return sorted(self.entries)

    def contains(self, name: object) -> bool:
        """True when ``name`` is a canonical name (not merely an alias)."""
        return normalize_name(name) in self.entries

    def canonical(self, raw: object) -> Optional[str]:
        """Resolve any known spelling to its canonical name, else ``None``."""
        return self._lookup.get(normalize_name(raw))

    def aliases(self, name: str) -> Tuple[str, ...]:
        entry = self.entries.get(normalize_name(name))
        return entry.aliases if entry else ()

    def is_flatground(self, name: object) -> bool:
        entry = self.entries.get(normalize_name(name))
        return bool(entry and entry.category == "flatground")


class Taxonomy:
    """The dictionary and the registry together, plus the rules that bind them.

    Constructing one verifies the two artefacts agree, so a trick can never be
    half-added: a rotation without a name, or a name with no way to derive it.
    """

    def __init__(self, dictionary: TrickDictionary, allowlist: FlatgroundAllowlist) -> None:
        self.dictionary = dictionary
        self.allowlist = allowlist
        self._check_agreement()

    def _check_agreement(self) -> None:
        missing = [name for name in self.dictionary.expressible_names() if not self.allowlist.contains(name)]
        if missing:
            raise ScopeError(
                "these rotation-dictionary names have no row in flatground_allowlist.csv: "
                f"{missing}. Add them, or remove them from tricks.json."
            )
        unknown = [name for name in self.allowlist.canonical_names() if name not in self.dictionary.tricks]
        if unknown:
            raise ScopeError(
                "these allowlisted names have no tricks.json entry: "
                f"{unknown}. Give each one a rotation, or mark it \"rotation_expressible\": "
                "false with a \"note\" explaining how it is represented instead."
            )

    @classmethod
    def load(
        cls,
        tricks_path: Path | str = DEFAULT_TRICKS_PATH,
        allowlist_path: Path | str = DEFAULT_ALLOWLIST_PATH,
    ) -> "Taxonomy":
        return cls(TrickDictionary.load(tricks_path), FlatgroundAllowlist.load(allowlist_path))

    # --- normalisation -----------------------------------------------------

    def normalize_label(self, raw: object) -> str:
        """Resolve any known spelling to a canonical name; raise on anything else."""
        canonical = self.allowlist.canonical(raw)
        if canonical is None:
            raise ScopeError(
                f"{raw!r} is not a flatground trick name. Add it to "
                "data/flatground_allowlist.csv (and to tricks.json if it has a rotation "
                "form), otherwise it cannot be ingested."
            )
        return canonical

    def label_from_rotation(self, rotation: Rotation) -> str:
        """Canonical name for a rotation triple; raise if the dictionary lacks it."""
        name = self.dictionary.label_from_rotation(rotation)
        if name is None:
            raise ScopeError(
                f"rotation ({rotation}) is not in data/tricks.json. Add it there (and a "
                "matching allowlist row), or the clip has no name to be given."
            )
        return name

    def rotation_for_label(self, label: object) -> Rotation:
        """Rotation triple for a canonical name; raise if the trick has no triple."""
        canonical = self.normalize_label(label)
        rotation = self.dictionary.rotation_for_label(canonical)
        if rotation is None:
            note = self.dictionary.tricks[canonical].note or "No explanation recorded."
            raise ScopeError(
                f"label {canonical!r} is allowlisted but has no rotation triple, so it cannot "
                f"be recovered from model output. {note}"
            )
        return rotation

    def name_for_mirrored(self, rotation: Rotation) -> Optional[str]:
        """Name for the same trick read by a rider in the opposite stance.

        Returns ``None`` when the mirror is not a named trick in this
        dictionary, rather than guessing. **Mirroring does not close**: 6 of the
        35 rotation-expressible names mirror to a partner that no dataset
        publishes (a frontside biggerspin heelflip, a frontside tre double flip,
        ...). This is the documented caveat on the plan's mirror-with-label-swap
        augmentation -- mirroring such a clip is still valid *input*
        augmentation, but its swapped label has no name to land on.
        """
        return self.dictionary.label_from_rotation(rotation.mirrored())

    def mirror_closure(self) -> Dict[str, object]:
        """Which names survive a stance mirror, for the docs and the tests."""
        named: List[str] = []
        unmirrored: List[str] = []
        for name in self.dictionary.expressible_names():
            rotation = self.dictionary.rotation_for_label(name)
            partner = self.dictionary.label_from_rotation(rotation.mirrored())
            (named if partner is not None else unmirrored).append(name)
        return {"named": named, "unmirrored": unmirrored}

    # --- the guardrail -----------------------------------------------------

    def _row_violations(self, row: Mapping[str, object], where: str) -> List[str]:
        empty = [
            column
            for column in ROTATION_COLUMNS
            if pd.isna(row.get(column)) or str(row.get(column, "")).strip() == ""
        ]
        if empty:
            return [f"{where}: rotation columns {empty} are empty (every row needs a triple)"]

        try:
            rotation = Rotation.from_row(row)
        except ScopeError as exc:
            return [f"{where}: {exc}"]

        label = normalize_name(row.get("label", ""))
        problems: List[str] = []
        derived = self.dictionary.label_from_rotation(rotation)
        if derived is None:
            problems.append(f"{where}: rotation ({rotation}) is not in tricks.json")
        elif derived != label:
            problems.append(
                f"{where}: label {label!r} disagrees with its own rotations "
                f"({rotation} derives {derived!r})"
            )
        if not self.allowlist.contains(label):
            problems.append(f"{where}: label {label!r} is not on the flatground allowlist")

        # Cross-check the name path against the rotation path: the upstream
        # spelling must resolve to the same canonical the triple derives.
        source = row.get("label_source")
        if source is not None and not pd.isna(source) and str(source).strip():
            resolved = self.allowlist.canonical(source)
            if resolved is not None and resolved != label:
                problems.append(
                    f"{where}: upstream name {str(source)!r} resolves to {resolved!r}, "
                    f"but its rotations derive {label!r}"
                )
        return problems

    def _stance_violations(self, frame: pd.DataFrame) -> List[str]:
        """Checks on the two distinct stance columns.

        The dangerous failure is not a missing value but a *wrong* one: copying a
        published riding direction into the goofy/regular toggle is not a schema
        error, it silently mirrors every sign, swapping kick<->heel and fs<->bs
        on every clip. That gets its own check.
        """
        problems: List[str] = []

        if "stance_published" in frame.columns:
            published = frame["stance_published"].fillna("").astype(str).str.strip().str.lower()
            published = {value for value in published if value}
            unknown = published - set(PUBLISHED_RIDING_VALUES)
            if unknown:
                problems.append(
                    f"stance_published has values outside {list(PUBLISHED_RIDING_VALUES)}: "
                    f"{sorted(unknown)}"
                )

        if "stance_input" not in frame.columns:
            problems.append("manifest is missing the stance_input column")
            return problems

        stance = frame["stance_input"].fillna("").astype(str).str.strip().str.lower()
        filled = {value for value in stance if value}
        bad = sorted(filled - set(STANCE_VALUES))
        if bad:
            problems.append(
                f"stance_input must be one of {list(STANCE_VALUES)} or empty, found {bad}. "
                "It is the goofy/regular toggle, not a riding direction: fakie/switch/nollie "
                "cannot fix the sign frame."
            )
        leaked = sorted((filled & set(PUBLISHED_RIDING_VALUES)) - {"regular"})
        if leaked:
            problems.append(
                f"stance_input contains riding directions {leaked}; using one as the stance "
                "would flip kick<->heel and fs<->bs on every clip."
            )
        return problems

    def validate_frame(self, frame: pd.DataFrame, max_report: int = 12) -> List[str]:
        """Return every way ``frame`` breaks the scope guardrail (empty when clean)."""
        violations: List[str] = self._stance_violations(frame)
        missing = [column for column in ("label", *ROTATION_COLUMNS) if column not in frame.columns]
        if missing:
            return violations + [f"manifest is missing columns {missing}"]

        off_list = sorted(set(frame["label"].astype(str)) - set(self.allowlist.canonical_names()))
        if off_list:
            violations.append(
                f"labels off the flatground allowlist: {off_list}. Normalise at ingestion with "
                "Taxonomy.normalize_label() instead of emitting upstream spellings."
            )

        for column in ("clip_id", "sha256"):
            if column in frame.columns:
                repeated = frame.loc[frame[column].duplicated(), column].unique().tolist()
                if repeated:
                    violations.append(f"duplicate {column}: {repeated[:5]}")

        for position, row in frame.iterrows():
            violations.extend(self._row_violations(row, str(row.get("clip_id", f"row {position}"))))

        if len(violations) > max_report:
            hidden = len(violations) - max_report
            violations = violations[:max_report] + [f"... and {hidden} further violation(s)"]
        return violations

    def validate_or_raise(self, frame: pd.DataFrame) -> None:
        """Raise :class:`ScopeError` listing every guardrail violation."""
        violations = self.validate_frame(frame)
        if violations:
            raise ScopeError(
                "manifest failed the flatground scope guardrail (plan section 5):\n  - "
                + "\n  - ".join(violations)
            )

    def describe(self) -> Dict[str, object]:
        """Counts used by the docs and the tests."""
        rotation_free = [
            name for name in self.allowlist.canonical_names() if not self.dictionary.is_expressible(name)
        ]
        return {
            "canonical_names": len(self.allowlist.canonical_names()),
            "rotation_expressible": len(self.dictionary.expressible_names()),
            "rotation_free": rotation_free,
            "aliases": sum(len(entry.aliases) for entry in self.allowlist.entries.values()),
        }

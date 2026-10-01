"""Pose and board feature extraction (plan section 7, M1).

Two things a skateboard clip contains that a generic video encoder does not:

- **The rider.** 17 body keypoints per frame. YOLO-pose gives these; the useful
  work is expressing them *relative to the rider* (hip-centred, scaled by
  torso length) so camera distance and angle stop mattering, which is what plan
  section 7 means by "the part that must be right".
- **The board.** A rotated rectangle whose long-axis angle is the single most
  direct measurement of a flip's rotation order. See the caveat in
  :func:`board_features` -- this is the hard half and it has no off-the-shelf
  model, so it is built as a *segmentation* problem rather than guessed at.

Both are cached to ``.npz`` keyed on clip id **and** extractor version, so
changing the feature definition cannot silently reuse stale features. (M0 learned
that lesson the expensive way when a cache keyed on ``clip_id`` alone reused
features across a resolution change.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .taxonomy import STANCE_VALUES, Taxonomy
from .video import sample_frames

#: COCO pose keypoint order, as ultralytics emits it. Named rather than indexed
#: by magic numbers so the feature code below is readable.
KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KEYPOINT_COUNT = len(KEYPOINT_NAMES)
#: Confidence below which a keypoint is treated as missing rather than as a
#: coordinate. Averaging in an occluded elbow produces a smooth, confident-looking
#: number that means nothing, so missingness is recorded instead.
KEYPOINT_MIN_CONF = 0.3

#: Bumped whenever the feature definition changes, so the cache invalidates.
#:
#: **v2 (2026-09-28)**: resolution is now native (854x480, was a distorted 640x640
#: that stretched height 33 % and shrank width 25 %) and the board stream grew
#: from 5 to 6 dims, adding **foreshortening** -- the quantity that actually
#: tracks a kickflip (plan 7.1, 12.16).
#:
#: The bump is the important part. M0's cache was keyed on `clip_id` alone and
#: silently reused features after a resolution change; every downstream score
#: still came out, so the only symptom was inexplicably flat results. Keying on
#: the extractor version makes that class of bug structurally impossible rather
#: than a thing future-me has to remember.
EXTRACTOR_VERSION = 2

#: Working resolution for extraction. The clips are stored 854x480; resizing to a
#: square distorts the board and costs 27 % of the measured frames (plan 12.15).
NATIVE_SIZE: Tuple[int, int] = (854, 480)

#: Frames sampled per clip. 48 rather than 12 because foreshortening is a
#: *transient* dip: at 12 frames across a 2 s clip the dip is sampled too sparsely
#: to survive averaging.
DEFAULT_FRAMES = 48


def load_pose_model(name: str = "yolo11n-pose.pt", device: Optional[object] = None):
    """Load a YOLO pose model, on the GPU when one is present.

    ``device`` is passed through to ultralytics unchanged. It must be an ``int``
    (0) or a torch device string ("cuda"); ultralytics rejects the string "0",
    so this deliberately does not stringify a caller that already chose right.

    Returns ``None`` rather than raising if ultralytics is missing, so a caller
    can report "pose extraction unavailable" instead of crashing mid-run. Weights
    download on first use.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    model = YOLO(name)
    if device is None:
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
    return model.to(device)


def _unit_scale(keypoints: np.ndarray) -> float:
    """Torso length in pixels, the unit every measurement is expressed in.

    ``keypoints`` is ``(17, 3)`` as x, y, confidence, so the spatial slice is
    taken first. A skater is vertical in frame, so the hip midpoint sits roughly
    one torso-length below the shoulder midpoint; that distance is the unit, and
    it survives the camera moving rather than the person's absolute size.
    """
    points = keypoints[:, :2]
    left, right = KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("right_shoulder")
    lhip, rhip = KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("right_hip")
    if not np.isfinite(points[[left, right]]).all():
        return 0.0
    shoulder_mid = 0.5 * (points[left] + points[right])
    if not np.isfinite(points[[lhip, rhip]]).all():
        return max(float(np.linalg.norm(points[left] - points[right])), 1.0)
    hip_mid = 0.5 * (points[lhip] + points[rhip])
    return max(float(np.linalg.norm(hip_mid - shoulder_mid)), 1.0)


#: COCO ships a `skateboard` class, so *detecting* a board is free. Measuring its
#: rotation is not -- see :func:`board_features`.
BOARD_COCO_CLASS = "skateboard"
BOARD_MIN_CONF = 0.25


def detect_board(model, frame: np.ndarray, device: int = 0):
    """Return ``(x1, y1, x2, y2)`` for the most confident board, else ``None``.

    Detection only. The box is axis-aligned, so it locates the board but carries
    **no rotation information** -- and rotation is the entire reason the board
    stream exists. See :func:`board_features` for how that is handled.
    """
    result = model.predict(frame, verbose=False, device=device)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best, best_conf = None, BOARD_MIN_CONF
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                              result.boxes.cls.cpu().numpy(),
                              result.boxes.conf.cpu().numpy()):
        if model.names[int(cls)] == BOARD_COCO_CLASS and conf >= best_conf:
            best, best_conf = box, conf
    return best


def board_features(box: Optional[np.ndarray], frame_shape: Tuple[int, int]) -> np.ndarray:
    """One frame of board features: ``(centre_x, centre_y, w, h, visible)``.

    **This is deliberately not the rotation signal, and the plan should not be
    read as saying it is.** Plan section 7 asks for four corners and a long-axis
    angle, which is the right idea; the honest state today is that a COCO box
    cannot supply either. A box is axis-aligned by construction, so:

    - its aspect ratio mixes the board's true angle with how much foreshortening
      the camera angle introduces -- a board tilted 45 deg and one flat both give
      a box, and the boxes are nearly identical;
    - the four box corners are the corners of an axis-aligned rectangle, not the
      board's actual corners, so they are wrong in a way that is *invisible* to a
      downstream model.

    What is returned is location and size, which are real and useful (they say
    where the board is and whether it is visible at all), plus ``visible`` so a
    clip with no detectable board is visibly empty rather than quietly zero.
    The rotation features are left to M2, where the plan already puts them
    behind an oracle-plot gate -- a fitted board-angle extractor has to be shown
    to carry signal before it is trusted to.

    Coordinates are normalised by frame size so they are resolution-independent,
    matching :func:`body_features`.
    """
    out = np.zeros(5, dtype=np.float32)
    if box is None:
        return out
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    out[0] = (0.5 * (x1 + x2)) / max(width, 1)
    out[1] = (0.5 * (y1 + y2)) / max(height, 1)
    out[2] = (x2 - x1) / max(width, 1)
    out[3] = (y2 - y1) / max(height, 1)
    out[4] = 1.0
    return out


def cache_key(clip_id: str, extractor_version: int = EXTRACTOR_VERSION) -> str:
    """Cache filename for one clip's features.

    Keyed on the extractor version as well as the clip id. M0's cache keyed on
    ``clip_id`` alone, so changing the resolution or the feature definition
    silently reused the previous run's numbers -- the worst failure mode in a
    feature pipeline, because every downstream score still comes out.
    """
    return f"{clip_id}_v{extractor_version}"


def save_features(cache_dir: Path, key: str, body: np.ndarray, board: np.ndarray) -> Path:
    """Persist one clip's features to ``.npz`` and return the path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    np.savez_compressed(path, body=body.astype(np.float32), board=board.astype(np.float32))
    return path


def load_features(cache_dir: Path, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load cached features, or ``None`` on a miss (corrupt files included)."""
    path = cache_dir / f"{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            return data["body"], data["board"]
    except (OSError, KeyError, ValueError):
        # A truncated .npz from an interrupted run must read as a miss, not crash
        # a 671-clip pass on clip 400.
        return None


def unwrap_angles(degrees: List[float]) -> List[float]:
    """Turn per-frame angles into a continuous accumulated sweep.

    The per-frame angle is in (-90, 90] (a rectangle has no facing), but a flip
    is a rotation that crosses that boundary many times. Any naive mean or slope
    of the raw angles is therefore meaningless -- a board sweeping smoothly
    through 360 deg looks, frame by frame, like noise bouncing off the wrap point.

    The angles are unwrapped into a continuous series so a *net sweep* can be
    measured: unwrap a kickflip and it travels roughly +360; a heelflip roughly
    -360. That signed sweep is the feature, and it is what makes the mirror pair
    separable.
    """
    if not degrees:
        return []
    out = [float(degrees[0])]
    for angle in degrees[1:]:
        previous = out[-1]
        delta = (float(angle) - previous + 90.0) % 180.0 - 90.0
        out.append(previous + delta)
    return out


def board_axes(mask_or_corners):
    """Both board axes as lengths, plus a **foreshortening ratio**.

    This is the feature the plan specified and M2 did not build. A kickflip rolls
    about the board's **long** axis, so in the image the board *foreshortens* — it
    gets shorter — while its long-axis **angle barely changes**. Measuring the
    angle alone is therefore nearly blind to a kickflip (plan 7.1 / 12.15).

    The foreshortening ratio is ``long_len / (long_len + short_len)``: the share of
    the rectangle's extent taken by the long axis. It approaches 0.5 for a board
    seen broadside (4:1 -> 0.80) and falls toward 0.29 for a square (1:1). It is
    self-normalising, so it does not depend on camera distance.

    Returns ``None`` for a degenerate mask, so "not measured" stays distinct from
    "measured as flat".
    """
    import cv2

    array = np.asarray(mask_or_corners)
    if array.ndim == 2:
        points = cv2.findNonZero(array.astype(np.uint8))
        if points is None or len(points) < 4:
            return None
        (_, _), (w, h), _ = cv2.minAreaRect(points)
    else:
        corners = array.astype(np.float32).reshape(4, 2)
        edges = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
        w, h = sorted(edges)[-2:]
    if w < 2 or h < 2:
        return None

    long_len, short_len = max(w, h), min(w, h)
    return {
        "long_len": long_len,
        "short_len": short_len,
        "aspect": long_len / short_len,
        "foreshortening": long_len / (long_len + short_len),
    }


def foreshortening_series(model, frames, device: int = 0):
    """Per-frame foreshortening ratio from segmented board masks.

    The signal a kickflip actually produces: a dip toward 0.5 as the deck rolls
    edge-on, recovering on the catch. ``None`` where the board is not segmented,
    so a missing measurement is never reported as a flat board.
    """
    out = []
    for frame in frames:
        result = model.predict(frame, verbose=False, device=device)[0]
        if result.masks is None or result.boxes is None:
            out.append(None)
            continue
        best, best_conf = None, 0.25
        for mask, cls, conf in zip(result.masks.data, result.boxes.cls, result.boxes.conf):
            if model.names[int(cls)] == "skateboard" and float(conf) > best_conf:
                best, best_conf = mask, float(conf)
        if best is None:
            out.append(None)
            continue
        binary = (best.cpu().numpy() > 0.5).astype(np.uint8)
        axes = board_axes(binary)
        out.append(None if axes is None else float(axes["foreshortening"]))
    return out


def rotation_window(
    angles: List[Optional[float]], min_span: int = 3, max_span: int = 14
) -> Optional[Tuple[int, int]]:
    """The frame range where the board is rotating *fastest*.

    A trick clip is ~2 s but the flip is ~0.3 s; the rest is approach, pop and
    roll-away. An earlier version selected the **highest total-variation** window
    and picked the slow drift instead of the flip -- drift accumulates more total
    travel over 60 frames than a 0.3 s spin does, which collapsed kick-vs-heel
    separation from -108 deg to +2 deg (plan 12.12).

    **Peak rate, not total travel.** A flip is defined by high |dtheta/dt|, so the
    objective is the largest single-step angular change inside the window, and the
    window is kept short. Drift is slow by definition and cannot win.

    Returns ``None`` when no rotation stands out, which is a real case (an
    ollie) and must not be reported as a zero-degree rotation.
    """
    if not angles:
        return None
    measured = [(index, value) for index, value in enumerate(angles) if value is not None]
    if len(measured) < min_span + 1:
        return None

    unwrapped = unwrap_angles([value for _, value in measured])
    steps = [abs(b - a) for a, b in zip(unwrapped, unwrapped[1:])]
    if not steps or max(steps) < 1.0:
        return None

    best, best_score = None, 0.0
    for start in range(0, len(steps)):
        for width in range(min_span, min(max_span, len(steps) - start) + 1):
            window_steps = steps[start : start + width]
            if not window_steps:
                continue
            # Peak rate dominates; total travel is only a tiebreak. A mean rate
            # would re-admit drift, and total travel alone re-admits it worse.
            score = max(window_steps) + 1e-6 * sum(window_steps)
            if score > best_score:
                best_score = score
                best = (measured[start][0], measured[start + width][0])
    return best


def sample_native_window(
    path,
    window: Tuple[float, float],
    fps: float,
    count: int = 24,
    size: Tuple[int, int] = (640, 640),
):
    """Re-decode ``window`` seconds of the clip at **native** frame rate.

    This is the fix for the aliasing that has capped every board-rotation
    measurement so far. A flip lasts ~0.3 s; sampling 60 frames across a ~2 s
    clip gives ~6 samples of it, and direction is only recoverable above ~5
    samples per rotation (plan 12.13's simulation). Worse, ~38 % of frames are
    unmeasurable, so a 14-sample window spans ~30 original frames and the
    effective rate is lower again.

    Re-decoding the window instead of *selecting* from existing samples is the
    only version that adds information. Selection cannot: it can only discard,
    and the -108 deg whole-clip result shows discarding is what hurt.

    ``window`` is in seconds as ``(start, end)`` and may be fractional -- a
    0.3 s flip is 9 frames at 30 fps.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = capture.get(cv2.CAP_PROP_FPS) or fps
        start = max(0, int(round(window[0] * native_fps)))
        end = int(round(window[1] * native_fps))
        if total > 0:
            end = min(total - 1, end)
        span = end - start + 1
        if span <= 0:
            return np.zeros((0, size[1], size[0], 3), dtype=np.uint8), []

        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        want = min(count, span)
        frames: List[np.ndarray] = []
        for _ in range(want):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            return np.zeros((0, size[1], size[0], 3), dtype=np.uint8), []
        while len(frames) < want:  # pad if the decode came up short
            frames.append(frames[-1])
        return np.stack(frames), list(range(start, start + len(frames)))
    finally:
        capture.release()


def window_in_seconds(window, frame_indices, fps: float):
    """Convert a window of *sample indices* into seconds using the sample grid.

    ``frame_indices[i]`` is the source frame sample ``i`` came from, so the
    conversion stays exact even though samples are evenly spaced in time rather
    than contiguous.
    """
    if not frame_indices or window[0] < 0:
        return 0.0, 0.0
    start_frame = frame_indices[window[0]]
    end_frame = frame_indices[min(window[1], len(frame_indices) - 1)]
    return start_frame / max(fps, 1e-6), (end_frame + 1) / max(fps, 1e-6)


def segmented_angle(model, image, device: int = 0) -> Optional[float]:
    """Board long-axis angle from a YOLO *segmentation* mask, or ``None``.

    The measurably better arm: section 12.9 recorded that this finds the board in
    ~84 % of frames versus Otsu's much lower rate, with clean 2.5:1 mask aspect
    (0.37-0.40) where a real board is ~4:1. The 0.62 "square blob" that 12.9
    mistakenly attributed to the segmenter came from the Otsu arm.

    Returns ``None`` rather than a placeholder whenever the board is absent or the
    mask is degenerate, so "not measured" never masquerades as "measured 0".
    """
    import cv2

    result = model.predict(image, verbose=False, device=device)[0]
    if result.masks is None or result.boxes is None:
        return None
    best, best_conf = None, 0.25
    for mask, cls, conf in zip(result.masks.data, result.boxes.cls, result.boxes.conf):
        if model.names[int(cls)] != "skateboard" or float(conf) <= best_conf:
            continue
        best, best_conf = mask, float(conf)
    if best is None:
        return None
    binary = (best.cpu().numpy() > 0.5).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 4:
        return None
    (_, _), (w, h), angle = cv2.minAreaRect(points)
    if w < 2 or h < 2:
        return None
    if w < h:
        angle += 90.0
    return float((angle + 90.0) % 180.0 - 90.0)


def board_summary(board: np.ndarray) -> np.ndarray:
    """Compress the per-frame board stream to four summary scalars.

    The 2640-dim board stream *lowered* holdout macro-F1 from 0.2048 to 0.1476
    (plan 12.17) -- not because the board carries no information, but because
    2448 pose dims already outnumber 110 training rows, so extra noisy columns
    cost more variance than they add. The fix is fewer, better numbers.

    1. **dip depth** -- ``median - min`` of the foreshortening series. This is the
       quantity plan 12.16 actually demonstrated on real clips (a flat board at
       0.83 dipping to 0.59 through a kickflip), and the one ``net_sweep`` failed
       to capture: kick-family and all other classes share a mean net_sweep of
       -1.15/-1.14, because sweep is an accumulated-path measure rather than an
       amplitude.
    2. **dip timing** -- where in the clip the dip sits, as a fraction. Tricks
       happen mid-clip, so a board dipping at frame 5 of 48 is something else.
    3. **coverage** -- fraction of frames where the board was measurable, so a
       clip with no detections is visibly empty rather than looking like a board
       that never moved.
    4. **peak foreshortening** -- the maximum, so amplitude is available on both
       sides rather than only as a dip.

    Angles are deliberately excluded: near-blind to a kickflip (plan 7.1), and the
    bulk of the harmful dimensionality.
    """
    # The angle/foreshortening series is the (frames, frames) block starting at
    # column 5: column 5+i is frame i's angle. Indexed [i, 5+i]. Slicing rows
    # instead of columns -- board[:, 5:5+frames] -- reads the wrong axis entirely
    # and reports the dip at frame 0; a test asserting the dip is mid-clip is what
    # caught it.
    per_frame = np.array([board[i, 5 + i] for i in range(board.shape[0])], dtype=np.float32)
    # An unmeasurable frame is stored as NaN (features.board_features_from_angles
    # writes NaN, not zero, precisely so it can be told apart from a measured 0).
    # Coverage therefore counts finite entries, and an all-zero block -- every
    # frame *measured* as 0 -- reports full coverage with zero depth, which is the
    # honest reading.
    usable = per_frame[np.isfinite(per_frame)]
    if usable.size < 2:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [
            float(np.median(usable) - usable.min()),
            float(usable.argmin()) / max(len(usable) - 1, 1),
            float(usable.size) / max(board.shape[0], 1),
            float(usable.max()),
        ],
        dtype=np.float32,
    )


def measure_clip_two_pass(
    path, fps: float, seg_model, pass1_frames: int = 24, native_frames: int = 20
):
    """Two-pass board measurement: locate the flip, then re-decode it densely.

    Pass 1 samples the whole clip coarsely and finds the **rotation window**
    (:func:`rotation_window`, peak |dtheta/dt|). Pass 2 re-decodes just that
    window at native frame rate and measures the sweep there.

    This is the only approach tried that actually **adds** information. Every
    earlier attempt selected from an already-taken 60-frame grid, which can only
    discard samples -- and discarding is measurably what hurt (plan 12.12: the
    -108 deg whole-clip separation collapsed to +2 deg once a window was chosen).

    Returns ``(sweep, window_seconds, n_native_measured, n_native_total)``.
    """
    batch = sample_frames(path, count=pass1_frames, size=(640, 640))
    pass1 = [segmented_angle(seg_model, frame, 0) for frame in batch]
    window = rotation_window(pass1)

    if window is None:
        usable = [value for value in pass1 if value is not None]
        sweep = net_sweep(usable) if len(usable) >= 2 else 0.0
        return sweep, (0.0, 0.0), len(usable), pass1_frames

    # Approximate seconds from the uniform pass-1 grid, then re-decode natively.
    # Bounded tightly: `rotation_window` is measured in *pass-1 sample indices*, and
    # with 24 samples over a ~2 s clip each index is ~0.08 s, so an unconverted
    # index (the earlier bug here) padded the window to the whole clip and
    # measured nothing useful. The flip is ~0.3 s, so the window is capped at
    # ~0.6 s and only widened if the decoded span came up empty.
    per_sample = max(fps, 1e-6) / max(pass1_frames - 1, 1)
    start_s = max(0.0, window[0] * per_sample - 0.05)
    end_s = window[1] * per_sample + 0.10
    end_s = min(max(end_s, start_s + 0.2), start_s + 0.6)

    native, _ = sample_native_window(path, (start_s, end_s), fps, count=native_frames)
    if native.shape[0] == 0:
        usable = [value for value in pass1 if value is not None]
        return (net_sweep(usable) if len(usable) >= 2 else 0.0), (start_s, end_s), 0, pass1_frames

    angles2 = [segmented_angle(seg_model, frame, 0) for frame in native]
    measured = [value for value in angles2 if value is not None]
    sweep = net_sweep(measured) if len(measured) >= 2 else 0.0
    return sweep, (start_s, end_s), len(measured), len(native)


def active_window(
    angles: List[Optional[float]], min_span: int = 4, max_span: int = 20
) -> Optional[Tuple[int, int]]:
    """The (start, end) index range where the board is actually rotating.

    A trick clip is ~2 s, but the flip itself is ~0.3 s; the rest is approach,
    pop and roll-away. Measuring a fixed grid over the whole clip therefore
    spends most of its samples on frames where nothing is rotating, and a double
    flip can alias into a single one.

    Selects the **highest-variation contiguous window** of the measured series, so
    the sample budget goes where the signal is. The window is chosen on *total
    variation* rather than "first to last detected", because the board is usually
    still visible before and after the rotation and those frames would dilute the
    quantity being measured.

    ``max_span`` bounds the window so a long slow drift across the whole clip
    cannot be selected as "the rotation" -- without it, a series whose variation
    grows monotonically makes the full span the answer every time, which is
    precisely the un-windowed behaviour this is meant to replace.

    Returns ``None`` when there is no measurable rotation, which is a real and
    common case (an ollie) and must not be reported as a zero-degree rotation.
    """
    if not angles:
        return None
    # Compact to the measured frames, keeping original indices for provenance.
    # Dropping unmeasurable frames here is what keeps None out of every
    # downstream slice.
    measured = [(index, value) for index, value in enumerate(angles) if value is not None]
    if len(measured) < min_span:
        return None

    unwrapped = unwrap_angles([value for _, value in measured])
    best: Optional[Tuple[int, int]] = None
    best_variation = 0.0
    for start in range(0, len(unwrapped) - min_span + 1):
        for end in range(start + min_span, min(len(unwrapped), start + max_span) + 1):
            variation = sum(
                abs(b - a) for a, b in zip(unwrapped[start:end], unwrapped[start + 1 : end + 1])
            )
            if variation > best_variation:
                best_variation, best = variation, (measured[start][0], measured[end - 1][0])
    if best is None or best_variation < 1.0:
        return None
    return best


def windowed_sweep(angles: List[Optional[float]]) -> Tuple[float, int, int]:
    """(signed sweep in degrees, window start, window end) over the active window.

    Falls back to the whole series when no window stands out, so a clip with a
    gentle rotation still produces a measurement rather than silently nothing.
    The unmeasurable frames are removed before slicing, so no ``None`` can reach
    :func:`net_sweep`.
    """
    window = rotation_window(angles)
    usable = [value for value in angles if value is not None]
    if window is None:
        if len(usable) < 2:
            return 0.0, -1, -1
        return net_sweep(usable), 0, len(angles) - 1

    start, end = window
    measured = [value for value in angles[start : end + 1] if value is not None]
    if len(measured) < 2:
        return net_sweep(usable), start, end
    return net_sweep(measured), start, end


def net_sweep(degrees: List[float]) -> float:
    """Total rotation travelled across a clip, in degrees, signed by direction.

    **Total variation, not endpoint difference.** A board that rotates a full 360
    deg ends up exactly where it started, so an endpoint difference reports ~0 for
    a textbook kickflip. That bug was live for two measurement rounds before it
    was caught by reading an actual trajectory: the observed kickflip series
    travelled 223 deg while its endpoints differed by 27. The statistic has to
    measure the *path*, because the phenomenon is a closed loop.

    Sign comes from the largest sustained excursion rather than the net endpoint,
    since a real clip drifts and its endpoints do not determine which way the board
    went. Snapped to a multiple of 180 because a rectangle has no facing, so a
    full flip reads 360, a half 180, and the sign is the kick/heel information.
    """
    unwrapped = unwrap_angles(degrees)
    if len(unwrapped) < 2:
        return 0.0

    # Guard against non-finite input reaching the arithmetic. `board_features_
    # from_angles` writes NaN for an unmeasurable frame, and NaN silently
    # poisons every sum below until it fails as an int conversion much further
    # from the cause than it should.
    finite = [value for value in degrees if value is not None and np.isfinite(value)]
    if len(finite) < 2:
        return 0.0
    unwrapped = unwrap_angles(finite)
    if len(unwrapped) < 2:
        return 0.0

    travelled = float(sum(abs(b - a) for a, b in zip(unwrapped, unwrapped[1:])))
    # Direction: the signed excursion of the unwrapped series from its starting
    # value, which survives the drift that endpoint differencing does not.
    excursion = unwrapped[-1] - unwrapped[0]
    if abs(excursion) < 1e-9 and travelled > 0:
        # Endpoints coincide (the closed-loop case). Fall back to the sign of the
        # largest single step, which for a monotonic loop carries the direction.
        excursion = max(unwrapped, key=lambda v: abs(v - unwrapped[0])) - unwrapped[0]
    if abs(excursion) < 1e-9:
        return 0.0
    magnitude = round(travelled / 180.0) * 180.0
    return float(magnitude if excursion > 0 else -magnitude)


def board_angle(corners: np.ndarray) -> float:
    """Long-axis angle in degrees in (-90, 90], from 4 corners of a rotated rect.

    The long axis is taken as the *longer* of the two sides, so the reported angle
    is the board's own axis rather than its short edge. Ambiguity of 180 deg is
    inherent and deliberate: a rectangle does not say which end is the nose, and
    pretending otherwise is how a sign convention gets quietly invented. The
    signed sweep in :func:`net_sweep` is what carries direction.
    """
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    edges = [np.linalg.norm(points[(i + 1) % 4] - points[i]) for i in range(4)]
    # OpenCV's minAreaRect orders corners, but not consistently enough to trust;
    # take the longest edge found between any pair.
    best, best_len = 0.0, -1.0
    for i in range(4):
        for j in range(i + 1, 4):
            length = float(np.linalg.norm(points[j] - points[i]))
            if length > best_len:
                best_len, best = length, float(np.arctan2(points[j][1] - points[i][1],
                                                         points[j][0] - points[i][0]))
    degrees = np.degrees(best)
    return float((degrees + 90.0) % 180.0 - 90.0)


def segment_board(frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """Isolate the board inside its detection box, returning a binary mask.

    Deliberately classical (Otsu + morphology + largest component) rather than a
    learned segmenter. A skateboard is a small, high-contrast, roughly-flat
    object inside a box that already contains it, so thresholding is adequate and
    -- more importantly -- has no failure mode that produces a *confident wrong
    mask*. A segmentation model that misses the board hands back a plausible
    mask of a foot, and the resulting angle is silently garbage.

    Returns ``None`` when the box is unusable or the mask is implausibly small,
    so "could not measure" stays distinguishable from "measured zero degrees".
    """
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Otsu on a blurred copy: a board seen edge-on during a flip is only a few
    # pixels wide, and hard edges produce speckle that breaks connectivity.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(areas.argmax())
    area = float(areas.max())
    # A board filling < 4 % of its own box is a mis-detection (a shoe, a shadow),
    # not a board. Returning None here is what keeps a bad mask out of the angle.
    if area < 0.04 * mask.size:
        return None
    return (labels == largest).astype(np.uint8) * 255


def board_corners(mask: np.ndarray) -> Optional[np.ndarray]:
    """Smallest rotated rectangle enclosing the board mask -> 4 corners.

    This is the step that recovers the rotation: a board spinning in the air is
    mostly a thin diagonal line, and a bounding box around it barely changes as it
    turns, whereas the minimum-area rectangle follows the board's own axis.
    """
    import cv2

    points = cv2.findNonZero(mask)
    if points is None or len(points) < 4:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    if w < 2 or h < 2:
        return None
    return np.asarray(cv2.boxPoints(((cx, cy), (w, h), angle)), dtype=np.float32)


def board_axis_angle(mask: np.ndarray) -> Optional[float]:
    """Long-axis angle in (-90, 90] straight from ``minAreaRect``.

    Taken from the rectangle's own ``ang`` rather than by differencing its corner
    points, which was tried first and is wrong: OpenCV orders the four corners
    but not in a way that guarantees consecutive points are the long edge, so
    differencing them picks up the *short* edge (or a diagonal) and reports a
    constant offset. ``minAreaRect``'s angle is exact for a clean mask --
    measured 45.00 deg for a synthetic board at 45 -- so it is used directly.

    Mod 180, because a rectangle has no facing. A skateboard's nose and tail are
    indistinguishable from its outline alone, and inventing a direction from it is
    exactly how a sign convention gets quietly fabricated. The *signed sweep* in
    :func:`net_sweep` is what carries kick-vs-heel.
    """
    import cv2

    points = cv2.findNonZero(mask)
    if points is None or len(points) < 4:
        return None
    (_, _), (w, h), angle = cv2.minAreaRect(points)
    if w < 2 or h < 2:
        return None
    # minAreaRect reports the angle of whichever side it treats as "width", which
    # swaps near 45 deg. Normalising through the long axis keeps it continuous.
    if w < h:
        angle += 90.0
    return float((angle + 90.0) % 180.0 - 90.0)


def extract_clip(
    clip_id: str,
    path: Path,
    pose_model,
    board_model,
    frames: int = DEFAULT_FRAMES,
    size: Tuple[int, int] = NATIVE_SIZE,
    device: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one clip's body and board streams.

    Frames are decoded at ``size`` and passed to the models one at a time: ultralytics
    rejects an ``(N, H, W, 3)`` array as a single image, so a batch needs the list
    form and explicit streaming, which also keeps peak VRAM flat over 671 clips.

    A frame where nothing is detected yields a zero row, not a skipped frame. The
    sequence length is fixed, so "the model found nothing here" stays a value the
    downstream model can see rather than a shift in time alignment.
    """
    from .video import sample_frames

    batch = sample_frames(path, count=frames, size=size)
    body = np.zeros((frames, KEYPOINT_COUNT * 3), dtype=np.float32)
    board = np.zeros((frames, 5), dtype=np.float32)
    angles: List[Optional[float]] = []

    for index, frame in enumerate(batch):
        if pose_model is not None:
            result = pose_model.predict(frame, verbose=False, device=device)[0]
            keypoints = None
            if result.keypoints is not None and len(result.keypoints.data):
                keypoints = result.keypoints.data[0].cpu().numpy()
                # Several people can appear in a competition frame; the largest is
                # the one performing. Sorting by box area rather than confidence,
                # because a confident background spectator is still not the subject.
                if len(result.keypoints.data) > 1 and result.boxes is not None and len(result.boxes):
                    areas = result.boxes.conf.cpu().numpy() * 0  # placeholder, see below
                    widths = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]).cpu().numpy()
                    heights = (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1]).cpu().numpy()
                    largest = int((widths * heights).argmax())
                    keypoints = result.keypoints.data[largest].cpu().numpy()
            body[index] = body_features(keypoints)

        if board_model is not None:
            box = detect_board(board_model, frame, device=device)
            angles.append(_measure_angle(frame, box))
    return body, board_features_from_angles(board, angles, frames)


def _measure_angle(frame: np.ndarray, box: Optional[np.ndarray]) -> Optional[float]:
    """Board's long-axis angle in this frame, or ``None`` if it could not be measured.

    Never returns a placeholder. A frame where the board was not detected, or
    where segmentation found nothing plausible, stays ``None`` all the way into
    the feature vector, so "not measured" and "measured zero degrees" -- which
    would mean an ollie -- are distinguishable.
    """
    if box is None:
        return None
    mask = segment_board(frame, box)
    if mask is None:
        return None
    return board_axis_angle(mask)


def board_features_from_angles(
    filled: np.ndarray, angles: List[Optional[float]], frames: int
) -> np.ndarray:
    """Board stream = detection features, per-frame angles, and a signed sweep.

    Layout per frame: the 5 box features, then that frame's angle (NaN where
    unmeasurable, so a missing measurement is distinguishable from a measured 0),
    then two summary scalars on the first row -- the **signed net sweep** and the
    fraction of frames that produced an angle at all.

    The summary scalars are what carry trick identity. A kickflip and a heelflip
    differ only in the *sign* of the sweep, so a feature set that kept the
    per-frame angles but dropped the sign would score them identically -- which is
    the precise failure this stream was built to fix.
    """
    out = np.zeros((frames, 5 + frames + 2), dtype=np.float32)
    out[:, :5] = filled
    series = [np.nan if angle is None else angle for angle in angles]
    for index, value in enumerate(series[:frames]):
        out[index, 5 + index] = value

    # An unmeasurable frame must be a **zero plus the coverage flag**, never a NaN
    # left in the vector. StandardScaler would carry the NaN into the scaler's
    # mean and variance, and one NaN poisons the whole fitted transform -- a
    # silent, catastrophic failure that produces scores rather than an error.
    # Zero is safe *because* the coverage scalar in the last column says how much
    # of the series was real.
    out[:, 5 : 5 + frames] = np.nan_to_num(out[:, 5 : 5 + frames], nan=0.0)

    # `usable` must exclude NaN as well as None, for the same reason.
    usable = [angle for angle in series if angle is not None and np.isfinite(angle)]
    # Normalised to 1.0 per full rotation so the LR sees a bounded input.
    out[0, 5 + frames] = net_sweep(usable) / 360.0 if len(usable) >= 2 else 0.0
    out[0, 5 + frames + 1] = len(usable) / max(frames, 1)
    return out


def pose_only_features(cache_dir: Path, key: str) -> Optional[np.ndarray]:
    """Flatten one clip's cached pose stream into a single feature vector.

    The M1 test of plan section 7 is *temporal*: the claim is that the
    **trajectory** across the window carries the signal, not any single frame
    (section 7 says so explicitly, and video.py's sampling is built for it). So
    the vector is the full sequence flattened, not a per-frame summary -- pooling
    to a mean here would discard exactly what is being tested.

    Frames are resampled to a fixed length so a clip of any duration produces the
    same shape, as plan section 7 requires.
    """
    loaded = load_features(cache_dir, key)
    if loaded is None:
        return None
    body, _ = loaded
    if body.size == 0:
        return None
    return body.reshape(-1).astype(np.float32)


def load_feature_table(
    manifest: pd.DataFrame,
    cache_dir: Path,
    extractor_version: int = EXTRACTOR_VERSION,
    include_board: bool = False,
    board_summary_dims: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build the ``(X, y, clip_ids)`` design matrix from the feature cache.

    Clips with no cached features are **dropped, not zero-filled**. A missing clip
    and a clip where the skater was genuinely absent both flatten to zeros, and
    training on a field of zeros teaches the model that "no data" is a meaningful
    pose -- which would be a fabricated signal rather than a measured one.

    ``include_board`` adds the board stream. By default it adds the **4 summary
    scalars** from :func:`board_summary` rather than the full per-frame stream:
    the full stream measurably hurt holdout score (plan 12.17), because 2640 extra
    dims against 110 training rows costs more variance than the signal is worth.
    ``board_summary_dims=False`` restores the full stream for reproducing that
    measurement.
    """
    features: List[np.ndarray] = []
    labels: List[str] = []
    clip_ids: List[str] = []
    for _, record in manifest.iterrows():
        loaded = load_features(cache_dir, cache_key(record["clip_id"], extractor_version))
        if loaded is None:
            continue
        body, board = loaded
        if body.size == 0:
            continue
        row = body.reshape(-1)
        if include_board:
            extra = board_summary(board) if board_summary_dims else board.reshape(-1)
            row = np.concatenate([row, extra])
        features.append(row)
        labels.append(str(record["label"]))
        clip_ids.append(str(record["clip_id"]))
    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=object), []
    return np.stack(features), np.array(labels, dtype=object), clip_ids


def body_quality(body: np.ndarray) -> Dict[str, float]:
    """How much of a clip's pose stream is actually usable.

    Reported per clip so a low number is visible before training, rather than
    showing up as a mysteriously weak model. A clip where the skater leaves frame
    has no signal to learn from, and that should be a decision, not a mystery.
    """
    if body.size == 0:
        return {"frames": 0.0, "keypoint_fill": 0.0, "mean_confidence": 0.0, "usable": False}
    confidences = body[:, 2::3]
    filled = float((confidences > 0).mean())
    mean_conf = float(confidences[confidences > 0].mean()) if (confidences > 0).any() else 0.0
    # A frame counts as usable when at least half the keypoints are present.
    per_frame = (confidences > 0).sum(axis=1)
    good_frames = float((per_frame >= KEYPOINT_COUNT / 2).mean())
    return {
        "frames": float(body.shape[0]),
        "keypoint_fill": filled,
        "mean_confidence": mean_conf,
        "usable": bool(good_frames >= 0.5),
    }


def body_features(keypoints: np.ndarray) -> np.ndarray:
    """One frame of body features: hip-centred, scaled, confidence-filtered.

    ``keypoints`` is ``(17, 3)`` as ``(x, y, confidence)`` in pixels, which is
    what ultralytics returns. The result is ``(17 * 3,)``: x and y per keypoint
    plus its confidence, so a downstream model can *learn* to discount an
    unreliable keypoint rather than being told to.

    Returns zeros when the pose is unusable, and ``unit`` 0.0 marks it, so a
    whole clip of bad detections is visibly empty instead of subtly wrong.
    """
    out = np.zeros(KEYPOINT_COUNT * 3, dtype=np.float32)
    if keypoints is None or not np.isfinite(keypoints).any():
        return out

    lhip, rhip = KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("right_hip")
    if np.isfinite(keypoints[[lhip, rhip]]).all():
        origin = 0.5 * (keypoints[lhip, :2] + keypoints[rhip, :2])
    else:
        origin = keypoints[:, :2][np.isfinite(keypoints[:, :2]).all(axis=1)].mean(axis=0) \
            if np.isfinite(keypoints[:, :2]).any() else np.zeros(2)

    scale = _unit_scale(keypoints)
    if scale <= 0.0:
        return out

    for index, row in enumerate(keypoints):
        x, y, conf = float(row[0]), float(row[1]), float(row[2])
        if not (np.isfinite(x) and np.isfinite(y)) or conf < KEYPOINT_MIN_CONF:
            continue
        out[index * 3] = (x - origin[0]) / scale
        out[index * 3 + 1] = (y - origin[1]) / scale
        out[index * 3 + 2] = conf
    return out

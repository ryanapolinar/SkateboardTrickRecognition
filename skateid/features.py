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
EXTRACTOR_VERSION = 1


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


def extract_clip(
    clip_id: str,
    path: Path,
    pose_model,
    board_model,
    frames: int = 12,
    size: Tuple[int, int] = (640, 640),
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
            board[index] = board_features(box, frame.shape)

    return body, board


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

"""Frame sampling shared by every feature extractor.

Clips are decoded with OpenCV rather than a deep-learning video loader so feature
extraction works with the core dependency set alone; the torch-based embedders in
:mod:`skateid.baselines` consume the arrays this module returns.

The samples are deliberately *evenly spaced over the whole clip* rather than
centred on a peak. Plan section 4 is explicit that a trick ends looking the way it
started, and that the information lives in the trajectory across the window, so a
single "action frame" would throw away exactly what the model needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np

#: Default sampling grid. Small enough that a whole dataset pass is cheap.
DEFAULT_FRAME_COUNT = 16
DEFAULT_SIZE: Tuple[int, int] = (112, 112)  # (width, height), as cv2.resize takes it


def _prepare(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize and convert one BGR frame to RGB."""
    import cv2

    if (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_frames(
    path: Path | str,
    count: int = DEFAULT_FRAME_COUNT,
    size: Tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Return ``count`` evenly spaced RGB frames as ``(count, H, W, 3)`` uint8.

    Seeking is used when the container reports a frame count, which is roughly an
    order of magnitude faster than decoding the whole clip. Short clips that
    cannot yield ``count`` frames are padded by repeating the last one, so every
    clip produces a fixed-shape array.
    """
    import cv2

    if count < 1:
        raise ValueError("count must be >= 1")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: List[np.ndarray] = []

        if total > 0:
            if count == 1:
                indices = [total // 2]
            else:
                step = total / count
                indices = [min(total - 1, int(index * step + step / 2)) for index in range(count)]
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(_prepare(frame, size))
        else:
            while len(frames) < count:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append(_prepare(frame, size))

        if not frames:
            raise OSError(f"no decodable frames in {path}")
        while len(frames) < count:
            frames.append(frames[-1])
        return np.stack(frames[:count])
    finally:
        capture.release()


def frames_to_jpegs(frames: np.ndarray, quality: int = 85) -> List[bytes]:
    """Encode frames as JPEG bytes, for backends that take images over a wire."""
    import cv2

    encoded: List[bytes] = []
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise OSError("failed to JPEG-encode a frame")
        encoded.append(buffer.tobytes())
    return encoded

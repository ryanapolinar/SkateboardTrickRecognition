"""B1 and B2: the two baselines that keep the model honest.

Neither of these trains on this project's representation. They exist so that every
claim about pose/board features (M1) has something to be measured against.

B1 - frozen feature extractor + linear probe
--------------------------------------------
A pretrained video model, frozen, with only a logistic regression on top. It has
never seen a skateboard trick and uses none of our pose/board machinery.

It is worth having for four reasons:

* **It bounds what generality buys.** If the pose+board pipeline cannot beat a
  frozen embedder, the hand-crafted representation is not earning its complexity,
  and we want to learn that in ~30 minutes of compute rather than after days of
  training. If it does beat it, the difference is the *measured* value of the
  skateboarding-specific representation instead of an opinion.
* **It is the number a reviewer asks for.** "Linear probe on frozen features" is
  the standard cheap protocol in video action recognition, so the result is
  comparable to published work rather than a bespoke metric.
* **It tests the dataset, not just the model.** A frozen embedder keys on
  appearance. If it scores well above the majority floor, the labels are probably
  separable by shortcut cues - venue, camera, clothing, skater - rather than by
  the rotation. That is a leakage alarm we want early, which is why B1 is scored
  on exactly the same split as everything else.
* **It is the ceiling for "no motion model".** Mean-pooling a clip's frames is a
  bag of frames: it ignores rotation order by construction. Whatever it cannot do
  is what the trajectory representation of plan section 4 is *for*.

B2 - vision-language model, zero-shot
-------------------------------------
A generalist that has read the internet, asked to name the trick with no training
data. It is the only baseline that emits a *name*, which makes it the first real
consumer of the taxonomy: the free-text answer is resolved with
``Taxonomy.normalize_label``, i.e. exactly the alias table the manifest uses.

* If it lands near chance, a frozen generalist genuinely cannot do this and the
  task needs the rotation reasoning this project is built around.
* If it lands high, the task may be easier than assumed - or the VLM may be
  reading the venue rather than the trick, which B1's shortcut check can confirm.
* Practically it is also a labelling aid: a VLM can pre-label clips for the
  active-learning and M3 second-opinion paths.

Honest scope
------------
VideoMAE/V-JEPA2 weights and hosted-VLM API keys are not guaranteed to be
available, so every backend reports *why* it cannot run instead of quietly
returning something. ``motion_stats`` runs today on the core dependency set and is
a genuinely weak frozen extractor: useful as a floor, never as the B1 number in a
report. The ``mock`` backends exist for tests and their numbers are plumbing
checks that must never be published as results.
"""

from __future__ import annotations

import abc
import base64
import dataclasses
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from skateid.eval import MajorityClassBaseline, evaluate_predictions
from skateid.taxonomy import Taxonomy, normalize_name
from skateid.video import DEFAULT_FRAME_COUNT, DEFAULT_SIZE, frames_to_jpegs, sample_frames


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# --- B1: frozen embedders --------------------------------------------------


class ClipEmbedder(abc.ABC):
    """Turns a clip's frames into one fixed-length vector. Never trained here."""

    name: str = "abstract"
    dim: int = 0
    #: Importable module names this backend needs.
    requirements: Tuple[str, ...] = ()
    #: Weights pulled on first use, for the operator's benefit.
    downloads: str = ""
    #: Required (width, height) of sampled frames, or None to accept any size.
    input_size: Optional[Tuple[int, int]] = None
    #: Required frame count, or None to accept any. Backbones with fixed temporal
    #: position embeddings (e.g. VideoMAE) only accept their training frame count.
    input_frames: Optional[int] = None

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        """``(usable, reason)`` so callers can skip loudly instead of crashing."""
        missing = [module for module in cls.requirements if not _module_available(module)]
        if missing:
            return False, f"missing optional dependencies: {', '.join(missing)}"
        return True, ""

    @abc.abstractmethod
    def embed(self, frames: np.ndarray) -> np.ndarray:
        """Embed ``(T, H, W, 3)`` uint8 frames as a 1-D float vector."""


class MotionStatsEmbedder(ClipEmbedder):
    """Weak, dependency-free extractor: colour, motion energy and centroid drift.

    Runs on numpy alone, so B1's plumbing is exercisable on any checkout. It has
    no notion of a board or a rotation, which is exactly why it is a floor rather
    than a headline. Prefer ``videomae``/``resnet18`` for a real B1 number.
    """

    name = "motion_stats"
    dim = 18

    def embed(self, frames: np.ndarray) -> np.ndarray:
        pixels = frames.astype(np.float32) / 255.0
        gray = pixels.mean(axis=3)  # (T, H, W)

        features: List[float] = [float(gray.mean()), float(gray.std())]
        for channel in range(3):
            plane = pixels[..., channel]
            features += [float(plane.mean()), float(plane.std())]

        if len(gray) > 1:
            deltas = np.abs(np.diff(gray, axis=0))  # (T-1, H, W)
            energy = deltas.reshape(deltas.shape[0], -1).mean(axis=1)
            features += [float(energy.mean()), float(energy.std()), float(energy.max())]
            weights = energy / (energy.sum() + 1e-6)
            centroid = float((weights * np.arange(len(energy))).sum())
            features.append(centroid / max(len(energy) - 1, 1))

            # Centroid drift in each half of the frame: a crude translation proxy
            # that at least reacts to the board moving under the skater.
            xs = np.linspace(0.0, 1.0, gray.shape[2], dtype=np.float32)
            half = gray.shape[1] // 2
            for rows in (slice(0, half), slice(half, None)):
                mass = gray[:, rows, :]
                # Each slice needs its own height axis: the second one is shorter.
                ys = np.linspace(0.0, 1.0, mass.shape[1], dtype=np.float32)
                total = mass.reshape(mass.shape[0], -1).sum(axis=1)
                cx = (mass.sum(axis=1) @ xs) / (total + 1e-6)
                cy = (mass.sum(axis=2) @ ys) / (total + 1e-6)
                features += [float(np.diff(cx).mean()), float(np.diff(cy).mean())]
        else:
            # One frame carries no motion at all; keep the vector shape fixed.
            features += [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]

        timeline = gray.reshape(len(gray), -1).mean(axis=1)
        slope = float(np.polyfit(np.arange(len(timeline)), timeline, 1)[0]) if len(timeline) > 1 else 0.0
        features.append(slope)
        features.append(float(np.abs(np.gradient(gray, axis=1)).mean()))

        vector = np.asarray(features, dtype=np.float32)
        if vector.shape[0] != self.dim:
            raise AssertionError(f"{self.name} produced {vector.shape[0]} features, expected {self.dim}")
        return vector


# ImageNet statistics, used by every torchvision backbone below.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchvisionEmbedder(ClipEmbedder):
    """Frozen ImageNet backbone, features averaged over frames.

    Needs the ``deeplearning`` extra. Mean-pooling frames is a deliberate
    choice, not a limitation worked around: it makes this baseline a genuine
    bag-of-frames model with no notion of rotation order, so it defines the
    "no motion model" ceiling the trajectory representation has to beat.
    """

    requirements = ("torch", "torchvision")
    downloads = "ImageNet weights (~45 MB for resnet18) on first use"
    input_size = (224, 224)

    #: Backbones whose pooled width is known without running the model.
    _DIMS = {"resnet18": 512, "resnet50": 2048, "mvit_v2_s": 768, "swin_t": 768}

    def __init__(self, arch: str = "resnet18", device: Optional[str] = None) -> None:
        import torch
        import torchvision.models as models

        if arch not in self._DIMS:
            raise ValueError(f"unsupported arch {arch!r}; known: {sorted(self._DIMS)}")
        self.arch = arch
        self.name = arch
        self.dim = self._DIMS[arch]
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = getattr(models, arch)(weights=models.get_model_weights(arch).DEFAULT)
        # Drop the classifier so forward() returns pooled features.
        for attribute in ("fc", "head", "heads", "classifier"):
            if hasattr(model, attribute):
                setattr(model, attribute, torch.nn.Identity())
                break
        self._model = model.eval().to(self.device)

    def embed(self, frames: np.ndarray) -> np.ndarray:
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div_(255.0)
        if tuple(tensor.shape[-2:]) != (224, 224):
            tensor = torch.nn.functional.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False
            )
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            pooled = self._model(tensor.to(self.device))
        return pooled.mean(dim=0).cpu().numpy().astype(np.float32)


class VideoMAEEmbedder(ClipEmbedder):
    """Frozen VideoMAE, temporal-mean pooled. The strongest available B1 backend.

    Needs the ``deeplearning`` extra plus ``transformers<5``. The pin is not
    cosmetic: VideoMAE's published checkpoint stores its state dict with the
    legacy ``{0...11}`` layer-broadcast keys, which torch 2.x no longer expands
    into per-layer entries. Under transformers 5.x those keys are dropped, the
    attention biases stay **randomly initialised**, and the encoder looks frozen
    while quietly being a partly-random network. That failure is invisible in the
    resulting score, so :meth:`_load_strict` refuses to return a model that did
    not load completely.
    """

    name = "videomae"
    dim = 768
    requirements = ("torch", "transformers")
    downloads = "HuggingFace 'MCG-NJU/videomae-base' (~350 MB) on first use"
    input_size = (224, 224)
    #: VideoMAE's temporal position embeddings are fixed for its training clip
    #: length: 16 frames = 8 tubelets x 196 patches = 1568 positions. Feeding 8
    #: frames produces 784 and the forward pass dies with an opaque tensor-size
    #: error, so the frame count is a property of the backend, not a free choice.
    input_frames = 16
    #: Number of checkpoint keys the encoder legitimately discards (the MAE decoder).
    discarded_keys: int = 0

    def __init__(self, checkpoint: str = "MCG-NJU/videomae-base", device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoImageProcessor, VideoMAEModel

        self.checkpoint = checkpoint
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)
        self._model = self._load_strict(VideoMAEModel, checkpoint).to(self.device)
        self._model.eval()

    def _load_strict(self, model_cls, checkpoint: str):
        """Load pretrained weights, refusing a partially-loaded encoder.

        ``unexpected_keys`` are tolerated: VideoMAE is a masked autoencoder, so
        its checkpoint carries a pretraining *decoder* (and head/mask token) that
        an encoder-only probe does not build. Discarding those is correct. A
        *missing* key is not: it means a parameter kept its random init, so the
        "frozen pretrained" encoder is partly noise and any score from it is
        meaningless. Fail loudly instead.
        """
        model, info = model_cls.from_pretrained(checkpoint, output_loading_info=True)
        missing = list(info.get("missing_keys") or [])
        mismatched = list(info.get("mismatched_keys") or [])
        errors = list(info.get("error_msgs") or [])
        if missing or mismatched or errors:
            raise RuntimeError(
                f"{self.name}: the checkpoint {checkpoint!r} did not load completely, so its "
                "features would be partly random and any score meaningless. "
                f"missing={missing[:5]} ({len(missing)}), "
                f"mismatched={mismatched[:5]} ({len(mismatched)}), errors={errors[:2]}. "
                "This checkpoint uses legacy '{0...11}' state-dict keys, which need "
                "transformers<5; check that the pin in pyproject.toml is installed."
            )
        # The encoder is fully covered by the missing-keys check above, so the
        # only unexpected keys that matter are ones the encoder *should* have
        # taken. Everything else is expected: VideoMAE is a masked autoencoder,
        # so its checkpoint also carries a pretraining decoder (64 keys), its
        # `mask_token`, and the `encoder_to_decoder` projection -- none of which
        # an encoder-only probe builds, and all of which are correct to discard.
        unexpected = list(info.get("unexpected_keys") or [])
        encoder_keys = [key for key in unexpected if key.startswith("encoder.")]
        if encoder_keys:
            raise RuntimeError(
                f"{self.name}: the checkpoint {checkpoint!r} contains encoder parameters that "
                f"did not match this architecture, so the encoder is not the pretrained one: "
                f"{encoder_keys[:5]} ({len(encoder_keys)} total)"
            )
        self.discarded_keys = len(unexpected)
        return model

    def embed(self, frames: np.ndarray) -> np.ndarray:
        torch = self._torch
        # Hand the processor uint8 frames and let it do its own rescale/normalize.
        # Pre-dividing by 255 here would make the processor rescale twice, which
        # squashes every pixel by 255x and quietly ruins the features.
        inputs = self._processor(list(frames), return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**{key: value.to(self.device) for key, value in inputs.items()})
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


#: Registry name -> (class, constructor kwargs). ``resnet50``/``mvit_v2_s``/``swin_t``
#: reuse the torchvision class so only the backbone name changes.
_EMBEDDER_REGISTRY: Dict[str, Tuple[type, Dict[str, object]]] = {
    "motion_stats": (MotionStatsEmbedder, {}),
    "resnet18": (TorchvisionEmbedder, {"arch": "resnet18"}),
    "resnet50": (TorchvisionEmbedder, {"arch": "resnet50"}),
    "mvit_v2_s": (TorchvisionEmbedder, {"arch": "mvit_v2_s"}),
    "swin_t": (TorchvisionEmbedder, {"arch": "swin_t"}),
    "videomae": (VideoMAEEmbedder, {}),
}


def embedder_names() -> List[str]:
    return sorted(_EMBEDDER_REGISTRY)


def count_tag(frame_count: int, size: Tuple[int, int]) -> str:
    """Cache-directory suffix recording the sampling grid, e.g. ``16f_112x112``.

    The frame count and size are part of the feature's definition, so they belong
    in the cache key. Omitting them lets a re-run at a different resolution
    silently reuse features extracted at the old one.
    """
    width, height = size
    return f"{frame_count}f_{width}x{height}"


def available_embedders() -> Dict[str, Tuple[bool, str]]:
    """``name -> (usable, reason)`` for every registered backend."""
    return {name: cls.availability() for name, (cls, _) in _EMBEDDER_REGISTRY.items()}


def build_embedder(name: str = "motion_stats", **overrides: object) -> ClipEmbedder:
    """Instantiate a registered embedder, or explain what is missing."""
    if name not in _EMBEDDER_REGISTRY:
        raise KeyError(f"unknown embedder {name!r}; known: {embedder_names()}")
    cls, kwargs = _EMBEDDER_REGISTRY[name]
    usable, reason = cls.availability()
    if not usable:
        raise RuntimeError(f"embedder {name!r} is unavailable: {reason}")
    return cls(**{**kwargs, **overrides})  # type: ignore[arg-type]


@dataclasses.dataclass
class FeatureMatrix:
    """Frozen features for a set of clips, in manifest order."""

    clip_ids: List[str]
    labels: List[str]
    vectors: np.ndarray


def extract_features(
    manifest: pd.DataFrame,
    embedder: ClipEmbedder,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    size: Tuple[int, int] = DEFAULT_SIZE,
    cache_dir: Path | str = Path("cache/features"),
    limit: Optional[int] = None,
    log: Optional[Callable[[str], None]] = print,
) -> Tuple[FeatureMatrix, int]:
    """Embed every clip in ``manifest``, caching one ``.npy`` per clip.

    Caching by ``clip_id`` makes a long extraction resumable and lets B1 be
    re-scored with a different probe without re-decoding 671 videos. The cache
    directory is keyed on the backend *and* the sampling grid, so changing
    ``frame_count`` or the frame size can never silently reuse features that were
    extracted under different settings. Returns the matrix and the number of
    clips whose file was missing.
    """
    size = getattr(embedder, "input_size", None) or size
    required_frames = getattr(embedder, "input_frames", None)
    if required_frames and frame_count != required_frames:
        if log:
            log(f"  {embedder.name} needs exactly {required_frames} frames; using that, not {frame_count}.")
        frame_count = required_frames
    target = Path(cache_dir) / f"{embedder.name}_{count_tag(frame_count, size)}"
    target.mkdir(parents=True, exist_ok=True)
    rows = manifest if limit is None else manifest.head(limit)

    clip_ids: List[str] = []
    labels: List[str] = []
    vectors: List[np.ndarray] = []
    absent = 0

    for position, row in enumerate(rows.itertuples(index=False), start=1):
        clip_id = str(row.clip_id)
        cached = target / f"{clip_id}.npy"
        if cached.exists():
            vector = np.load(cached)
        else:
            clip = Path(str(row.file_path))
            if not clip.exists():
                absent += 1
                continue
            vector = embedder.embed(sample_frames(clip, count=frame_count, size=size))
            np.save(cached, vector)
        vectors.append(np.asarray(vector, dtype=np.float32))
        clip_ids.append(clip_id)
        labels.append(str(row.label))
        if log and position % 100 == 0:
            log(f"  embedded {position}/{len(rows)} clips")

    if vectors:
        matrix = np.vstack(vectors)
    else:
        matrix = np.zeros((0, max(embedder.dim, 1)), dtype=np.float32)
    return FeatureMatrix(clip_ids, labels, matrix), absent


class LinearProbeBaseline:
    """B1: logistic regression on frozen features. Nothing is fine-tuned."""

    def __init__(self, *, C: float = 1.0, max_iter: int = 2000, standardize: bool = True) -> None:
        self.scaler = StandardScaler() if standardize else None
        self.model = LogisticRegression(C=C, max_iter=max_iter)
        self.classes_: List[str] = []

    def fit(self, vectors: np.ndarray, labels: Sequence[str]) -> "LinearProbeBaseline":
        features = self.scaler.fit_transform(vectors) if self.scaler else vectors
        self.model.fit(features, list(labels))
        self.classes_ = [str(value) for value in self.model.classes_]
        return self

    def predict(self, vectors: np.ndarray) -> List[str]:
        features = self.scaler.transform(vectors) if self.scaler else vectors
        return [str(value) for value in self.model.predict(features)]


def run_b1(
    manifest: pd.DataFrame,
    *,
    embedder: str = "motion_stats",
    split_col: str = "split_holdout",
    frame_count: int = DEFAULT_FRAME_COUNT,
    limit: Optional[int] = None,
    cache_dir: Path | str = Path("cache/features"),
    log: Optional[Callable[[str], None]] = print,
) -> Dict[str, object]:
    """Fit and score the B1 probe, or report why it could not run."""
    result: Dict[str, object] = {"baseline": "b1", "embedder": embedder, "split": split_col}
    try:
        backend = build_embedder(embedder)
    except (KeyError, RuntimeError) as exc:
        result.update(status="skipped", reason=str(exc))
        return result

    train_frame = manifest[manifest[split_col] == "train"]
    test_frame = manifest[manifest[split_col] == "test"]
    if train_frame.empty or test_frame.empty:
        result.update(status="skipped", reason=f"split '{split_col}' has an empty side")
        return result

    features, absent = extract_features(
        manifest, backend, frame_count=frame_count, cache_dir=cache_dir, limit=limit, log=log
    )
    if not features.clip_ids:
        result.update(status="skipped", reason="no clip files were on disk")
        return result

    table = pd.DataFrame(features.vectors, columns=[f"f{i}" for i in range(features.vectors.shape[1])])
    table.insert(0, "label", features.labels)
    table.insert(0, "clip_id", features.clip_ids)
    usable = manifest[["clip_id", split_col]].merge(table, on="clip_id", how="inner")

    columns = [column for column in usable.columns if column.startswith("f")]
    train = usable[usable[split_col] == "train"]
    test = usable[usable[split_col] == "test"]
    if train.empty or test.empty:
        result.update(status="skipped", reason="embedding covered only one side of the split")
        return result

    probe = LinearProbeBaseline().fit(train[columns].to_numpy(), train["label"].tolist())
    predictions = probe.predict(test[columns].to_numpy())
    metrics = evaluate_predictions(test["label"], predictions)

    floor = MajorityClassBaseline().fit(train["label"])
    floor_metrics = evaluate_predictions(test["label"], floor.predict(len(test)))

    result.update(
        status="ok",
        dim=int(features.vectors.shape[1]),
        features=backend.name,
        n_train=int(len(train)),
        n_test=int(len(test)),
        n_classes=int(test["label"].nunique()),
        clips_missing=absent,
        metrics=metrics,
        majority_macro_f1=float(floor_metrics["macro_f1"]),
        predictions=predictions,
        test_labels=test["label"].tolist(),
    )
    return result


# --- B2: vision-language zero-shot -----------------------------------------

VLM_PROMPT_TEMPLATE = (
    "You are shown {count} evenly spaced frames, in time order, from one short video "
    "clip of a single flatground skateboard trick. There is no ramp, rail, ledge or "
    "obstacle in the clip.\n"
    "The skateboarder is riding {stance} stance.\n"
    "Answer with exactly one name from this list and nothing else:\n"
    "{options}\n"
    "If no flatground trick is recognisable, answer 'unknown'."
)


def build_vlm_prompt(taxonomy: Taxonomy, stance: str = "regular") -> str:
    """The B2 prompt. Names come from the registry, never from hard-coded strings."""
    options = ", ".join(taxonomy.dictionary.expressible_names())
    return VLM_PROMPT_TEMPLATE.format(count="{count}", stance=stance, options=options)


@dataclasses.dataclass
class VLMAnswer:
    """One clip's answer: the model's raw text plus what it resolved to."""

    clip_id: str
    raw: str
    label: Optional[str]

    @property
    def resolved(self) -> bool:
        return self.label is not None


class VLMBackend(abc.ABC):
    """Asks a vision-language model to name one trick from a few frames."""

    name: str = "abstract"
    requirements: Tuple[str, ...] = ()
    #: Environment variable holding the API key, empty for local backends.
    env_key: str = ""
    default_model: str = ""

    def __init__(self, model: str = "") -> None:
        self.model = model or self.default_model

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        missing = [module for module in cls.requirements if not _module_available(module)]
        if missing:
            return False, f"missing optional dependencies: {', '.join(missing)}"
        if cls.env_key and not os.environ.get(cls.env_key):
            return False, f"{cls.env_key} is not set"
        return True, ""

    @abc.abstractmethod
    def answer(self, prompt: str, images: Sequence[bytes]) -> str:
        """Return the model's raw text answer for one clip."""


def _data_uri(image: bytes, media_type: str = "image/jpeg") -> str:
    return f"data:{media_type};base64,{base64.b64encode(image).decode('ascii')}"


def _post_json(url: str, payload: dict, headers: Dict[str, str], timeout: int = 180) -> dict:
    """Minimal JSON POST. Stdlib only, so B2 adds no dependency for a hosted model."""
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class MockVLMBackend(VLMBackend):
    """Test-only backend. ``answer`` may be a string, or ``(prompt, images) -> str``."""

    name = "mock"

    def __init__(self, answer: object = "unknown") -> None:
        super().__init__("mock")
        self._answer = answer

    def answer(self, prompt: str, images: Sequence[bytes]) -> str:
        if callable(self._answer):
            return str(self._answer(prompt, images))
        return str(self._answer)


class OpenAIVLMBackend(VLMBackend):
    """Hosted. Requires OPENAI_API_KEY."""

    name = "openai"
    env_key = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"
    url = "https://api.openai.com/v1/chat/completions"

    def answer(self, prompt: str, images: Sequence[bytes]) -> str:
        content: List[dict] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": _data_uri(image)}} for image in images]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 16,
            "temperature": 0,
        }
        data = _post_json(self.url, payload, {"Authorization": f"Bearer {os.environ[self.env_key]}"})
        return data["choices"][0]["message"]["content"]


class AnthropicVLMBackend(VLMBackend):
    """Hosted. Requires ANTHROPIC_API_KEY."""

    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_model = "claude-3-5-sonnet-latest"
    url = "https://api.anthropic.com/v1/messages"

    def answer(self, prompt: str, images: Sequence[bytes]) -> str:
        content: List[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(image).decode("ascii"),
                },
            }
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model,
            "max_tokens": 16,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        data = _post_json(
            self.url,
            payload,
            {"x-api-key": os.environ[self.env_key], "anthropic-version": "2023-06-01"},
        )
        return data["content"][0]["text"]


class GoogleVLMBackend(VLMBackend):
    """Hosted. Requires GOOGLE_API_KEY."""

    name = "google"
    env_key = "GOOGLE_API_KEY"
    default_model = "gemini-2.0-flash"

    def answer(self, prompt: str, images: Sequence[bytes]) -> str:
        parts: List[dict] = [{"text": prompt}]
        parts += [
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image).decode("ascii")}}
            for image in images
        ]
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={os.environ[self.env_key]}"
        )
        data = _post_json(
            url,
            {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0, "maxOutputTokens": 16}},
            {},
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OllamaVLMBackend(VLMBackend):
    """Local, key-free. Any vision model Ollama has pulled, e.g. ``llava`` or ``qwen2.5vl``."""

    name = "ollama"
    default_model = "llava"

    def __init__(self, model: str = "", host: str = "") -> None:
        super().__init__(model)
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        # No key and no import: whether it works depends on a running server, which
        # is discovered at call time rather than guessed at here.
        return True, ""

    def answer(self, prompt: str, images: Sequence[bytes]) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "images": [base64.b64encode(image).decode("ascii") for image in images],
            "options": {"temperature": 0},
        }
        data = _post_json(f"{self.host}/api/generate", payload, {})
        return str(data.get("response", ""))


_VLM_REGISTRY = {
    "mock": MockVLMBackend,
    "openai": OpenAIVLMBackend,
    "anthropic": AnthropicVLMBackend,
    "google": GoogleVLMBackend,
    "ollama": OllamaVLMBackend,
}


def vlm_names() -> List[str]:
    return sorted(_VLM_REGISTRY)


def available_vlms() -> Dict[str, Tuple[bool, str]]:
    return {name: backend.availability() for name, backend in _VLM_REGISTRY.items()}


def build_vlm(name: str = "mock", **overrides: object) -> VLMBackend:
    if name not in _VLM_REGISTRY:
        raise KeyError(f"unknown VLM backend {name!r}; known: {vlm_names()}")
    backend = _VLM_REGISTRY[name]
    usable, reason = backend.availability()
    if not usable:
        raise RuntimeError(f"VLM backend {name!r} is unavailable: {reason}")
    return backend(**overrides)  # type: ignore[arg-type]


def _evenly_pick(frames: np.ndarray, count: int) -> np.ndarray:
    """Take ``count`` frames spread across an already-sampled clip."""
    if count >= len(frames):
        return frames
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return frames[np.unique(indices)]


def resolve_answer(raw: str, taxonomy: Taxonomy) -> Optional[str]:
    """Map a model's free text onto a canonical name via the manifest's alias table.

    Matching is word-bounded, not plain substring, so a generic alias cannot
    capture a longer answer: "backside flip" resolves as its own name rather than
    collapsing onto ``flip``/``kickflip``, and "nollie" does not silently become
    ``ollie`` just because it contains it. Among matches the longest wins, so
    "it is a 360 flip I think" resolves to ``tre_flip``.

    Returning ``None`` is an abstention: scored as wrong, reported separately.
    """
    text = normalize_name(raw)
    if not text:
        return None
    exact = taxonomy.allowlist.canonical(text)
    if exact is not None:
        return exact

    candidates: List[Tuple[int, str]] = []
    for name in taxonomy.allowlist.canonical_names():
        for form in (name, *taxonomy.allowlist.aliases(name)):
            pattern = rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])"
            if re.search(pattern, text):
                candidates.append((len(form), name))
    return max(candidates)[1] if candidates else None


def run_b2(
    manifest: pd.DataFrame,
    *,
    backend: str = "mock",
    split_col: str = "split_holdout",
    frames_per_clip: int = 4,
    frame_count: int = DEFAULT_FRAME_COUNT,
    limit: Optional[int] = None,
    taxonomy: Optional[Taxonomy] = None,
    log: Optional[Callable[[str], None]] = print,
    **backend_kwargs: object,
) -> Dict[str, object]:
    """Ask a VLM to name each test clip, zero-shot, and score the answers."""
    tax = taxonomy or Taxonomy.load()
    result: Dict[str, object] = {"baseline": "b2", "backend": backend, "split": split_col}
    try:
        model = build_vlm(backend, **backend_kwargs)
    except (KeyError, RuntimeError) as exc:
        result.update(status="skipped", reason=str(exc))
        return result

    frame = manifest
    if split_col in manifest.columns:
        frame = manifest[manifest[split_col] == "test"]
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        result.update(status="skipped", reason=f"no clips on the test side of '{split_col}'")
        return result

    answers: List[VLMAnswer] = []
    labels: List[str] = []
    predictions: List[str] = []
    absent = 0
    failures = 0

    for position, row in enumerate(frame.itertuples(index=False), start=1):
        clip = Path(str(row.file_path))
        if not clip.exists():
            absent += 1
            continue
        stance = str(getattr(row, "stance", "") or "regular").lower() or "regular"
        prompt = build_vlm_prompt(tax, stance=stance)
        images = frames_to_jpegs(_evenly_pick(sample_frames(clip, count=frame_count), frames_per_clip))
        try:
            raw = model.answer(prompt, images)
        except Exception as exc:  # noqa: BLE001 - a dead server must not be a traceback
            # Fail the run on the first transport error rather than repeating the
            # same request for every remaining clip.
            result.update(
                status="failed",
                reason=f"the {model.name!r} backend errored on {clip.name}: {exc}",
                n_scored=len(predictions),
            )
            return result
        resolved = resolve_answer(raw, tax)
        if resolved is None:
            failures += 1
        answers.append(VLMAnswer(str(row.clip_id), raw, resolved))
        labels.append(str(row.label))
        predictions.append(resolved or "unknown")
        if log and position % 25 == 0:
            log(f"  asked {position}/{len(frame)} clips")

    if not predictions:
        result.update(status="skipped", reason="no clip files were on disk")
        return result

    metrics = evaluate_predictions(labels, predictions)
    result.update(
        status="ok",
        model=model.model,
        n_test=len(predictions),
        n_classes=len(set(labels)),
        clips_missing=absent,
        abstentions=failures,
        resolved_rate=(len(answers) - failures) / len(answers),
        metrics=metrics,
        predictions=predictions,
        test_labels=labels,
        answers=[dataclasses.asdict(answer) for answer in answers],
    )
    return result

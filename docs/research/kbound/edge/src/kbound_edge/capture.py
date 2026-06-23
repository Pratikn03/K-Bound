"""kbound_edge.capture -- Frame sources: OpenCV camera, synthetic, fake VideoCapture.

Three interchangeable sources behind one tiny :class:`FrameSource` interface
(``read() -> (ok, frame_bgr)``):

* :class:`OpenCVCameraSource` / :func:`open_opencv_source` -- wraps a real
  ``cv2.VideoCapture`` (webcam index, video file, or a phone-as-camera device).
* :class:`SyntheticFrameSource` -- deterministic generated frames with a
  fabricated 4-class structure and a configurable domain *shift* (brightness /
  contrast / noise / blur).  Used everywhere in the synthetic test chain.
* :class:`FakeVideoCapture` -- a drop-in stand-in that mimics the
  ``cv2.VideoCapture`` API (``isOpened/read/release/get``) but is backed by a
  :class:`SyntheticFrameSource`.  This lets the Tier-2 shadow loop exercise the
  exact OpenCV code path with NO camera attached.

LABEL HYGIENE: a :class:`SyntheticFrameSource` knows the latent class of each
frame, but the latent labels are reachable ONLY via the explicit ``.labels``
property (for OFFLINE training / calibration / evaluation).  ``read()`` returns
pixels only, so nothing on the online path can ever see a label.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Frame = np.ndarray  # HxWx3 uint8, BGR (OpenCV convention)


# Distinct base colours (BGR) for the 4 inspection classes.
_BASE_COLORS_BGR: Sequence[Tuple[int, int, int]] = (
    (200, 60, 60),    # class 0 -> bluish
    (60, 200, 60),    # class 1 -> greenish
    (60, 60, 200),    # class 2 -> reddish
    (60, 200, 200),   # class 3 -> yellowish
)


class FrameSource:
    """Minimal frame-source interface (a subset of cv2.VideoCapture)."""

    def read(self) -> Tuple[bool, Optional[Frame]]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def __iter__(self):
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame


def _make_frame(
    cls: int,
    size: int,
    rng: np.random.Generator,
    brightness: float,
    contrast: float,
    noise: float,
    blur: int,
    n_classes: int,
) -> Frame:
    """Render one synthetic BGR frame for latent class ``cls``."""
    color = np.array(_BASE_COLORS_BGR[cls % len(_BASE_COLORS_BGR)], dtype=np.float32)
    img = np.ones((size, size, 3), dtype=np.float32) * color

    # Class-specific bright blob in one quadrant -> spatial signal for the CNN.
    q = size // 2
    r = (cls // 2) % 2
    c = cls % 2
    img[r * q : r * q + q, c * q : c * q + q, :] += 60.0

    # Domain shift: contrast about mid-grey, then brightness offset.
    img = (img - 128.0) * float(contrast) + 128.0 + float(brightness) * 255.0

    # Sensor noise.
    if noise > 0:
        img = img + rng.normal(0.0, noise * 255.0, img.shape).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)

    if blur and blur >= 3 and blur % 2 == 1:
        try:
            import cv2

            img = cv2.GaussianBlur(img, (blur, blur), 0)
        except Exception:
            pass
    return img


class SyntheticFrameSource(FrameSource):
    """Deterministic synthetic frames with a fabricated 4-class structure.

    Parameters
    ----------
    num_frames : int
        Total frames to emit.
    n_classes : int, default=4
    image_size : int, default=64
        Square frame side (small for fast CPU tests; real clips use 224).
    seed : int, default=0
    brightness, contrast, noise : float
        Domain-shift knobs (0.0/1.0/0.05 = "clean").
    blur : int, default=0
        Odd kernel size for Gaussian blur (0 = none).
    class_seq : sequence of int, optional
        Explicit per-frame class sequence; if None, classes are drawn uniformly.
    """

    def __init__(
        self,
        num_frames: int,
        n_classes: int = 4,
        image_size: int = 64,
        seed: int = 0,
        brightness: float = 0.0,
        contrast: float = 1.0,
        noise: float = 0.05,
        blur: int = 0,
        class_seq: Optional[Sequence[int]] = None,
    ) -> None:
        self.num_frames = int(num_frames)
        self.n_classes = int(n_classes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.noise = float(noise)
        self.blur = int(blur)

        rng = np.random.default_rng(seed)
        if class_seq is None:
            self._labels = rng.integers(0, self.n_classes, size=self.num_frames)
        else:
            self._labels = np.asarray(list(class_seq)[: self.num_frames], dtype=int)

        self._frames: List[Frame] = [
            _make_frame(
                int(self._labels[i]), self.image_size, rng,
                self.brightness, self.contrast, self.noise, self.blur, self.n_classes,
            )
            for i in range(self.num_frames)
        ]
        self._i = 0

    def read(self) -> Tuple[bool, Optional[Frame]]:
        if self._i >= self.num_frames:
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def reset(self) -> None:
        self._i = 0

    @property
    def labels(self) -> np.ndarray:
        """OFFLINE-ONLY latent labels.  Never call this on the online path."""
        return self._labels.copy()

    @property
    def frames(self) -> List[Frame]:
        return list(self._frames)


class ListFrameSource(FrameSource):
    """A :class:`FrameSource` backed by a precomputed list of BGR frames.

    Used to splice several synthetic conditions into one mixed stream (e.g. for a
    Tier-2 shadow dry-run that should exercise adapt/abstain/freeze).
    """

    def __init__(self, frames: List[Frame], image_size: int = 64) -> None:
        self._frames = list(frames)
        self.num_frames = len(self._frames)
        self.image_size = int(image_size)
        self._i = 0

    def read(self) -> Tuple[bool, Optional[Frame]]:
        if self._i >= self.num_frames:
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def reset(self) -> None:
        self._i = 0


class OpenCVCameraSource(FrameSource):
    """Wrap a real ``cv2.VideoCapture`` (camera index, file path, or device)."""

    def __init__(self, index_or_path):
        import cv2

        self.cap = cv2.VideoCapture(index_or_path)

    def isOpened(self) -> bool:
        return bool(self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[Frame]]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


def open_opencv_source(index_or_path) -> OpenCVCameraSource:
    """Open a real OpenCV source (webcam index int, or video/device path str)."""
    return OpenCVCameraSource(index_or_path)


class VideoCaptureFrameSource(FrameSource):
    """Adapt ANY cv2.VideoCapture-like object (real or :class:`FakeVideoCapture`)."""

    def __init__(self, cap):
        self.cap = cap

    def read(self) -> Tuple[bool, Optional[Frame]]:
        return self.cap.read()

    def release(self) -> None:
        if hasattr(self.cap, "release"):
            self.cap.release()


class FakeVideoCapture:
    """A ``cv2.VideoCapture`` look-alike backed by a :class:`SyntheticFrameSource`.

    Implements just enough of the OpenCV API (``isOpened``, ``read``, ``release``,
    ``get``) for the runtime to drive it as if it were a real camera -- so Tier-2
    can be exercised end-to-end without hardware.
    """

    # A couple of cv2.CAP_PROP_* constants mirrored so callers need not import cv2.
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, source=None, **kwargs):
        self._src = source if source is not None else SyntheticFrameSource(**kwargs)
        self._released = False

    def isOpened(self) -> bool:
        return not self._released

    def read(self) -> Tuple[bool, Optional[Frame]]:
        if self._released:
            return False, None
        return self._src.read()

    def get(self, prop: int) -> float:
        if prop == self.CAP_PROP_FRAME_COUNT:
            return float(getattr(self._src, "num_frames", 0))
        if prop in (self.CAP_PROP_FRAME_WIDTH, self.CAP_PROP_FRAME_HEIGHT):
            return float(getattr(self._src, "image_size", 0))
        if prop == self.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def release(self) -> None:
        self._released = True

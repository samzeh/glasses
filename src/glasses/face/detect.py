"""Stage 1: detection, and the choice of primary subject.

The detector is a Protocol so the pipeline can be exercised without model
weights. ``ScrfdDetector`` is the real one; it imports insightface lazily so
the rest of the package stays installable without it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import DetectConfig
from .models import DETECTOR_FILE, default_providers, load_net, pack_dir
from .types import BBox, Detection, Landmarks5


@runtime_checkable
class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Find faces in a BGR frame, strongest first."""
        ...


class ScrfdDetector:
    """SCRFD as bundled with InsightFace. Returns boxes plus 5 landmarks.

    Kept behind the same Protocol as everything else so swapping in
    YOLOv8-face is a constructor change, not a pipeline change.

    The model loads on first ``detect`` rather than in ``__init__``, so
    constructing a pipeline is free and the ~1 s of ONNX session setup lands
    somewhere you can see it.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        *,
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
        config: DetectConfig | None = None,
    ) -> None:
        self.config = config or DetectConfig()
        self.model_pack = model_pack
        self._det_size = det_size
        self._providers = providers or default_providers()
        self._net = None

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._net is None:
            self._net = load_net(
                pack_dir(self.model_pack) / DETECTOR_FILE,
                providers=self._providers,
                input_size=self._det_size,
                det_thresh=self.config.min_detector_score,
            )
        return self._net

    def detect(self, frame: np.ndarray) -> list[Detection]:
        net = self._ensure_loaded()
        boxes, landmarks = net.detect(frame, max_num=0)

        if boxes is None or len(boxes) == 0:
            return []
        if landmarks is None:
            # Alignment is non-negotiable, so a detector that cannot produce
            # the 5 points is unusable here rather than degraded.
            raise RuntimeError(
                f"{self.model_pack}/{DETECTOR_FILE} returned no landmarks; "
                "the pipeline cannot align without them"
            )

        detections = [
            Detection(
                bbox=BBox(*(float(v) for v in box[:4])),
                landmarks=Landmarks5.from_array(np.asarray(kps)),
                score=float(box[4]),
            )
            for box, kps in zip(np.asarray(boxes), np.asarray(landmarks), strict=True)
            if float(box[4]) >= self.config.min_detector_score
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


def select_primary(
    detections: list[Detection],
    frame_shape: tuple[int, ...],
    config: DetectConfig | None = None,
) -> int | None:
    """Index of the largest centred face, or None if there are no detections.

    Size and centredness are combined rather than applied in sequence: a large
    face at the edge of the frame is usually someone walking past, and a small
    centred one is usually the person you are talking to from across a table.
    """
    if not detections:
        return None
    cfg = config or DetectConfig()

    height, width = float(frame_shape[0]), float(frame_shape[1])
    frame_center = (width / 2.0, height / 2.0)
    max_distance = float(np.hypot(width / 2.0, height / 2.0)) or 1.0
    max_area = max(d.bbox.area for d in detections) or 1.0

    best_index, best_score = 0, -np.inf
    for index, detection in enumerate(detections):
        cx, cy = detection.bbox.center
        distance = float(np.hypot(cx - frame_center[0], cy - frame_center[1]))
        centredness = 1.0 - min(1.0, distance / max_distance)
        size = detection.bbox.area / max_area
        score = size + cfg.center_weight * centredness
        if score > best_score:
            best_index, best_score = index, score
    return best_index

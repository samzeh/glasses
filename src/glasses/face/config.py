"""Tunable constants for the face pipeline.

Every number here is a starting point, not a finding. The PRD is explicit that
published thresholds are measured on clean benchmark photos and will not
transfer to an OV2640 behind a plastic lens. Anything marked CALIBRATE must be
re-measured against a held-out set of frames from the real glasses camera
before the numbers below mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical ArcFace 5-point reference, in pixels, for a 112x112 crop. Order is
# left eye, right eye, nose, left mouth corner, right mouth corner — "left" as
# it appears in the image, not the subject's left.
ARCFACE_REFERENCE_112: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


@dataclass(frozen=True)
class QualityConfig:
    """Stage 3. Rejecting a bad frame is cheaper than recovering from the bad
    embedding it produces, so this gate is deliberately strict. Expect it to
    reject 60-80% of ESP frames; that is why the capture burst exists."""

    min_detector_score: float = 0.7
    min_interocular_px: float = 40.0
    #: Variance of the Laplacian over the aligned crop. CALIBRATE — this is the
    #: single most camera-specific number in the file and the default is a
    #: placeholder from webcam-grade imagery.
    min_blur_variance: float = 60.0
    max_abs_yaw_deg: float = 30.0
    #: The term most likely to over-reject. Per-person face proportions move
    #: the pitch estimate by roughly +/-12 degrees on their own, so a third of
    #: this budget is spent on face shape rather than head angle. If the gate
    #: pass rate comes out low and ``pitch`` dominates the failures, suspect
    #: this before the camera.
    max_abs_pitch_deg: float = 30.0
    min_mean_luminance: float = 40.0
    max_mean_luminance: float = 220.0

    #: Where the nose sits between the eye line and the mouth line on a frontal
    #: face, as a fraction of the eye-to-mouth distance.
    #:
    #: CALIBRATE, and more urgently than the rest. Deriving this from the
    #: ArcFace canonical reference gives 0.495 and is wrong in practice: SCRFD
    #: puts its nose keypoint markedly lower, around 0.66 on real detections,
    #: so a reference-derived constant reads near-frontal faces as pitched
    #: down by 45 degrees and the pitch gate throws them away.
    #:
    #: The value below is the median over 6 detected faces — far too small a
    #: sample to trust. Run ``glasses calibrate-pose`` on your own frames and
    #: replace it. The constant belongs to whichever detector produces the
    #: landmarks, so it must be re-measured if SCRFD is swapped out.
    nose_frontal_fraction: float = 0.66
    #: How far that fraction travels over a quarter turn of pitch. Sets the
    #: scale of the pitch estimate. CALIBRATE.
    nose_pitch_span: float = 0.50


@dataclass(frozen=True)
class MatchConfig:
    """Stage 5. Thresholds are cosine similarity between L2-normalized
    embeddings, so they live in [-1, 1]."""

    #: Starting point from the PRD. Calibrate against a false-accept target,
    #: and re-check it at every gallery size — a threshold tuned at ten people
    #: produces steady misidentifications at two hundred.
    accept_threshold: float = 0.40
    #: Matches this far below the accept threshold surface as "possibly X"
    #: rather than as a name. A confident wrong name is actively harmful.
    review_margin: float = 0.05
    #: A person is scored by the mean of their top-k template similarities,
    #: never by a centroid: a centroid built from one good frontal and one bad
    #: profile is worse than either. k=3 per the PRD.
    top_k_templates: int = 3
    #: How many templates to pull from the gallery before grouping by person.
    #: Must be comfortably larger than top_k_templates * expected candidates.
    search_limit: int = 64


@dataclass(frozen=True)
class DetectConfig:
    """Stage 1. The primary subject is the largest centred face; the others are
    kept, because they matter for group encounters and for the refusal to bind
    facts when more than one non-wearer is visible."""

    min_detector_score: float = 0.5
    #: Weight on centredness when picking the primary subject. 0 picks purely
    #: by face area, 1 weights a centred face as heavily as a large one.
    center_weight: float = 0.5


@dataclass(frozen=True)
class PipelineConfig:
    detect: DetectConfig = field(default_factory=DetectConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    #: Aligned crop size. 112 is what ArcFace expects; AdaFace variants differ.
    align_size: int = 112

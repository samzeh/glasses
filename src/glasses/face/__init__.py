"""Server-side face recognition: detect, align, gate, embed, match, resolve."""

from ..identity import (
    Embedding,
    Gallery,
    GalleryEntry,
    InMemoryGallery,
    MatchConfig,
    MatchResult,
    MatchStatus,
    PersonScore,
    PgVectorGallery,
    TemplateHit,
    cosine_similarity,
    l2_normalize,
    match_embedding,
    resolve,
    score_persons,
)
from .config import DetectConfig, PipelineConfig, QualityConfig
from .detect import FaceDetector, ScrfdDetector, select_primary
from .embed import ArcFaceEmbedder, FaceEmbedder
from .pipeline import FacePipeline
from .quality import assess, head_pose, laplacian_variance
from .types import (
    AlignedFace,
    BBox,
    Detection,
    FaceResult,
    FrameResult,
    HeadPose,
    Landmarks5,
    QualityReport,
    StageTimings,
)

__all__ = [
    "AlignedFace",
    "ArcFaceEmbedder",
    "BBox",
    "Detection",
    "DetectConfig",
    "Embedding",
    "FaceDetector",
    "FaceEmbedder",
    "FacePipeline",
    "FaceResult",
    "FrameResult",
    "Gallery",
    "GalleryEntry",
    "HeadPose",
    "InMemoryGallery",
    "Landmarks5",
    "MatchConfig",
    "MatchResult",
    "MatchStatus",
    "PersonScore",
    "PgVectorGallery",
    "PipelineConfig",
    "QualityConfig",
    "QualityReport",
    "ScrfdDetector",
    "StageTimings",
    "TemplateHit",
    "assess",
    "cosine_similarity",
    "head_pose",
    "l2_normalize",
    "laplacian_variance",
    "match_embedding",
    "resolve",
    "score_persons",
    "select_primary",
]

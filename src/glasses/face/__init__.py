"""Server-side face recognition: detect, align, gate, embed, match, resolve."""

from .config import DetectConfig, MatchConfig, PipelineConfig, QualityConfig
from .detect import FaceDetector, ScrfdDetector, select_primary
from .embed import ArcFaceEmbedder, FaceEmbedder, cosine_similarity, l2_normalize
from .gallery import Gallery, GalleryEntry, InMemoryGallery, PgVectorGallery
from .match import match_embedding, resolve, score_persons
from .pipeline import FacePipeline
from .quality import assess, head_pose, laplacian_variance
from .types import (
    AlignedFace,
    BBox,
    Detection,
    Embedding,
    FaceResult,
    FrameResult,
    HeadPose,
    Landmarks5,
    MatchResult,
    MatchStatus,
    PersonScore,
    QualityReport,
    StageTimings,
    TemplateHit,
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

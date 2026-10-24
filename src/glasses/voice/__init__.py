"""Voice identity: who is speaking, as opposed to what they said.

Transcription is deliberately not here. This package answers "which enrolled
person is this voice", which no speech-to-text API does — they transcribe
sound, they do not track people across recordings.
"""

from .config import (
    SAMPLE_RATE,
    VoiceMatchConfig,
    VoicePipelineConfig,
    VoiceQualityConfig,
    WearerConfig,
)
from .embed import SpeakerEmbedder, WeSpeakerEmbedder
from .pipeline import VoicePipeline
from .quality import assess, split_long_segment
from .segment import DiarizingSegmenter, Segmenter, VadSegmenter, merge_adjacent
from .types import (
    RecordingResult,
    SegmentQuality,
    SpeechSegment,
    VoiceResult,
    VoiceTimings,
)

__all__ = [
    "SAMPLE_RATE",
    "DiarizingSegmenter",
    "RecordingResult",
    "SegmentQuality",
    "Segmenter",
    "SpeakerEmbedder",
    "SpeechSegment",
    "VadSegmenter",
    "VoiceMatchConfig",
    "VoicePipeline",
    "VoicePipelineConfig",
    "VoiceQualityConfig",
    "VoiceResult",
    "VoiceTimings",
    "WeSpeakerEmbedder",
    "WearerConfig",
    "assess",
    "merge_adjacent",
    "split_long_segment",
]

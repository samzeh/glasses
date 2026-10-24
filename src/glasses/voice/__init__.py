"""Voice identity: who is speaking, as opposed to what they said.

Identity is the part no speech-to-text API provides. Cartesia transcribes
sound very well and does not track people across recordings; that is this
package's job, and transcription hangs off it per segment rather than the
other way round.
"""

from .cartesia import CartesiaError, CartesiaTranscriber
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
from .transcribe import (
    Transcriber,
    Utterance,
    WhisperTranscriber,
    build_transcriber,
    render,
    transcribe_segments,
)
from .types import (
    RecordingResult,
    SegmentQuality,
    SpeechSegment,
    VoiceResult,
    VoiceTimings,
)

__all__ = [
    "SAMPLE_RATE",
    "CartesiaError",
    "CartesiaTranscriber",
    "DiarizingSegmenter",
    "RecordingResult",
    "SegmentQuality",
    "Segmenter",
    "SpeakerEmbedder",
    "SpeechSegment",
    "Transcriber",
    "Utterance",
    "VadSegmenter",
    "VoiceMatchConfig",
    "VoicePipeline",
    "VoicePipelineConfig",
    "VoiceQualityConfig",
    "VoiceResult",
    "VoiceTimings",
    "WeSpeakerEmbedder",
    "WhisperTranscriber",
    "WearerConfig",
    "assess",
    "build_transcriber",
    "merge_adjacent",
    "render",
    "transcribe_segments",
    "split_long_segment",
]

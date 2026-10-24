"""Tunable constants for the voice pipeline.

The headline number here is ``min_duration_s``. Speaker embeddings degrade
sharply on short speech, far more than most people expect — published work
reports EER rising by roughly half from a duration change of 3.6 s to 2.1 s,
before any noise is added. A conversation is mostly backchannels ("yeah",
"mhm", "right"), so without a duration floor the gallery fills with
embeddings of nothing.

Everything is CALIBRATE until measured on real glasses audio. A mic on
someone's face in a bar is a much worse input than the VoxCeleb clips these
models are benchmarked on, and far-field, reverberation and channel mismatch
all compound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..identity.config import MatchConfig

#: Every model in the chain expects 16 kHz mono. Not a preference: the
#: feature extractors hardcode it, and resampling at the edge is cheaper than
#: discovering a silent mismatch later.
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class VoiceQualityConfig:
    """The gate, and the single biggest lever in the pipeline.

    Same role as the face quality gate: reject the segment rather than
    recover from the bad embedding it produces. Expect it to throw away most
    of a real conversation, and that is correct — you need a handful of good
    seconds per person, not all of them.
    """

    #: Below this, the embedding is closer to noise than to an identity.
    #: 3.0 s is the knee in the published duration curves. CALIBRATE.
    min_duration_s: float = 3.0
    #: Longer than this and the segment probably spans a speaker change the
    #: segmenter missed. Split rather than embed across a turn boundary.
    max_duration_s: float = 30.0
    #: Crude SNR proxy: speech frame energy over the noise floor, in dB.
    #: CALIBRATE against real room noise; 10 dB is a placeholder.
    min_snr_db: float = 10.0
    #: Fraction of samples at full scale. A clipped waveform has harmonics
    #: that are not the speaker's, and the wearer's own voice clips most.
    max_clipped_fraction: float = 0.01
    #: RMS floor in dBFS. Rejects segments where VAD fired on a distant voice
    #: too quiet to characterise.
    min_rms_dbfs: float = -45.0


@dataclass(frozen=True)
class VoiceMatchConfig(MatchConfig):
    """Voice thresholds, separate from the face ones on purpose.

    Higher than the face threshold because voiceprints of different people
    sit closer together than faces do. 0.60 has two independent supports:
    it is the default in sherpa-onnx's own speaker-identification example,
    and it is the only value that behaves on the real-speech sample this was
    tested against, where a genuine pair scored 0.658 and an impostor 0.584.

    CALIBRATE, and treat the above as weak evidence: that sample is three
    recordings of two people, in Chinese, through an English-trained model.
    ``glasses voice-eval`` measures it properly on your own voices, and this
    is the number that decides how often you greet someone by a stranger's
    name — so err high. An "unknown" costs a prompt; a confident wrong name
    costs trust.
    """

    accept_threshold: float = 0.60
    review_margin: float = 0.07


@dataclass(frozen=True)
class WearerConfig:
    """The wearer is the one solved case, and leaning on it turns open-set
    diarization into a binary question.

    Their voice is the loudest thing the microphone ever hears and is partly
    bone-conducted, so it is both easy to recognise and unlike how they sound
    to anyone else. Enroll it once at setup; everything that is not the
    wearer is a conversational partner.
    """

    #: Matching the wearer is an easier problem than matching a stranger, so
    #: it gets a higher bar — a false "this is the wearer" silently discards
    #: someone else's speech.
    accept_threshold: float = 0.55


@dataclass(frozen=True)
class VoicePipelineConfig:
    quality: VoiceQualityConfig = field(default_factory=VoiceQualityConfig)
    match: VoiceMatchConfig = field(default_factory=VoiceMatchConfig)
    wearer: WearerConfig = field(default_factory=WearerConfig)

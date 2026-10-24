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
    sit closer together than faces do — and higher than you would guess.

    CALIBRATE. The operating range moves a long way with the microphone,
    which is the whole reason this cannot ship as a constant:

    * Public 16 kHz test clips, two speakers: genuine 0.66, impostor 0.58.
    * A MacBook Air built-in mic, two speakers at desk distance: genuine
      0.91, impostor 0.73.

    A threshold that works on one of those is badly wrong on the other.
    0.80 sits in the gap for the laptop case and is the shipped default
    because that is the setup this gets developed on; sherpa-onnx's own
    example uses 0.60, which would call a stranger by your name on that
    hardware.

    ``glasses voice-eval`` measures it on your own voices and prints the
    number to paste here. Err high while you wait: an "unknown" costs a
    prompt, a confident wrong name costs trust.
    """

    accept_threshold: float = 0.80
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

    #: Higher than the gallery threshold, not lower. A false "this is the
    #: wearer" silently discards someone else's speech, and silence is the
    #: worst failure mode here — it looks like the pipeline working.
    #: Measured: a different speaker on the same laptop mic scored 0.737
    #: against the wearer, so 0.55 classified a stranger as the wearer and
    #: dropped every one of her segments.
    accept_threshold: float = 0.85


@dataclass(frozen=True)
class VoicePipelineConfig:
    quality: VoiceQualityConfig = field(default_factory=VoiceQualityConfig)
    match: VoiceMatchConfig = field(default_factory=VoiceMatchConfig)
    wearer: WearerConfig = field(default_factory=WearerConfig)

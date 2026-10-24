"""Fetching the sherpa-onnx model files.

Chosen over the PRD's pyannote + SpeechBrain stack for three reasons that
are about deployability rather than accuracy:

* No gated download. ``pyannote/speaker-diarization-3.1`` sits behind an
  interactive HuggingFace licence click-through, which cannot be scripted.
  These files are plain GitHub release assets.
* No torch. The PRD's stack pulls ~2 GB of PyTorch; sherpa-onnx is 37 MB and
  brings its own ONNX runtime.
* Same runtime shape as the face side, which already runs ONNX.

The trade is accuracy: pyannote 3.1 still leads on diarization error rate.
If diarization becomes the bottleneck, the Segmenter protocol is the seam to
swap it behind.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

#: Silero VAD. Speech/not-speech only — it cannot tell speakers apart.
VAD_URL = f"{_RELEASES}/asr-models/silero_vad.onnx"

#: WeSpeaker CAM++ trained on VoxCeleb: English, 16 kHz, 192-d, and the
#: closest open equivalent of the ECAPA-TDNN the PRD names. ReDimNet is the
#: upgrade to evaluate — it shows the least cross-domain degradation of the
#: current crop, which is exactly the failure mode far-field glasses audio
#: will hit. Same role as AdaFace on the face side.
EMBEDDING_URL = f"{_RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx"

#: pyannote's segmentation net, ONNX-exported and redistributed under its
#: original MIT licence. This is the diarization half, and the one piece
#: genuinely derived from pyannote — without the gate.
SEGMENTATION_URL = (
    f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"


def model_dir() -> Path:
    """Where downloaded voice models live. ``GLASSES_MODEL_DIR`` overrides."""
    root = os.environ.get("GLASSES_MODEL_DIR")
    path = Path(root) if root else Path.home() / ".glasses" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  downloading {destination.name} …")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    # Rename only once the bytes are all there, so an interrupted download
    # cannot leave a truncated file that looks cached forever after.
    partial.rename(destination)
    return destination


def vad_model() -> Path:
    return _download(VAD_URL, model_dir() / "silero_vad.onnx")


def embedding_model() -> Path:
    return _download(EMBEDDING_URL, model_dir() / "wespeaker_en_voxceleb_CAMPP.onnx")


def segmentation_model() -> Path:
    target = model_dir() / "pyannote_segmentation_3_0.onnx"
    if target.exists():
        return target

    archive = _download(SEGMENTATION_URL, model_dir() / "segmentation.tar.bz2")
    with tarfile.open(archive, "r:bz2") as tar:
        member = tar.extractfile(SEGMENTATION_MEMBER)
        if member is None:
            raise RuntimeError(f"{SEGMENTATION_MEMBER} missing from {archive.name}")
        target.write_bytes(member.read())
    archive.unlink(missing_ok=True)
    return target


def weights_digest(path: Path | str) -> str:
    """First 12 hex characters of the weights file's SHA-256.

    Fills ``model_version``, for the same reason as on the face side: a
    package version can move without the weights changing and vice versa,
    and this is the thing that actually answers whether two embeddings are
    comparable.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def ensure_all() -> dict[str, Path]:
    """Fetch everything the voice pipeline needs."""
    return {
        "vad": vad_model(),
        "embedding": embedding_model(),
        "segmentation": segmentation_model(),
    }

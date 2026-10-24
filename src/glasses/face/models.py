"""Loading InsightFace ONNX nets, and identifying which weights you got.

``FaceAnalysis`` is not used. It asserts that a detector is present, so it
cannot load the recognition net alone, and it would pull in the landmark and
gender/age nets this pipeline never calls. Going to ``model_zoo`` directly
loads exactly one net per object.

``weights_digest`` is what fills ``model_version``. The insightface package
version would be the easy answer and is the wrong one: the weights can change
without it moving, and it can move without the weights changing. A digest of
the file is the thing that actually answers "are these two embeddings
comparable". It costs ~70 ms once per process for a 174 MB model.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

#: Filenames inside an InsightFace model pack.
DETECTOR_FILE = "det_10g.onnx"
RECOGNITION_FILE = "w600k_r50.onnx"


def default_providers() -> list[str]:
    """ONNX Runtime execution providers, best first.

    Worth doing automatically because the gain is large and free: on Apple
    Silicon, CoreML runs the same ArcFace ResNet-50 weights about 18x faster
    than the CPU provider (79 ms -> 4.4 ms for two faces). It is not a
    different model, so there is no accuracy trade — measured cosine between
    CPU and CoreML embeddings of the same crop is 0.9998, which is three
    orders of magnitude below the gap the accept threshold cares about.

    The cost is a one-off ~2.4 s on first inference while CoreML compiles the
    graph, which lands in process startup rather than in the hot loop.

    Override with ``GLASSES_PROVIDERS=CPUExecutionProvider`` to pin it, which
    is what to do if you ever need two machines to agree bit for bit.
    """
    override = os.environ.get("GLASSES_PROVIDERS")
    if override:
        return [name.strip() for name in override.split(",") if name.strip()]

    import onnxruntime

    available = onnxruntime.get_available_providers()
    preferred = [
        name for name in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if name in available
    ]
    return [*preferred, "CPUExecutionProvider"]


def pack_dir(model_pack: str = "buffalo_l") -> Path:
    """Path to a model pack, downloading it on first use."""
    from insightface.utils import storage

    return Path(storage.ensure_available("models", model_pack))


@lru_cache(maxsize=8)
def weights_digest(path: str) -> str:
    """First 12 hex characters of the weights file's SHA-256."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def load_net(path: Path, *, providers: list[str] | None = None, **prepare):  # type: ignore[no-untyped-def]
    from insightface import model_zoo

    net = model_zoo.get_model(str(path), providers=providers or default_providers())
    if net is None:
        raise RuntimeError(f"insightface could not load a model from {path}")
    net.prepare(ctx_id=0, **prepare)
    return net

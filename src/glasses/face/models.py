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
from functools import lru_cache
from pathlib import Path

#: Filenames inside an InsightFace model pack.
DETECTOR_FILE = "det_10g.onnx"
RECOGNITION_FILE = "w600k_r50.onnx"


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

    net = model_zoo.get_model(str(path), providers=providers or ["CPUExecutionProvider"])
    if net is None:
        raise RuntimeError(f"insightface could not load a model from {path}")
    net.prepare(ctx_id=0, **prepare)
    return net

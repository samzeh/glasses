"""Stage 2: similarity transform from the 5 landmarks onto the canonical
ArcFace crop.

Non-negotiable per the PRD: embeddings degrade sharply on unaligned crops. The
transform estimation is pure numpy so it is testable without OpenCV; the warp
uses OpenCV when it is installed and falls back to a numpy bilinear sampler
otherwise.
"""

from __future__ import annotations

import numpy as np

from .config import ARCFACE_REFERENCE_112
from .types import AlignedFace, Detection, Landmarks5


def reference_landmarks(size: int = 112) -> np.ndarray:
    """Canonical 5-point target for a ``size`` x ``size`` crop, (5, 2) float64."""
    ref = np.array(ARCFACE_REFERENCE_112, dtype=np.float64)
    if size != 112:
        ref = ref * (size / 112.0)
    return ref


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale,
    translation) mapping ``src`` onto ``dst``. Umeyama 1991.

    Reflections are excluded: a mirrored "alignment" would fit the landmarks
    well and produce a face the embedder has never seen.

    Returns a (2, 3) affine matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"expected matching (N, 2) arrays, got {src.shape} and {dst.shape}")

    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = (dst_demean.T @ src_demean) / n
    u, s, vt = np.linalg.svd(cov)

    correction = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1] = -1.0

    rotation = u @ np.diag(correction) @ vt
    src_var = src_demean.var(axis=0).sum()
    scale = 1.0 if src_var == 0 else float((s * correction).sum() / src_var)

    transform = np.zeros((2, 3), dtype=np.float64)
    transform[:, :2] = scale * rotation
    transform[:, 2] = dst_mean - scale * rotation @ src_mean
    return transform


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a (2, 3) affine matrix."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts @ transform[:, :2].T + transform[:, 2]


def align(
    frame: np.ndarray,
    landmarks: Landmarks5,
    *,
    size: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``frame`` so the landmarks land on the canonical reference.

    Returns ``(crop, transform)``.
    """
    transform = estimate_similarity_transform(landmarks.as_array(), reference_landmarks(size))
    return warp_affine(frame, transform, size), transform


def align_detection(frame: np.ndarray, detection: Detection, *, size: int = 112) -> AlignedFace:
    crop, transform = align(frame, detection.landmarks, size=size)
    return AlignedFace(image=crop, detection=detection, transform=transform)


def warp_affine(frame: np.ndarray, transform: np.ndarray, size: int) -> np.ndarray:
    """Warp into a ``size`` x ``size`` crop. Uses OpenCV when available."""
    try:
        import cv2
    except ImportError:
        return _warp_affine_numpy(frame, transform, size)
    return cv2.warpAffine(
        frame,
        transform.astype(np.float32),
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )


def _warp_affine_numpy(frame: np.ndarray, transform: np.ndarray, size: int) -> np.ndarray:
    """Bilinear inverse warp. Correctness-first fallback, not a fast path.

    Samples the source at the preimage of each output pixel; out-of-bounds
    samples come back as zeros, matching OpenCV's ``borderValue=0``.
    """
    frame = np.asarray(frame)
    squeeze = frame.ndim == 2
    if squeeze:
        frame = frame[:, :, None]
    h, w, channels = frame.shape

    full = np.vstack([transform, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(full)[:2]

    ys, xs = np.mgrid[0:size, 0:size]
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    src = grid @ inverse[:, :2].T + inverse[:, 2]
    sx, sy = src[:, 0], src[:, 1]

    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = sx - x0, sy - y0

    inside = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)

    src_f = frame.astype(np.float64)
    top = src_f[y0c, x0c] * (1 - wx)[:, None] + src_f[y0c, x1c] * wx[:, None]
    bottom = src_f[y1c, x0c] * (1 - wx)[:, None] + src_f[y1c, x1c] * wx[:, None]
    values = top * (1 - wy)[:, None] + bottom * wy[:, None]
    values[~inside] = 0.0

    out = values.reshape(size, size, channels)
    if squeeze:
        out = out[:, :, 0]
    return out.round().clip(0, 255).astype(frame.dtype)

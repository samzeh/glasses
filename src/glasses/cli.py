"""Command line entry point: ``glasses <command>``.

This is the M0 harness — enough to enroll yourself from the webcam or a few
photos, then see who the pipeline thinks it is looking at. It is not the product:
there is no ESP, no encounter, no audio. It exists so the thresholds can be
calibrated against real frames before any of that is built.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .db_cli import register as register_db
from .face import FacePipeline, PipelineConfig
from .face.types import FrameResult, MatchStatus
from .meet_cli import register as register_meet
from .store import LocalStore
from .voice_cli import register as register_voice

DEFAULT_STORE = Path("glasses-store")


# --- shared helpers --------------------------------------------------------


def _require_cv2():  # type: ignore[no-untyped-def]
    try:
        import cv2
    except ImportError:
        sys.exit(
            "This command needs the image stack. Install it with:\n    uv pip install -e '.[face]'"
        )
    return cv2


def build_pipeline(store: LocalStore, config: PipelineConfig | None = None) -> FacePipeline:
    from .face import ArcFaceEmbedder, ScrfdDetector

    return FacePipeline(
        detector=ScrfdDetector(),
        embedder=ArcFaceEmbedder(),
        gallery=store.gallery(),
        config=config or PipelineConfig(),
    )


def expand_images(paths: list[str]) -> list[Path]:
    """Accept files and directories; directories expand to the images in them."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes))
        else:
            out.append(path)
    return out


def read_frames(paths: list[Path]) -> Iterator[tuple[Path, np.ndarray]]:
    cv2 = _require_cv2()
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  ! could not read {path}", file=sys.stderr)
            continue
        yield path, frame


def describe_face(result: FrameResult, store: LocalStore) -> str:
    face = result.primary
    if face is None:
        return "no face"

    quality = face.quality
    detail = (
        f"iod={quality.interocular_px:.0f}px blur={quality.blur_variance:.0f} "
        f"yaw={quality.pose.yaw:+.0f} pitch={quality.pose.pitch:+.0f}"
    )
    if not quality.passed:
        return f"rejected ({', '.join(quality.failures)})  {detail}"

    match = face.match
    if match is None or match.status is MatchStatus.UNKNOWN:
        best = f" best={match.similarity:.3f}" if match else ""
        return f"unknown{best}  {detail}"

    name = store.name_of(match.person_id)
    prefix = "" if match.status is MatchStatus.MATCH else "possibly "
    return f"{prefix}{name} ({match.similarity:.3f})  {detail}"


# --- commands --------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Report whether this machine can actually run the pipeline."""
    print("dependencies")
    for module in ("numpy", "cv2", "onnxruntime", "insightface"):
        try:
            loaded = __import__(module)
            print(f"  {module:14s} {getattr(loaded, '__version__', 'ok')}")
        except ImportError:
            print(f"  {module:14s} MISSING  -> uv pip install -e '.[face]'")

    try:
        from .face.models import DETECTOR_FILE, RECOGNITION_FILE, pack_dir, weights_digest

        directory = pack_dir("buffalo_l")
        print(f"\nmodel pack    {directory}")
        for name in (DETECTOR_FILE, RECOGNITION_FILE):
            path = directory / name
            mark = weights_digest(str(path)) if path.exists() else "MISSING"
            print(f"  {name:18s} {mark}")
    except Exception as error:  # noqa: BLE001 - this command reports, never raises
        print(f"\nmodel pack    unavailable: {error}")

    store = LocalStore.load(args.store)
    print(f"\nstore         {args.store}")
    print(f"  people      {len(store)}")
    print(f"  templates   {len(store.templates)}")
    models = store.models()
    if len(models) > 1:
        print(f"  ! {len(models)} model versions present: {sorted(models)}")
        print("    A probe only ever sees one of them. Re-embed the gallery.")
    return 0


def too_similar(candidate, accepted: list, limit: float) -> bool:
    """Whether a candidate template duplicates one already accepted.

    Consecutive webcam frames are nearly identical, so capturing "ten frames"
    naively yields one view ten times. Ten copies of one angle is strictly
    worse than three genuinely different ones: the top-3 matcher averages
    near-duplicates into the same number it would have got from a single
    template, and the gallery grows for nothing.
    """
    return any(float(candidate.vector @ existing.vector) > limit for existing in accepted)


def capture_enrollment(args: argparse.Namespace, pipeline) -> tuple[list, list, list]:  # type: ignore[no-untyped-def]
    """Collect varied templates from a live camera, with feedback."""
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(
            f"Could not open camera {args.camera}.\n"
            "On macOS the terminal needs camera access: System Settings > "
            "Privacy & Security > Camera.",
            file=sys.stderr,
        )
        return [], [], []

    print(
        f"Capturing {args.frames} views of {args.name}.\n"
        "Look at the camera, then slowly turn your head left, right, up and "
        "down. Near-identical frames are skipped on purpose.\n"
        "q to stop early.\n"
    )

    embeddings, scores, sources = [], [], []
    deadline = time.time() + args.seconds
    try:
        while len(embeddings) < args.frames and time.time() < deadline:
            ok, frame = capture.read()
            if not ok:
                break

            face = pipeline.process_frame(frame).primary
            if face is None:
                status, colour = "no face", (90, 90, 90)
            elif face.embedding is None:
                status, colour = ", ".join(face.quality.failures), (0, 170, 220)
            elif too_similar(face.embedding, embeddings, args.max_similarity):
                status, colour = "too similar — move your head", (0, 170, 220)
            else:
                embeddings.append(face.embedding)
                scores.append(face.quality.score)
                sources.append(f"camera:{len(embeddings)}")
                status, colour = f"captured {len(embeddings)}/{args.frames}", (0, 200, 0)
                print(
                    f"  captured {len(embeddings)}/{args.frames}  quality={face.quality.score:.2f}"
                )

            if not args.headless:
                if face is not None:
                    box = face.detection.bbox
                    cv2.rectangle(
                        frame,
                        (int(box.x1), int(box.y1)),
                        (int(box.x2), int(box.y2)),
                        colour,
                        2,
                    )
                cv2.putText(
                    frame,
                    f"{len(embeddings)}/{args.frames}  {status}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colour,
                    2,
                )
                cv2.imshow("enroll", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if len(embeddings) < args.frames:
        print(f"\nstopped with {len(embeddings)} of {args.frames}")
    return embeddings, scores, sources


def cmd_enroll(args: argparse.Namespace) -> int:
    if bool(args.camera is not None) == bool(args.images):
        print(
            "Give either --camera (enroll from the webcam) or image paths, "
            "not both and not neither.",
            file=sys.stderr,
        )
        return 1

    store = LocalStore.load(args.store)
    pipeline = build_pipeline(store)

    existing = store.find_by_name(args.name)
    person_id = existing.id if existing else None
    if existing:
        print(
            f"adding to existing person {args.name} ({store.template_count(existing.id)} templates)"
        )

    rejected = 0
    if args.camera is not None:
        embeddings, scores, sources = capture_enrollment(args, pipeline)
    else:
        embeddings, scores, sources = [], [], []
        for path, frame in read_frames(expand_images(args.images)):
            result = pipeline.process_frame(frame)
            face = result.primary
            if face is None or face.embedding is None:
                reason = "no face" if face is None else ", ".join(face.quality.failures)
                print(f"  skip {path.name}: {reason}")
                rejected += 1
                continue
            if too_similar(face.embedding, embeddings, args.max_similarity):
                print(f"  skip {path.name}: too similar to a photo already taken")
                rejected += 1
                continue
            embeddings.append(face.embedding)
            scores.append(face.quality.score)
            sources.append(path.name)
            print(f"  ok   {path.name}  quality={face.quality.score:.2f}")

    if not embeddings:
        print(
            "\nNothing passed the quality gate, so nothing was enrolled.\n"
            "Use clear, well-lit, roughly frontal views, or loosen the gate "
            "while you calibrate it.",
            file=sys.stderr,
        )
        return 1

    person_id = store.enroll(
        args.name,
        embeddings,
        consent_note=args.consent,
        quality_scores=scores,
        source=",".join(sources),
        person_id=person_id,
    )
    store.save()
    print(
        f"\nenrolled {args.name} as {person_id[:8]} "
        f"with {len(embeddings)} template(s), {rejected} rejected"
    )
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    store = LocalStore.load(args.store)
    if not store.templates:
        print("The gallery is empty — enroll someone first.", file=sys.stderr)
        return 1

    pipeline = build_pipeline(store)
    for path, frame in read_frames(expand_images(args.images)):
        result = pipeline.process_frame(frame)
        print(f"{path.name}")
        if not result.faces:
            print("  no face")
            continue
        for index, face in enumerate(result.faces):
            tag = "primary" if face.is_primary else f"face {index}"
            single = FrameResult(faces=(face,), timings=result.timings)
            marker = "  " if face.is_primary else "  "
            print(f"{marker}{tag:8s} {describe_face(single, store)}")
        print(f"  {result.timings.total_ms:.0f} ms")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Live from a webcam. The closest thing to wearing the device."""
    cv2 = _require_cv2()
    store = LocalStore.load(args.store)
    pipeline = build_pipeline(store)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(
            f"Could not open camera {args.camera}.\n"
            "On macOS the terminal needs camera access: System Settings > "
            "Privacy & Security > Camera.",
            file=sys.stderr,
        )
        return 1

    print("watching — q or ctrl-c to stop")
    latencies: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = pipeline.process_frame(frame)
            latencies.append(result.timings.total_ms)

            for face in result.faces:
                box = face.detection.bbox
                passed = face.quality.passed
                matched = face.match is not None and face.match.is_identified
                colour = (0, 200, 0) if matched else ((0, 200, 200) if passed else (90, 90, 90))
                cv2.rectangle(
                    frame,
                    (int(box.x1), int(box.y1)),
                    (int(box.x2), int(box.y2)),
                    colour,
                    2,
                )
                single = FrameResult(faces=(face,), timings=result.timings)
                cv2.putText(
                    frame,
                    describe_face(single, store).split("  ")[0],
                    (int(box.x1), max(18, int(box.y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2,
                )

            cv2.putText(
                frame,
                f"{result.timings.total_ms:.0f} ms",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if not args.headless:
                cv2.imshow("glasses", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif len(latencies) % 30 == 0:
                print(f"  {describe_face(result, store)}")
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if latencies:
        array = np.array(latencies)
        print(
            f"\n{len(array)} frames  "
            f"p50 {np.percentile(array, 50):.0f} ms  "
            f"p95 {np.percentile(array, 95):.0f} ms"
        )
    return 0


def cmd_people(args: argparse.Namespace) -> int:
    store = LocalStore.load(args.store)
    if not store.people:
        print("nobody enrolled")
        return 0
    for person in store.people.values():
        print(
            f"{person.id[:8]}  {person.display_name:24s} "
            f"{store.template_count(person.id):3d} templates  "
            f"consented {person.consented_at[:10]}  {person.consent_note}"
        )
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    store = LocalStore.load(args.store)
    person = store.find_by_name(args.who) or store.people.get(args.who)
    if person is None:
        print(f"no such person: {args.who}", file=sys.stderr)
        return 1
    removed = store.forget(person.id)
    store.save()
    print(f"removed {person.display_name} and {removed} template(s)")
    return 0


def cmd_calibrate_pose(args: argparse.Namespace) -> int:
    """Measure where this detector actually puts the nose.

    ``nose_frontal_fraction`` is the zero point of the pitch estimate, and it
    belongs to the detector rather than to the aligner. Point this at a few
    dozen roughly frontal frames from the real camera and paste the result
    into ``QualityConfig``.
    """
    from .face import ScrfdDetector

    detector = ScrfdDetector()
    fractions: list[float] = []
    for _, frame in read_frames(expand_images(args.images)):
        for detection in detector.detect(frame):
            points = detection.landmarks.as_array()
            eye_mid = (points[0] + points[1]) / 2.0
            mouth_mid = (points[3] + points[4]) / 2.0
            span = mouth_mid[1] - eye_mid[1]
            if abs(span) > 1e-6:
                fractions.append(float((points[2, 1] - eye_mid[1]) / span))

    if not fractions:
        print("no faces found", file=sys.stderr)
        return 1

    array = np.array(fractions)
    print(f"faces            {array.size}")
    print(f"median           {np.median(array):.3f}   <- nose_frontal_fraction")
    print(f"mean +/- sd      {array.mean():.3f} +/- {array.std():.3f}")
    print(f"10th-90th pct    {np.percentile(array, 10):.3f} - {np.percentile(array, 90):.3f}")
    spread = np.degrees(np.arcsin(min(1.0, array.std() / 0.5)))
    print(f"\nFace-shape variation alone is worth about +/-{spread:.0f} deg of apparent pitch.")
    if array.size < 50:
        print(f"Only {array.size} faces — use at least 50 before trusting this.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Per-stage timings, against the PRD's 2-second hot-loop budget."""
    store = LocalStore.load(args.store)
    pipeline = build_pipeline(store)

    frames = list(read_frames(expand_images(args.images)))
    if not frames:
        print("no readable images", file=sys.stderr)
        return 1

    start = time.perf_counter()
    pipeline.process_frame(frames[0][1])
    print(f"cold start (model load included)  {(time.perf_counter() - start) * 1000:.0f} ms\n")

    runs = []
    for _ in range(args.repeat):
        for _, frame in frames:
            runs.append(pipeline.process_frame(frame).timings)

    def report(label: str, values: list[float], budget: float | None = None) -> None:
        array = np.array(values)
        note = ""
        if budget is not None:
            note = (
                f"   budget {budget:.0f}   {'OK' if np.percentile(array, 50) <= budget else 'OVER'}"
            )
        p50, p95 = np.percentile(array, 50), np.percentile(array, 95)
        print(f"  {label:10s} p50 {p50:6.1f}  p95 {p95:6.1f}{note}")

    print(f"warm, {len(runs)} frames (ms)")
    report("detect", [t.detect_ms for t in runs])
    report("align", [t.align_ms for t in runs])
    report("gate", [t.quality_ms for t in runs])
    report("embed", [t.embed_ms for t in runs])
    report("match", [t.match_ms for t in runs], 50.0)
    report("TOTAL", [t.total_ms for t in runs], 400.0)
    print("\nBudget is the PRD's server-side share: 400 ms for detect through")
    print("embed, 50 ms for the gallery match. Upload and render are not here.")
    return 0


# --- wiring ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glasses", description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE,
        help=f"gallery directory (default: {DEFAULT_STORE})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="verify models and dependencies")
    check.set_defaults(func=cmd_check)

    enroll = subparsers.add_parser("enroll", help="add a person, from the camera or photos")
    enroll.add_argument("--name", required=True)
    enroll.add_argument(
        "--consent",
        required=True,
        help="who agreed to this and when — recorded with the person, and required",
    )
    enroll.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="capture from this camera instead of reading photos (default device 0)",
    )
    enroll.add_argument("--frames", type=int, default=6, help="views to capture (default 6)")
    enroll.add_argument("--seconds", type=int, default=60, help="give up after this long")
    enroll.add_argument(
        "--max-similarity",
        type=float,
        default=0.97,
        help="reject a view this close to one already taken (default 0.97)",
    )
    enroll.add_argument("--headless", action="store_true", help="no preview window")
    enroll.add_argument("images", nargs="*")
    enroll.set_defaults(func=cmd_enroll)

    identify = subparsers.add_parser("identify", help="run the pipeline over images")
    identify.add_argument("images", nargs="+")
    identify.set_defaults(func=cmd_identify)

    watch = subparsers.add_parser("watch", help="live from a webcam")
    watch.add_argument("--camera", type=int, default=0)
    watch.add_argument("--headless", action="store_true", help="no window, print instead")
    watch.set_defaults(func=cmd_watch)

    people = subparsers.add_parser("people", help="list who is enrolled")
    people.set_defaults(func=cmd_people)

    forget = subparsers.add_parser("forget", help="delete a person and all their data")
    forget.add_argument("who", help="name or id")
    forget.set_defaults(func=cmd_forget)

    calibrate = subparsers.add_parser(
        "calibrate-pose", help="measure nose_frontal_fraction for your detector"
    )
    calibrate.add_argument("images", nargs="+")
    calibrate.set_defaults(func=cmd_calibrate_pose)

    bench = subparsers.add_parser("bench", help="per-stage latency")
    bench.add_argument("images", nargs="+")
    bench.add_argument("--repeat", type=int, default=5)
    bench.set_defaults(func=cmd_bench)

    register_voice(subparsers)
    register_meet(subparsers)
    register_db(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

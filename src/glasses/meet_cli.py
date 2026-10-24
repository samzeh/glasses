"""``glasses meet`` — run the capture loop against a camera and a mic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .binding.asd import AsdConfig
from .encounter.clock import Clock, seconds_to_us, us_to_seconds
from .face import FacePipeline
from .face.models import default_providers
from .meet import AudioRecorder, Capture, FaceTracker, MouthReader, assign_speakers, save_crop
from .storage import open_store
from .voice.audio import write_wav

VOICE = "voice"
FACE = "face"


def _require_cv2():  # type: ignore[no-untyped-def]
    try:
        import cv2
    except ImportError:
        sys.exit("This needs the image stack:\n    uv pip install -e '.[face,voice]'")
    return cv2


def cmd_meet(args: argparse.Namespace) -> int:
    cv2 = _require_cv2()
    from .face import ArcFaceEmbedder, ScrfdDetector
    from .voice import VadSegmenter, VoicePipeline, WeSpeakerEmbedder

    store = open_store(args.store)
    clock = Clock.start()
    providers = default_providers()

    print("loading models …")
    face_pipeline = FacePipeline(
        detector=ScrfdDetector(providers=providers),
        embedder=ArcFaceEmbedder(providers=providers),
        gallery=store.gallery(FACE),
    )
    voice_pipeline = VoicePipeline(
        segmenter=VadSegmenter(),
        embedder=WeSpeakerEmbedder(),
        gallery=store.gallery(VOICE),
        wearer_embeddings=store.wearer_embeddings(),
    )
    mouths = MouthReader(providers=providers)
    tracker = FaceTracker()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nrecording {args.seconds}s — talk, and let the other person talk. q to stop.\n")

    detections: list = []
    frames_seen = 0
    try:
        with AudioRecorder(clock, device=args.device) as recorder:
            started = clock.now_us()
            deadline = started + seconds_to_us(args.seconds)

            while clock.now_us() < deadline:
                ok, frame = camera.read()
                if not ok:
                    break
                at_us = clock.now_us()
                frames_seen += 1

                # Detection is 20x the cost of landmarks, so it runs
                # occasionally and tracking carries the boxes between.
                if frames_seen % args.detect_every == 1 or not detections:
                    detections = face_pipeline.detector.detect(frame)
                tracker.update(detections, frame, at_us)

                for track in tracker.active(at_us):
                    if track.current_detection is None:
                        continue
                    sample = mouths.sample(frame, track.current_detection, at_us)
                    if sample is not None:
                        track.mouth.append(sample)

                if not args.headless:
                    _draw(
                        cv2,
                        frame,
                        tracker,
                        at_us,
                        recorder.level_dbfs(),
                        us_to_seconds(deadline - at_us),
                    )
                    cv2.imshow("meet", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            audio = recorder.samples
            audio_start_us = recorder.start_us or started
    finally:
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = us_to_seconds(clock.now_us() - started)
    fps = frames_seen / elapsed if elapsed else 0.0
    print(
        f"captured {frames_seen} frames in {elapsed:.0f}s ({fps:.0f} fps), "
        f"{audio.size / 16000:.0f}s of audio"
    )
    if fps < 12:
        print("  ! frame rate too low for lip sync — expect unassigned utterances")

    if audio.size == 0:
        print("no audio captured", file=sys.stderr)
        return 1

    return _resolve(args, store, voice_pipeline, face_pipeline, tracker, audio, audio_start_us)


def _draw(cv2, frame, tracker, at_us, level_dbfs, remaining_s) -> None:  # type: ignore[no-untyped-def]
    for track in tracker.active(at_us):
        box = track.bbox
        cv2.rectangle(
            frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 200, 200), 2
        )
        cv2.putText(
            frame,
            f"#{track.id}",
            (int(box.x1), max(16, int(box.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 200),
            2,
        )
    cv2.putText(
        frame,
        f"{remaining_s:4.1f}s  mic {level_dbfs:5.0f} dBFS",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def _resolve(args, store, voice_pipeline, face_pipeline, tracker, audio, audio_start_us) -> int:  # type: ignore[no-untyped-def]
    print("\nfinding speech …")
    result = voice_pipeline.process(audio)

    captures: list[Capture] = []
    for item in result.segments:
        if item.embedding is None:
            continue
        start_us = audio_start_us + seconds_to_us(item.segment.start_s)
        end_us = audio_start_us + seconds_to_us(item.segment.end_s)
        capture = Capture(
            start_us=start_us,
            end_us=end_us,
            audio=item.segment.samples,
            asd=None,  # type: ignore[arg-type]
            track=None,
            is_wearer=item.is_wearer,
            voice_person_id=item.match.person_id if item.match else None,
            voice_name=store.name_of(item.match.person_id)
            if item.match and item.match.person_id
            else None,
            voice_similarity=item.match.similarity if item.match else 0.0,
        )
        captures.append(capture)

    if not captures:
        print("no usable speech — segments need 3+ seconds of continuous talking")
        return 1

    partners = [c for c in captures if not c.is_wearer]
    assign_speakers(
        partners,
        tracker,
        audio,
        audio_start_us,
        AsdConfig(min_correlation=args.min_correlation, min_margin=args.min_margin),
    )

    out_dir = Path(args.out)
    print(f"\n{len(captures)} utterance(s)\n")
    assigned = 0
    for index, capture in enumerate(captures, start=1):
        at = us_to_seconds(capture.start_us - audio_start_us)
        if capture.is_wearer:
            print(f"[{at:5.1f}s] WEARER ({capture.duration_s():.0f}s) — not captured")
            continue

        who = capture.voice_name or "unknown voice"
        line = f"[{at:5.1f}s] {who}"
        if capture.voice_similarity:
            line += f" (voice {capture.voice_similarity:.2f})"

        if capture.track is None:
            print(f"{line}  → UNASSIGNED: {capture.asd.reason}")
            for score in capture.asd.scores:
                print(f"               {score}")
            continue

        assigned += 1
        track = capture.track
        if track.best_frame is not None and track.best_detection is not None:
            capture.image_path = save_crop(
                track.best_frame, track.best_detection.bbox, out_dir / f"{index:03d}.jpg"
            )
        face_match = _identify_face(face_pipeline, store, track)
        print(f"{line}  → face #{track.id} {face_match}")
        print(f"               {capture.asd.reason}   {capture.image_path}")

        if args.save_audio:
            write_wav(out_dir / f"{index:03d}.wav", capture.audio)

    print(f"\n{assigned}/{len(partners)} partner utterance(s) matched to a face")
    print(f"crops in {out_dir}/ — open them and check it picked the right person")
    return 0


def _identify_face(face_pipeline, store, track) -> str:  # type: ignore[no-untyped-def]
    """Run the face pipeline on this track's best frame."""
    if track.best_frame is None:
        return "no frame"
    frame_result = face_pipeline.process_frame(track.best_frame)
    if not frame_result.faces:
        return "no face found in the saved frame"

    # The track's box is the one to report on; other faces in the frame
    # belong to other tracks.
    from .meet import iou

    best = max(frame_result.faces, key=lambda f: iou(f.detection.bbox, track.bbox))
    if not best.quality.passed:
        return f"face rejected ({', '.join(best.quality.failures)})"
    if best.match is None or best.match.person_id is None:
        return "face not recognised (new person)"
    return f"{store.name_of(best.match.person_id)} ({best.match.similarity:.2f})"


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    meet = subparsers.add_parser("meet", help="camera + mic: photograph whoever is speaking")
    meet.add_argument("--seconds", type=int, default=30)
    meet.add_argument("--camera", type=int, default=0)
    meet.add_argument("--device", type=int, default=None, help="audio input index")
    meet.add_argument("--out", default="captures", help="where to write face crops")
    meet.add_argument("--detect-every", type=int, default=5, help="frames between detections")
    meet.add_argument(
        "--min-correlation",
        type=float,
        default=0.30,
        help="lip-sync correlation the winner must reach (default 0.30). Lower "
        "it if real footage comes back unassigned with plausible-looking scores",
    )
    meet.add_argument(
        "--min-margin",
        type=float,
        default=0.15,
        help="how far the winner must beat the runner-up (default 0.15)",
    )
    meet.add_argument("--save-audio", action="store_true")
    meet.add_argument("--headless", action="store_true")
    meet.set_defaults(func=cmd_meet)

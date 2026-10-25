"""``glasses meet`` — run the capture loop against a camera and a mic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .attribution import (
    apply_remap,
    identities_by_speaker,
    labels_for,
    partner_remap,
    transcribable,
)
from .autoenroll import MIN_NAME_CONFIDENCE
from .binding.asd import AsdConfig
from .encounter.clock import Clock, seconds_to_us, us_to_seconds
from .face import FacePipeline
from .face.models import default_providers
from .meet import (
    AudioRecorder,
    Capture,
    FaceTracker,
    MouthReader,
    assign_speakers,
    people_in_frame,
    save_crop,
)
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
    # Split the voices so that the diarizer can assign them to the right speaker
    if args.diarize:
        from .voice import DiarizingSegmenter

        segmenter = DiarizingSegmenter(num_speakers=args.speakers)
    else:
        segmenter = VadSegmenter()
    voice_pipeline = VoicePipeline(
        segmenter=segmenter,
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
    # 640x480 was cheap enough to hold 30 fps, and it put faces under
    # the gate: at conversational distance a head is ~13% of the frame
    # width, so 640 gives ~80px of face and well under the 40px between
    # the eyes that the embedder needs. The face is then never embedded,
    # never matched, and the person is a stranger every time you meet
    # them. Detection cost rises with resolution; recognition failing
    # costs the whole product.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (args.width, args.height):
        print(f"  camera gave {actual_w}x{actual_h}, not {args.width}x{args.height}")

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

    if args.save_audio:
        # The whole take, not the per-capture clips below: segmentation and
        # lip-sync are what need debugging, and re-recording a conversation
        # to retry them loses the thing being debugged.
        full = Path(args.out) / "recording.wav"
        full.parent.mkdir(parents=True, exist_ok=True)
        write_wav(full, audio)
        print(f"  saved {full}")

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

    # Identity is resolved from the segments long enough to embed, then
    # carried to the short ones that share their speaker. Without this an
    # introduction — short by nature — never reaches a recogniser at all.
    identities = identities_by_speaker(result.segments, store)
    # The camera is the census. The wearer is behind it and never in
    # frame, so the faces it saw at once are the other people present —
    # a count the audio cannot produce, because a voice embedding that
    # cannot separate two people cannot count them.
    seen = people_in_frame(tracker)
    partners = args.partners if args.partners is not None else (seen or None)
    if partners:
        source = "you said" if args.partners is not None else "the camera saw"
        print(f"{source} {partners} other person/people")
    remap = partner_remap(result.segments, identities, partners=partners)
    identities = apply_remap(identities, remap)
    keep = transcribable(result.segments)
    labels = labels_for(keep, identities, remap)
    if remap:
        print(f"folded {len(remap)} surplus speaker cluster(s) into {args.partners}")

    captures: list[Capture] = []
    for item, (person_id, name, is_wearer) in zip(keep, labels, strict=True):
        start_us = audio_start_us + seconds_to_us(item.segment.start_s)
        end_us = audio_start_us + seconds_to_us(item.segment.end_s)
        captures.append(
            Capture(
                start_us=start_us,
                end_us=end_us,
                audio=item.segment.samples,
                asd=None,  # type: ignore[arg-type]
                track=None,
                speaker=item.segment.speaker,
                is_wearer=is_wearer,
                voice_person_id=person_id,
                voice_name=name or (store.name_of(person_id) if person_id else None),
                voice_similarity=item.match.similarity if item.match else 0.0,
            )
        )

    if not captures:
        print("no speech found in the recording")
        return 1

    short = sum(1 for c in captures if c.duration_s() < 3.0)
    print(f"{len(captures)} segment(s), {short} too short to identify on their own")

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

    if getattr(args, "remember", False):
        return _remember(args, store, face_pipeline, captures, audio_start_us)
    return 0


def _remember(args, store, face_pipeline, captures, audio_start_us) -> int:  # type: ignore[no-untyped-def]
    """Take bound captures through Gemini transcription and fact extraction.

    Identity here is the face's, not the voice's. That is the whole reason
    this path exists beside ``glasses remember`` — see ``recall``'s module
    docstring for the measurement behind it.
    """
    import uuid
    from datetime import UTC, datetime

    from .autoenroll import enroll_from_conversation, relabel_utterances
    from .facts import DEFAULT_MODEL, GeminiError, GeminiExtractor, has_api_key
    from .recall import face_templates_by_label, label_captures, utterances_from_captures
    from .voice.gemini import GeminiTranscriber

    if not has_api_key():
        print("\nGEMINI_API_KEY is required to remember this meeting.", file=sys.stderr)
        return 2

    encounter_id = args.encounter or f"enc-{uuid.uuid4().hex[:12]}"
    occurred_at = datetime.now(UTC).isoformat()
    wearer = store.wearer()
    wearer_name = wearer.display_name if wearer else "the wearer"

    print("\nidentifying faces …")
    identities = label_captures(captures, face_pipeline, store)
    for track_id, identity in sorted(identities.items()):
        who = identity.name or identity.note or "not recognised"
        extra = f" ({identity.similarity:.2f})" if identity.known else ""
        print(f"  face #{track_id}: {who}{extra}")

    transcriber = GeminiTranscriber(model=args.gemini_model or DEFAULT_MODEL)
    print(f"\ntranscribing with {transcriber.model_name} …")
    try:
        utterances = utterances_from_captures(
            captures, identities, transcriber, audio_start_us=audio_start_us
        )
    except GeminiError as error:
        print(f"  ! transcription failed: {error}", file=sys.stderr)
        return 1

    if not utterances:
        print("nothing intelligible to remember")
        return 1

    extractor = GeminiExtractor(model=args.gemini_model or DEFAULT_MODEL)
    try:
        facts = extractor.extract(
            utterances,
            encounter_id=encounter_id,
            wearer_name=wearer_name,
            occurred_at=occurred_at,
        )
    except GeminiError as error:
        print(f"\n  ! extraction failed: {error}", file=sys.stderr)
        return 1

    if not args.no_enroll and not args.dry_run:
        enrolled, facts = enroll_from_conversation(
            store,
            facts,
            [],
            [],
            min_confidence=args.name_confidence,
            source=f"meet ({encounter_id})",
            # Faces, not voiceprints: the camera is why this path exists.
            embeddings=face_templates_by_label(captures, identities),
            kind="face",
            # Everyone who gets here is a face the gallery already failed
            # to match, so they are a new person whatever they are called.
            allow_duplicate_names=True,
        )
        if enrolled:
            utterances = relabel_utterances(utterances, enrolled)
            print(f"\n  enrolled {len(enrolled)} new person/people from this conversation:")
            for item in enrolled:
                print(f"    {item}")

    print(f"\n  {len(facts)} fact(s):")
    for fact in facts:
        who = fact.person_name or "unattributed"
        print(f"    {who:<16} {fact.kind:<12} {fact.value}  ({fact.confidence:.2f})")
        if fact.evidence_span:
            print(f'      "{fact.evidence_span}"')

    if args.dry_run:
        print("\n--dry-run: no memories saved")
        return 0

    from .profiles import profile_path, save_encounter

    save_encounter(
        profile_path(store),
        utterances,
        facts,
        encounter_id=encounter_id,
        occurred_at=occurred_at,
        location=args.location,
    )
    people_saved = len({u.person_id for u in utterances if u.person_id and not u.is_wearer})
    print(f"\nsaved structured memories for {people_saved} known person/people")
    print('Now ask: glasses ask "what is this person building?"')
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
        return (
            f"face rejected ({', '.join(best.quality.failures)}) "
            f"iod={best.quality.interocular_px:.0f}px"
        )
    if best.match is None or best.match.person_id is None:
        return "face not recognised (new person)"
    return f"{store.name_of(best.match.person_id)} ({best.match.similarity:.2f})"


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    meet = subparsers.add_parser("meet", help="camera + mic: photograph whoever is speaking")
    meet.add_argument("--seconds", type=int, default=30)
    meet.add_argument("--camera", type=int, default=0)
    meet.add_argument(
        "--width", type=int, default=1280, help="capture width (faces need the pixels)"
    )
    meet.add_argument("--height", type=int, default=720, help="capture height")
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
    meet.add_argument(
        "--remember",
        action="store_true",
        help="transcribe with Gemini and save selected person facts",
    )
    meet.add_argument("--encounter", default=None, help="encounter id (default: a new one)")
    meet.add_argument("--location", default=None, help="where this happened")
    meet.add_argument("--gemini-model", default=None, help="Gemini model for memory processing")
    meet.add_argument(
        "--no-enroll", action="store_true", help="do not create people from introductions"
    )
    meet.add_argument(
        "--name-confidence",
        type=float,
        default=MIN_NAME_CONFIDENCE,
        help=f"how sure a heard name must be to enrol someone (default {MIN_NAME_CONFIDENCE})",
    )
    meet.add_argument("--dry-run", action="store_true", help="extract but save no memories")
    meet.add_argument(
        "--diarize",
        action="store_true",
        help="split the audio by speaker first — needed whenever two people "
        "talk, or a back-and-forth arrives as one segment holding both",
    )
    meet.add_argument(
        "--speakers",
        type=int,
        default=-1,
        help="pin the speaker count. Measured to make clustering WORSE on "
        "this model — leave it unless a recording proves otherwise",
    )
    meet.add_argument(
        "--partners",
        type=int,
        default=None,
        help="override the camera's count of how many other people were "
        "there. Normally left alone: the faces in frame are the census",
    )
    meet.add_argument("--save-audio", action="store_true", help="keep the recorded wav")
    meet.add_argument("--headless", action="store_true")
    meet.set_defaults(func=cmd_meet)

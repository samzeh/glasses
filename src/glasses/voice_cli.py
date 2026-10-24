"""Voice subcommands, wired into the ``glasses`` CLI.

Kept beside ``cli.py`` rather than inside it because the two halves share
only the store and the argument parser.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

from .identity.vectors import cosine_similarity
from .store import LocalStore
from .voice import (
    CartesiaError,
    DiarizingSegmenter,
    VadSegmenter,
    VoicePipeline,
    VoicePipelineConfig,
    WeSpeakerEmbedder,
)
from .voice.audio import load_audio, write_wav
from .voice.cartesia import API_KEY_ENV, DEFAULT_MODEL, has_api_key
from .voice.models import ensure_all, weights_digest

AUDIO_SUFFIXES = {".wav", ".wave", ".mp3", ".m4a", ".flac", ".aiff", ".aif", ".ogg", ".opus"}
VOICE = "voice"


def expand_audio(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES))
        else:
            out.append(path)
    return out


def build_voice_pipeline(
    store: LocalStore,
    *,
    diarize: bool = False,
    num_speakers: int | None = -1,
) -> VoicePipeline:
    # -1 means "work it out", and None arrives from an unset CLI flag.
    # sherpa rejects None outright, so the coercion happens here rather
    # than at each call site.
    count = -1 if num_speakers is None else num_speakers
    segmenter = DiarizingSegmenter(num_speakers=count) if diarize else VadSegmenter()
    return VoicePipeline(
        segmenter=segmenter,
        embedder=WeSpeakerEmbedder(),
        gallery=store.gallery(VOICE),
        wearer_embeddings=store.wearer_embeddings(),
        config=VoicePipelineConfig(),
    )


def _describe(result, store: LocalStore) -> str:  # type: ignore[no-untyped-def]
    quality = result.quality
    detail = (
        f"{quality.duration_s:5.1f}s snr={quality.snr_db:3.0f}dB rms={quality.rms_dbfs:4.0f}dBFS"
    )
    if not quality.passed:
        return f"rejected ({', '.join(quality.failures)})  {detail}"
    if result.is_wearer:
        return f"WEARER  {detail}"
    match = result.match
    if match is None or match.person_id is None:
        best = f" best={match.similarity:.3f}" if match else ""
        return f"unknown{best}  {detail}"
    prefix = "" if match.is_identified else "possibly "
    return f"{prefix}{store.name_of(match.person_id)} ({match.similarity:.3f})  {detail}"


# --- commands --------------------------------------------------------------


def cmd_voice_check(args: argparse.Namespace) -> int:
    print("voice models")
    try:
        for name, path in ensure_all().items():
            size = path.stat().st_size / 1e6
            print(f"  {name:14s} {size:6.1f} MB  {weights_digest(path)}  {path.name}")
    except Exception as error:  # noqa: BLE001 - this command reports, never raises
        print(f"  unavailable: {error}")
        return 1

    # Reports whether the key is present, never what it is: a demo failing
    # because nobody exported it should take one command to diagnose.
    print("\ntranscription")
    if has_api_key():
        print(f"  cartesia/{DEFAULT_MODEL}   {API_KEY_ENV} is set")
    else:
        print(f"  whisper (local)      {API_KEY_ENV} is not set")
        print("    Cartesia is the better transcript and the default when a")
        print(f"    key is present. export {API_KEY_ENV}=… to use it.")

    store = LocalStore.load(args.store)
    wearer = store.wearer()
    print(f"\nstore         {args.store}")
    print(f"  voiceprints {store.template_count_by_kind(VOICE)}")
    print(f"  wearer      {wearer.display_name if wearer else 'NOT ENROLLED'}")
    if wearer is None:
        print("    Without a wearer voiceprint, your own speech is treated as a")
        print("    conversational partner's. Run: glasses enroll-voice --wearer …")
    return 0


def _prompt_for(name: str, seconds: int, *, wearer: bool) -> None:
    who = "you" if wearer else name
    print(
        f"Recording {seconds}s of {who}.\n"
        "Talk continuously and naturally — read something aloud if it is "
        "easier. Pauses are fine, but the gate needs 3+ second stretches, so "
        "do not stop and start.\n"
    )


def cmd_enroll_voice(args: argparse.Namespace) -> int:
    if bool(args.mic) == bool(args.audio):
        print(
            "Give either --mic (record now) or audio paths, not both and not neither.",
            file=sys.stderr,
        )
        return 1

    store = LocalStore.load(args.store)
    pipeline = build_voice_pipeline(store, diarize=args.diarize)

    existing = store.find_by_name(args.name)
    person_id = existing.id if existing else None

    if args.mic:
        from .voice.capture import record_with_meter

        _prompt_for(args.name, args.seconds, wearer=args.wearer)
        samples = record_with_meter(args.seconds, device=args.device)
        if args.save:
            write_wav(args.save, samples)
            print(f"  saved {args.save}")
        return _enroll_from(args, store, pipeline, [("microphone", samples)], person_id)

    recordings = []
    for path in expand_audio(args.audio):
        try:
            recordings.append((path.name, load_audio(path)))
        except Exception as error:  # noqa: BLE001
            print(f"  ! {path.name}: {error}", file=sys.stderr)
    return _enroll_from(args, store, pipeline, recordings, person_id)


def _enroll_from(args, store, pipeline, recordings, person_id):  # type: ignore[no-untyped-def]
    embeddings, scores, sources = [], [], []
    skipped_as_wearer = 0
    for label, samples in recordings:
        result = pipeline.process(samples)
        kept = 0
        for item in result.segments:
            if item.embedding is None:
                continue
            if args.exclude_wearer and not args.wearer and item.is_wearer:
                # Enrolling someone else, but this segment matched the
                # enrolled wearer. Counted and reported rather than dropped
                # quietly: when it happens to every segment, the cause is
                # almost always that the wearer enrollment itself is wrong.
                skipped_as_wearer += 1
                continue
            embeddings.append(item.embedding)
            scores.append(item.quality.score)
            sources.append(f"{label}@{item.segment.start_s:.1f}s")
            kept += 1
        print(
            f"  {label}: {len(result.segments)} segments, {kept} usable "
            f"({result.speech_s:.0f}s speech of {result.audio_duration_s:.0f}s)"
        )
        for item in result.segments:
            if item.embedding is None:
                print(f"      skip {item.segment.start_s:6.1f}s  {_describe(item, store)}")

    if not embeddings:
        if skipped_as_wearer:  # only reachable with --exclude-wearer
            wearer = store.wearer()
            name = wearer.display_name if wearer else "the wearer"
            print(
                f"\nNothing was enrolled: all {skipped_as_wearer} usable "
                f"segment(s) matched {name}, the enrolled wearer.\n\n"
                f"If this recording really is {args.name} and not {name}, "
                f"then {name}'s wearer voiceprint is wrong — most likely it "
                "was enrolled from the wrong recording. Fix it with:\n"
                f'    glasses forget-voice "{name}"\n'
                f'    glasses enroll-voice --name "{name}" --consent … '
                "--wearer --mic",
                file=sys.stderr,
            )
            return 1
        print(
            "\nNothing passed the quality gate, so nothing was enrolled.\n"
            "The usual cause is duration: segments need 3+ seconds of "
            "continuous speech. Record a longer, uninterrupted stretch.",
            file=sys.stderr,
        )
        return 1

    if skipped_as_wearer:
        print(f"  ({skipped_as_wearer} segment(s) skipped as the wearer)")

    person_id = store.enroll(
        args.name,
        embeddings,
        consent_note=args.consent,
        quality_scores=scores,
        source=",".join(sources),
        person_id=person_id,
        kind=VOICE,
        is_wearer=args.wearer,
    )
    store.save()
    role = "wearer" if args.wearer else "person"
    print(f"\nenrolled {args.name} as {role} {person_id[:8]} with {len(embeddings)} voiceprint(s)")
    return 0


def cmd_identify_voice(args: argparse.Namespace) -> int:
    store = LocalStore.load(args.store)
    if store.template_count_by_kind(VOICE) == 0:
        print("No voiceprints enrolled — run enroll-voice first.", file=sys.stderr)
        return 1
    if bool(args.mic) == bool(args.audio):
        print("Give either --mic or audio paths, not both and not neither.", file=sys.stderr)
        return 1

    pipeline = build_voice_pipeline(store, diarize=args.diarize, num_speakers=args.speakers)

    if args.mic:
        from .voice.capture import record_with_meter

        print(f"Listening for {args.seconds}s. Talk, or let someone else talk.\n")
        samples = record_with_meter(args.seconds, device=args.device)
        if args.save:
            write_wav(args.save, samples)
        recordings = [("microphone", samples)]
    else:
        recordings = [(p.name, load_audio(p)) for p in expand_audio(args.audio)]

    for label, samples in recordings:
        result = pipeline.process(samples)
        print(f"{label}  ({result.audio_duration_s:.0f}s)")
        if not result.segments:
            print("  no speech")
            continue
        for item in result.segments:
            turn = "" if item.segment.speaker is None else f"spk{item.segment.speaker} "
            print(f"  {item.segment.start_s:6.1f}s  {turn}{_describe(item, store)}")
        print(f"  {result.timings.total_ms:.0f} ms")
    return 0


def _label_of(item, store: LocalStore):  # type: ignore[no-untyped-def]
    person_id = item.match.person_id if item.match else None
    name = store.name_of(person_id) if person_id else None
    return (person_id, name, item.is_wearer)


def transcribe_with_fallback(transcriber, segments, labels, *, fallback):  # type: ignore[no-untyped-def]
    """Transcribe every segment, dropping to ``fallback()`` if the hosted
    recogniser fails.

    A key that expired overnight should cost transcript quality, not the
    transcript. The transcriber that actually produced the text comes back
    with it, so the caller stops paying Cartesia's timeout on a key already
    known to be dead.
    """
    from .voice import transcribe_segments

    try:
        return transcribe_segments(transcriber, segments, labels=labels), transcriber
    except CartesiaError as error:
        local = fallback()
        print(f"  ! {error}", file=sys.stderr)
        print(f"  falling back to {local.model_name}", file=sys.stderr)
        return transcribe_segments(local, segments, labels=labels), local


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Who said what. Each speaker segment is transcribed on its own, so the
    attribution needs no timestamp alignment to be exact."""
    from .voice import WhisperTranscriber, build_transcriber, render

    store = LocalStore.load(args.store)
    pipeline = build_voice_pipeline(store, diarize=args.diarize, num_speakers=args.speakers)
    transcriber = build_transcriber(
        local_model=args.model,
        language=args.language,
        local_only=args.local,
    )
    print(f"transcribing with {transcriber.model_name}")

    if args.mic:
        from .voice.capture import record_with_meter

        print(f"Listening for {args.seconds}s.\n")
        samples = record_with_meter(args.seconds, device=args.device)
        if args.save:
            write_wav(args.save, samples)
        recordings = [("microphone", samples)]
    else:
        recordings = [(p.name, load_audio(p)) for p in expand_audio(args.audio)]

    for label, samples in recordings:
        result = pipeline.process(samples)
        usable = [i for i in result.segments if i.embedding is not None]
        utterances, transcriber = transcribe_with_fallback(
            transcriber,
            [i.segment for i in usable],
            [_label_of(i, store) for i in usable],
            fallback=lambda: WhisperTranscriber(args.model),
        )
        print(f"--- {label} ---")
        print(render(utterances) if utterances else "  (nothing intelligible)")
        unknown = sum(1 for u in utterances if u.person_id is None and not u.is_wearer)
        if unknown:
            print(f"\n  {unknown} utterance(s) from someone not enrolled")
        print()
    return 0


def cmd_voice_eval(args: argparse.Namespace) -> int:
    """Calibrate the accept threshold on your own voices.

    Takes a directory of ``<person>/<recording>.wav`` and reports the
    genuine/impostor separation plus the operating point at a target FAR.
    This is the command that turns the shipped threshold from a guess into a
    measurement, and the voice threshold badly needs it.
    """
    from .evaluation import operating_point_at_far, score_pairs

    root = Path(args.directory)
    people = sorted(p for p in root.iterdir() if p.is_dir())
    if len(people) < 2:
        print(f"need at least 2 person folders under {root}", file=sys.stderr)
        return 1

    store = LocalStore.load(args.store)
    pipeline = build_voice_pipeline(store)

    vectors, labels = [], []
    for person in people:
        count = 0
        for path in sorted(p for p in person.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
            for item in pipeline.process(load_audio(path)).embedded:
                vectors.append(item.embedding.vector)
                labels.append(person.name)
                count += 1
        print(f"  {person.name:20s} {count} usable segment(s)")

    if len(vectors) < 4:
        print("\nnot enough usable segments to measure anything", file=sys.stderr)
        return 1

    genuine, impostor = score_pairs(np.array(vectors), labels)
    if genuine.size == 0 or impostor.size == 0:
        print("\nneed at least two recordings of one person, and two people", file=sys.stderr)
        return 1

    print(
        f"\ngenuine pairs   n={genuine.size:4d}  mean {genuine.mean():.3f}  min {genuine.min():.3f}"
    )
    print(
        f"impostor pairs  n={impostor.size:4d}  mean {impostor.mean():.3f}  "
        f"max {impostor.max():.3f}"
    )
    separable = genuine.min() > impostor.max()
    print(f"separable       {'yes' if separable else 'NO — overlapping distributions'}")

    point = operating_point_at_far(genuine, impostor, args.far)
    print(f"\nat FAR <= {args.far:.0%}:  {point}")
    print(f"\n  accept_threshold = {point.threshold:.3f}   <- put this in VoiceMatchConfig")
    if genuine.size < 20 or impostor.size < 100:
        print("\nToo few pairs to trust. Aim for 5+ people with 3+ recordings each.")
    return 0


def cmd_voice_compare(args: argparse.Namespace) -> int:
    """Pairwise similarity between recordings. The quickest way to see
    whether two voices are separable at all before tuning anything."""
    store = LocalStore.load(args.store)
    pipeline = build_voice_pipeline(store)

    vectors: dict[str, np.ndarray] = {}
    for path in expand_audio(args.audio):
        embedded = pipeline.process(load_audio(path)).embedded
        if not embedded:
            print(f"  ! {path.name}: nothing passed the gate", file=sys.stderr)
            continue
        vectors[path.stem] = embedded[0].embedding.vector

    if len(vectors) < 2:
        print("need at least two usable recordings", file=sys.stderr)
        return 1

    width = max(len(k) for k in vectors)
    for a, b in itertools.combinations(vectors, 2):
        print(f"  {a:{width}s}  {b:{width}s}  {cosine_similarity(vectors[a], vectors[b]):.3f}")
    return 0


def cmd_forget_voice(args: argparse.Namespace) -> int:
    """Remove a person's voiceprints, keeping their face templates.

    `forget` is all-or-nothing by design -- it is the deletion-on-request
    path. This is the narrower repair tool for when one channel was enrolled
    wrongly and re-recording it is the fix.
    """
    store = LocalStore.load(args.store)
    person = store.find_by_name(args.who) or store.people.get(args.who)
    if person is None:
        print(f"no such person: {args.who}", file=sys.stderr)
        return 1

    removed = store.forget_templates(person.id, kind=VOICE)
    store.save()
    print(
        f"removed {removed} voiceprint(s) from {person.display_name}; "
        f"{store.template_count(person.id, 'face')} face template(s) kept"
    )
    return 0


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    check = subparsers.add_parser("voice-check", help="verify voice models and the wearer")
    check.set_defaults(func=cmd_voice_check)

    enroll = subparsers.add_parser("enroll-voice", help="add a person's voice from recordings")
    enroll.add_argument("--name", required=True)
    enroll.add_argument(
        "--consent",
        required=True,
        help="who agreed to this and when — recorded with the person, and required",
    )
    enroll.add_argument(
        "--wearer",
        action="store_true",
        help="this is you. Enroll once at setup: it turns open-set diarization "
        "into a binary question and keeps your own speech out of other "
        "people's profiles",
    )
    enroll.add_argument("--diarize", action="store_true", help="recording has several voices")
    enroll.add_argument(
        "--exclude-wearer",
        action="store_true",
        help="drop segments matching the enrolled wearer. For enrolling "
        "someone from a recording of a conversation you were part of. Off by "
        "default: when you pass --name you have said who this is, and the "
        "system should not overrule you",
    )
    enroll.add_argument(
        "--mic", action="store_true", help="record from the microphone now, no files needed"
    )
    enroll.add_argument("--seconds", type=int, default=20, help="how long to record (default 20)")
    enroll.add_argument("--device", type=int, default=None, help="input device index")
    enroll.add_argument("--save", help="also write the recording to this WAV")
    enroll.add_argument("audio", nargs="*")
    enroll.set_defaults(func=cmd_enroll_voice)

    identify = subparsers.add_parser("identify-voice", help="who is speaking in a recording")
    identify.add_argument("--diarize", action="store_true", help="split by speaker first")
    identify.add_argument("--speakers", type=int, default=-1, help="pin the speaker count")
    identify.add_argument("--mic", action="store_true", help="listen on the microphone now")
    identify.add_argument("--seconds", type=int, default=10, help="how long to listen")
    identify.add_argument("--device", type=int, default=None, help="input device index")
    identify.add_argument("--save", help="also write what it heard to this WAV")
    identify.add_argument("audio", nargs="*")
    identify.set_defaults(func=cmd_identify_voice)

    transcribe = subparsers.add_parser("transcribe", help="who said what, attributed")
    transcribe.add_argument("--diarize", action="store_true", help="split by speaker first")
    transcribe.add_argument("--speakers", type=int, default=-1, help="pin the speaker count")
    transcribe.add_argument(
        "--local",
        action="store_true",
        help="use the local Whisper instead of Cartesia, even with a key set",
    )
    transcribe.add_argument(
        "--language", default="en", help="ISO-639-1 language for Cartesia (default en)"
    )
    transcribe.add_argument(
        "--model", default="base.en", help="whisper size for the local path (default base.en)"
    )
    transcribe.add_argument("--mic", action="store_true", help="listen on the microphone now")
    transcribe.add_argument("--seconds", type=int, default=20, help="how long to listen")
    transcribe.add_argument("--device", type=int, default=None, help="input device index")
    transcribe.add_argument("--save", help="also write what it heard to this WAV")
    transcribe.add_argument("audio", nargs="*")
    transcribe.set_defaults(func=cmd_transcribe)

    forget_voice = subparsers.add_parser(
        "forget-voice", help="remove a person's voiceprints, keeping their face"
    )
    forget_voice.add_argument("who", help="name or id")
    forget_voice.set_defaults(func=cmd_forget_voice)

    evaluate = subparsers.add_parser(
        "voice-eval", help="calibrate accept_threshold on <person>/<recording> folders"
    )
    evaluate.add_argument("directory")
    evaluate.add_argument("--far", type=float, default=0.01, help="target false-accept rate")
    evaluate.set_defaults(func=cmd_voice_eval)

    compare = subparsers.add_parser("voice-compare", help="pairwise similarity between recordings")
    compare.add_argument("audio", nargs="+")
    compare.set_defaults(func=cmd_voice_compare)

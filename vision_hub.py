import os
import platform
import socket
import sys

import serial
import serial.tools.list_ports
import cv2
import numpy as np
import time

from glasses import env
from glasses.binding.asd import AsdConfig
from glasses.encounter.clock import Clock, seconds_to_us, us_to_seconds
from glasses.encounter.record import build_encounter, face_observations, speech_observations
from glasses.face import ArcFaceEmbedder, FacePipeline, ScrfdDetector
from glasses.face.models import default_providers
from glasses.meet import AudioRecorder, Capture, FaceTracker, MouthReader, assign_speakers
from glasses.storage import open_store, store_label
from glasses.voice import VadSegmenter, VoicePipeline, WeSpeakerEmbedder

# Before anything reads DATABASE_URL. Only the CLI called this, so running
# this script in a shell nobody exported into chose the local file store and
# said nothing about it — the encounters went to JSON instead of Tiger.
env.load()

BAUD_RATE = 115200
#: Which device this is, written on every encounter. One hub per pair of
#: glasses, so the host name is the honest answer until there are more.
DEVICE_ID = os.environ.get("GLASSES_DEVICE_ID") or socket.gethostname()
CAMERA_INDEX = int(os.environ.get("GLASSES_CAMERA", "1"))


def find_serial_port():
    """The ESP32's port, by name, on whichever OS this is.

    Hardcoding one is how this came to be pinned to a Windows COM port that
    does not exist on the machine it now runs on.
    """
    override = os.environ.get("GLASSES_SERIAL_PORT")
    if override:
        return override
    #: The USB-serial bridges these boards ship with, plus the S3's native USB.
    wanted = ("usbmodem", "usbserial", "wchusbserial", "SLAB_USBtoUART", "ttyUSB", "ttyACM")
    for port in serial.tools.list_ports.comports():
        if any(name in port.device for name in wanted):
            return port.device
    if platform.system() == "Windows":
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if ports:
            return ports[0]
    return None


def open_camera(index=CAMERA_INDEX):
    """A capture handle, without the Windows-only backend on macOS.

    ``cv2.CAP_DSHOW`` is DirectShow. Passing it elsewhere fails to open the
    camera at all, which is silent until the first ``cap.read()`` returns
    nothing.
    """
    capture = (
        cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if platform.system() == "Windows"
        else cv2.VideoCapture(index)
    )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if platform.system() == "Windows":
        # Fixed exposure keeps the mouth signal stable, but these two are
        # unsupported on the AVFoundation backend and make it return blank
        # frames rather than ignoring them.
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        capture.set(cv2.CAP_PROP_EXPOSURE, -4)
    return capture


print("Loading Face and Voice machine learning models...")
store = open_store()
print(f"Store: {store_label()}")
#: Only the Postgres store can hold an encounter. Offline runs still
#: recognise faces; they just have nowhere to put the history.
records_encounters = hasattr(store, "record_encounter")
if not records_encounters:
    print("No DATABASE_URL — recognising only, encounters will not be recorded.")
providers = default_providers()

face_pipeline = FacePipeline(
    detector=ScrfdDetector(providers=providers),
    embedder=ArcFaceEmbedder(providers=providers),
    gallery=store.gallery("face")
)

voice_pipeline = VoicePipeline(
    segmenter=VadSegmenter(),
    embedder=WeSpeakerEmbedder(),
    gallery=store.gallery("voice"),
    wearer_embeddings=store.wearer_embeddings()
)

mouths = MouthReader(providers=providers)

SERIAL_PORT = find_serial_port()
if SERIAL_PORT is None:
    sys.exit(
        "No ESP32 serial port found. Plug the glasses in, or set "
        "GLASSES_SERIAL_PORT to the device (e.g. /dev/cu.usbmodem1234)."
    )

print(f"Connecting to ESP32 on {SERIAL_PORT}...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print("Connected! Waiting for touch sensor input...")

def resolve_identities(tracker):
    """Who each tracked face is, resolved once for the whole session.

    One embedding per track rather than per utterance: the same face often
    speaks several times and the answer cannot change between them.
    """
    identities = {}
    for track in tracker.tracks.values():
        if track.best_frame is None:
            continue
        frame_result = face_pipeline.process_frame(track.best_frame)
        if not frame_result.faces:
            continue
        best = max(frame_result.faces, key=lambda f: f.detection.bbox.width)
        if best.quality.passed and best.match and best.match.person_id:
            identities[track.id] = (best.match.person_id, float(best.match.similarity))
    return identities


def record_session(tracker, captures, identities, clock, started_us, ended_us, frame_width):
    """Write the session to the store, if the store can hold one.

    A failure here is reported and swallowed: losing an encounter is bad,
    and taking the hub down mid-conversation with it is worse.
    """
    if not records_encounters:
        return
    faces = face_observations(
        list(tracker.tracks.values()), identities, frame_width=frame_width
    )
    speech = speech_observations(captures)
    if not faces and not speech:
        print("[DB] nothing observed, no encounter recorded.")
        return

    encounter = build_encounter(
        device_id=DEVICE_ID,
        started_us=started_us,
        ended_us=ended_us,
        faces=faces,
        speech=speech,
    )
    try:
        store.record_encounter(encounter, clock)
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        print(f"[DB] FAILED to record encounter: {error}")
        return
    print(
        f"[DB] encounter {encounter.id} [{encounter.status}] recorded: "
        f"{len(faces)} face, {len(speech)} speech observation(s)"
    )


def resolve_session(
    audio, audio_start_us, tracker, clock, started_us, ended_us, frame_width=None
):
    print("\n[Processing] Analyzing audio and binding to faces...")
    captures = []
    result = voice_pipeline.process(audio) if audio.size > 0 else None
    for item in result.segments if result else []:
        if item.embedding is None:
            continue
        captures.append(Capture(
            start_us=audio_start_us + seconds_to_us(item.segment.start_s),
            end_us=audio_start_us + seconds_to_us(item.segment.end_s),
            audio=item.segment.samples,
            asd=None,
            track=None,
            is_wearer=item.is_wearer,
            voice_person_id=item.match.person_id if item.match else None,
            voice_name=store.name_of(item.match.person_id) if item.match and item.match.person_id else None,
            voice_similarity=item.match.similarity if item.match else 0.0
        ))
        
    if not captures:
        print("No usable speech detected (requires 3+ continuous seconds).")

    partners = [c for c in captures if not c.is_wearer]
    if partners:
        assign_speakers(partners, tracker, audio, audio_start_us, AsdConfig(min_correlation=0.30, min_margin=0.15))

    identities = resolve_identities(tracker)

    for capture in captures:
        at = us_to_seconds(capture.start_us - audio_start_us)
        if capture.is_wearer:
            print(f"[{at:5.1f}s] WEARER (Audio: {capture.duration_s():.1f}s)")
            continue
            
        who = capture.voice_name or "unknown voice"
        line = f"[{at:5.1f}s] {who}"
        if capture.voice_similarity:
            line += f" (voice similarity: {capture.voice_similarity:.2f})"
            
        if capture.track is None:
            print(f"{line}  -> UNASSIGNED: {capture.asd.reason}")
            continue
            
        track = capture.track
        face_id = "new person"
        if track.id in identities:
            person_id, similarity = identities[track.id]
            face_id = f"{store.name_of(person_id)} ({similarity:.2f})"

        print(f"{line}  -> face #{track.id} [{face_id}]")

    record_session(tracker, captures, identities, clock, started_us, ended_us, frame_width)

is_recording = False
pending_command = None

while True:
    try:
        command = pending_command
        pending_command = None

        if not command and ser.in_waiting > 0:
            command = ser.readline().decode('utf-8').strip()

        # MULTIMODAL RECORDING MODE (Tap & Hold)
        if command == "CAM_ON":
            print("\nHardware Trigger: Camera & Mic ON. Recording session...")
            cap = open_camera()

            clock = Clock.start()
            tracker = FaceTracker()
            frames_seen = 0
            detections = []
            frame_width = None
            is_recording = True

            with AudioRecorder(clock) as recorder:
                started = clock.now_us()
                prev_frame_time = 0
                
                while True:
                    if ser.in_waiting > 0:
                        inner_cmd = ser.readline().decode('utf-8').strip()
                        if inner_cmd == "CAM_OFF":
                            print("Hardware Trigger: Camera & Mic OFF.")
                            is_recording = False
                            break
                        elif inner_cmd:
                            print(f"ESP32: {inner_cmd}")

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_width = frame.shape[1]
                    new_frame_time = time.time()
                    time_diff = new_frame_time - prev_frame_time
                    fps = int(1 / time_diff) if time_diff > 0 else 0
                    prev_frame_time = new_frame_time

                    at_us = clock.now_us()
                    frames_seen += 1

                    if frames_seen % 5 == 1 or not detections:
                        detections = face_pipeline.detector.detect(frame)
                    
                    tracker.update(detections, frame, at_us)
                    
                    for track in tracker.active(at_us):
                        if track.current_detection is not None:
                            sample = mouths.sample(frame, track.current_detection, at_us)
                            if sample is not None:
                                track.mouth.append(sample)
                                
                        box = track.bbox
                        cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 200, 200), 2)
                        cv2.putText(frame, f"Track #{track.id}", (int(box.x1), max(16, int(box.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

                    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.imshow("Multimodal Feed - Tap ESP32 to end", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            audio = recorder.samples
            audio_start_us = recorder.start_us or started
            ended = clock.now_us()
            cap.release()
            cv2.destroyWindow("Multimodal Feed - Tap ESP32 to end")

            # Runs even with no audio: the faces seen are still an encounter,
            # and a silent one is the expired case the schema has a state for.
            resolve_session(
                audio, audio_start_us, tracker, clock, started, ended, frame_width=frame_width
            )

        # CONTINUOUS IDENTIFY MODE (Single Tap)
        elif command == "SNAPSHOT" and not is_recording:
            print("\nHardware Trigger: Continuous Identify Mode ON...")
            cap = open_camera()

            prev_frame_time = 0
            last_lcd_update = 0

            while True:
                if ser.in_waiting > 0:
                    inner_cmd = ser.readline().decode('utf-8').strip()
                    if inner_cmd == "SNAPSHOT":
                        print("Hardware Trigger: Continuous Identify Mode OFF.")
                        ser.write(b"LCD:Ready\n")
                        break
                    elif inner_cmd == "CAM_ON":
                        print("Hardware Trigger: Overriding Identify Mode with Recording Mode...")
                        pending_command = "CAM_ON"
                        break
                    elif inner_cmd:
                        print(f"ESP32: {inner_cmd}")

                ret, frame = cap.read()
                if not ret:
                    break

                new_frame_time = time.time()
                time_diff = new_frame_time - prev_frame_time
                fps = int(1 / time_diff) if time_diff > 0 else 0
                prev_frame_time = new_frame_time

                result = face_pipeline.process_frame(frame)
                lcd_name = "Unknown"
                
                if result.faces:
                    for face in result.faces:
                        box = face.detection.bbox
                        if face.match is not None and face.match.is_identified:
                            name = store.name_of(face.match.person_id)
                            lcd_name = name
                            display_text = f"{name} ({face.match.similarity:.2f})"
                            color = (0, 255, 0)
                        else:
                            display_text = "Unknown person"
                            color = (0, 0, 255)
                            
                        cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
                        cv2.putText(frame, display_text, (int(box.x1), max(16, int(box.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Send update to LCD every 0.5 seconds to avoid flooding serial
                if time.time() - last_lcd_update > 0.5:
                    ser.write(f"LCD:{lcd_name}\n".encode('utf-8'))
                    last_lcd_update = time.time()

                cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Live Recognition - Single Tap ESP32 to end", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    ser.write(b"LCD:Ready\n")
                    break

            cap.release()
            try:
                cv2.destroyWindow("Live Recognition - Single Tap ESP32 to end")
            except cv2.error:
                pass  # Window was already closed or never created
            time.sleep(0.01)

        elif command:
            print(f"ESP32: {command}")

    except KeyboardInterrupt:
        break

ser.close()
"""Hardware test: the C270 is always looking for a face, and shows what it finds.

No touch needed. Every frame goes through the face pipeline; each face is
boxed in a laptop window (green = recognised, red = unknown) and the result
goes to the glasses OLED when the ESP32 is plugged in: the person's profile,
"Unknown person", or blank when nobody is there. The terminal logs each
change and a stats line every 5 s; q or Ctrl-C prints a summary.

    .venv/bin/python face_watch.py            # camera + window + OLED
    FACE_WATCH_NO_WINDOW=1 .venv/bin/python face_watch.py   # headless

It does not touch vision_hub.py and records nothing.
"""

from __future__ import annotations

import os
import sys
import time

import cv2

from glasses.device import GLASSES_CAMERA, GlassesLink, find_camera_index, open_camera
from glasses.env import load as load_env
from glasses.face import ArcFaceEmbedder, FacePipeline, ScrfdDetector
from glasses.face.models import default_providers
from glasses.oled import UNKNOWN, PersonDisplay
from glasses.storage import open_store, store_label

load_env()

WINDOW = os.environ.get("FACE_WATCH_NO_WINDOW") != "1"
#: How long a face may vanish before the OLED goes back to "Looking...".
#: Bridges single missed detections so the display does not flicker.
LOST_AFTER_S = 1.5
STATS_EVERY_S = 5.0


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def main() -> int:
    index = find_camera_index()
    if index is None:
        print(f"FAIL: no {GLASSES_CAMERA} camera found. Plug in the Logitech.")
        return 1
    print(f"[ok] {GLASSES_CAMERA} is OpenCV camera {index}")

    print("Loading face models...")
    store = open_store()
    print(f"[ok] store: {store_label()} ({len(getattr(store, 'people', {}) or {})} people enrolled)")
    providers = default_providers()
    pipeline = FacePipeline(
        detector=ScrfdDetector(providers=providers),
        embedder=ArcFaceEmbedder(providers=providers),
        gallery=store.gallery("face"),
    )

    link = None
    try:
        link = GlassesLink()
        print(f"[ok] glasses on {link.port} ({link.oled or 'no boot report'})")
    except Exception as error:  # noqa: BLE001 - laptop-only test is still useful
        print(f"[warn] no glasses display, laptop window only: {error}")

    display = PersonDisplay(link, store)
    display.clear()

    cap = open_camera(index)
    if not cap.isOpened():
        print(f"FAIL: could not open camera {index}")
        return 1

    # Totals for the summary.
    started = time.monotonic()
    frames = face_frames = failed_reads = 0
    first_face_s = None
    longest_gap_s = 0.0
    # Rolling window for the stats line.
    win_frames = win_faces = 0
    win_ms = 0.0
    win_start = started
    last_face_at = None

    print(f"\n[{stamp()}] Watching. Step in front of the camera. q / Ctrl-C to stop.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads in (1, 10, 100):
                    print(
                        f"[{stamp()}] camera gave no frame ({failed_reads}x). If macOS said "
                        "'not authorized', allow Terminal under Privacy & Security > Camera."
                    )
                if failed_reads >= 100:
                    break
                time.sleep(0.05)
                continue

            now = time.monotonic()
            frames += 1
            win_frames += 1
            result = pipeline.process_frame(frame)
            win_ms += result.timings.total_ms

            if result.faces:
                face_frames += 1
                win_faces += 1
                if first_face_s is None:
                    first_face_s = now - started
                if last_face_at is not None and now - last_face_at < LOST_AFTER_S:
                    longest_gap_s = max(longest_gap_s, now - last_face_at)
                last_face_at = now

                for face in result.faces:
                    box = face.detection.bbox
                    if face.match is not None and face.match.is_identified:
                        text = f"{store.name_of(face.match.person_id)} ({face.match.similarity:.2f})"
                        colour = (0, 255, 0)
                    elif not face.quality.passed:
                        text = "Face (low quality)"
                        colour = (0, 165, 255)
                    else:
                        text = "Unknown person"
                        colour = (0, 0, 255)
                    cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), colour, 2)
                    cv2.putText(frame, text, (int(box.x1), max(16, int(box.y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            display.update(result.faces, now)

            if now - win_start >= STATS_EVERY_S:
                span = now - win_start
                print(
                    f"[{stamp()}] {win_frames / span:4.1f} fps | face in {100 * win_faces / max(1, win_frames):3.0f}% "
                    f"of frames | pipeline {win_ms / max(1, win_frames):5.1f} ms/frame"
                )
                win_frames = win_faces = 0
                win_ms = 0.0
                win_start = now

            if WINDOW:
                on_oled = {None: "(blank)", UNKNOWN: "Unknown person"}.get(
                    display.showing, store.name_of(display.showing) if display.showing else ""
                )
                cv2.putText(frame, f"OLED: {on_oled}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("face_watch - q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if WINDOW:
            cv2.destroyAllWindows()
        display.clear()
        if link:
            link.close()

    elapsed = time.monotonic() - started
    print("\n=== face_watch summary ===")
    print(f"ran {elapsed:.0f}s, {frames} frames ({frames / max(elapsed, 1e-6):.1f} fps), {failed_reads} failed reads")
    print(f"face in {face_frames}/{frames} frames ({100 * face_frames / max(1, frames):.0f}%)")
    print(f"first face after {first_face_s:.1f}s" if first_face_s is not None else "no face ever detected")
    print(f"longest bridged dropout: {longest_gap_s:.2f}s")
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())

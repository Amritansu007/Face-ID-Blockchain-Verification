#!/usr/bin/env python3
"""
Webcam diagnostic - finds which stage crashes.

    python webcam_check.py

Runs five stages in increasing order of risk and prints progress before each
one. If the process dies without a traceback, the LAST LINE PRINTED tells you
which stage is at fault. Report that line.
"""

import os
import sys
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def line(n, msg):
    print("\n--- STAGE %s: %s ---" % (n, msg), flush=True)


def main():
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    line(0, "imports")
    import cv2
    import numpy as np
    print("opencv %s | python %s" % (cv2.__version__, sys.version.split()[0]), flush=True)
    print("platform: %s" % sys.platform, flush=True)

    line(1, "open camera at default settings, read 10 frames")
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam, backend)
    if not cap.isOpened():
        print("FAILED to open camera %d." % cam, flush=True)
        print("Try a different index: python webcam_check.py 1", flush=True)
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print("opened at %dx%d" % (w, h), flush=True)
    good = 0
    for i in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            good += 1
    print("read %d/10 frames OK" % good, flush=True)
    if good == 0:
        print("Camera opens but yields no frames - another app may be using it,"
              " or privacy settings block it.", flush=True)
        cap.release()
        return 1

    line(2, "request 1280x720 + MJPG, read 10 more")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        print("MJPG fourcc set", flush=True)
    except Exception as e:
        print("fourcc not supported: %s" % e, flush=True)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print("now reporting %dx%d" % (w2, h2), flush=True)
    good2, last = 0, None
    for i in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            good2 += 1
            last = frame
    print("read %d/10 frames OK at %s" % (
        good2, "unknown" if last is None else "%dx%d" % (last.shape[1], last.shape[0])),
        flush=True)

    line(3, "save a frame to disk")
    if last is None:
        print("no frame captured at 720p - rerun with default resolution", flush=True)
        cap.release()
        return 1
    ok, buf = cv2.imencode(".jpg", last)
    if ok:
        buf.tofile("webcam_test_frame.jpg")
        print("wrote webcam_test_frame.jpg (%d bytes)" % len(buf), flush=True)
    cap.release()
    print("camera released", flush=True)

    line(4, "import DeepFace and detect on the saved frame (no GUI)")
    from deepface import DeepFace
    print("DeepFace imported", flush=True)
    try:
        res = DeepFace.extract_faces(last, detector_backend="mtcnn",
                                     enforce_detection=False, align=False)
        faces = [f for f in res if f.get("confidence", 0) > 0
                 and f["facial_area"]["w"] > 20]
        print("mtcnn found %d face(s)" % len(faces), flush=True)
        if faces:
            fa = faces[0]["facial_area"]
            print("largest box: %dx%d px" % (fa["w"], fa["h"]), flush=True)
    except Exception:
        print("detection raised:", flush=True)
        traceback.print_exc()

    line(5, "cv2.imshow preview - THE USUAL SUSPECT")
    print("A window should appear for ~3 seconds. If the process dies here,", flush=True)
    print("run part_a.py with --no-preview.", flush=True)
    try:
        cap2 = cv2.VideoCapture(cam, backend)
        for i in range(60):
            ok, frame = cap2.read()
            if not ok:
                break
            cv2.putText(frame, "preview test %d/60" % (i + 1), (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow("webcam_check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap2.release()
        cv2.destroyAllWindows()
        print("preview survived", flush=True)
    except Exception:
        print("preview raised:", flush=True)
        traceback.print_exc()

    print("\nALL STAGES COMPLETED - no hard crash.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
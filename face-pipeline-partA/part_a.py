#!/usr/bin/env python3
"""
Part A - Face Scan -> Quality-Gated Crops -> Provenance Manifest
Built on DeepFace (RetinaFace detection + Facenet512 embeddings).

Pipeline position:  [THIS]  ->  Part B (reverse image search)  ->  Part C (chain)

    python part_a.py scan   --webcam            --subject self --consent
    python part_a.py scan   --image photo.jpg   --subject self --consent
    python part_a.py verify --manifest out/manifest.json

Outputs to --out (default ./out):
    context.jpg     padded crop  - BEST for reverse image search, try first
    tight.jpg       close crop   - fallback
    upscaled.jpg    upscaled     - fallback for small faces
    source.jpg      the exact frame that was used
    manifest.json   canonical, hashable provenance record for Part C
"""

from __future__ import annotations

import os

# must be set before tensorflow is imported, or the console fills with noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import hashlib
import json
import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import numpy as np

try:
    from deepface import DeepFace
except ImportError:
    sys.exit("deepface is not installed - see README_partA.md")

MANIFEST_VERSION = "1.3"
EMBEDDING_MODEL = "Facenet512"
PRIMARY_DETECTOR = "retinaface"    # 5 keypoints, high accuracy, ~3s/frame
FAST_DETECTOR = "mtcnn"            # ~0.13s/frame, used for the live webcam loop
# NOTE: the 'opencv', 'ssd' and 'yunet' backends are NOT usable with
# opencv-python 5.x - OpenCV 5 stopped bundling the haarcascade XML files and
# moves the DNN/YuNet helpers into opencv-contrib. MTCNN is the only fast
# backend that works on a plain opencv-python install, so it is the fallback.
FALLBACK_DETECTORS = ["mtcnn"]
ENCODING_DECIMALS = 6
QUALITY_PASS_THRESHOLD = 55

# approximate sizes of the weight files, for detecting partial downloads
EXPECTED_WEIGHTS = {
    "retinaface.h5": 119_000_000,
    "facenet512_weights.h5": 95_000_000,
}


# --------------------------------------------------------------------------
# hashing / canonical serialisation
# --------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> bytes:
    """Deterministic bytes for any dict. Sorted keys, no whitespace, UTF-8.
    Part C MUST hash exactly these bytes or verification will not reproduce."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


# --------------------------------------------------------------------------
# presentation-attack (liveness) heuristics
# --------------------------------------------------------------------------
# The pipeline otherwise accepts a printed photo or a phone screen held to the
# camera. These are LIGHTWEIGHT PASSIVE CHECKS, not a trained PAD model:
#
#   moire    - LCD/OLED pixel grids and printer halftones are periodic, so a
#              recapture shows sharp peaks in the mid-frequency FFT annulus.
#              Real skin has a smooth 1/f falloff with no such peaks.
#   texture  - re-captured images lose fine skin micro-texture; measured as
#              high-pass residual energy.
#   motion   - webcam only. A live face micro-moves relative to the scene; a
#              held photo moves rigidly with the hand, or not at all.
#
# ADVISORY BY DEFAULT. These heuristics have real false-positive rates, so
# they must never silently block a scan - see --require-liveness.

@dataclass
class Liveness:
    mode: str = "passive"
    moire_peak: float = 0.0
    texture_energy: float = 0.0
    motion_ratio: float = -1.0
    face_motion: float = -1.0
    bg_motion: float = -1.0
    score: int = 0
    verdict: str = "inconclusive"
    notes: list = field(default_factory=list)


def _moire_peak(face_bgr) -> float:
    """Ratio of the strongest mid-frequency FFT peak to that band's median.
    ~16 for real faces, higher for screen/print recaptures."""
    g = cv2.cvtColor(cv2.resize(face_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float64)
    g -= g.mean()
    g *= np.outer(np.hanning(256), np.hanning(256))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.hypot(yy - 128, xx - 128)
    band = (r > 30) & (r < 110)          # skip DC and the extreme corners
    vals = mag[band]
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals)) + 1e-9
    return float(np.percentile(vals, 99.9) / med)


def _texture_energy(face_bgr) -> float:
    """High-pass residual energy. Low values suggest a flat recapture."""
    g = cv2.cvtColor(cv2.resize(face_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float64)
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    return float(hp.std())


def _motion_stats(frames, box):
    """(face_motion, bg_motion) as mean abs inter-frame difference.

    Three cases worth separating:
      face~0 and bg~0        -> nothing moving at all: a static image
      face ~= bg  (ratio ~1) -> whole scene moves together: photo in a hand
      face >  bg  (ratio >1) -> face moves independently: a live person
    """
    if len(frames) < 4:
        return -1.0, -1.0
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    face_d, bg_d, prev = [], [], None
    step = max(1, len(frames) // 12)
    for f in frames[::step]:
        g = cv2.cvtColor(cv2.resize(f, (320, 240)), cv2.COLOR_BGR2GRAY).astype(np.float64)
        sy, sx = 240.0 / f.shape[0], 320.0 / f.shape[1]
        fy0, fy1 = max(0, int(y * sy)), min(240, int((y + h) * sy))
        fx0, fx1 = max(0, int(x * sx)), min(320, int((x + w) * sx))
        if prev is not None and fy1 > fy0 + 2 and fx1 > fx0 + 2:
            d = np.abs(g - prev)
            mask = np.ones_like(d, dtype=bool)
            mask[fy0:fy1, fx0:fx1] = False
            face_d.append(float(d[fy0:fy1, fx0:fx1].mean()))
            bg_d.append(float(d[mask].mean()) if mask.any() else 0.0)
        prev = g
    if not face_d:
        return -1.0, -1.0
    return float(np.mean(face_d)), float(np.mean(bg_d))


def assess_liveness(face_bgr, frames=None, box=None) -> Liveness:
    """Thresholds below were calibrated on a small sample and SIMULATED
    recaptures, not a real presentation-attack dataset. Treat the verdict as a
    hint, never as an authentication decision."""
    lv = Liveness()
    lv.moire_peak = _moire_peak(face_bgr)
    lv.texture_energy = _texture_energy(face_bgr)

    # real faces measured ~16; simulated print ~30, screen ~47
    s_moire = 1.0 - _norm(lv.moire_peak, 22.0, 45.0)
    s_tex = _norm(lv.texture_energy, 2.0, 7.0)

    if frames and box:
        lv.mode = "passive+temporal"
        face_m, bg_m = _motion_stats(frames, box)
        lv.face_motion, lv.bg_motion = face_m, bg_m
        if face_m >= 0:
            lv.motion_ratio = float(face_m / (bg_m + 1e-6))
            if face_m < 0.35:
                s_motion = 0.0            # nothing moved: static image
            else:
                s_motion = _norm(lv.motion_ratio, 1.35, 2.2)
            lv.score = int(round(100 * (0.30 * s_moire + 0.20 * s_tex
                                        + 0.50 * s_motion)))
        else:
            lv.score = int(round(100 * (0.55 * s_moire + 0.45 * s_tex)))
    else:
        lv.score = int(round(100 * (0.55 * s_moire + 0.45 * s_tex)))

    if lv.moire_peak > 28:
        lv.notes.append("strong periodic pattern - possible screen or print")
    if lv.texture_energy < 2.5:
        lv.notes.append("very flat texture - possible recapture")
    if lv.mode == "passive+temporal" and lv.face_motion >= 0:
        if lv.face_motion < 0.35:
            lv.notes.append("no movement at all across frames - static image?")
        elif lv.motion_ratio < 1.35:
            lv.notes.append("face moves rigidly with the scene - held photo?")

    lv.verdict = ("likely_live" if lv.score >= 60 else
                  "possible_spoof" if lv.score < 35 else "inconclusive")
    return lv


# --------------------------------------------------------------------------
# biometric template protection
# --------------------------------------------------------------------------
# A raw Facenet512 embedding is NOT an anonymous ID. Template-inversion
# attacks can reconstruct a recognisable face from one, so an embedding is
# itself biometric data. Biometrics also cannot be reissued: a leaked password
# is replaceable, a leaked face is not. Publishing one - or a hash committing
# to one - on an immutable public ledger is therefore a bad default.
#
# BioHashing fixes this. The embedding is projected through a salt-seeded
# orthonormal random basis and binarised. The result:
#   * irreversible  - binarisation discards magnitude, the projection is lossy
#   * comparable    - Hamming distance still tracks cosine distance
#   * revocable     - a new salt yields a completely unlinkable template
# The salt is the secret. It is written to a separate key file, never into the
# manifest; the manifest carries only its SHA-256 so you can prove which salt
# was used without revealing it.

_LAST_CAPTURE = None          # frames from the most recent webcam scan

TEMPLATE_SCHEME = "biohash-v1"
TEMPLATE_BITS = 256


def protect_template(embedding, salt: bytes, bits: int = TEMPLATE_BITS) -> str:
    """Embedding + salt -> irreversible, revocable bit template (hex)."""
    v = np.asarray(embedding, dtype=np.float64).ravel()
    v = v / (np.linalg.norm(v) + 1e-12)
    seed = int.from_bytes(hashlib.sha256(salt).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((v.size, bits))
    # orthonormal columns preserve relative distances far better than raw
    # gaussian columns do
    q, _ = np.linalg.qr(basis)
    projected = v @ q[:, :bits]
    packed = np.packbits((projected > 0).astype(np.uint8))
    return packed.tobytes().hex()


def template_distance(hex_a: str, hex_b: str) -> float:
    """Normalised Hamming distance in [0,1]. ~0 same face, ~0.5 unrelated."""
    a = np.unpackbits(np.frombuffer(bytes.fromhex(hex_a), dtype=np.uint8))
    b = np.unpackbits(np.frombuffer(bytes.fromhex(hex_b), dtype=np.uint8))
    n = min(a.size, b.size)
    return float(np.count_nonzero(a[:n] != b[:n]) / n)


def new_salt() -> bytes:
    return os.urandom(32)


# --------------------------------------------------------------------------
# image IO
# --------------------------------------------------------------------------
# cv2.imread / cv2.imwrite go through a narrow-char C++ API on Windows and
# silently fail on any path containing non-ASCII characters (accented letters,
# CJK, etc). Routing through numpy's file IO avoids that entirely.

def imread_unicode(path):
    """Read an image. Returns None on failure, like cv2.imread."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img, params=None) -> bool:
    """Write an image. Returns True on success, like cv2.imwrite."""
    try:
        ext = Path(path).suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img, params if params else [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# quality scoring
# --------------------------------------------------------------------------

@dataclass
class Quality:
    face_px: int = 0
    sharpness: float = 0.0
    brightness: float = 0.0
    clipping: float = 0.0
    yaw_offset: float = 0.0
    roll_deg: float = 0.0
    detector_confidence: float = 0.0
    score: int = 0
    passed: bool = False
    warnings: list = field(default_factory=list)


def _norm(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))


def _pose_from_keypoints(fa: dict) -> tuple:
    """Return (yaw_offset, roll_degrees) from DeepFace facial_area keypoints.

    yaw_offset: horizontal drift of the nose from the eye midpoint, divided by
      interocular distance. ~0.0 frontal, >0.30 a strong turn. Needs 'nose',
      which only retinaface provides - returns 0.0 on other backends.
    roll_deg: tilt of the line between the eyes.
    """
    le, re = fa.get("left_eye"), fa.get("right_eye")
    if not le or not re:
        return 0.0, 0.0
    le = np.asarray(le, dtype=float)
    re = np.asarray(re, dtype=float)

    dx, dy = (le[0] - re[0]), (le[1] - re[1])
    roll = float(abs(math.degrees(math.atan2(dy, dx))))
    if roll > 90:
        roll = 180.0 - roll

    interocular = float(np.linalg.norm(le - re))
    nose = fa.get("nose")
    if nose is None or interocular < 1e-6:
        return 0.0, roll
    mid_x = (le[0] + re[0]) / 2.0
    yaw = float(abs(float(nose[0]) - mid_x) / interocular)
    return yaw, roll


def compute_quality(bgr, fa: dict, det_conf: float) -> Quality:
    """fa is DeepFace's facial_area dict: x, y, w, h [, keypoints]."""
    x, y, w, h = int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])
    H, W = bgr.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    q = Quality()
    q.detector_confidence = float(det_conf)
    q.face_px = int(min(y2 - y, x2 - x))
    if q.face_px <= 0:
        q.warnings.append("degenerate face box")
        return q

    crop = bgr[y:y2, x:x2]
    # fixed resize first, so sharpness is scale-invariant
    gray = cv2.cvtColor(cv2.resize(crop, (200, 200)), cv2.COLOR_BGR2GRAY)

    q.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    q.brightness = float(gray.mean())
    q.clipping = float(((gray <= 5) | (gray >= 250)).mean())
    q.yaw_offset, q.roll_deg = _pose_from_keypoints(fa)

    has_nose = fa.get("nose") is not None
    s_size = _norm(q.face_px, 80, 250)
    s_sharp = _norm(q.sharpness, 40, 300)
    s_bright = 1.0 - _norm(abs(q.brightness - 128.0), 20, 90)
    s_clip = 1.0 - _norm(q.clipping, 0.01, 0.15)
    s_yaw = (1.0 - _norm(q.yaw_offset, 0.05, 0.35)) if has_nose else 0.6
    s_roll = 1.0 - _norm(q.roll_deg, 5, 30)
    s_conf = _norm(q.detector_confidence, 0.5, 0.95)

    q.score = int(round(100 * (
        0.24 * s_size + 0.24 * s_sharp + 0.13 * s_bright + 0.08 * s_clip +
        0.15 * s_yaw + 0.08 * s_roll + 0.08 * s_conf
    )))
    q.passed = q.score >= QUALITY_PASS_THRESHOLD

    if q.face_px < 100:
        q.warnings.append("face is only %dpx - reverse search will struggle" % q.face_px)
    if q.sharpness < 60:
        q.warnings.append("image looks blurry")
    if q.brightness < 70:
        q.warnings.append("underexposed")
    elif q.brightness > 190:
        q.warnings.append("overexposed")
    if has_nose and q.yaw_offset > 0.30:
        q.warnings.append("head is turned - frontal shots match far better")
    if q.roll_deg > 20:
        q.warnings.append("head tilted %.0f degrees" % q.roll_deg)
    if q.detector_confidence < 0.7:
        q.warnings.append("low detector confidence")
    return q


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def _is_no_face_error(exc) -> bool:
    """DeepFace raises ValueError for a genuine no-detection. Anything else
    (missing/corrupt weights, OOM, bad ONNX) is an infrastructure failure and
    must NOT be reported to the user as 'no face found'."""
    msg = str(exc).lower()
    return isinstance(exc, ValueError) and (
        "could not be detected" in msg or "face could not" in msg)


def detect_faces(bgr, backend: str = PRIMARY_DETECTOR, allow_fallback: bool = True,
                 quiet: bool = False):
    """Return (list_of_face_dicts, backend_used). Empty list if nothing found."""
    order = [backend]
    if allow_fallback:
        order += [b for b in FALLBACK_DETECTORS if b != backend]

    load_failures = {}
    for b in order:
        try:
            res = DeepFace.extract_faces(bgr, detector_backend=b,
                                         enforce_detection=True, align=False)
        except Exception as e:
            if not _is_no_face_error(e):
                load_failures[b] = e
                if not quiet:
                    print("  ! detector '%s' FAILED TO LOAD: %s"
                          % (b, str(e).strip().splitlines()[0][:160]))
                    print("    (this is not a detection failure - the model "
                          "could not be initialised)")
            res = []
        # deepface can return a full-frame pseudo-face; filter that out
        res = [f for f in res
               if f.get("confidence", 0) > 0 and f["facial_area"]["w"] > 20]
        if res:
            if b != backend and not quiet:
                reason = ("failed to load" if backend in load_failures
                          else "found no face")
                print("  %s %s, falling back to %s" % (backend, reason, b))
            return res, b

    if load_failures and not quiet:
        print("\n  All attempted detectors that failed to load: %s"
              % ", ".join(load_failures))
        print("  Most likely a partial or corrupt weight download. Delete the "
              "cache and retry:")
        print("    Windows:  Remove-Item -Recurse -Force $HOME\\.deepface\\weights")
        print("    Linux/mac: rm -rf ~/.deepface/weights")
    return [], "none"


def largest_face(faces):
    return max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])


# --------------------------------------------------------------------------
# crops
# --------------------------------------------------------------------------

def _crop_with_margin(bgr, fa: dict, margin: float):
    x, y, w, h = int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])
    H, W = bgr.shape[:2]
    dy, dx = int(h * margin), int(w * margin)
    return bgr[max(0, y - dy):min(H, y + h + dy),
               max(0, x - dx):min(W, x + w + dx)]


def make_crops(bgr, fa: dict) -> dict:
    """Three variants, ranked for reverse image search.

    'context' leads deliberately: Vision Web Detection matches the whole image,
    and indexed social photos include hair/shoulders/background. A tight face
    crop discards exactly the signal the index was built on.
    """
    context = _crop_with_margin(bgr, fa, 0.75)
    tight = _crop_with_margin(bgr, fa, 0.15)
    short = min(context.shape[:2])
    if short < 512:
        f = 512.0 / short
        upscaled = cv2.resize(context, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = context.copy()
    return {"context": context, "tight": tight, "upscaled": upscaled}


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

def embed_face(tight_bgr) -> list:
    """Facenet512 embedding of the already-cropped face. detector_backend
    'skip' means DeepFace treats the whole array as the face."""
    rep = DeepFace.represent(tight_bgr, model_name=EMBEDDING_MODEL,
                             detector_backend="skip", enforce_detection=False)
    return list(rep[0]["embedding"])


# --------------------------------------------------------------------------
# webcam
# --------------------------------------------------------------------------

def _draw_scan_overlay(frame, kept, n_frames, remaining, sharp, bright):
    """Guide oval + live meters. Deliberately ML-free: running a detector while
    the camera is open crashes the process on Windows (see _capture_frames)."""
    h, w = frame.shape[:2]
    ok_sharp, ok_bright = sharp > 55, 70 <= bright <= 190
    good = ok_sharp and ok_bright
    colour = (0, 210, 0) if good else (0, 165, 255)

    # target oval - "put your face in here"
    cx, cy = w // 2, int(h * 0.46)
    ax, ay = int(w * 0.17), int(h * 0.30)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, colour, 3)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):          # corner ticks
        px, py = cx + sx * ax, cy + sy * ay
        cv2.line(frame, (px, py), (px - sx * 28, py), colour, 3)
        cv2.line(frame, (px, py), (px, py - sy * 28), colour, 3)

    cv2.putText(frame, "ALIGN FACE IN THE OVAL", (cx - 165, cy - ay - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    def meter(y, label, value, lo, hi, is_ok):
        bw, bh, x0 = 210, 14, 14
        cv2.rectangle(frame, (x0, y), (x0 + bw, y + bh), (70, 70, 70), 1)
        filled = int(bw * max(0.0, min(1.0, (value - lo) / float(hi - lo))))
        c = (0, 210, 0) if is_ok else (0, 165, 255)
        cv2.rectangle(frame, (x0 + 1, y + 1), (x0 + filled, y + bh - 1), c, -1)
        cv2.putText(frame, "%s %.0f" % (label, value), (x0 + bw + 10, y + bh - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    meter(h - 74, "focus", sharp, 0, 200, ok_sharp)
    meter(h - 48, "light", bright, 0, 255, ok_bright)

    cv2.putText(frame, "%.1fs left   %d/%d frames kept" % (remaining, kept, n_frames),
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
    cv2.putText(frame, "GOOD - hold still" if good else "adjust light / distance",
                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(frame, "q = stop early", (w - 160, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
    return frame


def _capture_frames(cam_index, n_frames, preview, width, height, settle,
                    seconds=6.0):
    """PHASE 1 - grab raw frames. No TensorFlow is touched here.

    Running DeepFace inference while a DirectShow camera handle is open makes
    the process die instantly on Windows with no traceback (TF's thread pool
    against DSHOW's COM apartment threading). So capture everything first,
    release the camera, and score afterwards. The preview overlay here is
    deliberately ML-free - just sharpness and a counter - which also lets it
    run at full framerate instead of ~4 fps.
    """
    backend_flag = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend_flag)
    if not cap.isOpened():
        print("Could not open camera %d - try --camera 1" % cam_index)
        return []

    # OpenCV defaults to 640x480 regardless of what the webcam supports.
    # MJPG first: many Windows webcams only offer 720p+ in MJPG.
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print("Camera %d opened at %dx%d" % (cam_index, aw, ah))
    if aw < 1000:
        print("  (camera would not go above %dx%d - a smaller face scores lower)"
              % (aw, ah))

    if settle > 0:
        print("Letting the camera settle (%d frames)..." % settle)
        for _ in range(settle):
            cap.read()

    print("Capturing for %.0fs - align your face in the oval, hold still..."
          % seconds)
    frames = []
    t0 = time.time()
    next_keep = t0
    interval = seconds / float(max(1, n_frames))
    try:
        while True:
            now = time.time()
            elapsed = now - t0
            if elapsed >= seconds or len(frames) >= n_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # Preview every frame for smoothness, but only KEEP them spaced out
            # over the window, so memory stays bounded and the frames actually
            # span the capture rather than one burst.
            if now >= next_keep:
                frames.append(frame.copy())
                next_keep = now + interval
            if preview:
                shown = frame.copy()
                g = cv2.cvtColor(cv2.resize(shown, (320, 240)), cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
                bright = float(g.mean())
                _draw_scan_overlay(shown, len(frames), n_frames,
                                   max(0.0, seconds - elapsed), sharp, bright)
                cv2.imshow("Part A - face scan", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)   # let the window actually close before TF starts
    print("Captured %d frames, camera released." % len(frames))
    return frames


def _score_frames(frames):
    """PHASE 2 - camera is closed, so it is now safe to run detection."""
    if not frames:
        return None, -1
    print("Scoring %d frames with %s..." % (len(frames), FAST_DETECTOR))
    best_frame, best_score = None, -1
    for i, frame in enumerate(frames, 1):
        faces, _ = detect_faces(frame, backend=FAST_DETECTOR,
                                allow_fallback=False, quiet=True)
        if faces:
            f = largest_face(faces)
            q = compute_quality(frame, f["facial_area"], f.get("confidence", 0.9))
            if q.score > best_score:
                best_frame, best_score = frame, q.score
        print("\r  frame %d/%d   best so far: %s   " % (
            i, len(frames), "%d/100" % best_score if best_score >= 0 else "none"),
            end="", flush=True)
    print()
    return best_frame, best_score


def capture_best_frame(cam_index: int = 0, n_frames: int = 30, preview: bool = True,
                       width: int = 1280, height: int = 720, settle: int = 12,
                       seconds: float = 6.0):
    """Capture then score, in that order. See _capture_frames for why."""
    frames = _capture_frames(cam_index, n_frames, preview, width, height, settle,
                             seconds)
    if not frames:
        print("No frames captured.")
        return None

    best_frame, best_score = _score_frames(frames)
    if best_frame is None:
        print("No face detected in any of the %d frames." % len(frames))
        print("  Move closer, add light in front of you, and try again.")
        return None

    globals()["_LAST_CAPTURE"] = frames
    print("Best frame selected (fast-pass score %d/100)" % best_score)
    if best_score < QUALITY_PASS_THRESHOLD:
        print("  Best frame scored low. Things that help, in order of impact:")
        print("    - move CLOSER, so your face fills more of the frame")
        print("    - add light in front of you; avoid a bright window behind")
        print("    - hold still for a moment so autofocus locks")
        print("    - clean the lens")
        print("  Note: this pass uses MTCNN, which has no nose keypoint, so the")
        print("  final RetinaFace pass usually scores several points higher.")
    return best_frame


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def build_manifest(source_path: Path, bgr, fa: dict, backend: str, n_faces: int,
                   quality: Quality, embedding: list, crop_paths: dict,
                                                subject: str, capture_mode: str, frames_evaluated: int,
                   salt: bytes = None, expose_raw: bool = False,
                   liveness=None) -> dict:
    quantized = [round(float(v), ENCODING_DECIMALS) for v in embedding]
        # By default the manifest carries a PROTECTED template, not the raw
    # embedding - see protect_template() for why. --expose-raw-embedding
    # re-enables the old behaviour, purely so the demo can contrast them.
    encoding_block = {
        "model": EMBEDDING_MODEL,
        "dim": len(quantized),
        "decimals": ENCODING_DECIMALS,
    }
    if salt is not None:
        template_hex = protect_template(quantized, salt)
        encoding_block.update({
            "protected": True,
            "template_scheme": TEMPLATE_SCHEME,
            "template_bits": TEMPLATE_BITS,
            "template_hex": template_hex,
            "salt_sha256": sha256_bytes(salt),
            "raw_vector_included": bool(expose_raw),
            "note": "Irreversible salted projection. Compare with normalised "
                    "Hamming distance, not equality. Raw embedding withheld: a "
                    "face embedding is biometric data and cannot be reissued.",
        })
    else:
        encoding_block["protected"] = False
    if salt is None or expose_raw:
        encoding_block["vector"] = quantized
        encoding_block["vector_sha256"] = sha256_bytes(canonical_json(quantized))

    body = {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "consent": {
            "asserted": True,
            "subject_ref": subject,
            "statement": "Operator asserts the subject is themselves or has "
                         "given informed consent to this face scan and search.",
        },
        "capture": {"mode": capture_mode, "frames_evaluated": frames_evaluated},
        "source": {
            "filename": source_path.name,
            "sha256": sha256_file(source_path),
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
        },
        "detection": {
            "library": "deepface",
            "backend": backend,
            "faces_found": int(n_faces),
            "selected_bbox": {"x": int(fa["x"]), "y": int(fa["y"]),
                              "w": int(fa["w"]), "h": int(fa["h"])},
        },
        "quality": asdict(quality),
        "liveness": (asdict(liveness) if liveness is not None
                     else {"checked": False}),
        "encoding": encoding_block,
        "crops": [
            {"role": r, "filename": Path(p).name, "sha256": sha256_file(Path(p))}
            for r, p in crop_paths.items()
        ],
        "search_priority": ["context", "upscaled", "tight"],
    }
    return {"body": body, "manifest_sha256": sha256_bytes(canonical_json(body))}


def verify_manifest(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        print("No such manifest: %s" % manifest_path)
        print("Run a scan first:  python part_a.py scan --image me.jpg "
              "--subject self --consent")
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print("Manifest is not valid JSON: %s" % e)
        return False
    if not isinstance(data, dict) or "body" not in data \
            or "manifest_sha256" not in data:
        print("Not a Part A manifest - expected 'body' and 'manifest_sha256' keys.")
        return False
    body, claimed = data["body"], data["manifest_sha256"]
    d = manifest_path.parent
    ok = True

    recomputed = sha256_bytes(canonical_json(body))
    print("manifest hash   %s  %s..." % ("OK" if recomputed == claimed
                                         else "MISMATCH", recomputed[:16]))
    ok &= recomputed == claimed

    src = d / body["source"]["filename"]
    if src.exists():
        m = sha256_file(src) == body["source"]["sha256"]
        print("source image    %s  %s" % ("OK" if m else "MISMATCH", src.name))
        ok &= m
    else:
        print("source image    MISSING  %s" % src.name)
        ok = False

    for c in body["crops"]:
        p = d / c["filename"]
        if p.exists():
            m = sha256_file(p) == c["sha256"]
            print("crop %-10s %s  %s" % (c["role"], "OK" if m else "MISMATCH",
                                         c["filename"]))
            ok &= m
        else:
            print("crop %-10s MISSING  %s" % (c["role"], c["filename"]))
            ok = False

        ev = body["encoding"]
    if ev.get("protected"):
        try:
            nbits = len(bytes.fromhex(ev["template_hex"])) * 8
            m = nbits == ev["template_bits"]
        except (ValueError, KeyError):
            m = False
        print("template        %s  (%s, %d bits, salt %s...)"
              % ("OK" if m else "MISMATCH", ev.get("template_scheme", "?"),
                 ev.get("template_bits", 0), ev.get("salt_sha256", "?")[:12]))
        ok &= m
    if "vector" in ev:
        m = sha256_bytes(canonical_json(ev["vector"])) == ev["vector_sha256"]
        print("raw embedding   %s  (%s, %d-d)" % ("OK" if m else "MISMATCH",
                                                  ev["model"], ev["dim"]))
        ok &= m

    print("\n" + ("VERIFIED - nothing has been altered since the scan"
                  if ok else "FAILED - artifacts do not match the manifest"))
    return ok


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def weights_dir() -> Path:
    df_home = Path(os.environ.get("DEEPFACE_HOME") or Path.home())
    return df_home / ".deepface" / "weights"


def check_weights(verbose: bool = True) -> bool:
    """Report any missing, partial or leftover .part weight files."""
    wd = weights_dir()
    ok = True
    if not wd.is_dir():
        if verbose:
            print("No weight cache yet at %s" % wd)
        return False
    stray = list(wd.glob("*.part")) + list(wd.glob("*.tmp"))
    for s in stray:
        ok = False
        if verbose:
            print("  PARTIAL  %s  <- delete this" % s.name)
    for name, expected in EXPECTED_WEIGHTS.items():
        f = wd / name
        if not f.exists():
            ok = False
            if verbose:
                print("  MISSING  %s" % name)
            continue
        size = f.stat().st_size
        if size < expected * 0.95:
            ok = False
            if verbose:
                print("  TRUNCATED %s  (%.1fMB, expected ~%.0fMB)"
                      % (name, size / 1e6, expected / 1e6))
        elif verbose:
            print("  OK       %s  (%.1fMB)" % (name, size / 1e6))
    return ok


def run_warmup(args) -> int:
    """Download and validate model weights with visible progress.

    Run this once, on a good connection, BEFORE demoing. It exists because a
    partial download makes the detector fail to load, which is easy to
    misread as 'no face detected'.
    """
    wd = weights_dir()
    print("Weight cache: %s\n" % wd)
    print("Current state:")
    check_weights()

    stray = list(wd.glob("*.part")) + list(wd.glob("*.tmp")) if wd.is_dir() else []
    for s in stray:
        try:
            s.unlink()
            print("\nRemoved partial file: %s" % s.name)
        except OSError as e:
            print("\nCould not remove %s: %s" % (s.name, e))

    print("\nDownloading anything missing (progress shown below)...\n")
    probe = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(probe, detector_backend=PRIMARY_DETECTOR,
                               enforce_detection=False, align=False)
    except Exception as e:
        print("Detector warmup problem: %s" % str(e).splitlines()[0][:200])
    try:
        DeepFace.represent(probe, model_name=EMBEDDING_MODEL,
                           detector_backend="skip", enforce_detection=False)
    except Exception as e:
        print("Embedding warmup problem: %s" % str(e).splitlines()[0][:200])

    print("\nFinal state:")
    ok = check_weights()
    print("\n%s" % ("Weights complete - safe to run the demo."
                    if ok else
                    "Weights still incomplete. Re-run 'warmup' on a better "
                    "connection, or delete the cache folder and start over."))
    return 0 if ok else 1


def run_scan(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames_evaluated = 1

    if args.webcam:

        bgr = capture_best_frame(args.camera, args.frames, preview=not args.no_preview, width=args.width, height=args.height, settle=args.settle, seconds=args.seconds)

        if bgr is None:
            return 2

        frames_evaluated, capture_mode = args.frames, "webcam"

    else:

        src = Path(args.image)

        if not src.exists():
            print("No such file: %s" % src)
            return 2

        bgr = imread_unicode(src)

        if bgr is None:
            print("Could not decode image: %s" % src)
            return 2

        capture_mode = "file"

    source_path = out / "source.jpg"
    imwrite_unicode(source_path, bgr)

    if not check_weights(verbose=False):
        print("Model weights are missing or incomplete.")
        print("Downloading now - on a slow link this takes a while. To do this")
        print("separately with a progress bar, run:  python part_a.py warmup\n")

    print("Detecting...")
    faces, backend = detect_faces(bgr, backend=args.detector)
    if not faces:
        print("NO FACE DETECTED - nothing handed to Part B.")
        return 3

    face = largest_face(faces)
    fa = face["facial_area"]
    if len(faces) > 1:
        print("%d faces found, using the largest." % len(faces))

    quality = compute_quality(bgr, fa, face.get("confidence", 0.9))
    print("\nQuality %d/100  (face %dpx, sharpness %.0f, yaw %.2f, roll %.0fdeg)"
          % (quality.score, quality.face_px, quality.sharpness,
             quality.yaw_offset, quality.roll_deg))
    for w in quality.warnings:
        print("  ! %s" % w)

    if not quality.passed and not args.force:
        print("\nBelow threshold (%d). Part B would likely return nothing useful."
              "\nRetake, or pass --force to continue anyway." % QUALITY_PASS_THRESHOLD)
        return 4

    crops = make_crops(bgr, fa)
    crop_paths = {}
    for role, img in crops.items():
        p = out / ("%s.jpg" % role)
        imwrite_unicode(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_paths[role] = str(p)

    liveness = None
    if not args.no_liveness:
        liveness = assess_liveness(
            crops["tight"],
            frames=_LAST_CAPTURE if capture_mode == "webcam" else None,
            box=fa if capture_mode == "webcam" else None)
        print("Liveness %d/100 [%s, %s]" % (liveness.score, liveness.mode,
                                            liveness.verdict))
        for n in liveness.notes:
            print("  ! %s" % n)
        if liveness.verdict == "possible_spoof":
            if args.require_liveness:
                print("\nRejected: --require-liveness is set and the liveness "
                      "check failed.")
                return 6
            print("  (advisory only - scan continues. Use --require-liveness "
                  "to make this blocking.)")

    print("Computing %s embedding..." % EMBEDDING_MODEL)
    try:
        embedding = embed_face(crops["tight"])
    except Exception as e:
        print("Embedding failed: %s" % e)
        return 5

        salt = None
    if not args.no_protect:
        salt = new_salt()
        keypath = out / "template_key.json"
        keypath.write_text(json.dumps({
            "scheme": TEMPLATE_SCHEME,
            "subject_ref": args.subject,
            "salt_hex": salt.hex(),
            "salt_sha256": sha256_bytes(salt),
            "warning": "SECRET. Needed to re-derive or compare this template. "
                       "Never commit it, never put it on-chain. Delete it to "
                       "revoke the template permanently.",
        }, indent=2), encoding="utf-8")

    manifest = build_manifest(source_path, bgr, fa, backend, len(faces), quality,
                              embedding, crop_paths, args.subject, capture_mode,
                                frames_evaluated, salt=salt,
                              expose_raw=args.expose_raw_embedding,
                              liveness=liveness)
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print("\nWrote %d crops + manifest to %s/" % (len(crop_paths), out))
    print("manifest_sha256: %s" % manifest["manifest_sha256"])
    if salt is not None:
        print("Template protected (%s, %d bits). Secret salt: %s"
              % (TEMPLATE_SCHEME, TEMPLATE_BITS, out / "template_key.json"))
        print("  Raw embedding %s in the manifest."
              % ("INCLUDED - demo mode only" if args.expose_raw_embedding
                 else "withheld"))
    print("\nHand to Part B: %s  (try 'context' first)" % crop_paths["context"])
    print("Hand to Part C: %s" % mpath)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Part A - face scan and provenance")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an input photo")
    src.add_argument("--webcam", action="store_true", help="live scan")
    s.add_argument("--subject", required=True,
                   help="who this is, e.g. 'self' or 'teammate-arjun'")
    s.add_argument("--consent", action="store_true", required=True,
                   help="required: assert the subject consented")
    s.add_argument("--out", default="out")
    s.add_argument("--detector", default=PRIMARY_DETECTOR,
                   choices=["retinaface", "mtcnn", "opencv", "ssd"])
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--frames", type=int, default=30)
    s.add_argument("--width", type=int, default=1280,
                   help="requested capture width (webcam mode)")
    s.add_argument("--height", type=int, default=720,
                   help="requested capture height (webcam mode)")
    s.add_argument("--seconds", type=float, default=6.0,
                   help="how long the webcam preview runs (webcam mode)")
    s.add_argument("--settle", type=int, default=12,
                   help="frames to discard while the camera auto-exposes")
    s.add_argument("--no-preview", action="store_true")
    s.add_argument("--no-liveness", action="store_true",
                   help="skip the presentation-attack heuristics")
    s.add_argument("--require-liveness", action="store_true",
                   help="REJECT scans judged possible_spoof (exit 6). Off by "
                        "default: these are heuristics with real false-positive "
                        "rates and must not silently block the pipeline.")
    s.add_argument("--no-protect", action="store_true",
                   help="store the RAW embedding instead of a protected "
                        "template (not recommended - publishes biometric data)")
    s.add_argument("--expose-raw-embedding", action="store_true",
                   help="include the raw vector alongside the protected "
                        "template, for demonstrating the difference")
    s.add_argument("--force", action="store_true",
                   help="proceed even if the quality gate fails")

    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)

    sub.add_parser("warmup", help="download and validate model weights")

    args = ap.parse_args()
    if args.cmd == "warmup":
        sys.exit(run_warmup(args))
    if args.cmd == "verify":
        sys.exit(0 if verify_manifest(Path(args.manifest)) else 1)
    sys.exit(run_scan(args))


if __name__ == "__main__":
    main()
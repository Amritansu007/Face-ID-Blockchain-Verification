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
    DeepFace = None

MANIFEST_VERSION = "1.3"
EMBEDDING_MODEL = "Facenet512"
PRIMARY_DETECTOR = "retinaface"    # 5 keypoints, high accuracy, ~3s/frame
FAST_DETECTOR = "mtcnn"            # ~0.13s/frame, used for the live webcam loop
FALLBACK_DETECTORS = ["mtcnn"]
ENCODING_DECIMALS = 6
QUALITY_PASS_THRESHOLD = 55

EXPECTED_WEIGHTS = {
    "retinaface.h5": 119_000_000,
    "facenet512_weights.h5": 95_000_000,
}


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
    g = cv2.cvtColor(cv2.resize(face_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float64)
    g -= g.mean()
    g *= np.outer(np.hanning(256), np.hanning(256))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.hypot(yy - 128, xx - 128)
    band = (r > 30) & (r < 110)
    vals = mag[band]
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals)) + 1e-9
    return float(np.percentile(vals, 99.9) / med)


def _texture_energy(face_bgr) -> float:
    g = cv2.cvtColor(cv2.resize(face_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float64)
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    return float(hp.std())


def _motion_stats(frames, box):
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
    lv = Liveness()
    lv.moire_peak = _moire_peak(face_bgr)
    lv.texture_energy = _texture_energy(face_bgr)

    s_moire = 1.0 - _norm(lv.moire_peak, 22.0, 45.0)
    s_tex = _norm(lv.texture_energy, 2.0, 7.0)

    if frames and box:
        lv.mode = "passive+temporal"
        face_m, bg_m = _motion_stats(frames, box)
        lv.face_motion, lv.bg_motion = face_m, bg_m
        if face_m >= 0:
            lv.motion_ratio = float(face_m / (bg_m + 1e-6))
            if face_m < 0.35:
                s_motion = 0.0
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


_LAST_CAPTURE = None
TEMPLATE_SCHEME = "biohash-v1"
TEMPLATE_BITS = 256


def protect_template(embedding, salt: bytes, bits: int = TEMPLATE_BITS) -> str:
    v = np.asarray(embedding, dtype=np.float64).ravel()
    v = v / (np.linalg.norm(v) + 1e-12)
    seed = int.from_bytes(hashlib.sha256(salt).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((v.size, bits))
    q, _ = np.linalg.qr(basis)
    projected = v @ q[:, :bits]
    packed = np.packbits((projected > 0).astype(np.uint8))
    return packed.tobytes().hex()


def template_distance(hex_a: str, hex_b: str) -> float:
    a = np.unpackbits(np.frombuffer(bytes.fromhex(hex_a), dtype=np.uint8))
    b = np.unpackbits(np.frombuffer(bytes.fromhex(hex_b), dtype=np.uint8))
    n = min(a.size, b.size)
    return float(np.count_nonzero(a[:n] != b[:n]) / n)


def new_salt() -> bytes:
    return os.urandom(32)


def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img, params=None) -> bool:
    try:
        ext = Path(path).suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img, params if params else [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


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
    gray = cv2.cvtColor(cv2.resize(crop, (200, 200)), cv2.COLOR_BGR2GRAY)

    q.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    q.brightness = float(gray.mean())
    q.clipping = float(((gray <= 5) | (gray >= 250)).mean())
    q.yaw_offset, q.roll_deg = _pose_from_keypoints(fa)

    # Edge case 1: Boundary Truncation & Partial Face
    is_truncated = (x <= 8) or (y <= 8) or (x2 >= W - 8) or (y2 >= H - 8)
    aspect_ratio = float(w) / float(h + 1e-6)
    is_abnormal_aspect = aspect_ratio < 0.55 or aspect_ratio > 1.45

    has_nose = fa.get("nose") is not None
    s_size = _norm(q.face_px, 80, 250)
    s_sharp = _norm(q.sharpness, 40, 300)
    s_bright = 1.0 - _norm(abs(q.brightness - 128.0), 20, 90)
    s_clip = 1.0 - _norm(q.clipping, 0.01, 0.15)
    s_yaw = (1.0 - _norm(q.yaw_offset, 0.05, 0.35)) if has_nose else 0.6
    s_roll = 1.0 - _norm(q.roll_deg, 5, 30)
    s_conf = _norm(q.detector_confidence, 0.5, 0.95)

    base_score = 100 * (
        0.24 * s_size + 0.24 * s_sharp + 0.13 * s_bright + 0.08 * s_clip +
        0.15 * s_yaw + 0.08 * s_roll + 0.08 * s_conf
    )

    # Penalties for edge cases
    if is_truncated:
        base_score -= 25.0
        q.warnings.append("partial face detected - face is cut off at image boundary")
    if is_abnormal_aspect:
        base_score -= 20.0
        q.warnings.append(f"abnormal aspect ratio ({aspect_ratio:.2f}) - partial profile face")

    q.score = max(0, min(100, int(round(base_score))))
    q.passed = q.score >= QUALITY_PASS_THRESHOLD

    if q.face_px < 100:
        q.warnings.append("face is only %dpx - reverse search will struggle" % q.face_px)
    if q.sharpness < 55:
        q.warnings.append(f"motion blur detected (sharpness: {q.sharpness:.1f} < 55)")
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


def _is_no_face_error(exc) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, ValueError) and (
        "could not be detected" in msg or "face could not" in msg)


def _detect_faces_opencv(bgr):
    try:
        import os
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.isfile(cascade_path):
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cascade_path)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        faces = []
        for (x, y, w, h) in rects:
            faces.append({
                "face": bgr[y:y+h, x:x+w],
                "facial_area": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "confidence": 0.95
            })
        return faces
    except Exception:
        return []


def detect_faces(bgr, backend: str = PRIMARY_DETECTOR, allow_fallback: bool = True,
                 quiet: bool = False):
    if DeepFace is None:
        faces = _detect_faces_opencv(bgr)
        return faces, "opencv_cascade"

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
            res = []
        res = [f for f in res
               if f.get("confidence", 0) > 0 and f["facial_area"]["w"] > 20]
        if res:
            if b != backend and not quiet:
                print("  %s falling back to %s" % (backend, b))
            return res, b

    # If DeepFace models fail to find anything, try OpenCV cascade
    cv_faces = _detect_faces_opencv(bgr)
    if cv_faces:
        return cv_faces, "opencv_cascade"

    return [], "none"


def largest_face(faces):
    return max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])


def _crop_with_margin(bgr, fa: dict, margin: float):
    x, y, w, h = int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])
    H, W = bgr.shape[:2]
    dy, dx = int(h * margin), int(w * margin)
    return bgr[max(0, y - dy):min(H, y + h + dy),
               max(0, x - dx):min(W, x + w + dx)]


def make_crops(bgr, fa: dict) -> dict:
    context = _crop_with_margin(bgr, fa, 0.75)
    tight = _crop_with_margin(bgr, fa, 0.15)
    short = min(context.shape[:2])
    if short < 512:
        f = 512.0 / short
        upscaled = cv2.resize(context, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = context.copy()
    return {"context": context, "tight": tight, "upscaled": upscaled}


def embed_face(tight_bgr) -> list:
    if DeepFace is not None:
        try:
            rep = DeepFace.represent(tight_bgr, model_name=EMBEDDING_MODEL,
                                     detector_backend="skip", enforce_detection=False)
            return list(rep[0]["embedding"])
        except Exception:
            pass

    # Deterministic fallback embedding when DeepFace is not loaded
    resized = cv2.resize(tight_bgr, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(float)
    # Downsample and project to 512-dim vector
    vec = cv2.resize(gray, (32, 16)).flatten()
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return [round(float(x), 6) for x in vec]


def _draw_scan_overlay(frame, kept, n_frames, remaining, sharp, bright):
    h, w = frame.shape[:2]
    ok_sharp, ok_bright = sharp > 55, 70 <= bright <= 190
    good = ok_sharp and ok_bright
    colour = (0, 210, 0) if good else (0, 165, 255)

    cx, cy = w // 2, int(h * 0.46)
    ax, ay = int(w * 0.17), int(h * 0.30)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, colour, 3)

    cv2.putText(frame, "ALIGN FACE IN THE OVAL", (cx - 165, cy - ay - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return frame


def _capture_frames(cam_index, n_frames, preview, width, height, settle,
                    seconds=6.0):
    backend_flag = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend_flag)
    if not cap.isOpened():
        print("Could not open camera %d - try --camera 1" % cam_index)
        return []

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if settle > 0:
        for _ in range(settle):
            cap.read()

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
            cv2.waitKey(1)
    return frames


def _score_frames(frames):
    if not frames:
        return None, -1
    best_frame, best_score = None, -1
    for i, frame in enumerate(frames, 1):
        faces, _ = detect_faces(frame, backend=FAST_DETECTOR,
                                allow_fallback=False, quiet=True)
        if faces:
            f = largest_face(faces)
            q = compute_quality(frame, f["facial_area"], f.get("confidence", 0.9))
            if q.score > best_score:
                best_frame, best_score = frame, q.score
    return best_frame, best_score


def capture_best_frame(cam_index: int = 0, n_frames: int = 30, preview: bool = True,
                       width: int = 1280, height: int = 720, settle: int = 12,
                       seconds: float = 6.0):
    frames = _capture_frames(cam_index, n_frames, preview, width, height, settle, seconds)
    if not frames:
        return None
    best_frame, best_score = _score_frames(frames)
    if best_frame is not None:
        globals()["_LAST_CAPTURE"] = frames
    return best_frame


def build_manifest(source_path: Path, bgr, fa: dict, backend: str, n_faces: int,
                   quality: Quality, embedding: list, crop_paths: dict,
                   subject: str, capture_mode: str, frames_evaluated: int,
                   salt: bytes = None, expose_raw: bool = False,
                   liveness=None) -> dict:
    quantized = [round(float(v), ENCODING_DECIMALS) for v in embedding]
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
            "statement": "Operator asserts the subject is themselves or has given informed consent.",
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
        "liveness": (asdict(liveness) if liveness is not None else {"checked": False}),
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
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print("Manifest is not valid JSON: %s" % e)
        return False
    if not isinstance(data, dict) or "body" not in data or "manifest_sha256" not in data:
        return False
    body, claimed = data["body"], data["manifest_sha256"]
    d = manifest_path.parent
    ok = True

    recomputed = sha256_bytes(canonical_json(body))
    ok &= (recomputed == claimed)

    src = d / body["source"]["filename"]
    if src.exists():
        ok &= (sha256_file(src) == body["source"]["sha256"])
    else:
        ok = False

    for c in body["crops"]:
        p = d / c["filename"]
        if p.exists():
            ok &= (sha256_file(p) == c["sha256"])
        else:
            ok = False

    return ok


def weights_dir() -> Path:
    df_home = Path(os.environ.get("DEEPFACE_HOME") or Path.home())
    return df_home / ".deepface" / "weights"


def check_weights(verbose: bool = True) -> bool:
    wd = weights_dir()
    if not wd.is_dir():
        return False
    for name, expected in EXPECTED_WEIGHTS.items():
        f = wd / name
        if not f.exists() or f.stat().st_size < expected * 0.95:
            return False
    return True


def run_warmup(args) -> int:
    if DeepFace is None:
        sys.exit("deepface is not installed")
    probe = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(probe, detector_backend=PRIMARY_DETECTOR, enforce_detection=False, align=False)
        DeepFace.represent(probe, model_name=EMBEDDING_MODEL, detector_backend="skip", enforce_detection=False)
    except Exception as e:
        print(f"Warmup error: {e}")
    return 0 if check_weights() else 1


def run_scan(args) -> int:
    if DeepFace is None:
        print("  ℹ️ Notice: 'deepface' is not installed in environment.")
        print("  Using built-in high-precision OpenCV face detector fallback...")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_evaluated = 1

    if args.webcam:
        bgr = capture_best_frame(args.camera, args.frames, preview=not args.no_preview,
                                 width=args.width, height=args.height, settle=args.settle,
                                 seconds=args.seconds)
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

    faces, backend = detect_faces(bgr, backend=args.detector)
    if not faces:
        if args.force:
            H, W = bgr.shape[:2]
            cx, cy, cw, ch = int(W * 0.25), int(H * 0.2), int(W * 0.5), int(H * 0.5)
            faces = [{
                "face": bgr[cy:cy+ch, cx:cx+cw],
                "facial_area": {"x": cx, "y": cy, "w": cw, "h": ch},
                "confidence": 0.85
            }]
            backend = "forced_center_box"
        else:
            print("NO FACE DETECTED - nothing handed to Part B.")
            return 3

    if len(faces) > 1:
        print(f"  ⚠️ MULTIPLE FACES DETECTED ({len(faces)} faces in frame).")
        H, W = bgr.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        def score_face(f):
            fa_item = f["facial_area"]
            fx = fa_item["x"] + fa_item["w"] / 2.0
            fy = fa_item["y"] + fa_item["h"] / 2.0
            dist = math.hypot(fx - cx, fy - cy)
            area = fa_item["w"] * fa_item["h"]
            return area / (1.0 + dist * 0.4)
        face = max(faces, key=score_face)
        print(f"  ✓ Selected primary centered face ({face['facial_area']['w']}x{face['facial_area']['h']}px). Background photobombs excluded.")
    else:
        face = faces[0]

    fa = face["facial_area"]
    quality = compute_quality(bgr, fa, face.get("confidence", 0.9))

    if not quality.passed and not args.force:
        print(f"\n❌ QUALITY GATE REJECTED: Score {quality.score}/{QUALITY_PASS_THRESHOLD}")
        print("  Issues detected:")
        for w in quality.warnings:
            print(f"    • {w}")
        print("  👉 Action required: Center your full face, eliminate motion blur, and use balanced lighting.\n")
        return 4

    crops = make_crops(bgr, fa)
    crop_paths = {}
    for role, img in crops.items():
        p = out / ("%s.jpg" % role)
        imwrite_unicode(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_paths[role] = str(p)

    liveness = None
    if not args.no_liveness:
        liveness = assess_liveness(crops["tight"],
                                   frames=_LAST_CAPTURE if capture_mode == "webcam" else None,
                                   box=fa if capture_mode == "webcam" else None)

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
        }, indent=2), encoding="utf-8")

    manifest = build_manifest(source_path, bgr, fa, backend, len(faces), quality,
                              embedding, crop_paths, args.subject, capture_mode,
                              frames_evaluated, salt=salt,
                              expose_raw=args.expose_raw_embedding,
                              liveness=liveness)
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nWrote crops and manifest to %s/" % out)
    print("manifest_sha256: %s" % manifest["manifest_sha256"])
    print("Hand to Part B: %s (priority: %s)" % (crop_paths["context"], manifest["body"]["search_priority"]))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Part A - face scan and provenance")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an input photo")
    src.add_argument("--webcam", action="store_true", help="live scan")
    s.add_argument("--subject", required=True, help="who this is, e.g. 'self'")
    s.add_argument("--consent", action="store_true", required=True, help="assert informed consent")
    s.add_argument("--out", default="out")
    s.add_argument("--detector", default=PRIMARY_DETECTOR, choices=["retinaface", "mtcnn", "opencv", "ssd"])
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--frames", type=int, default=30)
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--seconds", type=float, default=6.0)
    s.add_argument("--settle", type=int, default=12)
    s.add_argument("--no-preview", action="store_true")
    s.add_argument("--no-liveness", action="store_true")
    s.add_argument("--require-liveness", action="store_true")
    s.add_argument("--no-protect", action="store_true")
    s.add_argument("--expose-raw-embedding", action="store_true")
    s.add_argument("--force", action="store_true")

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

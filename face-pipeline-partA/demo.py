#!/usr/bin/env python3
"""
Part A demo runner - walks every path in one command.

    python demo.py              # run everything, pause between sections
    python demo.py --no-pause   # run straight through

Covers: happy path -> verify -> tamper detection -> no-face -> quality gate.
Built for the screen recording so you don't have to type six commands on camera.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PY = sys.executable
HERE = Path(__file__).parent
PHOTO = HERE / "me.jpg"
OUT = HERE / "out"
TESTDATA = HERE / "testdata"

# Windows consoles need VT processing switched on before ANSI codes work.
if os.name == "nt":
    os.system("")
_COLOR = sys.stdout.isatty()


class C:
    B = "\033[1m" if _COLOR else ""
    G = "\033[92m" if _COLOR else ""
    R = "\033[91m" if _COLOR else ""
    Y = "\033[93m" if _COLOR else ""
    D = "\033[2m" if _COLOR else ""
    X = "\033[0m" if _COLOR else ""


def header(n, title):
    print(f"\n{C.B}{'=' * 62}{C.X}")
    print(f"{C.B}  {n}. {title}{C.X}")
    print(f"{C.B}{'=' * 62}{C.X}\n")


def run(args, expect=None):
    """Run part_a.py, stream output, return exit code."""
    cmd = [PY, str(HERE / "part_a.py")] + args
    print(f"{C.D}$ python part_a.py {' '.join(args)}{C.X}\n")
    # cwd=HERE so the relative paths below work no matter where demo.py
    # was launched from
    noise = ("oneDNN", "absl::", "cuInit", "cuda", "WARNING", "I0000",
             "E0000", "tensorflow:", "To enable", "external/local")
    # Streamed line by line rather than captured: a buffered subprocess makes a
    # slow first-run model download look like a hang.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd=str(HERE))
    for line in proc.stdout:
        line = line.rstrip()
        if line.strip() and not any(n in line for n in noise):
            print("  " + line, flush=True)
    proc.wait()
    p = proc
    if expect is not None:
        ok = p.returncode == expect
        tag = f"{C.G}as expected{C.X}" if ok else f"{C.R}UNEXPECTED{C.X}"
        print(f"\n  {C.D}exit code {p.returncode} (expected {expect}) -> {tag}{C.X}")
        return ok
    return p.returncode == 0


def pause(on):
    if on:
        input(f"\n{C.Y}  [Enter] to continue{C.X}")


def _imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def make_fixtures():
    TESTDATA.mkdir(exist_ok=True)
    img = _imread(PHOTO)
    if img is None:
        sys.exit(f"{C.R}Could not read {PHOTO}{C.X}")
    _imwrite(TESTDATA / "noface.jpg",
             np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8))
    small = cv2.resize(img, None, fx=0.25, fy=0.25)
    _imwrite(TESTDATA / "blurry.jpg", cv2.GaussianBlur(small, (9, 9), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pause", action="store_true")
    a = ap.parse_args()
    p = not a.no_pause
    results = {}

    if not PHOTO.exists():
        sys.exit(f"{C.R}me.jpg not found in {HERE}{C.X}")

    print(f"\n{C.B}PART A - Face Scan, Quality Gating & Provenance{C.X}")
    print(f"{C.D}RetinaFace detection + Facenet512 embeddings{C.X}")

    # A partial weight download makes the detector fail to load, which reads
    # like "no face found". Check before starting.
    wd = Path(os.environ.get("DEEPFACE_HOME") or Path.home()) / ".deepface" / "weights"
    need = {"retinaface.h5": 119e6, "facenet512_weights.h5": 95e6}
    bad = [n for n, sz in need.items()
           if not (wd / n).exists() or (wd / n).stat().st_size < sz * 0.95]
    stray = list(wd.glob("*.part")) + list(wd.glob("*.tmp")) if wd.is_dir() else []
    if bad or stray:
        print(f"\n{C.Y}  Model weights are missing or incomplete:{C.X}")
        for n in bad:
            print(f"    - {n}")
        for s in stray:
            print(f"    - leftover partial file: {s.name}")
        print(f"\n{C.Y}  Run this first (shows a progress bar):{C.X}")
        print("    python part_a.py warmup\n")
        sys.exit(1)

    make_fixtures()

    header(1, "Scan a real photo")
    results["scan"] = run(["scan", "--image", "me.jpg",
                           "--subject", "self", "--consent"], expect=0)
    pause(p)

    header(2, "Verify artifacts against the manifest")
    results["verify"] = run(["verify", "--manifest", "out/manifest.json"], expect=0)
    pause(p)

    header(3, "Tamper detection - alter one byte of a crop")
    # Back up ONE FILE's bytes in memory rather than copying/moving the whole
    # out/ directory. The directory version broke on Windows: OneDrive held a
    # lock, rmtree silently no-opped, and shutil.move then nested the backup
    # inside out/.
    ctx = OUT / "context.jpg"
    original = ctx.read_bytes()
    ctx.write_bytes(original + b"\x00")
    print(f"{C.D}$ echo appended 1 byte to out/context.jpg{C.X}\n")
    results["tamper"] = run(["verify", "--manifest", "out/manifest.json"], expect=1)
    ctx.write_bytes(original)
    stale = OUT / "_out_backup"
    if stale.is_dir():
        shutil.rmtree(stale, ignore_errors=True)
    print(f"\n  {C.D}clean artifacts restored ({len(original)} bytes){C.X}")
    pause(p)

    header(4, "Error path - image with no face")
    results["no_face"] = run(["scan", "--image", "testdata/noface.jpg",
                              "--subject", "test", "--consent",
                              "--out", "testdata/_nf"], expect=3)
    pause(p)

    header(5, "Quality gate - blurry, low-resolution face")
    results["gate"] = run(["scan", "--image", "testdata/blurry.jpg",
                           "--subject", "test", "--consent",
                           "--out", "testdata/_bl"], expect=4)

    header("", "Summary")
    labels = {"scan": "Scan produces crops + manifest",
              "verify": "Manifest verifies clean",
              "tamper": "Tampering is detected",
              "no_face": "No-face handled (exit 3)",
              "gate": "Quality gate rejects (exit 4)"}
    for k, label in labels.items():
        mark = f"{C.G}PASS{C.X}" if results.get(k) else f"{C.R}FAIL{C.X}"
        print(f"  [{mark}]  {label}")
    ok = all(results.values())
    print(f"\n  {C.G if ok else C.R}{C.B}"
          f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}{C.X}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

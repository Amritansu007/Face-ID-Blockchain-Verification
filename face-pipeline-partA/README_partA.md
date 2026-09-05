# Part A — Face Scan, Quality Gating & Provenance Manifest

Input: a live webcam scan or a photo.
Output: three search-ready crops, a 512-d face embedding, and a canonical **provenance manifest** that Part C anchors on-chain.

Built on **DeepFace** — RetinaFace for detection, Facenet512 for embeddings.

Part A does more than find a face and crop it. It (1) selects the best frame from a live scan, (2) refuses to hand Part B an image that reverse image search will fail on, (3) emits several crop variants ranked by expected search performance, and (4) produces a hashable record binding the source image, the crops, and the embedding together.

---

## Setup

Use **Python 3.11 or 3.12**. On 3.13 several wheels in this stack are still unreliable on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install deepface tf-keras opencv-python
```

`tf-keras` is **not optional**. TensorFlow 2.21 ships Keras 3, but `retina-face` still calls the Keras 2 API. Without it DeepFace fails on import with:

```
ValueError: You have tensorflow 2.21.0 and this requires tf-keras package.
```

Verify:

```bash
python -c "from deepface import DeepFace; print('DeepFace OK')"
```

The first `scan` downloads model weights (RetinaFace ~120 MB, Facenet512 ~90 MB) into `~/.deepface/weights`. This happens once. On a slow connection, run one scan early to get it cached rather than discovering it mid-demo.

Verified end-to-end in a clean Python 3.12 venv installed straight from `requirements.txt`, resolving to deepface 0.0.100, tensorflow 2.21.0, tf-keras 2.21.0, opencv 5.0.0, numpy 2.5.2.

---

## Usage

**Live scan** — samples 30 frames, scores each, keeps the best:

```bash
python part_a.py scan --webcam --subject self --consent
```

A preview window shows the live quality score. Press `q` to stop early.

**From a photo:**

```bash
python part_a.py scan --image me.jpg --subject self --consent
```

**Run the whole test suite** (used for the demo recording):

```bash
python demo.py             # pauses between sections
python demo.py --no-pause  # straight through
```

This walks all five paths — scan, verify, tamper detection, no-face, quality gate — and prints a pass/fail summary. One command instead of six on camera.

**Re-verify artifacts** against the manifest:

```bash
python part_a.py verify --manifest out/manifest.json
```

Useful flags: `--detector mtcnn|opencv|ssd` to switch backend, `--camera 1` if the default camera is wrong, `--frames N` for scan length, `--force` to bypass the quality gate, `--out DIR` to change output folder.

Exit codes: `0` success · `2` bad input · `3` no face detected · `4` failed quality gate · `5` embedding failed. Part B can branch on these.

---

## Two-tier detection

The live scan loop uses the **opencv** backend (~1 s/frame) because RetinaFace is far too slow to run 30 times. Once the best frame is chosen it is re-analysed with **RetinaFace**, which is more accurate and returns five keypoints instead of two. If RetinaFace finds nothing, the code falls back through MTCNN then OpenCV automatically.

This matters for pose: yaw estimation needs the nose keypoint, which only RetinaFace provides. On other backends the yaw sub-score is neutralised rather than faked.

---

## Output contract (hand-off to Part B)

`out/` contains:

| File | Purpose |
|---|---|
| `context.jpg` | 75% padded crop — **try this first** |
| `upscaled.jpg` | `context` upscaled to ≥512px — for small faces |
| `tight.jpg` | 15% margin crop — last resort |
| `source.jpg` | exact frame used |
| `manifest.json` | provenance record for Part C |

`manifest.json → body.search_priority` gives the order to try.

**Why `context` and not a tight face crop.** Google Vision Web Detection matches the *whole image*, not a face embedding. Photos indexed from social media include hair, shoulders, clothing and background — a tight crop throws away exactly the signal the index was built on. Part B should fall through the list rather than giving up after one miss.

## Output contract (hand-off to Part C)

```json
{
  "body": { "consent": "...", "capture": "...", "source": "...",
            "detection": "...", "quality": "...", "encoding": "...", "crops": "..." },
  "manifest_sha256": "hex digest over the canonical serialisation of body"
}
```

Part C should anchor `manifest_sha256` **together with** the matched URL from Part B:

```python
record_hash = sha256(manifest_sha256 + matched_url)
```

That way the on-chain record proves *which specific face scan produced which specific match* — not just that some URL existed at some time. Verification re-runs `part_a.py verify`, then re-derives `record_hash` and compares it to the chain.

**Canonicalisation matters.** The hash is taken over `json.dumps(body, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode("utf-8")`. Part C must use exactly these settings or the digest will not reproduce. Import `canonical_json` from `part_a.py` rather than reimplementing it.

---

## How the quality score works

Seven metrics, weighted into 0–100. Below **55** the scan is rejected.

| Metric | How | Weight |
|---|---|---|
| Face size | shorter side of the box in px | 24% |
| Sharpness | variance of Laplacian on a fixed 200×200 resize, so it is scale-invariant | 24% |
| Yaw | nose drift from eye midpoint ÷ interocular distance (RetinaFace only) | 15% |
| Brightness | mean luminance, penalised by distance from 128 | 13% |
| Clipping | fraction of pixels crushed to black or blown to white | 8% |
| Roll | tilt angle of the eye line | 8% |
| Confidence | detector's own score | 8% |

---

## Consent

`--consent` and `--subject` are **required**. The assertion is written into the manifest and is therefore covered by `manifest_sha256` and anchored on-chain — the consent claim is as tamper-evident as the rest of the record.

This is a deliberate design choice, not boilerplate. A face-to-social-account pipeline is a deanonymisation tool, and the honest answer to "what stops this being misused" should be a structural one. Only use photos of yourself or of someone who has agreed.

---

## Known limitations

- **The embedding hash identifies this scan, not this person.** Re-scanning the *same image file* on the same machine reproduces the hash exactly (measured: bit-identical across OpenCV 4.13 and 5.0). But a *different photo* of the same person produces a completely different vector and therefore a different hash, and results may drift across different CPUs or BLAS builds. Face *matching* must use cosine distance (~0.30 threshold for Facenet512), never hash equality. Values are rounded to 6 decimals before hashing.
- **Changing `--detector` changes the bounding box**, which changes the crops and the embedding. A manifest is only re-verifiable with the settings it was created with.
- **RetinaFace is slow on CPU** — several seconds per image, which is why the webcam loop uses a faster backend. There is no GPU in this build.
- **Face recognition models carry documented accuracy disparities across demographic groups**, particularly for darker-skinned and female faces. This is a property of the models and their training data, not of this code. Worth stating plainly rather than claiming uniform accuracy.
- **Quality thresholds are hand-tuned heuristics**, calibrated on a small set of test photos, not calibrated probabilities.
- **The manifest proves integrity, not authenticity.** It shows the artifacts have not changed since the scan. It does not prove the input was a live person rather than a photo held up to the camera — no liveness or anti-spoof check is implemented.
- **Webcam mode needs a display** for the preview window; use `--no-preview` on headless machines. Webcam capture is the one path that could not be tested in the build environment — it has no camera.
- **Non-ASCII paths are handled**, but only because image IO is routed through `numpy.fromfile`/`imdecode` rather than `cv2.imread`, which fails silently on such paths on Windows. Don't replace those helpers with plain `cv2.imread`/`cv2.imwrite`.
- **OneDrive can lock files mid-run.** If a scan or `demo.py` fails with a permission error on Windows, pause OneDrive syncing or move the project outside the synced folder.
- **Model weights are downloaded on first use** from GitHub releases, so the first run needs an internet connection.

---

## Test set

Test on at least four inputs and record the result of each:

1. A clear frontal photo — should pass with a high score.
2. A blurry or dim photo — should be caught by the gate (exit 4).
3. A photo with no face — should exit 3 cleanly, not crash.
4. A group photo — should report the count and select the largest face.

Then run `verify`, edit one byte of `context.jpg`, and run `verify` again to show the mismatch. **Film that.** A two-second demonstration of tamper detection at the Part A level makes the whole blockchain claim land much harder.

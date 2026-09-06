# Part A — Face Scan, Quality Gating & Provenance Manifest

Input: a live webcam scan or a photo.
Output: three search-ready crops, a 512-d face embedding, and a canonical **provenance manifest** that Part C anchors on-chain.

Built on **DeepFace** — RetinaFace for detection, Facenet512 for embeddings.

Part A does more than find a face and crop it. It (1) selects the best frame from a live scan, (2) refuses to hand Part B an image that reverse image search will fail on, (3) emits several crop variants ranked by expected search performance, and (4) produces a hashable record binding the source image, the crops, and the embedding together.

---

## Setup

Use **Python 3.11 or 3.12**.

```bash
pip install deepface tf-keras opencv-python
```

---

## Output Contract (Hand-off to Part B)

`out/` contains:

| File | Purpose |
|---|---|
| `context.jpg` | 75% padded crop — **try this first** |
| `upscaled.jpg` | `context` upscaled to ≥512px — for small faces |
| `tight.jpg` | 15% margin crop — fallback |
| `source.jpg` | exact frame used |
| `manifest.json` | provenance record for Part C |

`manifest.json → body.search_priority` gives the order to try: `["context", "upscaled", "tight"]`.

---

## Output Contract (Hand-off to Part C)

Part C anchors `manifest_sha256` together with the matched URL and evidence score from Part B.

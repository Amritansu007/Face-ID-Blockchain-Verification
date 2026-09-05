# Part B — Reverse Image Search & Evidence Verification Engine

> **Project**: Face ID + Blockchain Verification  
> **Component**: Part B (Intelligence & Evidence Layer)  
> **Role**: Online Image Occurrence Verification, Evidence Scoring & Candidate Ranking Engine

---

## 🎯 Architecture & Pipeline

```
                     USER IMAGE (Cropped Face from Part A)
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Pre-Processing & Quality Diagnostics       │
             │ • EXIF Auto-Orientation (Orientation 1..8)│
             │ • BGRA Alpha Channel Blending             │
             │ • Auto-Compress if > 500 KB               │
             │ • Blur (Laplacian) & Lighting Checks      │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Cryptographic Local SHA-256 Cache Check   │
             │ Instant sub-second response on cache hit  │
             └─────────────────────┬─────────────────────┘
                                   │ (Cache Miss)
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Google Lens Search (via SerpApi)          │
             │ • Query exact_matches                     │
             │ • Query visual_matches                    │
             │ • Exponential backoff retry               │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Candidate Pool Normalization              │
             │ • Strip tracking params (utm_*, ref, etc.)│
             │ • Deduplicate identical URLs & CDN links  │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Image Fetching with CDN 403 Fallback      │
             │ • Download candidate image                │
             │ • Anti-hotlink fallback to Lens thumbnail │
             └─────────────────────┬─────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
   Multi-Scale ORB          RANSAC Homography          HSV Color Hist
  (Native + 70% Crop        (Over-Determined             Correlation
   + 0.75x & 1.25x)          Inliers >= 6..8)        (Bhattacharyya sim)
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Normalized Evidence Scoring Engine        │
             │ Exact (+35) + Visual (+35) + Geometry     │
             │ (+20) + Social (+5) + Meta (+5) = 100 max │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Candidate Evidence Ranking (#1 .. #N)     │
             │ Explainable checklist for every candidate │
             └─────────────────────┬─────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  VERIFIED_MATCH             POSSIBLE_MATCH             NO_RELIABLE_MATCH
  (Score >= 65 +             (Moderate overlap;         (Fail-Safe Refusal;
   Confirmed Overlap)         insufficient geometry)     prevents false records)
        │                                                     │
        ▼                                                     ▼
  Commit Canonical Hash to Part C                     🛑 Safe Blockchain Refusal
```

---

## 🧠 Core Philosophy & Boundaries (Crucial for Judges)

> [!IMPORTANT]
> **Online Image Occurrence Verification vs. Biological Biometrics**:
> Part B explicitly does **not** perform facial recognition on unknown persons. It performs **image-level visual, geometric, and color verification** (ORB + RANSAC + Color Histogram).
> * **What it proves**: "This exact photo or a crop/variation of it appears on this web page."
> * **What it does NOT claim**: "These two distinct photos are biologically the same human being."
> * **Blockchain Privacy**: Raw photos and biometric face embeddings are **never** committed to the blockchain. Only canonical evidence hashes, verification metadata, and matched URLs are recorded.
> * **Search vs. Verification**: Google Lens is our **candidate-generation layer**, not the decision-maker. Search engine ranking reflects web popularity and SEO, not verification. Part B independently scores and verifies every candidate.

---

## ⚖️ Evidence Scoring Engine (0–100 Scale)

Evidence scores are formatted as **`X/100`** (e.g., `88.5/100`), **never as percentages**, ensuring technical honesty.

$$\text{Evidence Score} = \min(100, S_{\text{exact}} + S_{\text{visual}} + S_{\text{geometric}} + S_{\text{social}} + S_{\text{meta}})$$

| Signal Component | Max Points | Evaluation Method |
| :--- | :---: | :--- |
| **Google Lens Exact Match** | **+35** | Query matched indexed exact occurrence (`exact_matches`) |
| **Visual Similarity** | **+35** | Multi-Scale ORB + Lowe's Ratio Test ($0.75$). 5 matches = 20, 15 matches = 60, 25+ matches = 100 |
| **Geometric Consistency** | **+20** | RANSAC homography over-determined inliers ($\ge 6\text{--}8$ inliers). |
| **Social Media Context** | **+5** | Verified social domain (Instagram, LinkedIn, X, GitHub) **only if** visual overlap exists |
| **Metadata Completeness** | **+5** | Title verified (+3) and publisher source identified (+2) |

---

## ⚡ Superpowers & Advanced Features

### 1. Pre-Flight Quality Diagnostics
* **Blur Detection**: Laplacian variance metric ($Var < 30 \implies$ Blurry).
* **Illumination Check**: Mean intensity ($< 35 \implies$ Underexposed; $> 225 \implies$ Overexposed).
* **Resolution Check**: Rejects crops $< 48\times 48\text{ px}$.

### 2. Auto-Compression for Oversized (>500 KB) Inputs
If a user uploads a high-resolution 5 MB smartphone photo, Part B automatically downscales and JPEG-compresses it in memory under SerpApi's 500 KB limit.

### 3. EXIF Auto-Orientation & BGRA Transparency Blending
Corrects phone camera EXIF orientation tags (tags 3, 6, 8) so the face is upright before keypoint extraction, and blends 4-channel PNG transparent backgrounds into neutral white.

### 4. Anti-Hotlinking CDN Thumbnail Fallback
Instagram, Pinterest, and Facebook often block direct image downloads with HTTP 403. Part B automatically falls back to Google's cached thumbnail URL.

### 5. Cryptographic Local SHA-256 Search Cache
Caches search results keyed by the input image's SHA-256 hash. Cuts repeated query latency from ~6s to **0.01s** and saves API credits.

### 6. Interactive Visual HTML Evidence Report (`--report-html`)
Generates a standalone, styled HTML report (`verification_report.html`) complete with candidate cards, thumbnail previews, confidence pills, and full evidence checklists.

---

## 🏆 Comprehensive Edge Cases Handled

| # | Edge Case | How Part B Handles It |
|---|---|---|
| 1 | **No face detected / empty crop** | Part A rejects upstream; Part B checks decodability & minimum dimensions. |
| 2 | **No reverse-image match online** | Lens returns empty result; handled gracefully as 0 candidates, not a fatal crash. |
| 3 | **Unrelated Google Lens results** | ORB + RANSAC visual checks reject false positives, scoring them as `LOW_CONFIDENCE`. |
| 4 | **First result is wrong/sponsored** | All candidates are scored and ranked; best evidence wins, not top search link. |
| 5 | **Cropped, scaled, or thumbnail avatar** | Multi-scale pyramid (0.75x, 1.25x) and center-crop ORB detect matches despite resizing. |
| 6 | **Duplicate search results** | URL normalization strips tracking params (`utm_*`, `ref`, `igshid`) and deduplicates results. |
| 7 | **Social result without visual match** | Social domain bonus is strictly withheld unless independent visual confirmation exists. |
| 8 | **Direct candidate image 403 Forbidden** | Anti-hotlinking fallback automatically downloads Google's hosted thumbnail. |
| 9 | **API failure / rate limiting (429)** | Exponential backoff retry logic (up to 3 attempts) for both upload and search requests. |
| 10 | **All candidates low confidence** | **Fail-Safe Decision Engine**: Returns `NO_RELIABLE_MATCH` and stops Part C from creating a false blockchain record. |
| 11 | **Phone photo with EXIF rotation** | Native EXIF parser extracts orientation tag and auto-rotates image upright. |
| 12 | **Oversized camera image (>500 KB)** | Automatic iterative compression scales and compresses under 500 KB limit. |
| 13 | **Transparent PNG/WebP (BGRA)** | Alpha channel blended onto white background to avoid black box artifacts. |
| 14 | **Blurry or pitch-black input** | Pre-flight diagnostics flag blurriness (Laplacian var) and underexposure. |
| 15 | **Network outage during repeat demo** | Local SHA-256 search cache serves previously verified results in 0.01s offline. |

---

## 🥇 Pitch & Judge Defense (Q&A)

### "What exactly did you build in Part B?"
> *"Google Lens is our candidate-generation layer, not our final decision-maker. Part B collects candidate occurrences, independently evaluates their visual overlap using multi-scale ORB, geometric consistency using RANSAC homography, and color correlation, combines those signals into an evidence score, ranks every candidate, and can reject all candidates if the evidence isn't sufficient."*

### "Why not just use the first Google Lens result?"
> *"Because search ranking isn't verification. A search engine can return visually or contextually similar but unrelated results based on web popularity. We therefore perform an independent verification step before considering a result reliable."*

### "What if nothing matches?"
> *"That's a valid and intended outcome. We return `NO_RELIABLE_MATCH` rather than forcing a false positive on the blockchain ledger."*

### "What happens if multiple images are uploaded?"
> *"Our current pipeline processes one image per verification request. Multiple images are treated as separate verification inputs rather than combining different people into one identity. A future extension could process multiple images of the same consenting subject independently and aggregate their search evidence."*

---

## 📁 File Structure

```
partB/
├── reverse_image_search.py   # Core Verification & Evidence Ranking Engine
├── test_reverse_search.py    # 11-Test Offline Unit Suite + Live Test Runner
├── integration_example.py    # End-to-End Part A → Part B → Part C Pipeline Demo
├── requirements.txt          # Minimal production dependencies (OpenCV, Requests, NumPy)
├── .env.example              # API key template
├── .gitignore                # Credentials & environment ignore rules
├── README_partB.md           # System documentation & hackathon guide
├── verification_report.html  # Generated standalone visual HTML evidence report
└── test_images/
    └── test.jpg              # Sample face image for testing
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure SerpApi Key

Copy `.env.example` to `.env` and add your SerpApi key:

```bash
cp .env.example .env
# Edit .env:
# SERPAPI_API_KEY=your_key_here
```

### 3. Run Automated Offline Tests (11 Tests)

```bash
python test_reverse_search.py --offline
```

### 4. Run Live Reverse Image Search

```bash
# Formatted interactive evidence report in terminal
python reverse_image_search.py test_images/test.jpg

# Generate interactive standalone HTML report
python reverse_image_search.py test_images/test.jpg --report-html

# Output canonical JSON for Part C / Frontend
python reverse_image_search.py test_images/test.jpg --json
```

### 5. Run Full End-to-End Pipeline Demo (Part A → B → C)

```bash
python integration_example.py
```

---

## 🔗 Part C Blockchain Integration Contract

```json
{
  "matched_url": "https://instagram.com/verified_subject/profile_pic",
  "title": "Verified Profile • Instagram",
  "source": "Instagram",
  "source_image": "cropped_face.jpg",
  "image_hash_sha256": "9cb452b8f5aa9e4d...",
  "match_status": "VERIFIED_MATCH",
  "match_type": "EXACT_PLUS_VISUAL",
  "evidence_score": 88.5,
  "visual_score": 84.0,
  "geometric_score": 75.0,
  "selection_reason": "Exact-match signal confirmed with 22 visual features and 12 geometric inliers.",
  "quality_diagnostics": {
    "blur_variance": 9679.4,
    "is_blurry": false,
    "mean_brightness": 106.0,
    "lighting_condition": "NORMAL",
    "is_acceptable": true
  },
  "evidence_breakdown": {
    "exact_match_signal": true,
    "orb_matches_count": 22,
    "ransac_inliers": 12,
    "visual_score": 84.0,
    "geometric_score": 75.0,
    "color_correlation": 0.92,
    "social_domain": true
  },
  "search_statistics": {
    "exact_results_count": 1,
    "visual_results_count": 4,
    "total_raw_candidates": 5,
    "unique_candidates": 4,
    "images_downloaded": 4,
    "thumbnail_fallbacks": 1,
    "verified_matches": 1,
    "possible_matches": 1,
    "low_confidence_matches": 2,
    "from_cache": false
  },
  "timing": {
    "upload_sec": 0.78,
    "lens_exact_sec": 1.12,
    "lens_visual_sec": 1.34,
    "verification_sec": 2.45,
    "scoring_sec": 0.01,
    "total_sec": 5.70
  },
  "candidates": [...]
}
```
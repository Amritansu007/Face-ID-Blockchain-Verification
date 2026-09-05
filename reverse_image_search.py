"""
Part B — Reverse Image Search & Verification Engine (Advanced Production Edition)
Project: Face ID + Blockchain Verification

Architecture & Flow:
    User Image (Cropped Face from Part A)
          ↓
    Image Pre-Processing & Quality Diagnostics (EXIF auto-rotation, BGRA alpha blending,
    auto-compression if > 500 KB, blur & illumination checks)
          ↓
    Local Cryptographic SHA-256 Cache Check (Instant sub-second hit or miss)
          ↓
    SerpApi Upload (multipart 'image' field with exponential retry)
          ↓
    Google Lens Query (Exact Matches + Visual Matches with retry)
          ↓
    Candidate Pool (Deduplicated by canonical URL & image URL, tracking params stripped)
          ↓
    Anti-Hotlinking CDN Fallback (Uses Google Lens thumbnail if direct image gives 403)
          ↓
    Multi-Signal Verification Engine:
      - Multi-scale ORB (Full query, center crop, 0.75x, 1.25x)
      - RANSAC Homography Geometric Consistency (Over-determined inliers)
      - HSV Color Histogram Correlation (Bhattacharyya metric)
          ↓
    Evidence Scoring Engine (Normalized 0–100 score + signal breakdown)
          ↓
    Candidate Ranking (All candidates sorted by evidence score)
          ↓
    Decision Engine (VERIFIED_MATCH / POSSIBLE_MATCH / NO_RELIABLE_MATCH)
          ↓
    Output Delivery:
      - Formatted Console Table
      - Canonical JSON for Part C Blockchain Attestation
      - Interactive Visual HTML Evidence Report
"""

import os
import sys
import re
import time
import json
import struct
import warnings
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# Suppress harmless urllib3 v2 LibreSSL warning on macOS Python 3.9
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import requests
import cv2
import numpy as np


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

SERPAPI_IMAGE_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search"

MAX_IMAGE_BYTES = 500 * 1024  # SerpApi upload limit: 500 KB
REQUEST_TIMEOUT = 15          # Seconds per HTTP request
MAX_RETRIES = 3               # Network retry attempts with exponential backoff
BACKOFF_FACTOR = 1.5          # Backoff multiplier: 1.5s, 2.25s, ...

MAX_CANDIDATES_TO_VERIFY = 15 # Maximum candidate images to download and verify
DEFAULT_IMAGE_DIM = 800       # Maximum dimension for ORB image processing
CACHE_FILE = ".search_cache.json"
CACHE_TTL_SECONDS = 86400     # Cache valid for 24 hours

# Known social and professional media domains
SOCIAL_DOMAINS = {
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "tiktok.com",
    "youtube.com",
    "pinterest.com",
    "reddit.com",
    "github.com",
    "threads.net",
}

# Tracking query parameters to strip during URL deduplication
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "igshid", "ref", "source", "t", "s"
}


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class ImageQualityDiagnostics:
    blur_variance: float
    is_blurry: bool
    mean_brightness: float
    lighting_condition: str      # "NORMAL", "TOO_DARK", "OVEREXPOSED"
    dimensions: Tuple[int, int]  # (width, height)
    is_acceptable: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvidenceBreakdown:
    exact_match_signal: bool
    orb_matches_count: int
    ransac_inliers: int
    visual_score: float          # 0–100 scale
    geometric_score: float       # 0–100 scale
    color_correlation: float     # 0.0–1.0 scale
    social_domain: bool
    has_title: bool
    has_source: bool
    checklist: List[str] = field(default_factory=list)


@dataclass
class CandidateResult:
    rank: int
    url: Optional[str]
    title: Optional[str]
    source: Optional[str]
    image_url: Optional[str]
    evidence_score: float        # 0–100 scale
    visual_score: float          # 0–100 scale
    geometric_score: float       # 0–100 scale
    color_correlation: float     # 0.0–1.0 scale
    match_status: str            # VERIFIED_MATCH, POSSIBLE_MATCH, LOW_CONFIDENCE
    match_type: Optional[str]
    selection_reason: str
    evidence_breakdown: EvidenceBreakdown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchStatistics:
    exact_results_count: int = 0
    visual_results_count: int = 0
    total_raw_candidates: int = 0
    unique_candidates: int = 0
    images_downloaded: int = 0
    thumbnail_fallbacks: int = 0
    verified_matches: int = 0
    possible_matches: int = 0
    low_confidence_matches: int = 0
    from_cache: bool = False


@dataclass
class TimingBreakdown:
    upload_sec: float = 0.0
    lens_exact_sec: float = 0.0
    lens_visual_sec: float = 0.0
    verification_sec: float = 0.0
    scoring_sec: float = 0.0
    total_sec: float = 0.0


@dataclass
class MatchResult:
    matched_url: Optional[str]
    title: Optional[str]
    source: Optional[str]
    source_image: str
    image_hash_sha256: str
    match_status: str            # VERIFIED_MATCH, POSSIBLE_MATCH, NO_RELIABLE_MATCH
    match_type: Optional[str]
    evidence_score: float        # 0–100 scale
    visual_score: float          # 0–100 scale
    geometric_score: float       # 0–100 scale
    selection_reason: str
    quality_diagnostics: ImageQualityDiagnostics
    evidence_breakdown: Optional[Dict[str, Any]]
    search_statistics: SearchStatistics
    timing: TimingBreakdown
    candidates: List[Dict[str, Any]]
    manifest_sha256: Optional[str] = None
    record_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_url": self.matched_url,
            "title": self.title,
            "source": self.source,
            "source_image": self.source_image,
            "image_hash_sha256": self.image_hash_sha256,
            "manifest_sha256": self.manifest_sha256,
            "record_hash": self.record_hash,
            "match_status": self.match_status,
            "match_type": self.match_type,
            "evidence_score": self.evidence_score,
            "visual_score": self.visual_score,
            "geometric_score": self.geometric_score,
            "selection_reason": self.selection_reason,
            "quality_diagnostics": asdict(self.quality_diagnostics),
            "evidence_breakdown": self.evidence_breakdown,
            "search_statistics": asdict(self.search_statistics),
            "timing": asdict(self.timing),
            "candidates": self.candidates,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ==============================================================================
# CRYPTOGRAPHIC LOCAL SEARCH CACHE
# ==============================================================================

class SearchCacheManager:
    """Manages local SHA-256 keyed cache to avoid burning SerpApi credits."""
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def get(self, image_hash: str) -> Optional[Dict[str, Any]]:
        entry = self.data.get(image_hash)
        if not entry:
            return None
        timestamp = entry.get("timestamp", 0)
        if time.time() - timestamp > CACHE_TTL_SECONDS:
            return None
        return entry.get("result")

    def set(self, image_hash: str, result_dict: Dict[str, Any]) -> None:
        self.data[image_hash] = {
            "timestamp": time.time(),
            "result": result_dict
        }
        self._save()


# ==============================================================================
# ENVIRONMENT & API KEY HANDLING
# ==============================================================================

def load_dotenv_fallback(dotenv_path: str = ".env") -> None:
    """Lightweight .env parser avoiding external dependency issues."""
    if not os.path.isfile(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


def get_api_key(required: bool = True) -> Optional[str]:
    """Retrieve the SerpApi key from the environment or .env file."""
    load_dotenv_fallback()
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key and required:
        raise RuntimeError(
            "SERPAPI_API_KEY is not set.\n"
            "Set it via environment variable or in a .env file:\n"
            "  export SERPAPI_API_KEY='your_api_key_here'\n"
            "Get a key at: https://serpapi.com/manage-api-key"
        )
    return api_key


# ==============================================================================
# IMAGE PRE-PROCESSING, QUALITY DIAGNOSTICS & EXIF NORMALIZATION
# ==============================================================================

def _get_exif_orientation(image_bytes: bytes) -> int:
    """Extract EXIF orientation tag (0x0112) using standard library struct."""
    try:
        if len(image_bytes) < 4 or image_bytes[:2] != b"\xff\xd8":
            return 1  # Not a JPEG or too short

        idx = 2
        while idx < len(image_bytes) - 4:
            marker, length = struct.unpack(">HH", image_bytes[idx:idx+4])
            if marker == 0xffe1:  # APP1 (EXIF)
                exif_data = image_bytes[idx+4:idx+2+length]
                if exif_data.startswith(b"Exif\x00\x00"):
                    tiff_header = exif_data[6:]
                    endian = "<" if tiff_header[:2] == b"II" else ">"
                    first_ifd = struct.unpack(endian + "I", tiff_header[4:8])[0]
                    num_entries = struct.unpack(endian + "H", tiff_header[first_ifd:first_ifd+2])[0]
                    entry_offset = first_ifd + 2
                    for _ in range(num_entries):
                        tag = struct.unpack(endian + "H", tiff_header[entry_offset:entry_offset+2])[0]
                        if tag == 0x0112:  # Orientation
                            return struct.unpack(endian + "H", tiff_header[entry_offset+8:entry_offset+10])[0]
                        entry_offset += 12
                break
            idx += 2 + length
    except Exception:
        pass
    return 1


def _apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Rotates OpenCV image to upright orientation based on EXIF tag."""
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def assess_image_quality(image: np.ndarray) -> ImageQualityDiagnostics:
    """Pre-flight quality check: blurriness, brightness, and resolution."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Blur Detection (Laplacian Variance)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_var < 30.0

    # 2. Illumination / Brightness
    mean_bright = float(np.mean(gray))
    if mean_bright < 35.0:
        lighting = "TOO_DARK"
    elif mean_bright > 225.0:
        lighting = "OVEREXPOSED"
    else:
        lighting = "NORMAL"

    warnings_list = []
    if is_blurry:
        warnings_list.append(f"Image is blurry (Laplacian variance: {blur_var:.1f} < 30.0)")
    if lighting == "TOO_DARK":
        warnings_list.append(f"Image is severely underexposed (brightness: {mean_bright:.1f}/255)")
    elif lighting == "OVEREXPOSED":
        warnings_list.append(f"Image is overexposed/washed out (brightness: {mean_bright:.1f}/255)")
    if w < 48 or h < 48:
        warnings_list.append(f"Resolution is very small ({w}x{h} px)")

    is_acceptable = (not is_blurry) and (lighting == "NORMAL") and (w >= 48 and h >= 48)

    return ImageQualityDiagnostics(
        blur_variance=round(blur_var, 1),
        is_blurry=is_blurry,
        mean_brightness=round(mean_bright, 1),
        lighting_condition=lighting,
        dimensions=(w, h),
        is_acceptable=is_acceptable,
        warnings=warnings_list,
    )


def auto_compress_and_normalize_image(image_path: str) -> Tuple[np.ndarray, str, str, bool]:
    """
    Validates, auto-orients EXIF, normalizes BGRA, and auto-compresses oversized images.
    Returns: (cv2_image, usable_file_path_for_upload, sha256_hash, was_compressed)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    if ext not in supported:
        raise ValueError(
            f"Unsupported format '{ext}'. Allowed formats: JPG, JPEG, PNG, WebP."
        )

    with open(image_path, "rb") as f:
        raw_bytes = f.read()

    file_size = len(raw_bytes)
    image_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Decode with unchanged flag to preserve alpha if present
    raw_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        raise ValueError("Image could not be decoded by OpenCV. File may be corrupted.")

    # Handle BGRA alpha channel (blend onto neutral background)
    if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
        b, g, r, a = cv2.split(raw_img)
        alpha = a.astype(float) / 255.0
        white_bg = np.ones_like(b, dtype=float) * 255.0
        b = (b.astype(float) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        g = (g.astype(float) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        r = (r.astype(float) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        raw_img = cv2.merge([b, g, r])
    elif len(raw_img.shape) == 2:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

    # Handle EXIF orientation
    orientation = _get_exif_orientation(raw_bytes)
    if orientation in [3, 6, 8]:
        raw_img = _apply_exif_orientation(raw_img, orientation)

    # If file size is within limits (< 500 KB), use original file path
    if file_size <= MAX_IMAGE_BYTES:
        return raw_img, image_path, image_hash, False

    # AUTO-COMPRESSION for oversized images (e.g. 5 MB camera uploads)
    compressed_temp_path = f".tmp_compressed_{os.path.basename(image_path)}.jpg"
    h, w = raw_img.shape[:2]
    scale = 1.0
    if max(h, w) > 1400:
        scale = 1400.0 / float(max(h, w))

    resized = cv2.resize(raw_img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)

    quality = 90
    encoded = None
    while quality >= 30:
        success, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if success and len(encoded) <= MAX_IMAGE_BYTES * 0.95:
            break
        quality -= 10

    with open(compressed_temp_path, "wb") as f:
        f.write(encoded)

    return raw_img, compressed_temp_path, image_hash, True


def resize_image_aspect_ratio(image: np.ndarray, max_dim: int = DEFAULT_IMAGE_DIM) -> np.ndarray:
    """Resize image preserving aspect ratio if larger than max_dim."""
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image
    scale = max_dim / float(largest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ==============================================================================
# NETWORK & SERPAPI CLIENT WITH EXPONENTIAL RETRY
# ==============================================================================

def _requests_retry_post(url: str, files: dict, data: dict) -> requests.Response:
    """Perform HTTP POST with exponential backoff retry."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, files=files, data=data, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(BACKOFF_FACTOR ** attempt)
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_FACTOR ** attempt)
            else:
                raise RuntimeError(f"Network failure reaching SerpApi ({url}): {e}")
    if last_err:
        raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {last_err}")
    return resp


def _requests_retry_get(url: str, params: dict = None, headers: dict = None) -> requests.Response:
    """Perform HTTP GET with exponential backoff retry."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(BACKOFF_FACTOR ** attempt)
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_FACTOR ** attempt)
            else:
                raise RuntimeError(f"Network error connecting to {url}: {e}")
    if last_err:
        raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {last_err}")
    return resp


def upload_image_to_serpapi(image_path: str, api_key: str) -> str:
    """Upload image to SerpApi Image API. Field MUST be 'image'."""
    with open(image_path, "rb") as img_file:
        files = {
            "image": (
                os.path.basename(image_path),
                img_file,
                "application/octet-stream",
            )
        }
        data = {"api_key": api_key}
        response = _requests_retry_post(SERPAPI_IMAGE_URL, files=files, data=data)

    if not response.ok:
        try:
            err_msg = response.json().get("error", response.text)
        except Exception:
            err_msg = response.text
        raise RuntimeError(
            f"SerpApi image upload failed (HTTP {response.status_code}): {err_msg}"
        )

    try:
        payload = response.json()
    except Exception:
        raise RuntimeError("SerpApi image upload returned non-JSON response.")

    image_id = payload.get("image_id")
    if not image_id:
        raise RuntimeError(f"SerpApi upload response missing 'image_id': {payload}")

    return image_id


def query_google_lens(image_id: str, search_type: str, api_key: str) -> Dict[str, Any]:
    """Query Google Lens via SerpApi with safe empty result handling."""
    params = {
        "engine": "google_lens",
        "image_id": image_id,
        "type": search_type,
        "api_key": api_key,
    }

    response = _requests_retry_get(SERPAPI_SEARCH_URL, params=params)

    if not response.ok:
        try:
            err_msg = response.json().get("error", response.text)
        except Exception:
            err_msg = response.text
        if "hasn't returned any results" in str(err_msg).lower():
            return {}
        raise RuntimeError(
            f"Google Lens query failed (HTTP {response.status_code}): {err_msg}"
        )

    try:
        payload = response.json()
    except Exception:
        raise RuntimeError("Google Lens returned invalid JSON response.")

    if payload.get("error"):
        err_msg = str(payload.get("error"))
        if "hasn't returned any results" in err_msg.lower():
            return {}
        raise RuntimeError(f"Google Lens API Error: {err_msg}")

    return payload


# ==============================================================================
# CANDIDATE NORMALIZATION & INTELLIGENT DEDUPLICATION
# ==============================================================================

def normalize_url(url: Optional[str]) -> str:
    """Normalize URL by stripping tracking parameters, lowercase domain, removing trailing slash."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")

        query_dict = parse_qs(parsed.query, keep_blank_values=False)
        cleaned_query = {
            k: v for k, v in query_dict.items() if k.lower() not in TRACKING_PARAMS
        }
        query_str = urlencode(cleaned_query, doseq=True)

        return urlunparse((scheme, netloc, path, "", query_str, ""))
    except Exception:
        return url.strip().lower()


def get_domain(url: Optional[str]) -> str:
    """Extract clean domain name without www."""
    if not url:
        return ""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def is_social_domain(url: Optional[str]) -> bool:
    """Check if the URL belongs to a known social media or community site."""
    domain = get_domain(url)
    if not domain:
        return False
    return any(domain == s or domain.endswith("." + s) for s in SOCIAL_DOMAINS)


def extract_and_merge_candidates(
    exact_payload: Dict[str, Any],
    visual_payload: Dict[str, Any],
    stats: SearchStatistics
) -> List[Dict[str, Any]]:
    """Extract exact and visual matches, normalize, and deduplicate candidates."""
    raw_exact = exact_payload.get("exact_matches", [])
    raw_visual = visual_payload.get("visual_matches", [])

    stats.exact_results_count = len(raw_exact)
    stats.visual_results_count = len(raw_visual)
    stats.total_raw_candidates = len(raw_exact) + len(raw_visual)

    merged: Dict[str, Dict[str, Any]] = {}

    def process_item(item: dict, is_exact: bool):
        link = item.get("link")
        title = item.get("title")
        source = item.get("source")
        image_url = item.get("image")
        thumbnail_url = item.get("thumbnail")
        position = item.get("position", 999)

        norm_link = normalize_url(link)
        norm_img = normalize_url(image_url or thumbnail_url)
        dedup_key = norm_link or norm_img
        if not dedup_key:
            return

        if dedup_key in merged:
            existing = merged[dedup_key]
            if is_exact:
                existing["exact_match"] = True
            if not existing.get("image_url") and image_url:
                existing["image_url"] = image_url
            if not existing.get("thumbnail_url") and thumbnail_url:
                existing["thumbnail_url"] = thumbnail_url
            if not existing.get("title") and title:
                existing["title"] = title
            if not existing.get("source") and source:
                existing["source"] = source
        else:
            merged[dedup_key] = {
                "url": link,
                "normalized_url": norm_link,
                "title": title,
                "source": source,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "exact_match": is_exact or bool(item.get("exact_matches", False)),
                "position": position,
            }

    for item in raw_exact:
        process_item(item, is_exact=True)

    for item in raw_visual:
        process_item(item, is_exact=False)

    candidates = list(merged.values())
    stats.unique_candidates = len(candidates)
    return candidates


# ==============================================================================
# CANDIDATE IMAGE DOWNLOADER WITH ANTI-HOTLINKING THUMBNAIL FALLBACK
# ==============================================================================

def download_candidate_image_with_fallback(
    image_url: Optional[str],
    thumbnail_url: Optional[str] = None
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Downloads candidate image. If direct image fails (HTTP 403 / anti-hotlinking),
    automatically falls back to the Google-hosted thumbnail URL!
    Returns: (cv2_image_or_None, used_thumbnail_fallback_bool)
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    def _fetch(url: str) -> Optional[np.ndarray]:
        try:
            resp = _requests_retry_get(url, headers=headers)
            if not resp.ok:
                return None
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    # Try direct high-res image first
    if image_url:
        img = _fetch(image_url)
        if img is not None:
            return img, False

    # Fall back to Google Lens hosted thumbnail
    if thumbnail_url and thumbnail_url != image_url:
        img = _fetch(thumbnail_url)
        if img is not None:
            return img, True

    return None, False


# ==============================================================================
# DUAL VERIFICATION: MULTI-SCALE ORB + RANSAC + COLOR HISTOGRAM CORRELATION
# ==============================================================================

def compute_color_histogram_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes HSV 2D color histogram correlation (0.0 to 1.0)."""
    try:
        norm1 = resize_image_aspect_ratio(img1, 300)
        norm2 = resize_image_aspect_ratio(img2, 300)

        hsv1 = cv2.cvtColor(norm1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(norm2, cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

        correlation = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        return round(max(0.0, correlation), 2)
    except Exception:
        return 0.0


def _compute_orb_matches(
    gray_query: np.ndarray,
    gray_candidate: np.ndarray,
    orb: cv2.ORB
) -> Tuple[int, int, float]:
    """Runs ORB feature extraction, BFMatcher with Lowe's ratio test, and RANSAC homography."""
    kp1, des1 = orb.detectAndCompute(gray_query, None)
    kp2, des2 = orb.detectAndCompute(gray_candidate, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return 0, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    good_count = len(good_matches)
    if good_count < 5:
        return good_count, 0, 0.0

    inliers_count = 0
    inlier_ratio = 0.0

    if good_count >= 6:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        try:
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers_count = int(mask.ravel().sum())
                inlier_ratio = round(float(inliers_count) / float(good_count), 2)
        except cv2.error:
            pass

    return good_count, inliers_count, inlier_ratio


def verify_visual_overlap_multiscale(
    query_image: np.ndarray,
    candidate_image: np.ndarray
) -> Tuple[float, float, float, int, int]:
    """
    Multi-scale + Center Crop ORB Verification + Color Histogram Correlation.
    Returns: (visual_score, geometric_score, color_correlation, max_good_matches, max_inliers)
    """
    try:
        q_norm = resize_image_aspect_ratio(query_image, DEFAULT_IMAGE_DIM)
        c_norm = resize_image_aspect_ratio(candidate_image, DEFAULT_IMAGE_DIM)

        q_gray = cv2.cvtColor(q_norm, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.cvtColor(c_norm, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2000)

        views = [q_gray]

        # Central 70% crop focused on core face
        qh, qw = q_gray.shape[:2]
        if qh > 60 and qw > 60:
            y1, y2 = int(qh * 0.15), int(qh * 0.85)
            x1, x2 = int(qw * 0.15), int(qw * 0.85)
            views.append(q_gray[y1:y2, x1:x2])

        # Multi-scale views of candidate (0.75x, 1.25x)
        c_views = [c_gray]
        ch, cw = c_gray.shape[:2]
        if min(ch, cw) > 100:
            c_views.append(cv2.resize(c_gray, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA))
        if max(ch, cw) < 1200:
            c_views.append(cv2.resize(c_gray, (0, 0), fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR))

        best_good_matches = 0
        best_inliers = 0
        best_inlier_ratio = 0.0

        for q_v in views:
            for c_v in c_views:
                matches_cnt, inliers_cnt, ratio = _compute_orb_matches(q_v, c_v, orb)
                if matches_cnt > best_good_matches:
                    best_good_matches = matches_cnt
                if inliers_cnt > best_inliers:
                    best_inliers = inliers_cnt
                    best_inlier_ratio = ratio

        # Color histogram correlation
        color_corr = compute_color_histogram_correlation(q_norm, c_norm)

        # Visual Score: Fewer than 5 matches is noise (0.0)
        if best_good_matches < 5:
            visual_score = 0.0
        else:
            visual_score = min(100.0, best_good_matches * 4.0)

        # Geometric Consistency Score:
        # In projective 2D geometry, any 4 non-collinear points fit a homography trivially.
        # True geometric consistency requires an over-determined system (at least 6-8 inliers).
        if best_inliers <= 4:
            geometric_score = 0.0
            visual_score = min(visual_score, 15.0)
        else:
            geometric_score = min(100.0, (best_inliers - 4) * 8.33)
            if best_inliers >= 8 and best_inlier_ratio >= 0.4:
                visual_score = max(visual_score, geometric_score * 0.9)

        return round(visual_score, 1), round(geometric_score, 1), color_corr, best_good_matches, best_inliers
    except Exception:
        return 0.0, 0.0, 0.0, 0, 0


# ==============================================================================
# EVIDENCE ENGINE & SCORING SYSTEM
# ==============================================================================

def score_and_explain_candidate(
    candidate: Dict[str, Any],
    visual_score: float,
    geometric_score: float,
    color_correlation: float,
    good_matches: int,
    inliers: int
) -> Tuple[float, EvidenceBreakdown]:
    """Calculates normalized Evidence Score (0–100) with detailed explainable checklist."""
    exact_match = bool(candidate.get("exact_match", False))
    url = candidate.get("url")
    title = candidate.get("title")
    source = candidate.get("source")
    social = is_social_domain(url)

    score = 0.0
    checklist = []

    # 1. Google Lens Exact Match Signal (+35)
    if exact_match:
        score += 35.0
        checklist.append("✓ Google Lens exact-match signal detected (+35)")
    else:
        checklist.append("✗ No exact-match signal from Google Lens")

    # 2. Visual Overlap Score (+35)
    visual_points = min(35.0, (visual_score / 100.0) * 35.0)
    score += visual_points
    if good_matches >= 15:
        checklist.append(f"✓ Strong visual overlap: {good_matches} ORB features matched (+{visual_points:.1f})")
    elif good_matches >= 5:
        checklist.append(f"✓ Moderate visual overlap: {good_matches} ORB features matched (+{visual_points:.1f})")
    else:
        checklist.append(f"✗ Weak/no visual overlap: {good_matches} ORB features matched")

    # 3. Geometric Consistency Score (+20)
    geometric_points = min(20.0, (geometric_score / 100.0) * 20.0)
    score += geometric_points
    if inliers >= 8:
        checklist.append(f"✓ Strong geometric consistency: {inliers} RANSAC inliers (+{geometric_points:.1f})")
    elif inliers >= 5:
        checklist.append(f"✓ Partial geometric consistency: {inliers} RANSAC inliers (+{geometric_points:.1f})")
    else:
        checklist.append("✗ Insufficient geometric consistency (low RANSAC inliers)")

    # 4. Color Distribution Correlation
    if color_correlation >= 0.7:
        checklist.append(f"✓ High color histogram correlation: {color_correlation:.2f}")
    elif color_correlation >= 0.35:
        checklist.append(f"✓ Moderate color correlation: {color_correlation:.2f}")
    else:
        checklist.append(f"! Low color correlation: {color_correlation:.2f}")

    # 5. Social Platform Context (+5 only if visual confirmation exists)
    if social:
        if visual_score >= 20.0 or exact_match:
            score += 5.0
            checklist.append("✓ Known social/community platform with visual confirmation (+5)")
        else:
            checklist.append("! Social platform detected but excluded: insufficient visual evidence")

    # 6. Metadata Completeness (+5)
    if title:
        score += 3.0
        checklist.append("✓ Page title available (+3)")
    if source:
        score += 2.0
        checklist.append("✓ Source publisher verified (+2)")

    final_evidence_score = round(min(100.0, score), 1)

    breakdown = EvidenceBreakdown(
        exact_match_signal=exact_match,
        orb_matches_count=good_matches,
        ransac_inliers=inliers,
        visual_score=visual_score,
        geometric_score=geometric_score,
        color_correlation=color_correlation,
        social_domain=social,
        has_title=bool(title),
        has_source=bool(source),
        checklist=checklist,
    )

    return final_evidence_score, breakdown


def classify_candidate(
    candidate: Dict[str, Any],
    evidence_score: float,
    visual_score: float,
    geometric_score: float,
    good_matches: int,
    inliers: int
) -> Tuple[str, Optional[str], str]:
    """Classifies candidate into VERIFIED_MATCH, POSSIBLE_MATCH, or LOW_CONFIDENCE."""
    exact_match = bool(candidate.get("exact_match", False))

    if exact_match and (visual_score >= 25.0 or good_matches >= 8):
        return (
            "VERIFIED_MATCH",
            "EXACT_PLUS_VISUAL",
            f"Google Lens exact-match confirmed with {good_matches} visual features and {inliers} geometric inliers."
        )

    if visual_score >= 50.0 and geometric_score >= 40.0:
        return (
            "VERIFIED_MATCH",
            "STRONG_VISUAL_AND_GEOMETRIC",
            f"High visual overlap ({visual_score}/100) and geometric consistency ({inliers} inliers)."
        )

    if evidence_score >= 65.0 and visual_score >= 35.0:
        return (
            "VERIFIED_MATCH",
            "HIGH_EVIDENCE_SCORE",
            f"High cumulative evidence score ({evidence_score}/100) across search and visual signals."
        )

    if (visual_score >= 25.0 and evidence_score >= 40.0) or (exact_match and visual_score >= 15.0):
        return (
            "POSSIBLE_MATCH",
            "MODERATE_EVIDENCE",
            f"Moderate visual similarity ({visual_score}/100) or exact index match; lacks high geometric confidence."
        )

    return (
        "LOW_CONFIDENCE",
        None,
        f"Visual overlap ({visual_score}/100) or evidence score ({evidence_score}/100) below verification threshold."
    )


# ==============================================================================
# VISUAL HTML EVIDENCE REPORT GENERATOR
# ==============================================================================

def generate_html_report(match_result: MatchResult, output_html_path: str = "verification_report.html") -> str:
    """Generates an interactive standalone visual HTML report for hackathon demos."""
    badge_color = {
        "VERIFIED_MATCH": "#10b981",
        "POSSIBLE_MATCH": "#f59e0b",
        "NO_RELIABLE_MATCH": "#ef4444",
        "LOW_CONFIDENCE": "#6b7280",
    }

    candidates_html = ""
    for c in match_result.candidates:
        color = badge_color.get(c["match_status"], "#6b7280")
        ev_score = c["evidence_score"]
        v_score = c["visual_score"]
        g_score = c["geometric_score"]
        checklist_items = "".join(f"<li>{item}</li>" for item in c["evidence_breakdown"]["checklist"])

        img_tag = f'<img src="{c.get("image_url")}" alt="Thumbnail" class="candidate-thumb" onerror="this.style.display=\'none\'" />' if c.get("image_url") else ''

        candidates_html += f"""
        <div class="candidate-card">
            <div class="candidate-header">
                <span class="rank-badge">#{c["rank"]}</span>
                <span class="status-pill" style="background-color: {color};">{c["match_status"]}</span>
                <span class="score-pill">{ev_score}/100 Evidence</span>
            </div>
            <div class="candidate-body">
                {img_tag}
                <div class="candidate-info">
                    <h3 class="candidate-title">{c.get("title") or "Untitled Occurrence"}</h3>
                    <p class="candidate-source"><strong>Source:</strong> {c.get("source") or "Unknown"}</p>
                    <p class="candidate-url"><a href="{c.get("url")}" target="_blank">{c.get("url")}</a></p>
                    
                    <div class="metrics-grid">
                        <div>Visual Score: <strong>{v_score}/100</strong></div>
                        <div>Geometric Inliers: <strong>{c["evidence_breakdown"]["ransac_inliers"]}</strong></div>
                        <div>Color Correlation: <strong>{c.get("color_correlation", 0.0):.2f}</strong></div>
                    </div>
                    <ul class="checklist">{checklist_items}</ul>
                </div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evidence Verification Report — Part B</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{
            border-bottom: 1px solid var(--surface-hover);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        h1 {{ margin: 0 0 0.5rem 0; font-size: 1.8rem; display: flex; align-items: center; gap: 0.75rem; }}
        .subtitle {{ color: var(--text-muted); font-size: 0.95rem; margin: 0; }}
        .decision-banner {{
            background: var(--surface);
            border-left: 6px solid {badge_color.get(match_result.match_status, "#6b7280")};
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--surface);
            padding: 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--surface-hover);
        }}
        .stat-val {{ font-size: 1.5rem; font-weight: bold; color: var(--primary); }}
        .stat-label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; }}
        .candidate-card {{
            background: var(--surface);
            border: 1px solid var(--surface-hover);
            border-radius: 8px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}
        .candidate-header {{
            background: rgba(0,0,0,0.2);
            padding: 0.75rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .rank-badge {{ font-weight: bold; font-size: 1.1rem; color: var(--text); }}
        .status-pill {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: bold;
            color: #fff;
        }}
        .score-pill {{ margin-left: auto; font-weight: bold; color: var(--text-muted); }}
        .candidate-body {{ padding: 1.25rem; display: flex; gap: 1.25rem; }}
        .candidate-thumb {{ width: 120px; height: 120px; object-fit: cover; border-radius: 6px; }}
        .candidate-info {{ flex: 1; }}
        .candidate-title {{ margin: 0 0 0.5rem 0; font-size: 1.15rem; }}
        .candidate-source, .candidate-url {{ margin: 0 0 0.25rem 0; font-size: 0.9rem; color: var(--text-muted); }}
        .candidate-url a {{ color: var(--primary); text-decoration: none; word-break: break-all; }}
        .metrics-grid {{ display: flex; gap: 1.5rem; margin: 0.75rem 0; font-size: 0.9rem; }}
        .checklist {{ margin: 0.75rem 0 0 0; padding-left: 1.25rem; font-size: 0.85rem; color: var(--text-muted); }}
        footer {{ text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 3rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⭐ Face ID + Blockchain: Part B Evidence Engine</h1>
            <p class="subtitle">Online Image Occurrence Verification, Multi-Scale ORB + RANSAC & Candidate Ranking</p>
        </header>

        <div class="decision-banner">
            <div>
                <span class="status-pill" style="background: {badge_color.get(match_result.match_status, "#6b7280")}; font-size: 0.9rem;">
                    {match_result.match_status}
                </span>
                <h2 style="margin: 0.5rem 0 0.25rem 0;">Evidence Score: {match_result.evidence_score}/100</h2>
                <p style="margin: 0; color: var(--text-muted);">{match_result.selection_reason}</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.85rem; color: var(--text-muted);">Source Image</div>
                <div style="font-weight: bold;">{os.path.basename(match_result.source_image)}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">SHA-256: {match_result.image_hash_sha256[:16]}...</div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">{match_result.search_statistics.unique_candidates}</div>
                <div class="stat-label">Unique Candidates</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{match_result.search_statistics.images_downloaded}</div>
                <div class="stat-label">Verified Overlap</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{match_result.search_statistics.verified_matches}</div>
                <div class="stat-label">Verified Matches</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{match_result.timing.total_sec}s</div>
                <div class="stat-label">Total Latency {"(Cached)" if match_result.search_statistics.from_cache else ""}</div>
            </div>
        </div>

        <h2>Ranked Candidate Pool</h2>
        {candidates_html if candidates_html else "<p>No candidates available.</p>"}

        <footer>
            Part B Verification Engine • Cryptographic Evidence Record for Part C Blockchain Ledger
        </footer>
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path


def resolve_input_image_or_manifest(input_path: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """
    Resolves input_path:
      - If it's a directory (e.g. Part A's 'out/'), looks for manifest.json and search_priority crops.
      - If it's 'manifest.json', parses search_priority ('context' -> 'upscaled' -> 'tight') and finds the crop.
      - If it's a direct image file (e.g. 'test.jpg' or 'out/context.jpg'), returns it directly.
    Returns: (resolved_image_path, manifest_sha256_or_None, manifest_dict_or_None)
    """
    from pathlib import Path
    p = Path(input_path)
    manifest_data = None
    manifest_sha256 = None

    if p.is_dir():
        mfile = p / "manifest.json"
        if mfile.is_file():
            try:
                manifest_data = json.loads(mfile.read_text(encoding="utf-8"))
                manifest_sha256 = manifest_data.get("manifest_sha256")
            except Exception:
                pass
        priority = ["context", "upscaled", "tight"]
        if manifest_data and "body" in manifest_data and "search_priority" in manifest_data["body"]:
            priority = manifest_data["body"]["search_priority"]
        for role in priority:
            cfile = p / f"{role}.jpg"
            if cfile.is_file():
                return str(cfile), manifest_sha256, manifest_data
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            matches = list(p.glob(ext))
            if matches:
                return str(matches[0]), manifest_sha256, manifest_data
        raise FileNotFoundError(f"No usable image or crop found in directory: {input_path}")

    elif p.name.lower() == "manifest.json" and p.is_file():
        try:
            manifest_data = json.loads(p.read_text(encoding="utf-8"))
            manifest_sha256 = manifest_data.get("manifest_sha256")
            d = p.parent
            priority = manifest_data.get("body", {}).get("search_priority", ["context", "upscaled", "tight"])
            for role in priority:
                cfile = d / f"{role}.jpg"
                if cfile.is_file():
                    return str(cfile), manifest_sha256, manifest_data
        except Exception as e:
            raise ValueError(f"Could not parse manifest {input_path}: {e}")

    return str(p), None, None


# ==============================================================================
# MAIN VERIFICATION PIPELINE
# ==============================================================================

def find_match(
    image_path: str,
    api_key: Optional[str] = None,
    max_verify: int = MAX_CANDIDATES_TO_VERIFY,
    use_cache: bool = True,
    generate_html: bool = False,
    html_output_path: str = "verification_report.html",
    is_demo: bool = False,
    verbose: bool = True
) -> MatchResult:
    """
    Main Part B Pipeline (Production & Hackathon Supercharged):
      1. Resolves Part A hand-off (accepts directory, manifest.json, or cropped face image)
      2. Image Pre-Processing & Normalization (EXIF orientation, BGRA, auto-compression)
      3. Quality Diagnostics (Blur & illumination checks)
      4. Cryptographic Local Cache Lookup (Instant response if searched recently)
      5. SerpApi Upload (Multipart 'image' with exponential retry)
      6. Google Lens Exact + Visual Search
      7. Candidate Deduplication & Tracking Parameter Removal
      8. Multi-Scale ORB + RANSAC + Color Histogram Verification (with CDN thumbnail fallback)
      9. Normalized Evidence Scoring & Signal Breakdown
      10. Fail-Safe Decision Engine (NO_RELIABLE_MATCH)
      11. Anchors Part A manifest_sha256 into canonical output for Part C
    """
    total_start_time = time.time()
    timing = TimingBreakdown()
    stats = SearchStatistics()

    # Step 0: Resolve Part A hand-off (handles out/ directory or manifest.json)
    resolved_image_path, manifest_sha256, manifest_data = resolve_input_image_or_manifest(image_path)
    if verbose and manifest_sha256:
        print(f"📦 Linked Part A Manifest SHA-256: {manifest_sha256}")
        print(f"   Using prioritized crop: {os.path.basename(resolved_image_path)}")

    # Step 1: Pre-process & Validate input image
    (
        query_image,
        upload_path,
        image_hash,
        was_compressed
    ) = auto_compress_and_normalize_image(resolved_image_path)

    # Step 2: Quality Diagnostics
    quality = assess_image_quality(query_image)

    # Step 3: Check Local Cryptographic Cache
    cache_mgr = SearchCacheManager()
    if use_cache:
        cached_data = cache_mgr.get(image_hash)
        if cached_data:
            if verbose:
                print(f"\n⚡ CACHE HIT: Reusing previous search for {image_path} (SHA-256: {image_hash[:12]}...)")
            rec_hash = cached_data.get("record_hash")
            man_sha = manifest_sha256 or cached_data.get("manifest_sha256")
            if not rec_hash and man_sha and cached_data.get("matched_url"):
                rec_hash = hashlib.sha256(f"{man_sha}:{cached_data['matched_url']}".encode("utf-8")).hexdigest()

            res = MatchResult(
                matched_url=cached_data.get("matched_url"),
                title=cached_data.get("title"),
                source=cached_data.get("source"),
                source_image=resolved_image_path,
                image_hash_sha256=image_hash,
                match_status=cached_data.get("match_status"),
                match_type=cached_data.get("match_type"),
                evidence_score=cached_data.get("evidence_score", 0.0),
                visual_score=cached_data.get("visual_score", 0.0),
                geometric_score=cached_data.get("geometric_score", 0.0),
                selection_reason=cached_data.get("selection_reason", ""),
                quality_diagnostics=quality,
                evidence_breakdown=cached_data.get("evidence_breakdown"),
                search_statistics=SearchStatistics(**cached_data.get("search_statistics", {})),
                timing=timing,
                candidates=cached_data.get("candidates", []),
                manifest_sha256=man_sha,
                record_hash=rec_hash,
            )
            if generate_html:
                generate_html_report(res, html_output_path)
            return res

    if is_demo:
        if verbose:
            print("\n" + "=" * 70)
            print("🎮 DEMO MODE: SIMULATING LENS CANDIDATES FOR EVALUATION")
            print("=" * 70)
            print(f"Target Image : {image_path}")
            print(f"SHA-256 Hash : {image_hash}")
            print(f"Diagnostics  : Brightness {quality.mean_brightness:.1f}/255, Blur Var {quality.blur_variance:.1f}")

        timing.upload_sec = 0.65
        timing.lens_exact_sec = 0.95
        timing.lens_visual_sec = 1.10

        # Simulate candidate generation with local variations of query image
        # Candidate 1: Verified identical/cropped match
        # Candidate 2: Partial overlap
        # Candidate 3: Unrelated noise candidate
        h, w = query_image.shape[:2]
        cand1_img = cv2.resize(query_image[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)], (0, 0), fx=0.85, fy=0.85)
        cand2_img = cv2.resize(query_image, (0, 0), fx=0.5, fy=0.5)
        cand3_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(cand3_img, (150, 150), 60, (255, 255, 255), -1)

        simulated_candidates = [
            ("https://instagram.com/verified_subject/avatar", "Verified Subject (@verified_subject) • Instagram", "Instagram", True, cand1_img),
            ("https://linkedin.com/in/verified-subject", "Verified Subject - Engineering Lead - LinkedIn", "LinkedIn", False, cand2_img),
            ("https://unrelated-news.com/article/908", "Stock Portrait in News Editorial", "UnrelatedNews", False, cand3_img),
        ]

        stats.exact_results_count = 1
        stats.visual_results_count = 2
        stats.total_raw_candidates = 3
        stats.unique_candidates = 3

        scored_candidates: List[CandidateResult] = []
        t_verify_start = time.time()

        for c_url, c_title, c_source, c_exact, c_img in simulated_candidates:
            stats.images_downloaded += 1
            v_score, g_score, c_corr, matches_cnt, inliers_cnt = verify_visual_overlap_multiscale(query_image, c_img)
            cand_dict = {"url": c_url, "title": c_title, "source": c_source, "exact_match": c_exact}
            ev_score, breakdown = score_and_explain_candidate(cand_dict, v_score, g_score, c_corr, matches_cnt, inliers_cnt)
            status, m_type, reason = classify_candidate(cand_dict, ev_score, v_score, g_score, matches_cnt, inliers_cnt)

            if status == "VERIFIED_MATCH":
                stats.verified_matches += 1
            elif status == "POSSIBLE_MATCH":
                stats.possible_matches += 1
            else:
                stats.low_confidence_matches += 1

            scored_candidates.append(CandidateResult(
                rank=0, url=c_url, title=c_title, source=c_source, image_url=None,
                evidence_score=ev_score, visual_score=v_score, geometric_score=g_score,
                color_correlation=c_corr, match_status=status, match_type=m_type,
                selection_reason=reason, evidence_breakdown=breakdown
            ))

        timing.verification_sec = round(time.time() - t_verify_start, 2)
        scored_candidates.sort(key=lambda c: c.evidence_score, reverse=True)
        for i, c in enumerate(scored_candidates, 1):
            c.rank = i
        timing.scoring_sec = 0.005
        timing.total_sec = round(timing.upload_sec + timing.lens_exact_sec + timing.lens_visual_sec + timing.verification_sec + timing.scoring_sec, 2)

    else:
        if not api_key:
            api_key = get_api_key(required=True)

        if verbose:
            print("\n" + "=" * 70)
            print("🔍 PART B: REVERSE IMAGE SEARCH & EVIDENCE VERIFICATION ENGINE")
            print("=" * 70)
            print(f"Target Image : {image_path}")
            print(f"SHA-256 Hash : {image_hash}")
            print(f"Diagnostics  : Brightness {quality.mean_brightness:.1f}/255, Blur Var {quality.blur_variance:.1f}")
            if quality.warnings:
                for w in quality.warnings:
                    print(f"  ⚠️ Warning: {w}")
            if was_compressed:
                print(f"  ℹ️ Image exceeded 500 KB limit -> auto-compressed to {os.path.getsize(upload_path) / 1024:.1f} KB for SerpApi")

        # Step 4: Upload to SerpApi
        if verbose:
            print("\n[1/5] 📤 Uploading image to SerpApi Image API...")
        t0 = time.time()
        try:
            image_id = upload_image_to_serpapi(upload_path, api_key)
        finally:
            if was_compressed and os.path.isfile(upload_path):
                try:
                    os.remove(upload_path)
                except Exception:
                    pass

        timing.upload_sec = round(time.time() - t0, 2)
        if verbose:
            print(f"      ✓ Upload complete in {timing.upload_sec}s (image_id: {image_id})")

        # Step 5: Google Lens Exact Matches
        if verbose:
            print("\n[2/5] 🔎 Searching Google Lens exact matches...")
        t0 = time.time()
        exact_results = query_google_lens(image_id, "exact_matches", api_key)
        timing.lens_exact_sec = round(time.time() - t0, 2)
        if verbose:
            print(f"      ✓ Exact matches found: {len(exact_results.get('exact_matches', []))} ({timing.lens_exact_sec}s)")

        # Step 6: Google Lens Visual Matches
        if verbose:
            print("\n[3/5] 🔎 Searching Google Lens visual matches...")
        t0 = time.time()
        visual_results = query_google_lens(image_id, "visual_matches", api_key)
        timing.lens_visual_sec = round(time.time() - t0, 2)
        if verbose:
            print(f"      ✓ Visual matches found: {len(visual_results.get('visual_matches', []))} ({timing.lens_visual_sec}s)")

        candidates = extract_and_merge_candidates(exact_results, visual_results, stats)
        if verbose:
            print(f"      ✓ Unique deduplicated candidates in pool: {stats.unique_candidates}")

        # Step 7: Visual Verification (Multi-Scale ORB + RANSAC + Color Histogram)
        if verbose:
            print(f"\n[4/5] 🔬 Verifying candidates via Multi-Scale ORB + RANSAC + Color (up to {max_verify})...")

        t_verify_start = time.time()
        scored_candidates: List[CandidateResult] = []

        for idx, cand in enumerate(candidates[:max_verify]):
            img_url = cand.get("image_url")
            thumb_url = cand.get("thumbnail_url")

            cand_img, used_fallback = download_candidate_image_with_fallback(img_url, thumb_url)

            if cand_img is not None:
                stats.images_downloaded += 1
                if used_fallback:
                    stats.thumbnail_fallbacks += 1

                v_score, g_score, c_corr, matches_cnt, inliers_cnt = verify_visual_overlap_multiscale(
                    query_image, cand_img
                )
            else:
                v_score, g_score, c_corr, matches_cnt, inliers_cnt = 0.0, 0.0, 0.0, 0, 0

            ev_score, breakdown = score_and_explain_candidate(
                cand, v_score, g_score, c_corr, matches_cnt, inliers_cnt
            )

            status, m_type, reason = classify_candidate(
                cand, ev_score, v_score, g_score, matches_cnt, inliers_cnt
            )

            if status == "VERIFIED_MATCH":
                stats.verified_matches += 1
            elif status == "POSSIBLE_MATCH":
                stats.possible_matches += 1
            else:
                stats.low_confidence_matches += 1

            cand_result = CandidateResult(
                rank=0,
                url=cand.get("url"),
                title=cand.get("title"),
                source=cand.get("source"),
                image_url=img_url or thumb_url,
                evidence_score=ev_score,
                visual_score=v_score,
                geometric_score=g_score,
                color_correlation=c_corr,
                match_status=status,
                match_type=m_type,
                selection_reason=reason,
                evidence_breakdown=breakdown,
            )
            scored_candidates.append(cand_result)

        timing.verification_sec = round(time.time() - t_verify_start, 2)

        t_score_start = time.time()
        scored_candidates.sort(key=lambda c: c.evidence_score, reverse=True)
        for i, c in enumerate(scored_candidates, 1):
            c.rank = i
        timing.scoring_sec = round(time.time() - t_score_start, 3)

        timing.total_sec = round(time.time() - total_start_time, 2)

    # Step 8: Formatted Console Output
    if verbose:
        print("\n" + "=" * 70)
        print("📊 CANDIDATE EVIDENCE RANKING (ALL EVALUATED RESULTS)")
        print("=" * 70)
        if not scored_candidates:
            print("  No online candidates were found.")
        else:
            for c in scored_candidates:
                badge = "🟢 VERIFIED" if c.match_status == "VERIFIED_MATCH" else (
                    "🟡 POSSIBLE" if c.match_status == "POSSIBLE_MATCH" else "⚪ LOW CONFIDENCE"
                )
                print(f"\n#{c.rank} [{badge}] {c.title or 'Untitled'}")
                print(f"   Domain/Source  : {c.source or get_domain(c.url) or 'Unknown'}")
                print(f"   URL            : {c.url or 'N/A'}")
                print(f"   Evidence Score : {c.evidence_score}/100")
                print(f"   Visual Score   : {c.visual_score}/100 ({c.evidence_breakdown.orb_matches_count} ORB matches)")
                print(f"   Geometric Score: {c.geometric_score}/100 ({c.evidence_breakdown.ransac_inliers} RANSAC inliers)")
                print(f"   Color Correl   : {c.color_correlation:.2f}")
                print(f"   Status/Type    : {c.match_status} ({c.match_type or 'NONE'})")
                print("   Evidence Checklist:")
                for item in c.evidence_breakdown.checklist:
                    print(f"     {item}")

        print("\n" + "=" * 70)
        print("📈 SEARCH & VERIFICATION METRICS")
        print("=" * 70)
        print(f"  Exact Results    : {stats.exact_results_count}")
        print(f"  Visual Results   : {stats.visual_results_count}")
        print(f"  Unique Candidates: {stats.unique_candidates}")
        print(f"  Images Verified  : {stats.images_downloaded} ({stats.thumbnail_fallbacks} via CDN fallback)")
        print(f"  Verified Matches : {stats.verified_matches}")
        print(f"  Possible Matches : {stats.possible_matches}")
        print(f"  Low Confidence   : {stats.low_confidence_matches}")
        print("  Latency Breakdown:")
        print(f"    Upload: {timing.upload_sec}s | Lens: {timing.lens_exact_sec + timing.lens_visual_sec:.2f}s | "
              f"Verification: {timing.verification_sec}s | Total: {timing.total_sec}s")

    candidates_dict_list = [c.to_dict() for c in scored_candidates]

    # Step 9: Fail-Safe Decision Engine
    if not scored_candidates:
        match_res = MatchResult(
            matched_url=None,
            title=None,
            source=None,
            source_image=resolved_image_path,
            image_hash_sha256=image_hash,
            match_status="NO_RELIABLE_MATCH",
            match_type=None,
            evidence_score=0.0,
            visual_score=0.0,
            geometric_score=0.0,
            selection_reason="Google Lens returned zero exact or visual candidate occurrences.",
            quality_diagnostics=quality,
            evidence_breakdown=None,
            search_statistics=stats,
            timing=timing,
            candidates=[],
            manifest_sha256=manifest_sha256,
            record_hash=None,
        )
    else:
        best = scored_candidates[0]
        record_hash = None
        if manifest_sha256 and best.url:
            record_hash = hashlib.sha256(f"{manifest_sha256}:{best.url}".encode("utf-8")).hexdigest()

        if best.match_status == "LOW_CONFIDENCE" or best.evidence_score < 35.0:
            if verbose:
                print("\n" + "!" * 70)
                print("⚠️ DECISION: NO_RELIABLE_MATCH (Fail-Safe Triggered)")
                print("   Top candidate did not meet the reliability threshold required for blockchain verification.")
                print("!" * 70)
            match_res = MatchResult(
                matched_url=None,
                title=None,
                source=None,
                source_image=resolved_image_path,
                image_hash_sha256=image_hash,
                match_status="NO_RELIABLE_MATCH",
                match_type=None,
                evidence_score=best.evidence_score,
                visual_score=best.visual_score,
                geometric_score=best.geometric_score,
                selection_reason=(
                    f"Top candidate scored only {best.evidence_score}/100. "
                    "System safely rejected all candidates to prevent false-positive blockchain recording."
                ),
                quality_diagnostics=quality,
                evidence_breakdown=asdict(best.evidence_breakdown),
                search_statistics=stats,
                timing=timing,
                candidates=candidates_dict_list,
                manifest_sha256=manifest_sha256,
                record_hash=None,
            )
        else:
            if verbose:
                status_icon = "✅" if best.match_status == "VERIFIED_MATCH" else "⚠️"
                print("\n" + "=" * 70)
                print(f"{status_icon} FINAL MATCH RESULT: {best.match_status}")
                print("=" * 70)
                print(f"  Matched URL    : {best.url}")
                print(f"  Title          : {best.title}")
                print(f"  Source         : {best.source or get_domain(best.url)}")
                print(f"  Evidence Score : {best.evidence_score}/100")
                print(f"  Visual Score   : {best.visual_score}/100")
                print(f"  Geometric Score: {best.geometric_score}/100")
                print(f"  Match Type     : {best.match_type}")
                print(f"  Reason         : {best.selection_reason}")
                if manifest_sha256:
                    print(f"  Part A Manifest: {manifest_sha256}")
                    print(f"  Part C Hash    : {record_hash}")
                print("=" * 70 + "\n")

            match_res = MatchResult(
                matched_url=best.url,
                title=best.title,
                source=best.source or get_domain(best.url),
                source_image=resolved_image_path,
                image_hash_sha256=image_hash,
                match_status=best.match_status,
                match_type=best.match_type,
                evidence_score=best.evidence_score,
                visual_score=best.visual_score,
                geometric_score=best.geometric_score,
                selection_reason=best.selection_reason,
                quality_diagnostics=quality,
                evidence_breakdown=asdict(best.evidence_breakdown),
                search_statistics=stats,
                timing=timing,
                candidates=candidates_dict_list,
                manifest_sha256=manifest_sha256,
                record_hash=record_hash,
            )

    # Save to local cache
    if use_cache:
        cache_mgr.set(image_hash, match_res.to_dict())

    # Step 10: Generate HTML Report if requested
    if generate_html:
        report_path = generate_html_report(match_res, html_output_path)
        if verbose:
            print(f"📄 Visual HTML Evidence Report generated: {report_path}")

    return match_res


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Part B: Reverse Image Search & Evidence Verification Engine (Advanced Edition)"
    )
    parser.add_argument("image_path", help="Path to input cropped face image")
    parser.add_argument("--json", action="store_true", help="Output canonical JSON payload for Part C")
    parser.add_argument("--report-html", action="store_true", help="Generate standalone visual HTML evidence report")
    parser.add_argument("--html-output", default="verification_report.html", help="Path for HTML report file")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local SHA-256 search cache")
    parser.add_argument("--demo", action="store_true", help="Run simulated demonstration without requiring SerpApi key")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES_TO_VERIFY,
                        help="Maximum candidates to visually verify")

    args = parser.parse_args()

    try:
        result = find_match(
            image_path=args.image_path,
            max_verify=args.max_candidates,
            use_cache=not args.no_cache,
            generate_html=args.report_html,
            html_output_path=args.html_output,
            is_demo=args.demo,
            verbose=not args.json
        )

        if args.json:
            print(result.to_json(indent=2))

    except Exception as err:
        if args.json:
            error_payload = {
                "error": str(err),
                "match_status": "ERROR",
                "matched_url": None
            }
            print(json.dumps(error_payload, indent=2))
        else:
            print(f"\n❌ Error during verification: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
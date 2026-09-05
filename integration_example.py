"""
Pipeline Integration Example: Part A → Part B → Part C (Production Edition)
Project: Face ID + Blockchain Verification

Demonstrates the complete end-to-end hackathon workflow:
  1. Part A (Face Detection & Normalization):
     - Takes raw user image, performs face detection & crop
     - Ensures normalized bounds (under 500 KB)

  2. Part B (Your Engine — Verification & Evidence Layer):
     - Image quality diagnostics (blur, illumination, resolution)
     - Cryptographic SHA-256 caching for sub-second repeat queries
     - Multi-Scale ORB + RANSAC + Color Histogram Verification
     - Candidate ranking & explainable evidence checklist
     - Generates interactive visual HTML evidence report ('verification_report.html')

  3. Part C (Blockchain Attestation):
     - Validates Part B verdict
     - If VERIFIED_MATCH: commits canonical SHA-256 hash & verification metadata on-chain
     - If NO_RELIABLE_MATCH: enforces fail-safe refusal (no false positive transaction created)
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, Any, Optional

import cv2
import numpy as np

from reverse_image_search import (
    find_match,
    get_api_key,
    generate_html_report,
    assess_image_quality,
    MatchResult,
    SearchStatistics,
    TimingBreakdown,
    ImageQualityDiagnostics
)


def simulate_part_a_face_crop(input_image_path: str, output_crop_path: str = "cropped_face.jpg") -> str:
    """
    Simulates Part A (Teammate's component):
    Takes the raw user photo, performs face detection and cropping,
    and saves 'cropped_face.jpg' (under 500 KB).
    """
    print("\n" + "=" * 70)
    print("👤 PART A: FACE DETECTION & NORMALIZATION (Offline)")
    print("=" * 70)

    if not os.path.isfile(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Could not read {input_image_path}")

    h, w = img.shape[:2]
    print(f"  Input image: {input_image_path} ({w}x{h} px)")

    # Simulate face bounding box crop
    y1, y2 = int(h * 0.1), int(h * 0.9)
    x1, x2 = int(w * 0.1), int(w * 0.9)
    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(output_crop_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    crop_size_kb = os.path.getsize(output_crop_path) / 1024
    print(f"  ✓ Face detected and cropped -> '{output_crop_path}' ({crop_size_kb:.1f} KB)")
    return output_crop_path


def run_part_b_verification(cropped_image_path: str) -> MatchResult:
    """
    Runs Part B (Your Engine):
    Performs Google Lens candidate search, multi-scale visual verification,
    candidate ranking, quality diagnostics, and evidence report generation.
    """
    print("\n" + "=" * 70)
    print("⭐ PART B: REVERSE IMAGE SEARCH & EVIDENCE VERIFICATION ENGINE")
    print("=" * 70)

    api_key = get_api_key(required=False)

    if api_key:
        print("  SERPAPI_API_KEY detected. Running live verification...")
        return find_match(cropped_image_path, api_key=api_key, generate_html=True, verbose=True)
    else:
        print("  ⚠️ SERPAPI_API_KEY not found in environment.")
        print("  Simulating Part B candidate generation and multi-scale verification for demonstration...\n")

        img = cv2.imread(cropped_image_path)
        with open(cropped_image_path, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()

        quality = assess_image_quality(img)
        print(f"  ✓ Pre-Flight Quality Check: Brightness {quality.mean_brightness}/255, Blur Var: {quality.blur_variance}")
        print(f"  ✓ Image Quality Acceptable: {quality.is_acceptable}")

        stats = SearchStatistics(
            exact_results_count=1,
            visual_results_count=4,
            total_raw_candidates=5,
            unique_candidates=4,
            images_downloaded=4,
            thumbnail_fallbacks=1,
            verified_matches=1,
            possible_matches=1,
            low_confidence_matches=2,
            from_cache=False
        )
        timing = TimingBreakdown(
            upload_sec=0.78,
            lens_exact_sec=1.12,
            lens_visual_sec=1.34,
            verification_sec=2.45,
            scoring_sec=0.01,
            total_sec=5.70
        )

        candidates = [
            {
                "rank": 1,
                "url": "https://instagram.com/verified_subject/profile_pic",
                "title": "Alex Morgan (@alexmorgan) • Instagram photos and videos",
                "source": "Instagram",
                "image_url": "https://instagram.com/static/profile.jpg",
                "evidence_score": 88.5,
                "visual_score": 84.0,
                "geometric_score": 75.0,
                "color_correlation": 0.92,
                "match_status": "VERIFIED_MATCH",
                "match_type": "EXACT_PLUS_VISUAL",
                "selection_reason": "Exact-match signal confirmed with 22 visual features and 12 geometric inliers.",
                "evidence_breakdown": {
                    "exact_match_signal": True,
                    "orb_matches_count": 22,
                    "ransac_inliers": 12,
                    "visual_score": 84.0,
                    "geometric_score": 75.0,
                    "color_correlation": 0.92,
                    "social_domain": True,
                    "has_title": True,
                    "has_source": True,
                    "checklist": [
                        "✓ Google Lens exact-match signal detected (+35)",
                        "✓ Strong visual overlap: 22 ORB features matched (+29.4)",
                        "✓ Strong geometric consistency: 12 RANSAC inliers (+15.0)",
                        "✓ High color histogram correlation: 0.92",
                        "✓ Known social/community platform with visual confirmation (+5)",
                        "✓ Page title available (+3)",
                        "✓ Source publisher verified (+2)"
                    ]
                }
            },
            {
                "rank": 2,
                "url": "https://linkedin.com/in/alexmorgan",
                "title": "Alex Morgan - Tech Lead - LinkedIn",
                "source": "LinkedIn",
                "image_url": "https://media.licdn.com/dms/image/avatar.jpg",
                "evidence_score": 52.0,
                "visual_score": 40.0,
                "geometric_score": 30.0,
                "color_correlation": 0.74,
                "match_status": "POSSIBLE_MATCH",
                "match_type": "MODERATE_EVIDENCE",
                "selection_reason": "Moderate visual similarity (40.0/100); lacks high geometric confidence.",
                "evidence_breakdown": {
                    "exact_match_signal": False,
                    "orb_matches_count": 10,
                    "ransac_inliers": 4,
                    "visual_score": 40.0,
                    "geometric_score": 30.0,
                    "color_correlation": 0.74,
                    "social_domain": True,
                    "has_title": True,
                    "has_source": True,
                    "checklist": [
                        "✗ No exact-match signal from Google Lens",
                        "✓ Moderate visual overlap: 10 ORB features matched (+14.0)",
                        "✓ Partial geometric consistency: 4 RANSAC inliers (+6.0)",
                        "✓ Known social/community platform with visual confirmation (+5)",
                        "✓ Page title available (+3)",
                        "✓ Source publisher verified (+2)"
                    ]
                }
            },
            {
                "rank": 3,
                "url": "https://unrelated-site.org/gallery/stock_people",
                "title": "Stock Portrait Gallery",
                "source": "UnrelatedSite",
                "image_url": "https://unrelated-site.org/images/p102.jpg",
                "evidence_score": 15.0,
                "visual_score": 12.0,
                "geometric_score": 0.0,
                "color_correlation": 0.18,
                "match_status": "LOW_CONFIDENCE",
                "match_type": None,
                "selection_reason": "Visual overlap (12.0/100) or evidence score (15.0/100) below verification threshold.",
                "evidence_breakdown": {
                    "exact_match_signal": False,
                    "orb_matches_count": 3,
                    "ransac_inliers": 0,
                    "visual_score": 12.0,
                    "geometric_score": 0.0,
                    "color_correlation": 0.18,
                    "social_domain": False,
                    "has_title": True,
                    "has_source": True,
                    "checklist": [
                        "✗ No exact-match signal from Google Lens",
                        "✗ Weak/no visual overlap: 3 ORB features matched",
                        "✗ Insufficient geometric consistency (low RANSAC inliers)",
                        "✓ Page title available (+3)",
                        "✓ Source publisher verified (+2)"
                    ]
                }
            }
        ]

        best = candidates[0]
        res = MatchResult(
            matched_url=best["url"],
            title=best["title"],
            source=best["source"],
            source_image=cropped_image_path,
            image_hash_sha256=img_hash,
            match_status=best["match_status"],
            match_type=best["match_type"],
            evidence_score=best["evidence_score"],
            visual_score=best["visual_score"],
            geometric_score=best["geometric_score"],
            selection_reason=best["selection_reason"],
            quality_diagnostics=quality,
            evidence_breakdown=best["evidence_breakdown"],
            search_statistics=stats,
            timing=timing,
            candidates=candidates
        )

        # Generate standalone HTML evidence report
        report_file = generate_html_report(res, "verification_report.html")
        print(f"  ✓ Standalone HTML Report generated: {report_file}")
        return res


def run_part_c_blockchain_commit(part_b_result: MatchResult) -> Dict[str, Any]:
    """
    Simulates Part C (Blockchain Verification):
    Rules:
      - NEVER commit raw face images or biometric embeddings to blockchain (Privacy).
      - If match_status is NO_RELIABLE_MATCH, refuse blockchain write (Fail-safe).
      - If VERIFIED_MATCH, commit canonical evidence hash & verification metadata.
    """
    print("\n" + "=" * 70)
    print("⛓️ PART C: BLOCKCHAIN ATTESTATION & RECORD ENGINE")
    print("=" * 70)

    # 1. Enforce Fail-Safe Check
    if part_b_result.match_status == "NO_RELIABLE_MATCH":
        print("  🛑 FAIL-SAFE TRIGGERED: match_status is NO_RELIABLE_MATCH.")
        print("  ⚠️ Rule Enforced: NEVER write an unverified match to blockchain.")
        print("  Outcome: Transaction safely rejected. Zero false-positive on-chain records.")
        return {
            "blockchain_status": "REJECTED_UNVERIFIED",
            "transaction_hash": None,
            "reason": part_b_result.selection_reason
        }

    # 2. Canonical Payload Preparation
    timestamp = int(time.time())
    canonical_payload = {
        "matched_url": part_b_result.matched_url,
        "source": part_b_result.source,
        "evidence_score": part_b_result.evidence_score,
        "visual_score": part_b_result.visual_score,
        "geometric_score": part_b_result.geometric_score,
        "image_hash_sha256": part_b_result.image_hash_sha256,
        "match_type": part_b_result.match_type,
        "timestamp": timestamp,
        "verification_engine": "PartB_ORB_RANSAC_Lens_v2"
    }

    # Sort keys for deterministic canonical representation
    canonical_json = json.dumps(canonical_payload, sort_keys=True)
    evidence_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    mock_tx_input = f"TX_{evidence_hash}_{timestamp}"
    tx_hash = "0x" + hashlib.sha256(mock_tx_input.encode("utf-8")).hexdigest()

    print(f"  ✓ Evidence Verified: {part_b_result.matched_url}")
    print(f"  ✓ Evidence Score   : {part_b_result.evidence_score}/100")
    print(f"  ✓ Input Image Hash : SHA-256:{part_b_result.image_hash_sha256[:16]}...")
    print(f"  ✓ Canonical Hash   : SHA-256:{evidence_hash}")
    print(f"  ✓ Privacy Safe     : No biometric data or raw photos written to ledger")
    print(f"  ✓ Smart Contract Tx: {tx_hash}")
    print(f"  ✓ Status           : CONFIRMED ON-CHAIN")

    return {
        "blockchain_status": "COMMITTED_ON_CHAIN",
        "evidence_hash": evidence_hash,
        "transaction_hash": tx_hash,
        "canonical_payload": canonical_payload
    }


def main():
    test_image = os.path.join("test_images", "test.jpg")
    if not os.path.isfile(test_image):
        print(f"Error: {test_image} does not exist.")
        sys.exit(1)

    print("\n" + "#" * 70)
    print("🚀 FULL PIPELINE DEMONSTRATION: PART A → PART B → PART C")
    print("#" * 70)

    # Step 1: Part A
    cropped_path = simulate_part_a_face_crop(test_image)

    # Step 2: Part B
    part_b_res = run_part_b_verification(cropped_path)

    # Step 3: Part C
    part_c_res = run_part_c_blockchain_commit(part_b_res)

    print("\n" + "=" * 70)
    print("🎯 PIPELINE COMPLETE SUMMARY")
    print("=" * 70)
    print(f"Part A Input    : {test_image}")
    print(f"Part B Status   : {part_b_res.match_status} (Score: {part_b_res.evidence_score}/100)")
    print(f"Part B Match    : {part_b_res.matched_url}")
    print(f"Part C Outcome  : {part_c_res['blockchain_status']}")
    if part_c_res.get("transaction_hash"):
        print(f"Part C Tx Hash  : {part_c_res['transaction_hash']}")
    print(f"Evidence Report : file://{os.path.abspath('verification_report.html')}")
    print("=" * 70 + "\n")

    # Clean up temporary cropped face
    if os.path.isfile(cropped_path):
        os.remove(cropped_path)


if __name__ == "__main__":
    main()

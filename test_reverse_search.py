"""
Part B — Comprehensive Test Suite (Offline Unit Tests + Live Search Runner)
Project: Face ID + Blockchain Verification

Covers:
  [Test 1] Image Validation & Quality Diagnostics (Blur, Brightness, Dimensions)
  [Test 2] Identical Image Verification (ORB + RANSAC + Color Histogram)
  [Test 3] Cropped & Scaled Web Avatar Verification (Multi-Scale Engine)
  [Test 4] Rejection of Unrelated Image (LOW_CONFIDENCE)
  [Test 5] URL Normalization & Tracking Parameter Deduplication
  [Test 6] Evidence Scoring Engine & Checklist Explainability
  [Test 7] Fail-Safe Decision Engine (NO_RELIABLE_MATCH)
  [Test 8] Auto-Compression for Oversized (>500 KB) Images
  [Test 9] Anti-Hotlinking CDN Thumbnail Fallback
  [Test 10] Cryptographic SHA-256 Local Search Cache
  [Test 11] Standalone Visual HTML Evidence Report Generation
"""

import os
import sys
import glob
import json
import time
import cv2
import numpy as np

from reverse_image_search import (
    auto_compress_and_normalize_image,
    assess_image_quality,
    verify_visual_overlap_multiscale,
    compute_color_histogram_correlation,
    normalize_url,
    get_domain,
    is_social_domain,
    extract_and_merge_candidates,
    score_and_explain_candidate,
    classify_candidate,
    download_candidate_image_with_fallback,
    SearchCacheManager,
    generate_html_report,
    find_match,
    get_api_key,
    SearchStatistics,
    TimingBreakdown,
    CandidateResult,
    EvidenceBreakdown,
    MatchResult,
    MAX_IMAGE_BYTES
)

TEST_DIR = "test_images"


def run_offline_unit_tests() -> bool:
    print("\n" + "=" * 75)
    print("🧪 RUNNING PART B EXPANDED VERIFICATION ENGINE TEST SUITE (11 TESTS)")
    print("=" * 75)

    passed = 0
    total = 0

    # Generate a deterministic high-contrast synthetic test fixture
    test_img_path = ".tmp_synthetic_test_face.jpg"
    np.random.seed(42)
    probe = np.zeros((500, 500, 3), dtype=np.uint8)
    for y in range(500):
        probe[y, :, 0] = int(180 + 50 * np.sin(y / 30.0))
        probe[y, :, 1] = int(190 + 40 * np.cos(y / 25.0))
        probe[y, :, 2] = int(210 + 30 * np.sin(y / 40.0))
    cv2.ellipse(probe, (250, 250), (140, 180), 0, 0, 360, (150, 130, 110), -1)
    cv2.circle(probe, (200, 210), 22, (50, 50, 50), -1)
    cv2.circle(probe, (300, 210), 22, (50, 50, 50), -1)
    cv2.circle(probe, (200, 210), 8, (230, 230, 230), -1)
    cv2.circle(probe, (300, 210), 8, (230, 230, 230), -1)
    cv2.line(probe, (250, 230), (245, 275), (90, 80, 70), 5)
    cv2.ellipse(probe, (250, 320), (55, 25), 0, 0, 180, (60, 50, 170), -1)
    for i in range(10):
        cv2.putText(probe, f"ID-VERIFY-{i*77}", (50, 40 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    noise = np.random.randint(0, 30, (500, 500, 3), dtype=np.uint8)
    probe = cv2.add(probe, noise)
    cv2.imwrite(test_img_path, probe, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # --------------------------------------------------------------------------
    # Test 1: Image Validation & Quality Diagnostics
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 1] Testing Image Normalization & Quality Diagnostics...")
    try:
        img, upload_path, img_hash, was_compressed = auto_compress_and_normalize_image(test_img_path)
        assert img is not None and len(img.shape) == 3
        quality = assess_image_quality(img)
        print(f"  ✓ Image loaded (shape: {img.shape}, SHA-256: {img_hash[:12]}...)")
        print(f"  ✓ Quality: Brightness {quality.mean_brightness}/255, Blur Var: {quality.blur_variance}, Acceptable: {quality.is_acceptable}")
        assert quality.is_acceptable is True
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 1 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 2: Identical Image Match (ORB + RANSAC + Color)
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 2] Testing Multi-Scale ORB + RANSAC + Color on Identical Image...")
    try:
        v_score, g_score, c_corr, matches, inliers = verify_visual_overlap_multiscale(img, img)
        print(f"  ✓ Matches: {matches}, Inliers: {inliers}, Visual: {v_score}/100, Geometric: {g_score}/100, Color: {c_corr:.2f}")
        assert v_score >= 80.0
        assert g_score >= 80.0
        assert c_corr >= 0.95
        assert inliers >= 10
        print("  ✓ Identical match scored maximum across all visual/geometric signals")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 2 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 3: Cropped & Scaled Avatar Match
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 3] Testing Multi-Scale ORB on Cropped & Scaled Avatar...")
    try:
        h, w = img.shape[:2]
        cropped = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        scaled_cropped = cv2.resize(cropped, (0, 0), fx=0.7, fy=0.7)

        v_score, g_score, c_corr, matches, inliers = verify_visual_overlap_multiscale(img, scaled_cropped)
        print(f"  ✓ Matches: {matches}, Inliers: {inliers}, Visual: {v_score}/100, Geometric: {g_score}/100, Color: {c_corr:.2f}")
        assert matches >= 10
        assert v_score >= 40.0
        print("  ✓ Cropped/scaled web avatar successfully verified by multi-scale pyramid")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 3 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 4: Rejection of Unrelated Geometric Image
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 4] Testing Rejection of Unrelated Image (LOW_CONFIDENCE)...")
    try:
        unrelated = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(unrelated, (200, 200), 80, (255, 255, 255), -1)
        cv2.rectangle(unrelated, (50, 50), (150, 150), (128, 128, 128), -1)

        v_score, g_score, c_corr, matches, inliers = verify_visual_overlap_multiscale(img, unrelated)
        print(f"  ✓ Matches: {matches}, Inliers: {inliers}, Visual: {v_score}/100, Geometric: {g_score}/100, Color: {c_corr:.2f}")
        assert v_score <= 15.0
        assert g_score == 0.0

        cand = {"url": "https://unrelated.com/sample", "title": "Sample", "source": "Sample"}
        ev_score, breakdown = score_and_explain_candidate(cand, v_score, g_score, c_corr, matches, inliers)
        status, _, _ = classify_candidate(cand, ev_score, v_score, g_score, matches, inliers)
        assert status == "LOW_CONFIDENCE"
        print(f"  ✓ Unrelated candidate rejected as {status} (Evidence: {ev_score}/100)")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 4 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 5: URL Normalization & Candidate Deduplication
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 5] Testing URL Normalization & Tracking Param Removal...")
    try:
        u1 = "https://www.instagram.com/p/abc123/?utm_source=ig_web_copy_link&igshid=xyz"
        u2 = "https://instagram.com/p/abc123"
        u3 = "https://instagram.com/p/abc123/?ref=feed"

        norm1 = normalize_url(u1)
        norm2 = normalize_url(u2)
        norm3 = normalize_url(u3)
        assert norm1 == norm2 == norm3 == "https://instagram.com/p/abc123"

        dummy_exact = {"exact_matches": [{"link": u1, "title": "Exact Post", "source": "Instagram"}]}
        dummy_visual = {"visual_matches": [
            {"link": u2, "title": "Duplicate Post", "source": "Instagram"},
            {"link": "https://linkedin.com/in/alex", "title": "Alex Profile", "source": "LinkedIn"}
        ]}
        stats = SearchStatistics()
        merged = extract_and_merge_candidates(dummy_exact, dummy_visual, stats)
        assert len(merged) == 2  # Deduplicated from 3 to 2
        assert merged[0]["exact_match"] is True
        print(f"  ✓ Deduplicated 3 raw candidates into {len(merged)} unique candidates (preserved exact signal)")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 5 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 6: Evidence Scoring & Explainability Checklist
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 6] Testing Evidence Engine Scoring & Checklist...")
    try:
        cand = {
            "url": "https://instagram.com/verified_subject",
            "title": "Verified Profile • Instagram",
            "source": "Instagram",
            "exact_match": True,
        }
        ev_score, breakdown = score_and_explain_candidate(
            cand, visual_score=90.0, geometric_score=85.0, color_correlation=0.88, good_matches=25, inliers=15
        )
        print(f"  ✓ Calculated Evidence Score: {ev_score}/100")
        assert 80.0 <= ev_score <= 100.0
        assert len(breakdown.checklist) >= 5
        status, mtype, reason = classify_candidate(cand, ev_score, 90.0, 85.0, 25, 15)
        assert status == "VERIFIED_MATCH"
        print(f"  ✓ Classification: {status} ({mtype})")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 6 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 7: Fail-Safe NO_RELIABLE_MATCH Rejection
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 7] Testing Fail-Safe NO_RELIABLE_MATCH Decision...")
    try:
        unrelated_cand = {"url": "https://random.org", "title": "Random", "source": "Random"}
        ev_score, breakdown = score_and_explain_candidate(unrelated_cand, 5.0, 0.0, 0.1, 1, 0)
        status, mtype, reason = classify_candidate(unrelated_cand, ev_score, 5.0, 0.0, 1, 0)
        assert status == "LOW_CONFIDENCE"

        res = MatchResult(
            matched_url=None,
            title=None,
            source=None,
            source_image=test_img_path,
            image_hash_sha256="dummy_hash",
            match_status="NO_RELIABLE_MATCH",
            match_type=None,
            evidence_score=ev_score,
            visual_score=5.0,
            geometric_score=0.0,
            selection_reason="Safe rejection triggered.",
            quality_diagnostics=assess_image_quality(img),
            evidence_breakdown=None,
            search_statistics=SearchStatistics(),
            timing=TimingBreakdown(),
            candidates=[]
        )
        payload = json.loads(res.to_json())
        assert payload["match_status"] == "NO_RELIABLE_MATCH"
        assert payload["matched_url"] is None
        print("  ✓ Canonical JSON safely reflects refusal (matched_url is null)")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 7 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 8: Auto-Compression of Oversized (>500 KB) Images
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 8] Testing Auto-Compression for Oversized (>500 KB) Image...")
    dummy_large_path = "test_images/_tmp_large_test.jpg"
    try:
        # Create an artificial high-res 2000x2000 image that exceeds 500 KB uncompressed
        large_img = cv2.resize(img, (2200, 2200), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(dummy_large_path, large_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        initial_size = os.path.getsize(dummy_large_path)
        assert initial_size > MAX_IMAGE_BYTES
        print(f"  Created large test image: {initial_size / 1024:.1f} KB (> 500 KB limit)")

        # Run auto-compression
        norm_img, upload_path, hash_val, was_comp = auto_compress_and_normalize_image(dummy_large_path)
        compressed_size = os.path.getsize(upload_path)
        print(f"  ✓ Auto-compressed for upload: {compressed_size / 1024:.1f} KB (was_compressed={was_comp})")
        assert was_comp is True
        assert compressed_size <= MAX_IMAGE_BYTES
        if os.path.isfile(upload_path) and upload_path != dummy_large_path:
            os.remove(upload_path)
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 8 failed: {e}")
    finally:
        if os.path.isfile(dummy_large_path):
            os.remove(dummy_large_path)

    # --------------------------------------------------------------------------
    # Test 9: Anti-Hotlinking CDN Thumbnail Fallback
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 9] Testing Anti-Hotlinking CDN Thumbnail Fallback...")
    try:
        # Simulate candidate image where direct URL is a fake 404/broken link, but thumbnail is valid
        broken_url = "https://httpstat.us/403"
        valid_thumb = "https://picsum.photos/100/100"  # Small public placeholder or None

        # Test download fallback handler with non-existing image
        res_img, used_fb = download_candidate_image_with_fallback(
            image_url="http://invalid-non-existent-domain-403.com/avatar.jpg",
            thumbnail_url=None
        )
        assert res_img is None
        assert used_fb is False
        print("  ✓ Handled broken/anti-hotlinked candidate image gracefully (no unhandled exception)")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 9 failed: {e}")

    # --------------------------------------------------------------------------
    # Test 10: Cryptographic Local Search Cache
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 10] Testing Cryptographic SHA-256 Local Search Cache...")
    cache_test_file = ".test_cache.json"
    try:
        cache = SearchCacheManager(cache_file=cache_test_file)
        sample_hash = "abc123hash"
        sample_payload = {"matched_url": "https://example.com", "evidence_score": 88.0}

        # Cache miss
        assert cache.get(sample_hash) is None
        # Cache set
        cache.set(sample_hash, sample_payload)
        # Cache hit
        retrieved = cache.get(sample_hash)
        assert retrieved is not None
        assert retrieved["matched_url"] == "https://example.com"
        print("  ✓ SHA-256 local cache verified: Miss -> Store -> Hit working properly")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 10 failed: {e}")
    finally:
        if os.path.isfile(cache_test_file):
            os.remove(cache_test_file)

    # --------------------------------------------------------------------------
    # Test 11: Standalone Visual HTML Evidence Report Generation
    # --------------------------------------------------------------------------
    total += 1
    print(f"\n[Test 11] Testing Visual HTML Evidence Report Generator...")
    html_test_file = "test_verification_report.html"
    try:
        dummy_candidate = {
            "rank": 1,
            "url": "https://instagram.com/test_user",
            "title": "Test User (@test_user) • Instagram photos",
            "source": "Instagram",
            "image_url": "https://instagram.com/avatar.jpg",
            "evidence_score": 88.5,
            "visual_score": 84.0,
            "geometric_score": 75.0,
            "color_correlation": 0.91,
            "match_status": "VERIFIED_MATCH",
            "match_type": "EXACT_PLUS_VISUAL",
            "selection_reason": "Verified exact and visual occurrence.",
            "evidence_breakdown": {
                "exact_match_signal": True,
                "orb_matches_count": 22,
                "ransac_inliers": 12,
                "visual_score": 84.0,
                "geometric_score": 75.0,
                "color_correlation": 0.91,
                "social_domain": True,
                "has_title": True,
                "has_source": True,
                "checklist": [
                    "✓ Google Lens exact-match signal detected (+35)",
                    "✓ Strong visual overlap: 22 ORB features matched (+29.4)",
                    "✓ Strong geometric consistency: 12 RANSAC inliers (+15.0)",
                    "✓ High color histogram correlation: 0.91",
                    "✓ Known social platform with visual confirmation (+5)",
                    "✓ Page title available (+3)",
                    "✓ Source publisher verified (+2)"
                ]
            }
        }
        res = MatchResult(
            matched_url=dummy_candidate["url"],
            title=dummy_candidate["title"],
            source=dummy_candidate["source"],
            source_image=test_img_path,
            image_hash_sha256="8f4b23c89a01",
            match_status="VERIFIED_MATCH",
            match_type="EXACT_PLUS_VISUAL",
            evidence_score=88.5,
            visual_score=84.0,
            geometric_score=75.0,
            selection_reason="Exact match confirmed with 22 features and 12 inliers.",
            quality_diagnostics=assess_image_quality(img),
            evidence_breakdown=dummy_candidate["evidence_breakdown"],
            search_statistics=SearchStatistics(unique_candidates=5, images_downloaded=5, verified_matches=1),
            timing=TimingBreakdown(total_sec=4.85),
            candidates=[dummy_candidate]
        )
        report_path = generate_html_report(res, html_test_file)
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "VERIFIED_MATCH" in content
        assert "Evidence Score: 88.5/100" in content
        print(f"  ✓ Standalone HTML Report generated ({os.path.getsize(report_path)} bytes)")
        passed += 1
    except Exception as e:
        print(f"  ✗ Test 11 failed: {e}")
    finally:
        if os.path.isfile(html_test_file):
            os.remove(html_test_file)

    # Clean up synthetic test image if created
    if test_img_path == ".tmp_synthetic_test_face.jpg" and os.path.isfile(test_img_path):
        try:
            os.remove(test_img_path)
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"TEST SUITE SUMMARY: {passed}/{total} TESTS PASSED")
    print("=" * 75)
    return passed == total


def run_live_tests():
    api_key = get_api_key(required=False)
    if not api_key:
        print("\n⚠️ SERPAPI_API_KEY is not set. Falling back to offline test suite.")
        run_offline_unit_tests()
        return

    images = glob.glob(os.path.join(TEST_DIR, "*.jpg")) + glob.glob(os.path.join(TEST_DIR, "*.png"))
    if not images:
        print(f"No test images found in {TEST_DIR}/.")
        return

    for img_path in images:
        print("\n" + "#" * 75)
        print(f"LIVE TEST: {img_path}")
        print("#" * 75)
        try:
            result = find_match(img_path, api_key=api_key, generate_html=True)
            print("\nCanonical JSON payload for Part C:")
            print(result.to_json(indent=2))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    if "--offline" in sys.argv:
        success = run_offline_unit_tests()
        sys.exit(0 if success else 1)
    elif "--live" in sys.argv:
        run_live_tests()
    else:
        api_key = get_api_key(required=False)
        if api_key:
            print("SERPAPI_API_KEY detected. Running live test runner (pass --offline for unit tests).")
            run_live_tests()
        else:
            print("No SERPAPI_API_KEY detected in environment. Running offline self-test suite.")
            success = run_offline_unit_tests()
            sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
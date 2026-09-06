#!/usr/bin/env python3
"""
Unified End-to-End Pipeline: Part A (Face Detection) → Part B (Verification Engine) → Part C (Blockchain)
Project: Face ID + Blockchain Verification

Usage:
    # 1. Full pipeline from raw photo:
    python pipeline.py --image photo.jpg --subject self --consent

    # 2. Full pipeline from live webcam:
    python pipeline.py --webcam --subject self --consent

    # 3. Resume from existing Part A scan directory or manifest:
    python pipeline.py --manifest out/manifest.json
    python pipeline.py --scan-dir out/

    # 4. Offline hackathon demo mode:
    python pipeline.py --demo
"""

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Import Part B engine
from reverse_image_search import (
    find_match,
    get_api_key,
    generate_html_report,
    MatchResult
)

# Import Part C blockchain engine
from partC import (
    store_and_verify,
    build_data_hash,
    is_live_configured
)

HERE = Path(__file__).parent


def run_part_a(
    image_path: Optional[str] = None,
    use_webcam: bool = False,
    subject: str = "self",
    out_dir: str = "out",
    force_quality: bool = False
) -> Tuple[int, Path]:
    """
    Executes Part A face detection & provenance generation.
    Returns: (exit_code, manifest_path)
    """
    print("\n" + "=" * 70)
    print("👤 STEP 1: PART A — FACE SCAN, DETECTION & PROVENANCE")
    print("=" * 70)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "manifest.json"

    # Execute Part A engine (with full interactive webcam GUI, face alignment oval, and quality checks)
    cmd = [sys.executable, str(HERE / "part_a.py"), "scan", "--subject", subject, "--consent", "--out", out_dir]
    if use_webcam:
        cmd.append("--webcam")
    elif image_path:
        cmd.extend(["--image", image_path])
    if force_quality:
        cmd.append("--force")

    print(f"  Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  ⚠️ Part A failed or exited with code {proc.returncode}")
        return proc.returncode, manifest_path

    print(f"  ✓ Part A successfully completed. Manifest at: {manifest_path}")
    return 0, manifest_path


def run_part_b(
    manifest_or_crop_path: str,
    api_key: Optional[str] = None,
    is_demo: bool = False,
    report_html: bool = True
) -> MatchResult:
    """
    Executes Part B verification engine taking input from Part A.
    """
    print("\n" + "=" * 70)
    print("⭐ STEP 2: PART B — REVERSE IMAGE SEARCH & EVIDENCE ENGINE")
    print("=" * 70)

    result = find_match(
        image_path=manifest_or_crop_path,
        api_key=api_key,
        generate_html=report_html,
        html_output_path="verification_report.html",
        is_demo=is_demo,
        verbose=True
    )
    return result


def run_part_c(match_result: MatchResult) -> Dict[str, Any]:
    """
    Executes Part C blockchain attestation on Ethereum Sepolia.
    Enforces hard fail-safe gate: NEVER commit unverified matches (NO_RELIABLE_MATCH) to blockchain.
    """
    print("\n" + "=" * 70)
    print("⛓️ STEP 3: PART C — BLOCKCHAIN ATTESTATION (Ethereum Sepolia)")
    print("=" * 70)

    # Convert MatchResult to canonical Part C dictionary
    match_dict = match_result.to_dict()

    # Call Part C module directly
    part_c_res = store_and_verify(match_dict)

    if part_c_res.get("status") == "SKIPPED":
        print(f"  🛑 FAIL-SAFE TRIGGERED: {part_c_res.get('reason')}")
        print("  ⚠️ Rule Enforced: NEVER write an unverified match to blockchain.")
        print("  Outcome: Transaction safely rejected. Zero false-positive on-chain records.")
        return part_c_res

    mode = part_c_res.get("mode", "LIVE_ON_CHAIN")
    print(f"  ✓ Status            : {part_c_res.get('status')} ({mode})")
    print(f"  ✓ Network           : {part_c_res.get('network', 'Ethereum Sepolia')}")
    print(f"  ✓ Data Hash (bytes32): {part_c_res.get('data_hash')}")
    print(f"  ✓ Transaction Hash  : {part_c_res.get('tx_hash')}")
    if part_c_res.get("block_number"):
        print(f"  ✓ Block Number      : {part_c_res.get('block_number')}")
    if part_c_res.get("explorer_url"):
        print(f"  ✓ Explorer URL      : {part_c_res.get('explorer_url')}")

    verif = part_c_res.get("verification", {})
    if verif:
        print(f"  ✓ On-Chain Readback : {verif.get('indicator', 'Verified')}")
        print(f"  ✓ On-Chain URL      : {verif.get('on_chain_url')}")
        print(f"  ✓ On-Chain Submitter: {verif.get('on_chain_submitter')}")

    return part_c_res


def main():
    parser = argparse.ArgumentParser(description="End-to-End Face ID + Blockchain Pipeline (Part A → B → C)")
    parser.add_argument("--image", help="Path to raw user photo")
    parser.add_argument("--webcam", action="store_true", help="Capture live face scan from webcam")
    parser.add_argument("--subject", default="self", help="Subject identifier (e.g. 'self')")
    parser.add_argument("--consent", action="store_true", default=True, help="Assert user consent")
    parser.add_argument("--out", default="out", help="Output directory for Part A crops & manifest")
    parser.add_argument("--manifest", help="Directly provide existing Part A manifest.json")
    parser.add_argument("--scan-dir", help="Directly provide existing Part A output directory")
    parser.add_argument("--demo", action="store_true", help="Run simulated hackathon demonstration without API key")
    parser.add_argument("--json", action="store_true", help="Output final canonical JSON for Part C")
    parser.add_argument("--force", action="store_true", help="Bypass quality gates if needed")

    args = parser.parse_args()

    # Determine input mode
    manifest_path = None
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.scan_dir:
        manifest_path = Path(args.scan_dir) / "manifest.json"
    elif args.webcam or args.image:
        exit_code, manifest_path = run_part_a(
            image_path=args.image,
            use_webcam=args.webcam,
            subject=args.subject,
            out_dir=args.out,
            force_quality=args.force
        )
        if exit_code != 0:
            print(f"\n❌ Part A aborted with exit code {exit_code}. Pipeline stopped.")
            sys.exit(exit_code)
    elif args.demo:
        # For standalone demo mode without webcam or image, synthesize a sample face
        demo_img_path = Path(args.out) / "demo_probe.jpg"
        Path(args.out).mkdir(parents=True, exist_ok=True)
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
        for i in range(8):
            cv2.putText(probe, f"DEMO-SAMPLE-{i*99}", (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        cv2.imwrite(str(demo_img_path), probe)

        exit_code, manifest_path = run_part_a(
            image_path=str(demo_img_path),
            use_webcam=False,
            subject=args.subject,
            out_dir=args.out,
            force_quality=True
        )
        if exit_code != 0:
            print(f"\n❌ Part A aborted with exit code {exit_code}. Pipeline stopped.")
            sys.exit(exit_code)
    else:
        # Check if an existing scan directory with manifest.json exists
        default_manifest = Path(args.out) / "manifest.json"
        if default_manifest.is_file():
            print(f"ℹ️ No input specified. Found existing Part A manifest at '{default_manifest}'. Resuming...")
            manifest_path = default_manifest
        else:
            print("\n❌ Error: No input specified.")
            print("\nUsage options:")
            print("  1. Capture from webcam : python pipeline.py --webcam --subject self --consent")
            print("  2. Use an image file   : python pipeline.py --image path/to/photo.jpg --subject self --consent")
            print("  3. Resume from Part A  : python pipeline.py --manifest out/manifest.json")
            print("  4. Run simulated demo  : python pipeline.py --demo")
            sys.exit(1)

    # Execute Part B
    part_b_result = run_part_b(
        manifest_or_crop_path=str(manifest_path),
        is_demo=args.demo or (not get_api_key(required=False)),
        report_html=True
    )

    # Execute Part C
    part_c_result = run_part_c(part_b_result)

    if args.json:
        final_payload = {
            "part_a": {
                "manifest_sha256": part_b_result.manifest_sha256,
            },
            "part_b": part_b_result.to_dict(),
            "part_c": part_c_result
        }
        print("\n" + json.dumps(final_payload, indent=2))
    else:
        print("\n" + "=" * 70)
        print("🎯 PIPELINE EXECUTION COMPLETE — HACKATHON VERIFICATION SUITE")
        print("=" * 70)
        print(f"Part A Manifest  : {part_b_result.manifest_sha256}")
        print(f"Part B Status    : {part_b_result.match_status} ({part_b_result.evidence_score}/100)")
        print(f"Part B Match     : {part_b_result.matched_url}")
        
        # Feature 1 & 2 Summary
        if part_b_result.match_visualization_path:
            print(f"Feature 1 Proof  : Inlier matchlines saved to '{part_b_result.match_visualization_path}'")
        if part_b_result.deepfake_analysis:
            df = part_b_result.deepfake_analysis
            print(f"Feature 2 Deepfake: {df.get('verdict')} (Confidence: {df.get('confidence_score', 0):.1f}%, CMOS Noise: {df.get('sensor_noise_variance', 0):.2f})")
            
        print(f"Part C Status    : {part_c_result.get('status')} ({part_c_result.get('mode', 'LIVE')})")
        print(f"Part C DataHash  : {part_c_result.get('data_hash')}")
        print(f"Part C Tx Hash   : {part_c_result.get('tx_hash')}")
        if part_c_result.get("explorer_url"):
            print(f"Part C Explorer  : {part_c_result.get('explorer_url')}")
            
        # Feature 4 Mobile QR Code
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={part_c_result.get('explorer_url', 'https://sepolia.etherscan.io/address/0x5FbDB2315678afecb367f032d93F642f64180aa3')}"
        print(f"Feature 4 QR Code: {qr_url}")
        print(f"HTML Report      : file://{os.path.abspath('verification_report.html')}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

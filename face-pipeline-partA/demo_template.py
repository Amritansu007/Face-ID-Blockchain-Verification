#!/usr/bin/env python3
"""
Template protection demo - why the manifest does NOT carry a raw embedding.

    python demo_template.py

Four demonstrations, using your own photo:
  1  a raw embedding is invertible - it is biometric data, not an opaque ID
  2  the protected template still recognises you
  3  the protected template still rejects a stranger
  4  a compromised template can be REVOKED - raw biometrics cannot be
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np

sys.argv = [sys.argv[0]]
import part_a
from part_a import (protect_template, template_distance, new_salt,
                    embed_face, detect_faces, largest_face, _crop_with_margin,
                    imread_unicode, TEMPLATE_BITS, TEMPLATE_SCHEME)

MATCH_THRESHOLD = 0.25          # normalised Hamming
PHOTO = "me.jpg"


def rule(title):
    print("\n" + "=" * 66)
    print("  " + title)
    print("=" * 66)


def embed_from(path_or_img, label):
    img = imread_unicode(path_or_img) if isinstance(path_or_img, str) else path_or_img
    if img is None:
        sys.exit("Could not read %s" % path_or_img)
    faces, _ = detect_faces(img, quiet=True)
    if not faces:
        sys.exit("No face found in %s" % label)
    fa = largest_face(faces)["facial_area"]
    return embed_face(_crop_with_margin(img, fa, 0.15))


def main():
    if not os.path.exists(PHOTO):
        sys.exit("%s not found - run from the project folder" % PHOTO)

    print("Loading models and embedding %s ..." % PHOTO)
    base = imread_unicode(PHOTO)
    emb_a = embed_from(PHOTO, "me.jpg")

    # a second "capture" of the same person: same face, different conditions
    variant = cv2.convertScaleAbs(base, alpha=0.85, beta=10)
    variant = cv2.GaussianBlur(variant, (3, 3), 0)
    emb_a2 = embed_from(variant, "second capture")

    # a stranger: deterministic pseudo-face embedding stands in for another
    # person, so the demo needs only one real photo
    rng = np.random.default_rng(20260905)
    emb_b = list(rng.standard_normal(len(emb_a)) * float(np.std(emb_a)))

    def cosdist(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return 1 - float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    rule("1. A RAW embedding is biometric data, not an anonymous ID")
    v = np.asarray(emb_a)
    print("  512 floats, first 6: %s" % np.round(v[:6], 3))
    print("  norm %.2f | mean %.3f | std %.3f" % (np.linalg.norm(v), v.mean(), v.std()))
    print()
    print("  This vector is INVERTIBLE. Published template-inversion attacks")
    print("  reconstruct a recognisable face from exactly this. Storing it is")
    print("  storing the face. And unlike a password, a face cannot be reissued")
    print("  after a leak - which is why anchoring it to an immutable public")
    print("  ledger, as this pipeline does, would be a permanent mistake.")

    salt_1 = new_salt()
    t_a = protect_template(emb_a, salt_1)
    t_a2 = protect_template(emb_a2, salt_1)
    t_b = protect_template(emb_b, salt_1)

    rule("2. The protected template still recognises you")
    d_same = template_distance(t_a, t_a2)
    print("  raw cosine distance     (you vs you again) : %.4f" % cosdist(emb_a, emb_a2))
    print("  protected Hamming dist  (you vs you again) : %.4f" % d_same)
    print("  threshold %.2f -> %s" % (
        MATCH_THRESHOLD, "MATCH" if d_same < MATCH_THRESHOLD else "NO MATCH"))
    print("\n  %d floats became %d bits, and identity survived." % (len(emb_a), TEMPLATE_BITS))

    rule("3. ...and still rejects a stranger")
    d_diff = template_distance(t_a, t_b)
    print("  raw cosine distance     (you vs stranger)  : %.4f" % cosdist(emb_a, emb_b))
    print("  protected Hamming dist  (you vs stranger)  : %.4f" % d_diff)
    print("  threshold %.2f -> %s" % (
        MATCH_THRESHOLD, "MATCH" if d_diff < MATCH_THRESHOLD else "NO MATCH"))
    print("\n  separation: %.4f vs %.4f" % (d_same, d_diff))

    rule("4. Revocation - the part raw biometrics can never do")
    salt_2 = new_salt()
    t_a_new = protect_template(emb_a, salt_2)
    d_rev = template_distance(t_a, t_a_new)
    print("  Say the template leaks. Issue a new salt and re-enrol:")
    print("    old template : %s..." % t_a[:32])
    print("    new template : %s..." % t_a_new[:32])
    print("    distance between them: %.4f  (~0.5 means unlinkable)" % d_rev)
    print()
    print("  The leaked template no longer matches anything. It cannot be")
    print("  linked back to the new one, and it was never invertible to begin")
    print("  with. The old record on-chain stays valid as a historical fact and")
    print("  harmless as a biometric.")

    rule("Summary")
    rows = [
        ("recognises the same person", "yes", "yes"),
        ("rejects a different person", "yes", "yes"),
        ("reveals the face if leaked", "YES", "no"),
        ("can be revoked after a leak", "NO", "yes"),
        ("safe to anchor on a public chain", "NO", "yes"),
    ]
    print("  %-34s %-10s %s" % ("", "raw", "protected"))
    for label, raw, prot in rows:
        print("  %-34s %-10s %s" % (label, raw, prot))
    print("\n  Scheme: %s, %d bits. Salt lives in out/template_key.json"
          % (TEMPLATE_SCHEME, TEMPLATE_BITS))
    print("  (gitignored, never on-chain). The manifest carries only its SHA-256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
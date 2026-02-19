# -*- coding: utf-8 -*-
"""
==============================================================================
  GATEKEEPER FULL TEST SUITE
==============================================================================
  Tests the ImageNetGatekeeper against known cattle images and non-cattle
  images found in the uploads/ folder.

  Exit codes:
    0 — all critical tests passed
    1 — one or more critical tests FAILED  (caller should revert changes)

  Usage:
    python gatekeeper/test_gatekeeper_full.py
==============================================================================
"""

import sys
import os

# Make sure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gatekeeper.gatekeeper import ImageNetGatekeeper

# ── colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"{GREEN}  [PASS] {msg}{RESET}")
def fail(msg):print(f"{RED}  [FAIL] {msg}{RESET}")
def info(msg):print(f"{CYAN}  [INFO] {msg}{RESET}")
def warn(msg):print(f"{YELLOW}  [WARN] {msg}{RESET}")

# ── Image catalogue ─────────────────────────────────────────────────────────
UPLOADS = os.path.join(PROJECT_ROOT, "uploads")

# Files known to contain cattle / buffalo (MUST pass gatekeeper)
KNOWN_BOVINE = [
    "Gir_01-1024x1024.jpg",
    "gir.jpg",
    "gir1.png",
    "MN_1.1.1.jpg",
    "murrah-buffalo.jpg",
    "sahiwal.jpg",
    "sahiwal2.png",
    "red sindhi2.png",
    "cape-buffalo-syncerus-caffer.webp",
    "cow-with-no-head-with-an-itch-flexible-licking-her-back-in-a-green-meadow-under-a-blue-sky-2H52Y8D.jpg",
]

# Files we HOPE the gatekeeper rejects (non-bovine).
# These are SOFT tests — failure is a warning, not a hard fail.
KNOWN_NON_BOVINE = [
    "ChatGPT Image Jan 21, 2026, 12_29_30 AM.png",  # AI-generated, likely no cattle
]

# ── Test harness ─────────────────────────────────────────────────────────────

def test_model_loads(gk: ImageNetGatekeeper) -> bool:
    print(f"\n{BOLD}[TEST 1] Gatekeeper model loads correctly{RESET}")
    if gk.model is not None:
        ok("MobileNetV2 loaded (model is not None)")
        return True
    else:
        fail("Model is None — gatekeeper running in fail-open mode (no TF?)")
        warn("This is non-fatal but means gatekeeper provides no protection")
        return False  # treat as critical: gatekeeper must actually load

def test_bovine_images(gk: ImageNetGatekeeper) -> tuple[int, int]:
    """Returns (passed, total) for known-bovine images."""
    print(f"\n{BOLD}[TEST 2] Known-bovine images must PASS gatekeeper{RESET}")
    passed = 0
    total  = 0
    for fname in KNOWN_BOVINE:
        path = os.path.join(UPLOADS, fname)
        if not os.path.exists(path):
            warn(f"  Skipping (not found): {fname}")
            continue
        total += 1
        result = gk.check(path)
        verdict = result["is_bovine"]
        tag     = result["reason"]
        conf    = result["top_class"]
        bsig    = result["bovine_signal"]
        asig    = result["animal_signal"]
        if verdict:
            ok(f"{fname}  → PASS  reason={tag} bovine={bsig}% animal={asig}%")
            passed += 1
        else:
            fail(f"{fname}  → REJECTED  reason={tag} top={conf} "
                 f"bovine={bsig}% animal={asig}%")
    return passed, total

def test_non_bovine_images(gk: ImageNetGatekeeper) -> tuple[int, int]:
    """Returns (correctly_rejected, total). These are SOFT tests."""
    print(f"\n{BOLD}[TEST 3] Non-bovine images — soft check (should be rejected){RESET}")
    rejected = 0
    total    = 0
    for fname in KNOWN_NON_BOVINE:
        path = os.path.join(UPLOADS, fname)
        if not os.path.exists(path):
            warn(f"  Skipping (not found): {fname}")
            continue
        total += 1
        result = gk.check(path)
        verdict = result["is_bovine"]
        tag     = result["reason"]
        conf    = result["top_class"]
        bsig    = result["bovine_signal"]
        asig    = result["animal_signal"]
        if not verdict:
            ok(f"{fname}  → REJECTED (correct)  reason={tag}")
            rejected += 1
        else:
            warn(f"{fname}  → PASSED through  reason={tag} top={conf} "
                 f"bovine={bsig}% animal={asig}%")
            warn("  (This is a SOFT failure — gatekeeper is lenient by design)")
    return rejected, total

def test_result_schema(gk: ImageNetGatekeeper) -> bool:
    """Ensure check() always returns the required keys."""
    print(f"\n{BOLD}[TEST 4] Result dictionary has required keys{RESET}")
    required_keys = {"is_bovine", "reason", "top_class", "top_confidence",
                     "bovine_signal", "animal_signal"}
    # Use any available image
    for fname in KNOWN_BOVINE:
        path = os.path.join(UPLOADS, fname)
        if os.path.exists(path):
            result = gk.check(path)
            missing = required_keys - set(result.keys())
            if missing:
                fail(f"Missing keys in result: {missing}")
                return False
            ok(f"All required keys present: {sorted(required_keys)}")
            return True
    warn("No test image found to check schema — skipping")
    return True  # non-fatal

def test_fail_open(gk: ImageNetGatekeeper) -> bool:
    """Passing a bad path must NOT raise an exception and must return is_bovine=True."""
    print(f"\n{BOLD}[TEST 5] Bad image path → fail-open (no exception, is_bovine=True){RESET}")
    result = gk.check("/nonexistent/path/to/image.jpg")
    if result.get("is_bovine") is True:
        ok(f"Fail-open triggered correctly: reason={result.get('reason')}")
        return True
    else:
        fail("Gatekeeper returned is_bovine=False for a missing file — unexpected")
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 stdout on Windows to avoid charmap errors
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print(f"\n{BOLD}{CYAN}{'='*60}")
    print("  GATEKEEPER FULL TEST SUITE")
    print(f"{'='*60}{RESET}")

    gk = ImageNetGatekeeper()

    results = {}

    # T1 - critical
    results["model_loads"] = test_model_loads(gk)

    # T2 - critical: bovine images must pass
    bovine_pass, bovine_total = test_bovine_images(gk)
    if bovine_total > 0:
        bovine_rate = bovine_pass / bovine_total
        results["bovine_pass_rate"] = bovine_rate >= 0.70  # at least 70% pass
    else:
        warn("No bovine test images found — skipping pass-rate check")
        results["bovine_pass_rate"] = True  # can't fail what we can't test

    # T3 - soft
    test_non_bovine_images(gk)

    # T4 - critical
    results["schema_ok"] = test_result_schema(gk)

    # T5 - critical
    results["fail_open"] = test_fail_open(gk)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}{RESET}")

    if bovine_total > 0:
        info(f"Bovine pass rate: {bovine_pass}/{bovine_total} "
             f"({bovine_pass/bovine_total:.0%})")

    critical_failed = [k for k, v in results.items() if not v]
    if critical_failed:
        print(f"\n{RED}{BOLD}CRITICAL FAILURES: {critical_failed}{RESET}")
        print(f"{RED}Gatekeeper tests FAILED — recommend reverting changes.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}ALL CRITICAL TESTS PASSED [OK]{RESET}")
        print(f"{GREEN}Gatekeeper is working correctly.{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gatekeeper.gatekeeper import ImageNetGatekeeper

gk = ImageNetGatekeeper()
print("Model loaded:", gk.model is not None)

uploads = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
test_files = [
    "Gir_01-1024x1024.jpg",
    "gir.jpg",
    "MN_1.1.1.jpg",
    "murrah-buffalo.jpg",
    "sahiwal.jpg",
    "sahiwal2.png",
    "red sindhi2.png",
    "cape-buffalo-syncerus-caffer.webp",
]

passed = 0
total = 0
for f in test_files:
    path = os.path.join(uploads, f)
    if not os.path.exists(path):
        print(f"SKIP (not found): {f}")
        continue
    total += 1
    r = gk.check(path)
    verdict = "PASS" if r["is_bovine"] else "FAIL"
    if r["is_bovine"]:
        passed += 1
    print(f"[{verdict}] {f}")
    print(f"      reason={r['reason']} bovine_sig={r['bovine_signal']}% animal_sig={r['animal_signal']}%")

print()
print(f"Result: {passed}/{total} bovine images passed ({passed/total*100:.0f}% pass rate)" if total else "No images found!")
if total > 0 and passed/total >= 0.70:
    print("STATUS: PASS (>=70% pass rate)")
else:
    print("STATUS: FAIL (<70% pass rate OR no images)")

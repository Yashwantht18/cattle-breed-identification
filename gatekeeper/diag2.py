import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gatekeeper.gatekeeper import ImageNetGatekeeper

gk = ImageNetGatekeeper()
print("Model loaded:", gk.model is not None)
print()

uploads = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
test_bovine = [
    ("Gir_01-1024x1024.jpg",    "BOVINE"),
    ("gir.jpg",                  "BOVINE"),
    ("gir1.png",                 "BOVINE"),
    ("MN_1.1.1.jpg",             "BOVINE"),
    ("murrah-buffalo.jpg",       "BOVINE"),
    ("sahiwal.jpg",              "BOVINE"),
    ("sahiwal2.png",             "BOVINE"),
    ("red sindhi2.png",          "BOVINE"),
    ("cape-buffalo-syncerus-caffer.webp", "BOVINE"),
    ("cow-with-no-head-with-an-itch-flexible-licking-her-back-in-a-green-meadow-under-a-blue-sky-2H52Y8D.jpg", "BOVINE"),
    ("camera_capture.jpg",       "HUMAN"),   # the problem image — should REJECT
    ("ChatGPT Image Jan 21, 2026, 12_29_30 AM.png", "NON-BOVINE"),
]

bovine_pass = 0; bovine_total = 0
human_rej   = 0; human_total  = 0

for fname, expected in test_bovine:
    path = os.path.join(uploads, fname)
    if not os.path.exists(path):
        print(f"SKIP: {fname}")
        continue
    r = gk.check(path)
    verdict = "PASS" if r["is_bovine"] else "REJECT"
    correct = ""
    if expected == "BOVINE":
        bovine_total += 1
        if r["is_bovine"]:
            bovine_pass += 1
            correct = "OK"
        else:
            correct = "WRONG-should-pass"
    elif expected in ("HUMAN", "NON-BOVINE"):
        human_total += 1
        if not r["is_bovine"]:
            human_rej += 1
            correct = "OK"
        else:
            correct = "WRONG-should-reject"
    
    print(f"[{verdict}] [{correct}] {fname}")
    print(f"         reason={r['reason']} bovine={r['bovine_signal']}% animal={r['animal_signal']}% top={r['top_class']}({r['top_confidence']}%)")
    print()

print(f"Bovine pass rate:  {bovine_pass}/{bovine_total} ({bovine_pass/bovine_total*100:.0f}%)" if bovine_total else "No bovine images.")
print(f"Non-bovine reject: {human_rej}/{human_total} ({human_rej/human_total*100:.0f}%)" if human_total else "No non-bovine images.")

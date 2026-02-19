import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gatekeeper.gatekeeper import ImageNetGatekeeper

gk = ImageNetGatekeeper()
print("Gatekeeper initialized:", gk.model is not None)
print()

test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

if test_files:
    for img_name in test_files[:5]:
        img_path = os.path.join(test_dir, img_name)
        result = gk.check(img_path)
        status = "PASS" if result["is_bovine"] else "REJECTED"
        print(f"[{status}] {img_name}")
        print(f"        reason={result['reason']} | top={result['top_class']}({result['top_confidence']}%) | bovine_signal={result['bovine_signal']}% | animal_signal={result['animal_signal']}%")
        print()
else:
    print("No test images found in uploads/")

"""
download_and_balance.py
-----------------------
Step 1 : Count current clean images per breed in train/
Step 2 : For every breed below TARGET_PER_BREED, download more from Bing
Step 3 : Quality-check each downloaded image (blur, size, greyscale)
Step 4 : Print final balanced counts

Run from project root:
    python model_training/download_and_balance.py

Requirements:
    pip install icrawler opencv-python pillow
"""

import os, sys, math, shutil, time, colorsys
import numpy as np
from PIL import Image, ImageStat

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("[WARN] opencv-python missing. Run: pip install opencv-python")

try:
    from icrawler.builtin import BingImageCrawler
    ICRAWLER_OK = True
except ImportError:
    ICRAWLER_OK = False
    print("[ERROR] icrawler missing. Run: pip install icrawler")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
TRAIN_DIR    = os.path.join(PROJECT_ROOT,
               'Indian Bovine Breed Recognition.v1i.folder', 'train')

TARGET_PER_BREED = 300    # minimum images we want per breed in train/
                           # (oversampling will bring all to 500 at training time)

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Quality thresholds (same as clean_dataset.py)
MIN_SIZE       = 100
MIN_BLUR_VAR   = 80.0
MIN_BRIGHTNESS = 20
MAX_BRIGHTNESS = 240
MIN_SATURATION = 12
MAX_ASPECT     = 5.0

# ---------------------------------------------------------------------------
# Breed name → best Bing search query
# ---------------------------------------------------------------------------
BREED_QUERIES = {
    "Alambadi":          "Alambadi cattle breed India",
    "Amritmahal":        "Amritmahal cattle breed Karnataka India",
    "Ayrshire":          "Ayrshire cow breed",
    "Banni":             "Banni buffalo breed Gujarat India",
    "Bargur":            "Bargur cattle breed Tamil Nadu India",
    "Bhadawari":         "Bhadawari buffalo breed India",
    "Brown_Swiss":       "Brown Swiss cow dairy breed",
    "Chhattisgarhi":     "Chhattisgarhi cattle breed India",
    "Dangi":             "Dangi cattle breed Maharashtra India",
    "Deoni":             "Deoni cattle breed India",
    "Gir":               "Gir cow breed Gujarat India",
    "Guernsey":          "Guernsey cow dairy breed",
    "Hallikar":          "Hallikar cattle breed Karnataka India",
    "Hariana":           "Hariana cattle breed Haryana India",
    "Holstein_Friesian": "Holstein Friesian black white dairy cow",
    "Jaffrabadi":        "Jaffrabadi buffalo breed Gujarat India",
    "Jersey":            "Jersey cow dairy breed",
    "Kangayam":          "Kangayam cattle breed Tamil Nadu India",
    "Kankrej":           "Kankrej cattle breed Gujarat India",
    "Kasargod":          "Kasargod cattle dwarf cow Kerala India",
    "Kenkatha":          "Kenkatha cattle breed Madhya Pradesh India",
    "Kherigarh":         "Kherigarh cattle breed Uttar Pradesh India",
    "Khillari":          "Khillari cattle breed Maharashtra India",
    "Krishna_Valley":    "Krishna Valley cattle breed Karnataka India",
    "Malnad_gidda":      "Malnad Gidda cattle Karnataka dwarf India",
    "Mehsana":           "Mehsana buffalo breed Gujarat India",
    "Murrah":            "Murrah buffalo Haryana India dairy",
    "Nagori":            "Nagori cattle breed Rajasthan India",
    "Nagpuri":           "Nagpuri buffalo Central India Maharashtra",
    "Nili_Ravi":         "Nili Ravi buffalo Pakistan Punjab",
    "Nimari":            "Nimari cattle breed Madhya Pradesh India",
    "Ongole":            "Ongole cattle breed Andhra Pradesh India",
    "Pulikulam":         "Pulikulam cattle breed Tamil Nadu India",
    "Rathi":             "Rathi cattle breed Rajasthan India",
    "Red_Dane":          "Red Dane cow dairy breed",
    "Red_Sindhi":        "Red Sindhi cattle breed Pakistan India",
    "Sahiwal":           "Sahiwal cow breed Punjab Pakistan India",
    "Surti":             "Surti buffalo breed Gujarat India",
    "Tharparkar":        "Tharparkar cattle breed Rajasthan India",
    "Toda":              "Toda buffalo Nilgiri Hills India",
    "Umblachery":        "Umblachery cattle breed Tamil Nadu India",
    "Vechur":            "Vechur dwarf cattle breed Kerala India",
    "chilika":           "Chilika buffalo Odisha India",
    "gojri":             "Gojri buffalo Himachal Pradesh India",
    "kalahandi":         "Kalahandi buffalo Odisha India",
    "luit":              "Luit buffalo Assam India",
    "marathwada":        "Marathwada buffalo Maharashtra India",
    "pandharpuri":       "Pandharpuri buffalo Maharashtra India",
}

# ---------------------------------------------------------------------------
# Quality check (returns True if image is good)
# ---------------------------------------------------------------------------
def is_good_image(path):
    try:
        img_pil = Image.open(path).convert('RGB')
        img_pil.verify()
        img_pil = Image.open(path).convert('RGB')
    except Exception:
        return False

    w, h = img_pil.size
    if w < MIN_SIZE or h < MIN_SIZE:
        return False
    if max(w/h, h/w) > MAX_ASPECT:
        return False

    stat = ImageStat.Stat(img_pil)
    brightness = sum(stat.mean[:3]) / 3
    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
        return False

    # Saturation check
    try:
        pixels = list(img_pil.getdata())
        sat_vals = [colorsys.rgb_to_hsv(r/255, g/255, b/255)[1]*255
                    for r,g,b in pixels[::50]]
        if np.mean(sat_vals) < MIN_SATURATION:
            return False
    except Exception:
        pass

    # Blur check
    if CV2_OK:
        try:
            img_cv = cv2.imread(path)
            if img_cv is not None:
                grey = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(grey, cv2.CV_64F).var() < MIN_BLUR_VAR:
                    return False
        except Exception:
            pass

    return True

# ---------------------------------------------------------------------------
# Count clean images in a breed folder
# ---------------------------------------------------------------------------
def count_images(breed_dir):
    if not os.path.isdir(breed_dir):
        return 0
    return len([f for f in os.listdir(breed_dir)
                if os.path.splitext(f)[1].lower() in VALID_EXTS])

# ---------------------------------------------------------------------------
# Download images for one breed using Bing
# ---------------------------------------------------------------------------
def download_for_breed(breed, breed_dir, need_count):
    query = BREED_QUERIES.get(breed, f"{breed} cattle buffalo India")
    print(f"\n  [DOWNLOAD] {breed}: need {need_count} more images")
    print(f"  Query: '{query}'")

    # Download to a temp folder then quality-check before moving
    tmp_dir = os.path.join(breed_dir, '__tmp_download__')
    os.makedirs(tmp_dir, exist_ok=True)

    # Download 2x what we need (some will fail quality check)
    fetch_count = min(need_count * 2, 200)  # Bing max ~1000 but keep reasonable

    try:
        crawler = BingImageCrawler(
            storage={'root_dir': tmp_dir},
            downloader_threads=4,
            parser_threads=2,
        )
        crawler.crawl(
            keyword=query,
            max_num=fetch_count,
            min_size=(150, 150),
            file_idx_offset=0,
        )
    except Exception as e:
        print(f"  [WARN] Download error: {e}")

    # Quality check and move good images
    accepted = 0
    rejected = 0
    existing_count = count_images(breed_dir)

    for fname in sorted(os.listdir(tmp_dir)):
        src = os.path.join(tmp_dir, fname)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXTS:
            os.remove(src)
            continue

        if is_good_image(src):
            # Rename to avoid collisions with existing files
            new_name = f"downloaded_{breed}_{existing_count + accepted + 1:04d}.jpg"
            dst = os.path.join(breed_dir, new_name)
            shutil.move(src, dst)
            accepted += 1
        else:
            os.remove(src)
            rejected += 1

        if accepted >= need_count:
            break

    # Clean up tmp
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    print(f"  [OK] Downloaded {accepted} good images "
          f"(rejected {rejected} low-quality)")
    return accepted

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  DATASET BALANCER + HIGH-QUALITY DOWNLOADER")
    print(f"  Target: {TARGET_PER_BREED} clean images per breed in train/")
    print("=" * 65)

    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] train/ not found at {TRAIN_DIR}")
        sys.exit(1)

    # Step 1: Count current state
    breeds = sorted([d for d in os.listdir(TRAIN_DIR)
                     if os.path.isdir(os.path.join(TRAIN_DIR, d))])

    print(f"\n[INFO] Found {len(breeds)} breed folders in train/")
    print("\n--- Current image counts ---")

    needs_download = []
    for breed in breeds:
        bd = os.path.join(TRAIN_DIR, breed)
        count = count_images(bd)
        status = "OK" if count >= TARGET_PER_BREED else f"NEED +{TARGET_PER_BREED - count}"
        print(f"  {breed:<28} {count:>4} images  [{status}]")
        if count < TARGET_PER_BREED:
            needs_download.append((breed, TARGET_PER_BREED - count))

    print(f"\n[INFO] {len(needs_download)} breeds need more images")
    print(f"[INFO] {len(breeds) - len(needs_download)} breeds already have "
          f">= {TARGET_PER_BREED} images\n")

    if not needs_download:
        print("[DONE] All breeds already meet the target. No downloads needed.")
        return

    # Step 2: Download for breeds below target
    print("=" * 65)
    print("  DOWNLOADING HIGH-QUALITY IMAGES")
    print("=" * 65)

    for i, (breed, need) in enumerate(needs_download, 1):
        print(f"\n[{i}/{len(needs_download)}] Processing: {breed}")
        breed_dir = os.path.join(TRAIN_DIR, breed)
        got = download_for_breed(breed, breed_dir, need)
        final_count = count_images(breed_dir)
        print(f"  Final count for {breed}: {final_count} images")
        time.sleep(1)   # polite delay between breeds

    # Step 3: Final summary
    print("\n" + "=" * 65)
    print("  FINAL COUNTS AFTER DOWNLOAD")
    print("=" * 65)
    all_ok = True
    for breed in breeds:
        bd = os.path.join(TRAIN_DIR, breed)
        count = count_images(bd)
        status = "OK" if count >= TARGET_PER_BREED else f"STILL LOW ({count})"
        print(f"  {breed:<28} {count:>4}  [{status}]")
        if count < TARGET_PER_BREED:
            all_ok = False

    print()
    if all_ok:
        print("[SUCCESS] All breeds now have >= "
              f"{TARGET_PER_BREED} training images!")
        print("[NEXT] Run training:")
        print("       python model_training/train_mobilenet_fixed.py")
    else:
        print(f"[PARTIAL] Some breeds still below {TARGET_PER_BREED}.")
        print("          Bing returned fewer results for rare breeds.")
        print("          The oversampling in training will compensate.")

if __name__ == '__main__':
    main()

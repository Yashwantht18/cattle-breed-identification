"""
clean_dataset.py
----------------
Scans all training images and removes/flags low-quality ones.

Quality checks:
  1. File corrupt / unreadable
  2. Resolution too small (< 100x100 px)
  3. Nearly grayscale (coat colour is critical — greyscale loses breed info)
  4. Extremely dark or washed out (brightness check)
  5. Very blurry (Laplacian variance < threshold)
  6. Wrong aspect ratio (very narrow/wide — likely cropped artefact)

Run from project root:
    python model_training/clean_dataset.py
"""

import os
import sys
import numpy as np
from PIL import Image, ImageStat

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARN] opencv-python not installed — blur detection disabled.")
    print("       Run: pip install opencv-python")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Indian Bovine Breed Recognition.v1i.folder')

VALID_EXTS     = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif'}
MIN_SIZE       = 100        # px — each dimension must be >= this
MIN_BLUR_VAR   = 80.0       # Laplacian variance — below = blurry (needs cv2)
MIN_BRIGHTNESS = 20         # 0-255 — below = too dark
MAX_BRIGHTNESS = 240        # 0-255 — above = too washed out
MIN_SATURATION = 12         # 0-255 — below = nearly greyscale
MAX_ASPECT     = 5.0        # max(w/h, h/w) — above = extreme crop artefact
DRY_RUN        = False      # Set False to actually DELETE bad images

SPLITS = ['train', 'valid', 'test']

# ---------------------------------------------------------------------------
# QUALITY CHECK
# ---------------------------------------------------------------------------
def check_image(path):
    """
    Returns (is_good: bool, reasons: list[str])
    """
    reasons = []

    # 1. Can it be opened?
    try:
        img_pil = Image.open(path).convert('RGB')
        img_pil.verify()          # checks for corruption
        img_pil = Image.open(path).convert('RGB')  # reopen after verify
    except Exception as e:
        return False, [f"CORRUPT: {e}"]

    w, h = img_pil.size

    # 2. Resolution
    if w < MIN_SIZE or h < MIN_SIZE:
        reasons.append(f"TOO_SMALL: {w}x{h} (min {MIN_SIZE}x{MIN_SIZE})")

    # 3. Aspect ratio
    aspect = max(w / h, h / w)
    if aspect > MAX_ASPECT:
        reasons.append(f"BAD_ASPECT: {w}x{h} ratio={aspect:.1f}")

    # 4. Brightness
    stat    = ImageStat.Stat(img_pil)
    mean_r, mean_g, mean_b = stat.mean[:3]
    brightness = (mean_r + mean_g + mean_b) / 3
    if brightness < MIN_BRIGHTNESS:
        reasons.append(f"TOO_DARK: brightness={brightness:.1f}")
    if brightness > MAX_BRIGHTNESS:
        reasons.append(f"WASHED_OUT: brightness={brightness:.1f}")

    # 5. Saturation (coat colour is critical)
    img_hsv = img_pil.convert('HSV') if hasattr(img_pil, 'convert') else None
    try:
        import colorsys
        pixels = list(img_pil.getdata())
        sat_vals = []
        for r, g, b in pixels[::50]:   # sample every 50th pixel for speed
            _, s, _ = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            sat_vals.append(s * 255)
        mean_sat = np.mean(sat_vals)
        if mean_sat < MIN_SATURATION:
            reasons.append(f"GREYSCALE: saturation={mean_sat:.1f}")
    except Exception:
        pass

    # 6. Blur detection (requires OpenCV)
    if CV2_AVAILABLE:
        try:
            img_cv = cv2.imread(path)
            if img_cv is not None:
                grey      = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                blur_var  = cv2.Laplacian(grey, cv2.CV_64F).var()
                if blur_var < MIN_BLUR_VAR:
                    reasons.append(f"BLURRY: laplacian_var={blur_var:.1f} (min {MIN_BLUR_VAR})")
        except Exception:
            pass

    return len(reasons) == 0, reasons

# ---------------------------------------------------------------------------
# MAIN SCAN
# ---------------------------------------------------------------------------
def scan_split(split):
    split_dir = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"  [SKIP] {split}/ not found")
        return {}, {}

    bad_files   = {}   # breed -> list of (path, reasons)
    breed_stats = {}   # breed -> {total, bad, good}

    breeds = sorted(os.listdir(split_dir))
    for breed in breeds:
        breed_dir = os.path.join(split_dir, breed)
        if not os.path.isdir(breed_dir):
            continue

        files = [
            os.path.join(breed_dir, f)
            for f in os.listdir(breed_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
        total = len(files)
        bad   = []

        for fp in files:
            ok, reasons = check_image(fp)
            if not ok:
                bad.append((fp, reasons))

        breed_stats[breed] = {'total': total, 'bad': len(bad), 'good': total - len(bad)}
        if bad:
            bad_files[breed] = bad

    return breed_stats, bad_files


def main():
    print("=" * 65)
    print("  DATASET QUALITY CHECKER")
    print(f"  DRY_RUN = {DRY_RUN}  (set False to permanently delete bad images)")
    print("=" * 65)

    grand_total = 0
    grand_bad   = 0
    all_bad     = []

    for split in SPLITS:
        print(f"\n{'-'*65}")
        print(f"  SPLIT: {split.upper()}")
        print(f"{'-'*65}")

        stats, bad_files = scan_split(split)
        split_total = sum(s['total'] for s in stats.values())
        split_bad   = sum(s['bad']   for s in stats.values())

        grand_total += split_total
        grand_bad   += split_bad

        # Print breeds with bad images
        worst = sorted(stats.items(), key=lambda x: x[1]['bad'], reverse=True)
        for breed, s in worst:
            if s['bad'] > 0:
                pct = s['bad'] / s['total'] * 100
                print(f"  {breed:<25} total={s['total']:>4}  bad={s['bad']:>3}  ({pct:.0f}%)")
                if breed in bad_files:
                    for fp, reasons in bad_files[breed][:3]:   # show max 3
                        print(f"    >> {os.path.basename(fp)}: {', '.join(reasons)}")
                    if len(bad_files[breed]) > 3:
                        print(f"    >> ... and {len(bad_files[breed])-3} more")
                all_bad.extend([(fp, r) for fp, r in bad_files.get(breed, [])])

        print(f"\n  {split.upper()} SUMMARY: {split_bad} bad / {split_total} total "
              f"({split_bad/max(split_total,1)*100:.1f}% removed)")

    print(f"\n{'='*65}")
    print(f"  GRAND TOTAL: {grand_bad} bad images out of {grand_total} "
          f"({grand_bad/max(grand_total,1)*100:.1f}%)")
    print(f"  CLEAN IMAGES: {grand_total - grand_bad}")
    print(f"{'='*65}")

    if not DRY_RUN and all_bad:
        print(f"\n[DELETE] Removing {len(all_bad)} bad images...")
        deleted = 0
        errors  = 0
        for fp, _ in all_bad:
            try:
                os.remove(fp)
                deleted += 1
            except Exception as e:
                print(f"  [ERR] Could not delete {fp}: {e}")
                errors += 1
        print(f"[DONE] Deleted {deleted} files. Errors: {errors}")
        print("[NEXT] Re-run training to use the cleaned dataset.")
    elif DRY_RUN and all_bad:
        print(f"\n[DRY RUN] Would delete {len(all_bad)} files.")
        print("[ACTION]  Set DRY_RUN = False in this script to actually delete them.")

    # Save report
    report_path = os.path.join(PROJECT_ROOT, 'models', 'dataset_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Dataset Quality Report\n")
        f.write(f"Total scanned: {grand_total}\n")
        f.write(f"Bad images:    {grand_bad} ({grand_bad/max(grand_total,1)*100:.1f}%)\n")
        f.write(f"Clean images:  {grand_total - grand_bad}\n\n")
        f.write("Bad image list:\n")
        for fp, reasons in all_bad:
            f.write(f"  {fp}\n    Reasons: {', '.join(reasons)}\n")
    print(f"\n[SAVED] Quality report -> {report_path}")


if __name__ == '__main__':
    main()

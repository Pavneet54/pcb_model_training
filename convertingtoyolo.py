import cv2
import numpy as np
import shutil
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CLASS MAP — matched by pixel frequency to known class counts
# Your 25 colors sorted by pixel count:
#   (0,234,255)=115M  (255,127,0)=61M  (231,233,185)=42M
#   (255,255,0)=30M   (185,215,237)=19M (255,0,0)=19M ...
#
# Known class counts (from paper):
#   C=1746  R=724  U=761  J=1476  L=473  Q=616
#   P=200   D=362  IC=237 RN=330  CR=194 RA=404
#   M=74    T=47   V=41   TP=266  FB=69  S=37
#   BTN=72  CRA=54 QA=36  LED=39  F=44   SW=50  JP=31
# ══════════════════════════════════════════════════════════════
CLASS_MAP = {
    (0,   234, 255): 0,   # C   — Capacitor       (115M px — most common)
    (255, 127,   0): 1,   # R   — Resistor         (61M px)
    (231, 233, 185): 2,   # U   — IC chip          (42M px)
    (255, 255,   0): 3,   # J   — Connector        (30M px)
    (185, 215, 237): 4,   # L   — Inductor         (19M px)
    (255,   0,   0): 5,   # Q   — Transistor       (19M px)
    (185, 237, 224): 6,   # P   — Pad              (13M px)
    (79,  143,  35): 7,   # D   — Diode            (13M px)
    (204, 204, 204): 8,   # IC  — Integrated ckt    (7M px)
    (191, 255,   0): 9,   # RN  — Resistor network  (7M px)
    (170,   0, 255): 10,  # CR  — Crystal           (5M px)
    (170, 255, 195): 11,  # RA  — Resistor array    (5M px)
    (107,  35, 143): 12,  # M   — Mounting hole     (4M px)
    (0,   64,  255): 13,  # T   — Transformer       (3M px)
    (220, 185, 237): 14,  # V   — Via               (2M px)
    (106, 255,   0): 15,  # TP  — Test point        (2M px)
    (143,  35,  35): 16,  # FB  — Ferrite bead      (1M px)
    (0,  149,  255): 17,  # S   — Switch            (1M px)
    (220, 190, 255): 18,  # BTN — Button            (1M px)
    (237, 185, 185): 19,  # CRA — Crystal array    (654K px)
    (255, 250, 200): 20,  # QA  — Transistor array (584K px)
    (115, 115, 115): 21,  # LED — LED              (530K px)
    (143, 106,  35): 22,  # F   — Fuse             (223K px)
    (35,   98, 143): 23,  # SW  — Slide switch     (200K px)
    (245, 130,  48): 24,  # JP  — Jumper           (135K px — rarest)
}

CLASS_NAMES = [
    "C", "R", "U", "J", "L", "Q", "P", "D", "IC", "RN",
    "CR", "RA", "M", "T", "V", "TP", "FB", "S", "BTN", "CRA",
    "QA", "LED", "F", "SW", "JP"
]

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
DATA_ROOT = Path("/Users/pavneet/Desktop/industry45/PCBSegClassNet/data/segmentation")
OUT_ROOT  = Path("/Users/pavneet/Desktop/industry45/PCBSegClassNet/data/yolo")
SPLITS    = ["train", "val"]

EPSILON   = 0.005
MIN_AREA  = 10
MIN_PTS   = 3
TOLERANCE = 3

# ══════════════════════════════════════════════════════════════
# CONVERSION FUNCTION
# ══════════════════════════════════════════════════════════════
def mask_to_yolo(mask_path, out_txt):
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return 0, f"Cannot read: {mask_path.name}"

    h, w  = mask.shape[:2]
    lines = []

    for (r, g, b), cls_id in CLASS_MAP.items():
        bgr    = np.array([b, g, r], dtype=np.uint8)
        lower  = np.clip(bgr.astype(int) - TOLERANCE, 0, 255).astype(np.uint8)
        upper  = np.clip(bgr.astype(int) + TOLERANCE, 0, 255).astype(np.uint8)
        binary = cv2.inRange(mask, lower, upper)

        if binary.sum() == 0:
            continue

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue

            eps    = EPSILON * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            pts    = approx.reshape(-1, 2)

            if len(pts) < MIN_PTS:
                continue

            coords = []
            for px, py in pts:
                coords.append(round(float(px) / w, 6))
                coords.append(round(float(py) / h, 6))

            lines.append(f"{cls_id} " + " ".join(str(c) for c in coords))

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines))
    return len(lines), None


# ══════════════════════════════════════════════════════════════
# RUN CONVERSION
# ══════════════════════════════════════════════════════════════
for split in SPLITS:
    mask_dir = DATA_ROOT / split / "masks"
    img_dir  = DATA_ROOT / split / "images"
    out_lbl  = OUT_ROOT  / split / "labels"
    out_img  = OUT_ROOT  / split / "images"

    out_lbl.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

    masks = sorted(mask_dir.glob("*.png"))
    print(f"\n── {split}: {len(masks)} masks ──")

    converted    = 0
    total_objs   = 0
    errors       = []
    class_counts = {i: 0 for i in range(25)}

    for i, mask_path in enumerate(masks):
        out_txt = out_lbl / mask_path.with_suffix(".txt").name
        n, err  = mask_to_yolo(mask_path, out_txt)

        if err:
            errors.append(err)
            continue

        for line in out_txt.read_text().strip().splitlines():
            if line:
                class_counts[int(line.split()[0])] += 1

        total_objs += n
        converted  += 1

        if (i+1) % 500 == 0:
            print(f"   {i+1}/{len(masks)} done...")

    # Copy images
    for img_path in list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")):
        dst = out_img / img_path.name
        if not dst.exists():
            shutil.copy(str(img_path), str(dst))

    # Summary
    print(f"\n  Converted  : {converted}/{len(masks)}")
    print(f"  Total objs : {total_objs}")
    print(f"  Errors     : {len(errors)}")
    print(f"\n  Class distribution:")
    for cls_id, count in class_counts.items():
        if count > 0:
            bar  = "█" * min(count // 20, 35)
            rare = " ← rare" if count < 100 else ""
            print(f"    {CLASS_NAMES[cls_id]:6s} ({cls_id:2d}): {count:5d}  {bar}{rare}")

    if errors:
        print(f"\n  Errors:")
        for e in errors[:5]:
            print(f"    {e}")

print("\n✅ Done!")
print(f"   Labels → {OUT_ROOT}/train/labels/")
print(f"   Labels → {OUT_ROOT}/val/labels/")
print(f"\n⚠️  IMPORTANT: Check the class distribution above.")
print(f"   If any class shows 0 — the color mapping may need adjustment.")
print(f"   Run verify_conversion.py next to visually confirm.")



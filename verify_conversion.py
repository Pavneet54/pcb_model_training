
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import random

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
DATA_ROOT = Path("/Users/pavneet/Desktop/industry45/PCBSegClassNet/data/segmentation")
YOLO_ROOT = Path("/Users/pavneet/Desktop/industry45/PCBSegClassNet/data/yolo")
SPLIT     = "train"

CLASS_NAMES = [
    "C","R","U","J","L","Q","P","D","IC","RN",
    "CR","RA","M","T","V","TP","FB","S","BTN","CRA",
    "QA","LED","F","SW","JP"
]

VIZ_COLORS = [
    (255, 56, 56),(255,157, 56),(255,255, 56),( 56,255, 56),
    ( 56,255,255),( 56, 56,255),(255, 56,255),(255,128,  0),
    (  0,255,128),(128,  0,255),(255,  0,128),(128,255,  0),
    (  0,128,255),(255,200,200),(200,255,200),(200,200,255),
    (255,165,  0),(  0,255,165),(165,  0,255),(255, 80, 80),
    ( 80,255, 80),( 80, 80,255),(200,100,  0),(  0,200,100),
    (100,  0,200),
]

img_dir  = DATA_ROOT / SPLIT / "images"
mask_dir = DATA_ROOT / SPLIT / "masks"
lbl_dir  = YOLO_ROOT / SPLIT / "labels"

# ══════════════════════════════════════════════════════════════
# CHECK 1 — Counts
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("  CHECK 1 — File counts")
print("=" * 55)

imgs  = list(img_dir.glob("*.png"))  + list(img_dir.glob("*.jpg"))
masks = list(mask_dir.glob("*.png"))
lbls  = list(lbl_dir.glob("*.txt"))

print(f"  Images  : {len(imgs)}")
print(f"  Masks   : {len(masks)}")
print(f"  Labels  : {len(lbls)}")
print(f"  Match   : {'✅' if len(imgs)==len(lbls) else '❌'}")

# ══════════════════════════════════════════════════════════════
# CHECK 2 — Label format
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  CHECK 2 — Label format")
print("=" * 55)

errors      = []
empty       = 0
total_objs  = 0
pt_counts   = []
class_totals= {i:0 for i in range(25)}

for lbl in lbls:
    content = lbl.read_text().strip()
    if not content:
        empty += 1
        continue
    for line in content.splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        coords = parts[1:]
        if len(coords) % 2 != 0:
            errors.append(f"{lbl.name}: odd coords")
        bad = [c for c in coords if float(c)<0 or float(c)>1]
        if bad:
            errors.append(f"{lbl.name}: coord out of [0,1]")
        n_pts = len(coords)//2
        pt_counts.append(n_pts)
        class_totals[cls] += 1
        total_objs += 1

print(f"  Total labels : {len(lbls)}")
print(f"  Empty files  : {empty}  (background images — OK)")
print(f"  Total objects: {total_objs}")
print(f"  Format errors: {len(errors)}")
if errors:
    for e in errors[:5]:
        print(f"    ❌ {e}")
else:
    print(f"  Format       : ✅ all valid")

if pt_counts:
    print(f"\n  Polygon points:")
    print(f"    Min  : {min(pt_counts)}")
    print(f"    Max  : {max(pt_counts)}")
    print(f"    Mean : {sum(pt_counts)/len(pt_counts):.1f}")

# ══════════════════════════════════════════════════════════════
# CHECK 3 — Class distribution
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  CHECK 3 — All 25 classes present")
print("=" * 55)
missing = []
for i, name in enumerate(CLASS_NAMES):
    count  = class_totals[i]
    bar    = "█" * min(count//20, 30)
    rare   = " ← rare" if count < 100 else ""
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {name:6s}: {count:5d}  {bar}{rare}")
    if count == 0:
        missing.append(name)

if missing:
    print(f"\n  ❌ Missing: {missing}")
else:
    print(f"\n  ✅ All 25 classes present!")

# ══════════════════════════════════════════════════════════════
# CHECK 4 — Visual overlay on 5 samples
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  CHECK 4 — Visual overlay (5 samples)")
print("=" * 55)

all_imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
samples  = []
random.shuffle(all_imgs)
for img_path in all_imgs:
    lbl_path = lbl_dir / img_path.with_suffix(".txt").name
    if not lbl_path.exists():
        continue
    lines = [l for l in lbl_path.read_text().strip().splitlines() if l]
    if len(lines) >= 3:
        samples.append(img_path)
    if len(samples) == 5:
        break

fig, axes = plt.subplots(len(samples), 3, figsize=(18, 5*len(samples)))
fig.suptitle(
    "Conversion Verification: Original | Mask | YOLO Overlay",
    fontsize=13, fontweight="bold"
)

for row, img_path in enumerate(samples):
    img      = cv2.imread(str(img_path))
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w     = img_rgb.shape[:2]

    mask_path = mask_dir / img_path.with_suffix(".png").name
    lbl_path  = lbl_dir  / img_path.with_suffix(".txt").name

    # Column 1 — original
    axes[row,0].imshow(img_rgb)
    axes[row,0].set_title(f"Original: {img_path.name}", fontsize=8)
    axes[row,0].axis("off")

    # Column 2 — original mask
    if mask_path.exists():
        m = cv2.imread(str(mask_path))
        axes[row,1].imshow(cv2.cvtColor(m, cv2.COLOR_BGR2RGB))
    axes[row,1].set_title("Original mask", fontsize=8)
    axes[row,1].axis("off")

    # Column 3 — YOLO polygon overlay
    overlay = img_rgb.copy()
    patches = []
    seen    = set()
    lines   = lbl_path.read_text().strip().splitlines()

    for line in lines:
        parts  = line.strip().split()
        if not parts: continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts    = np.array([
            [int(coords[i]*w), int(coords[i+1]*h)]
            for i in range(0, len(coords)-1, 2)
        ], dtype=np.int32)

        col_rgb = VIZ_COLORS[cls_id % len(VIZ_COLORS)]
        col_bgr = (col_rgb[2], col_rgb[1], col_rgb[0])
        cv2.fillPoly(overlay, [pts], col_bgr)
        cv2.polylines(overlay, [pts], True, (255,255,255), 1)

        cx = int(pts[:,0].mean())
        cy = int(pts[:,1].mean())
        cv2.putText(overlay, CLASS_NAMES[cls_id],
                    (cx-8, cy+4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255,255,255), 1, cv2.LINE_AA)

        if cls_id not in seen:
            patches.append(mpatches.Patch(
                color=[c/255 for c in col_rgb],
                label=CLASS_NAMES[cls_id]
            ))
            seen.add(cls_id)

    blended = cv2.addWeighted(img_rgb, 0.4, overlay, 0.6, 0)
    axes[row,2].imshow(blended)
    axes[row,2].set_title(f"YOLO polygons ({len(lines)} objects)", fontsize=8)
    axes[row,2].legend(handles=patches, loc="upper right",
                       fontsize=6, framealpha=0.7)
    axes[row,2].axis("off")

plt.tight_layout()
out = YOLO_ROOT / "verify_conversion.png"
plt.savefig(str(out), dpi=120, bbox_inches="tight")
plt.show()
print(f"\n✅ Saved: {out}")

# ══════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  FINAL VERDICT")
print("=" * 55)
checks = {
    "All 5008 masks converted" : len(lbls) >= 5008,
    "Zero format errors"       : len(errors) == 0,
    "All 25 classes present"   : len(missing) == 0,
    "Points in range 3-200"    : pt_counts and max(pt_counts) < 200,
}
for check, ok in checks.items():
    print(f"  {'✅' if ok else '❌'}  {check}")

if all(checks.values()):
    print(f"\n  ✅ Conversion confirmed correct!")
    print(f"  Next step: upload to Kaggle and train YOLO11s")
else:
    print(f"\n  ❌ Fix issues above before training")

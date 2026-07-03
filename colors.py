
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

mask_dir = Path("/Users/pavneet/Desktop/industry45/PCBSegClassNet/data/segmentation/train/masks")
masks    = list(mask_dir.glob("*.png"))   # ALL masks not just 30

print(f"Scanning ALL {len(masks)} masks — takes ~2 min...")

all_colors = Counter()
for i, mask_path in enumerate(masks):
    img = cv2.imread(str(mask_path))
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels  = img_rgb.reshape(-1, 3)
    for px in pixels:
        r, g, b = int(px[0]), int(px[1]), int(px[2])
        if (r, g, b) != (0, 0, 0):
            all_colors[(r, g, b)] += 1
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(masks)} done, {len(all_colors)} colors found so far...")

print(f"\nALL colors found ({len(all_colors)} total):")
print(f"{'RGB':30s}  {'Count':>8s}")
print("-" * 42)
for color, count in all_colors.most_common(50):
    print(f"  {str(color):28s}  {count:>8d}")

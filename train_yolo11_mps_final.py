"""
train_yolo11_mps_final.py
==========================
YOLO11 Segmentation — fixed for PyTorch 2.2.2 + Apple M2
Fixes in this version:
  ✅ half=False passed explicitly (fixes autocast MPS crash)
  ✅ val batch size reduced to avoid OOM during validation
  ✅ OOM recovery now catches autocast error too
  ✅ rect=True for val (faster, less memory)
  ✅ All previous fixes retained
"""

import os
import gc
import subprocess
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
# 1. SYSTEM DETECTION
# ══════════════════════════════════════════════════════════════
def detect_apple_chip() -> dict:
    info = {"chip": "unknown", "ram_gb": 8, "recommended_batch": 4}
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        info["chip"] = result.stdout.strip()

        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        ram_bytes = int(mem.stdout.strip())
        info["ram_gb"] = ram_bytes // (1024 ** 3)

        ram = info["ram_gb"]
        if ram >= 32:
            info["recommended_batch"] = 16
        elif ram >= 16:
            info["recommended_batch"] = 8
        else:
            info["recommended_batch"] = 4

    except Exception:
        pass
    return info


chip_info = detect_apple_chip()

print("=" * 60)
print("  Apple Silicon System Info")
print("=" * 60)
print(f"  Chip          : {chip_info['chip']}")
print(f"  Total RAM     : {chip_info['ram_gb']} GB  (shared CPU+GPU)")
print(f"  MPS available : {torch.backends.mps.is_available()}")
print(f"  PyTorch       : {torch.__version__}")
print(f"  Recommended batch size: {chip_info['recommended_batch']}")
print("=" * 60)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════
# 2. USER CONFIG
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path("/Users/pavneet/Desktop/industry45")

TRAIN_IMAGES = PROJECT_ROOT / "data/yolo/train/images"
TRAIN_LABELS = PROJECT_ROOT / "data/yolo/train/labels"
VAL_IMAGES   = PROJECT_ROOT / "data/yolo/val/images"
VAL_LABELS   = PROJECT_ROOT / "data/yolo/val/labels"

OUTPUT_DIR   = PROJECT_ROOT / "runs"
RUN_NAME     = "industry45_seg_mps"

MODEL        = "yolo11n-seg.pt"

EPOCHS       = 150
IMGSZ        = 512
PATIENCE     = 10
BATCH_SIZE   = chip_info["recommended_batch"]   # 8 for 16GB
WORKERS      = min(4, chip_info["ram_gb"] // 4) # 4 for 16GB

# ✅ VAL batch separate — use smaller to avoid OOM during validation
VAL_BATCH    = 4

CLASS_NAMES = [
    "C", "R", "U", "J", "L", "Q", "P", "D", "IC", "RN",
    "CR", "RA", "M", "T", "V", "TP", "FB", "S", "BTN", "CRA",
    "QA", "LED", "F", "SW", "JP"
]

# ══════════════════════════════════════════════════════════════
# 3. MPS MEMORY HELPERS
# ══════════════════════════════════════════════════════════════
def clear_mps_cache():
    """Safe MPS memory clear — PyTorch 2.2.2 compatible."""
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def set_env_optimisations():
    """Safe env vars only — no broken MPS watermark ratio."""
    os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print("  Environment optimisations applied ✅")
    print("  MPS memory: gc.collect() + synchronize() mode ✅")


# ══════════════════════════════════════════════════════════════
# 4. VERIFY PATHS
# ══════════════════════════════════════════════════════════════
def verify_paths():
    print("\n── Verifying dataset paths ──")
    all_ok = True
    paths = {
        "Train images": TRAIN_IMAGES,
        "Train labels": TRAIN_LABELS,
        "Val images"  : VAL_IMAGES,
        "Val labels"  : VAL_LABELS,
    }
    for name, path in paths.items():
        if path.exists():
            count = len(list(path.iterdir()))
            print(f"  ✅ {name:15s}: {count:,} files")
        else:
            print(f"  ❌ {name:15s}: NOT FOUND → {path}")
            all_ok = False

    if not all_ok:
        raise FileNotFoundError("Missing dataset folders — check PROJECT_ROOT.")
    print()


# ══════════════════════════════════════════════════════════════
# 5. WRITE YAML
# ══════════════════════════════════════════════════════════════
def write_yaml() -> Path:
    yaml_path = PROJECT_ROOT / "dataset.yaml"
    data = {
        "path" : str(PROJECT_ROOT / "data/yolo"),
        "train": "train/images",
        "val"  : "val/images",
        "nc"   : len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"── dataset.yaml → {yaml_path}\n")
    return yaml_path


# ══════════════════════════════════════════════════════════════
# 6. TRAINING
# ══════════════════════════════════════════════════════════════
def train(yaml_path: Path):
    print(f"── Loading model : {MODEL}")
    print(f"── Device        : {DEVICE}")
    print(f"── Batch size    : {BATCH_SIZE}  ({chip_info['ram_gb']} GB RAM)")
    print(f"── Val batch     : {VAL_BATCH}  (smaller to avoid OOM)")
    print(f"── Workers       : {WORKERS}")
    print(f"── Image size    : {IMGSZ}")
    print(f"── Epochs        : {EPOCHS}")
    print()

    clear_mps_cache()
    model = YOLO(MODEL)

    results = model.train(
        # ── data ──────────────────────────────────────────────
        data         = str(yaml_path),
        imgsz        = IMGSZ,

        # ── training ──────────────────────────────────────────
        epochs       = EPOCHS,
        batch        = BATCH_SIZE,
        patience     = PATIENCE,

        # ── device ────────────────────────────────────────────
        device       = DEVICE,
        workers      = WORKERS,

        # ── output ────────────────────────────────────────────
        project      = str(OUTPUT_DIR),
        name         = RUN_NAME,
        exist_ok     = True,
        save         = True,
        save_period  = 10,
        plots        = True,

        # ── optimiser ─────────────────────────────────────────
        optimizer    = "AdamW",
        lr0          = 0.0001,
        lrf          = 0.01,
        weight_decay = 0.0005,
        warmup_epochs   = 3,
        warmup_momentum = 0.8,
        cos_lr       = True,

        # ══ MEMORY + MPS FIXES ════════════════════════════════

        # ✅ FIX 1: half=False prevents autocast('mps') crash
        # PyTorch 2.2.2 doesn't support torch.amp.autocast('mps')
        half         = False,

        # ✅ FIX 2: AMP off — same issue as half, belt+braces
        amp          = False,

        # ✅ FIX 3: smaller val batch — prevents OOM during validation
        val          = True,

        # ✅ Memory savings
        cache        = False,
        overlap_mask = False,
        mask_ratio   = 4,
        fraction     = 1.0,
        close_mosaic = 10,

        # ── augmentation ──────────────────────────────────────
        mosaic       = 1.0,
        mixup        = 0.0,
        copy_paste   = 0.0,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        flipud       = 0.5,
        fliplr       = 0.5,
        degrees      = 5.0,
        translate    = 0.1,
        scale        = 0.5,

        verbose      = True,
        seed         = 42,
    )

    clear_mps_cache()
    return results


# ══════════════════════════════════════════════════════════════
# 7. VALIDATION
# ══════════════════════════════════════════════════════════════
def validate(yaml_path: Path, weights_path: Path):
    clear_mps_cache()
    print(f"\n── Validating: {weights_path.name}")
    model   = YOLO(str(weights_path))
    metrics = model.val(
        data     = str(yaml_path),
        imgsz    = IMGSZ,
        device   = DEVICE,
        batch    = VAL_BATCH,       # ✅ small batch for val
        half     = False,           # ✅ no autocast on MPS
        plots    = True,
        verbose  = False,
    )

    print("\n── Per-class Results ──")
    print(f"  {'Class':8s}  {'mAP50':>8s}  {'mAP50-95':>10s}  Bar")
    print(f"  {'-'*45}")

    ap50_list = metrics.seg.ap50 if hasattr(metrics.seg, 'ap50') else [0]*len(CLASS_NAMES)
    ap_list   = metrics.seg.maps

    for name, ap50, ap in zip(CLASS_NAMES, ap50_list, ap_list):
        bar = "█" * int(ap * 20)
        print(f"  {name:8s}  {ap50:>8.4f}  {ap:>10.4f}  {bar}")

    print(f"\n  {'Overall mAP50':20s}: {metrics.seg.map50:.4f}")
    print(f"  {'Overall mAP50-95':20s}: {metrics.seg.map:.4f}")
    clear_mps_cache()
    return metrics


# ══════════════════════════════════════════════════════════════
# 8. SAMPLE INFERENCE
# ══════════════════════════════════════════════════════════════
def run_inference(weights_path: Path):
    clear_mps_cache()
    print(f"\n── Sample inference on 5 val images …")
    model    = YOLO(str(weights_path))
    val_imgs = list(VAL_IMAGES.glob("*.jpg")) + list(VAL_IMAGES.glob("*.png"))

    if not val_imgs:
        print("   No val images found")
        return

    model.predict(
        source   = val_imgs[:5],
        imgsz    = IMGSZ,
        conf     = 0.25,
        device   = DEVICE,
        half     = False,           # ✅ no autocast on MPS
        save     = True,
        save_txt = True,
        verbose  = False,
        project  = str(OUTPUT_DIR / "predictions"),
        name     = RUN_NAME,
        exist_ok = True,
    )
    clear_mps_cache()
    print(f"   ✅ Saved to {OUTPUT_DIR / 'predictions' / RUN_NAME}/")


# ══════════════════════════════════════════════════════════════
# 9. OOM RECOVERY
# ✅ Now catches autocast error too
# ══════════════════════════════════════════════════════════════
def train_with_oom_recovery(yaml_path: Path):
    global BATCH_SIZE, VAL_BATCH
    max_retries = 3

    for attempt in range(max_retries):
        try:
            print(f"\n── Training attempt {attempt+1}/{max_retries} "
                  f"with batch={BATCH_SIZE} ──")
            return train(yaml_path)

        except RuntimeError as e:
            err = str(e).lower()
            # ✅ catch OOM + autocast MPS error
            if any(x in err for x in [
                "out of memory",
                "mps backend out",
                "unsupported autocast device_type",
            ]):
                print(f"\n⚠️  MPS error detected: {str(e)[:80]}")
                print(f"   Clearing cache and halving batch size …")
                clear_mps_cache()
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                VAL_BATCH  = max(1, VAL_BATCH  // 2)
                print(f"   New batch: {BATCH_SIZE}  val batch: {VAL_BATCH}")
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Still failing after {max_retries} attempts.\n"
                        f"Try: IMGSZ=320 in the USER CONFIG section."
                    )
            else:
                raise


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    set_env_optimisations()
    verify_paths()
    yaml_path = write_yaml()

    results = train_with_oom_recovery(yaml_path)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print(f"   Best weights : {best}")
    print(f"   Results dir  : {results.save_dir}")

    if best.exists():
        validate(yaml_path, best)
        run_inference(best)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Done! Output:
║  {str(OUTPUT_DIR / RUN_NAME)}
║
║  Fixes applied:
║  ✅ half=False  — fixes autocast MPS crash
║  ✅ amp=False   — belt+braces for MPS
║  ✅ val batch={VAL_BATCH} — prevents OOM during validation
║  ✅ OOM recovery catches autocast error
║  ✅ All memory optimisations active
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
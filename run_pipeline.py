"""
run_pipeline.py
---------------
End-to-end pipeline for one fire:
  1. SIFT-align + normalize raw GeoTIFFs  (preprocess.py)
  2. Compute dNBR + USGS severity labels  (dnbr.py)
  3. Classical CV pipeline + IoU eval     (classical.py)
  4. Save all figures to outputs/<fire>/

Usage:
  python run_pipeline.py [fire_name]

Default fire: camp_fire
Expected raw inputs:
  data/raw/camp_fire_pre_2018.tif
  data/raw/camp_fire_post_2018.tif
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — saves files instead of showing windows

from pathlib import Path

# ── Config ────────────────────────────────────────────────
FIRE = sys.argv[1] if len(sys.argv) > 1 else "camp_fire"
YEAR = 2018
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs") / FIRE

OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Step 1: Preprocess ────────────────────────────────────
print("=" * 60)
print("STEP 1 — Preprocessing (SIFT align + normalize)")
print("=" * 60)

from preprocess import load_tiff, sift_align, normalize_pair, save_tiff

pre_raw_path  = RAW_DIR / f"{FIRE}_pre_{YEAR}.tif"
post_raw_path = RAW_DIR / f"{FIRE}_post_{YEAR}.tif"

if not pre_raw_path.exists() or not post_raw_path.exists():
    print(f"\nERROR: Raw tiffs not found at:")
    print(f"  {pre_raw_path}")
    print(f"  {post_raw_path}")
    print("\nDownload them from Google Drive → data/raw/ then re-run.")
    sys.exit(1)

print(f"Loading {pre_raw_path} ...")
pre_img, profile = load_tiff(pre_raw_path)
print(f"Loading {post_raw_path} ...")
post_img, _ = load_tiff(post_raw_path)
print(f"  Image shape: {pre_img.shape}")

print("Aligning with SIFT ...")
post_aligned = sift_align(pre_img, post_img)

print("Normalizing bands ...")
pre_norm, post_norm = normalize_pair(pre_img, post_aligned)

proc_fire_dir = PROC_DIR / FIRE
proc_fire_dir.mkdir(parents=True, exist_ok=True)
save_tiff(pre_norm,  profile, proc_fire_dir / "pre.tif")
save_tiff(post_norm, profile, proc_fire_dir / "post.tif")
print(f"Saved processed pair to {proc_fire_dir}/")

# ── Step 2: dNBR + USGS labels ────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — dNBR + USGS severity labels")
print("=" * 60)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dnbr import (
    compute_dnbr, dnbr_to_classes, otsu_threshold_dnbr,
    CLASS_LABELS, CLASS_COLORS, plot_dnbr_and_labels,
)
import rasterio

dnbr, nbr_pre, nbr_post = compute_dnbr(pre_norm, post_norm)
labels_usgs = dnbr_to_classes(dnbr)
labels_otsu, otsu_thresh = otsu_threshold_dnbr(dnbr)

print(f"dNBR range: [{np.nanmin(dnbr):.3f}, {np.nanmax(dnbr):.3f}]")
print(f"Otsu thresholds: {otsu_thresh}")
for i, cls in enumerate(CLASS_LABELS):
    pct = (labels_usgs == i).mean() * 100
    print(f"  {cls:12s}: {pct:.1f}%")

# Save rasters
single_profile = profile.copy()
single_profile.update(count=1, dtype=rasterio.float32)
with rasterio.open(OUT_DIR / "dnbr.tif", "w", **single_profile) as dst:
    dst.write(dnbr[np.newaxis])

label_profile = profile.copy()
label_profile.update(count=1, dtype=rasterio.int32)
with rasterio.open(OUT_DIR / "labels_usgs.tif", "w", **label_profile) as dst:
    dst.write(labels_usgs[np.newaxis])

# Save dNBR overview figure
plot_dnbr_and_labels(
    FIRE, pre_norm, post_norm, dnbr, labels_usgs,
    title_suffix="dNBR + USGS Labels",
    save_path=str(OUT_DIR / "dnbr_overview.png"),
)

# ── Step 3: Classical pipeline ────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Classical pipeline (Gaussian → Canny → Otsu)")
print("=" * 60)

from classical import (
    run_classical_pipeline, evaluate, plot_classical_pipeline,
)

dnbr_smooth, edges, pred_labels, thresholds = run_classical_pipeline(dnbr)
print(f"Otsu thresholds on smoothed dNBR: {thresholds}")

ious, acc = evaluate(pred_labels, labels_usgs)

plot_classical_pipeline(
    FIRE, dnbr, dnbr_smooth, edges, pred_labels, labels_usgs, ious,
    save_path=str(OUT_DIR / "classical_pipeline.png"),
)

# ── Step 4: Extra figures ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Saving extra diagnostic figures")
print("=" * 60)

# NBR scatter: pre vs post colored by USGS class
fig, ax = plt.subplots(figsize=(7, 6))
cmap_cls = mcolors.ListedColormap(CLASS_COLORS)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap_cls.N)

sample = np.random.default_rng(42).choice(nbr_pre.size, size=min(50_000, nbr_pre.size), replace=False)
sc = ax.scatter(
    nbr_pre.ravel()[sample],
    nbr_post.ravel()[sample],
    c=labels_usgs.ravel()[sample],
    cmap=cmap_cls, norm=norm, s=1, alpha=0.4,
)
ax.set_xlabel("NBR pre-fire")
ax.set_ylabel("NBR post-fire")
ax.set_title(f"{FIRE.replace('_',' ').title()} — NBR scatter (colored by USGS class)")
cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(CLASS_LABELS)
fig.tight_layout()
fig.savefig(OUT_DIR / "nbr_scatter.png", dpi=150)
plt.close(fig)
print(f"  Saved → {OUT_DIR}/nbr_scatter.png")

# Summary metrics table as text figure
fig, ax = plt.subplots(figsize=(5, 3))
ax.axis("off")
rows = [[cls, f"{ious[cls]:.4f}"] for cls in CLASS_LABELS]
rows.append(["mIoU", f"{ious['mIoU']:.4f}"])
rows.append(["Pixel Acc", f"{acc:.4f}"])
tbl = ax.table(
    cellText=rows,
    colLabels=["Class", "IoU"],
    loc="center",
    cellLoc="center",
)
tbl.scale(1, 1.5)
ax.set_title("Classical Pipeline — Evaluation Metrics", pad=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "metrics_table.png", dpi=150)
plt.close(fig)
print(f"  Saved → {OUT_DIR}/metrics_table.png")

# ── Done ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"All outputs saved to {OUT_DIR}/")
print("  dnbr_overview.png")
print("  classical_pipeline.png")
print("  nbr_scatter.png")
print("  metrics_table.png")
print("  dnbr.tif")
print("  labels_usgs.tif")
print("=" * 60)

# Wildfire Burn Severity Mapping via Satellite Imagery
CS 131 Final Project

## Overview
Automated burn severity classification using classical CV and deep learning on multispectral Landsat 8 imagery. Compares a classical pipeline (Gaussian filtering → Canny edge detection → Otsu thresholding on dNBR) against a U-Net segmentation model trained on raw multispectral bands.

## Setup
```bash
pip install -r requirements.txt
```

For GEE data access, authenticate first:
```bash
earthengine authenticate
```

## Project Structure
```
data/
  raw/         # raw GEE exports (gitignored)
  processed/   # aligned, normalized image pairs
notebooks/
  01_eda.ipynb          # exploratory data analysis
  02_classical.ipynb    # classical pipeline results
  03_unet.ipynb         # U-Net training and evaluation
src/
  preprocess.py   # SIFT alignment, band normalization, GEE export
  dnbr.py         # dNBR computation + label generation
  classical.py    # Gaussian + Canny + Otsu pipeline
  unet.py         # U-Net model + training loop
  evaluate.py     # per-class IoU and metrics
outputs/          # saved figures and maps
```

## Data
Uses the [MTBS dataset](https://www.mtbs.gov/) (Monitoring Trends in Burn Severity) from USGS, paired with Landsat 8 imagery accessed via Google Earth Engine. Subsetting to 5–10 large CA/OR fires from 2018–2022.

## Pipeline
1. **Preprocessing**: SIFT-based alignment of pre/post-fire image pairs, band normalization
2. **dNBR**: Compute differenced Normalized Burn Ratio → 4-class severity labels (unburned, low, moderate, high)
3. **Classical baseline**: Gaussian filter → Canny edges → Otsu threshold
4. **U-Net**: Train on stacked multispectral bands (RGB + NIR + SWIR), evaluate per-class IoU

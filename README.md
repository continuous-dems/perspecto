# VIZDEM: Wisdom for your DEMs

**Vizdem** is a lightweight, standalone Python tool for generating aesthetic and scientific visualizations from Digital Elevation Models (DEMs).

Originally born from the [CUDEM](https://github.com/cires-dem/cudem) project, Vizdem has been decoupled to run on a modern, "pure Python" geospatial stack (Rasterio, NumPy, Matplotlib). It allows you to create everything from "Joy Division" style ridgeline plots to rigorous, georeferenced hillshades with a single command.

> ⚠️ **BETA STATUS:** This project is in active development (v0.1.2).

## Features

* **Pure Python Stack:** Built on `rasterio`, `numpy`, and `matplotlib`. No complex compiled dependencies required for core features.
* **Joyplots:** Create artistic "Unknown Pleasures" style ridgeline plots from any DEM.
* **Hillshading:** * **`shade`**: Fast, artistic rendering for PNG/JPG images (using Matplotlib).
    * **`geoshade`**: Rigorous, buffered processing for GeoTIFF outputs (using Rasterio/NumPy). Preserves georeferencing for GIS workflows.
* **Hypsometry:** Instant histograms and CDF plots to understand your data distribution.
* **Colorbars:** Generate standalone legends/colorbars automatically matched to your data.
* **Fetchez Integration:** Seamlessly fetch color palettes (CPTs) from `cpt-city` via [fetchez](https://github.com/cires-dem/fetchez).

## Installation

### Prerequisites
* Python 3.9+
* [Fetchez](https://github.com/cires-dem/fetchez) (Recommended for CPT fetching)

### Install via Pip
```bash
git clone https://github.com/ciresdem/vizdem.git
cd vizdem
pip install -e .
```

## Usage
Vizdem is a Command Line Interface (CLI) tool. The general syntax is:
```bash
vizdem [COMMAND] [INPUT_DEM] [OPTIONS]
```

* ***Joyplots*** (joy)

Generate artistic ridgeline plots.

```bash
# Basic usage
vizdem joy input.tif --output art.png

# Customizing the look (High contrast, less overlap)
vizdem joy input.tif --step 20 --scale 0.2 --overlap 0.5 --color white --face-color black -out joy_dark.png
```

* ***Hillshading & Relief*** (shade)

Best for creating static images (PNG/JPG) for presentations or web use. Uses Matplotlib light sources.

```bash
# Simple grayscale hillshade
vizdem shade input.tif -out relief.png

# Color relief using a Matplotlib colormap
vizdem shade input.tif --cmap viridis --blend-mode soft -out color_relief.png

# Using a fetched CPT (requires fetchez)
vizdem shade input.tif --cmap city:gg/wiki-2.0 -out wikimap.png
```

* ***Georeferenced Shading*** (geoshade)

Best for GIS workflows. Produces a GeoTIFF that retains spatial metadata.

```bash
# Create a GeoTIFF hillshade
vizdem geoshade input.tif -out output_hs.tif

# Create a blended color relief GeoTIFF
vizdem geoshade input.tif --cmap terrain --blend-mode overlay -out output_blend.tif
```

* ***Histograms*** (hist)

Analyze the elevation distribution of your DEM.

```bash
# Basic Histogram (PDF) with statistics overlay
vizdem hist input.tif --stats True -out hist.png

# Cumulative Distribution Function (CDF)
vizdem hist input.tif --plot_type cdf --color red -out cdf.png
```

* ***Colorbars*** (colorbar)

Generate a standalone legend image. Reads the DEM to determine min/max range automatically.

```bash
# Horizontal colorbar (Matplotlib engine)
vizdem colorbar input.tif --cmap terrain --label "Elevation (m)" -out legend.png

# Vertical PyGMT colorbar (requires pygmt installed)
vizdem colorbar input.tif --engine pygmt --orient vertical -out legend_gmt.png
```

## Roadmap
[ ] Automatic DEM Fetching: Download data for a region automatically using fetchez.

[ ] Texture Overlay: Drape satellite imagery (Sentinel-2, NOAA) over hillshades.

[ ] Water Masking: Flatten water bodies using vector data.

[ ] 3D Perspective: Porting the POV-Ray modules for true 3D rendering.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ciresdem/vizdem/blob/main/LICENSE) file for details.

Copyright (c) 2010-2026 Regents of the University of Colorado
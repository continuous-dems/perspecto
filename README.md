# 🌍 PERSPECTO 🗺

**Wisdom for your DEMs**

**Perspecto** is a lightweight, standalone Python tool for generating aesthetic and scientific visualizations from Digital Elevation Models (DEMs).

Originally part the [CUDEM](https://github.com/ciresdem/cudem) project, Perspecto is now a standalone tool which allows you to create everything from "Joy Division" style ridgeline plots to rigorous, georeferenced hillshades with a single command.

> ⚠️ **BETA STATUS:** This project is in active development (v0.1.3).

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
git clone https://github.com/continuous-dems/perspecto.git
cd perspecto
pip install -e .
```

## Usage
Perspecto is a Command Line Interface (CLI) tool. The general syntax is:
```bash
perspecto [COMMAND] [INPUT_DEM] [OPTIONS]
```

* ***Hillshading & Relief*** (shade)

Best for creating static images (PNG/JPG) for presentations or web use. Uses Matplotlib light sources.

```bash
# Simple grayscale hillshade
perspecto shade input.tif -out relief.png

# Color relief using a Matplotlib colormap
perspecto shade input.tif --cmap viridis --blend-mode soft -out color_relief.png

# Using a fetched CPT (requires fetchez)
perspecto shade input.tif --cmap wiki-2.0 -out wikimap.png
```

* ***Georeferenced Shading*** (geoshade)

Best for GIS workflows. Produces a GeoTIFF that retains spatial metadata.

```bash
# Create a GeoTIFF hillshade
perspecto geoshade input.tif -out output_hs.tif

# Create a blended color relief GeoTIFF
perspecto geoshade input.tif --cmap terrain --blend-mode overlay -out output_blend.tif
```

* ***Histograms*** (hist)

Analyze the elevation distribution of your DEM.

```bash
# Basic Histogram (PDF) with statistics overlay
perspecto hist input.tif --stats True -out hist.png

# Cumulative Distribution Function (CDF)
perspecto hist input.tif --plot_type cdf --color red -out cdf.png
```

* ***Joyplots*** (joy)

Generate artistic ridgeline plots.

```bash
# Basic usage
perspecto joy input.tif --output art.png

# Customizing the look (High contrast, less overlap)
perspecto joy input.tif --step 20 --scale 0.2 --overlap 0.5 --color white --face-color black -out joy_dark.png
```

* ***Colorbars*** (colorbar)

Generate a standalone legend image. Reads the DEM to determine min/max range automatically.

```bash
# Horizontal colorbar (Matplotlib engine)
perspecto colorbar input.tif --cmap terrain --label "Elevation (m)" -out legend.png

# Vertical PyGMT colorbar (requires pygmt installed)
perspecto colorbar input.tif --engine pygmt --orient vertical -out legend_gmt.png
```

## Roadmap
[ ] Automatic DEM Fetching: Download data for a region automatically using fetchez.

[ ] Texture Overlay: Drape satellite imagery (Sentinel-2, NOAA) over hillshades.

[ ] 3D Perspective: Porting the POV-Ray modules for true 3D rendering.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/continuous-dems/perspecto/blob/main/LICENSE) file for details.

Copyright (c) 2010-2026 Regents of the University of Colorado
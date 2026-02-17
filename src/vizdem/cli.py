#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.cli
~~~~~~~~~~~~~~~~~~~~~~~

VizDEM CLI, using click

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import logging
import click

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(name)s: %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option()
def main():
    """
    vizdem: Wisdom for your DEMs.

    Tools to create aesthetic and scientific visualizations from elevation data.
    """
    pass

vlz = None

# --- JOYPLOT COMMAND ---
@main.command()
@click.argument('dem', type=click.Path(exists=True))
@click.option('--step', '-s', default=10, help='Row step (decimation). Higher = fewer lines.')
@click.option('--scale', '-z', default=0.1, help='Vertical exaggeration.')
@click.option('--overlap', '-o', default=1.0, help='Overlap factor (1.0 = standard).')
@click.option('--color', '-c', default='black', help='Line color.')
@click.option('--output', '-out', default=None, help='Output filename.')
def joy(dem, step, scale, overlap, color, output):
    """Joy Division (ridgeline) plot."""
    from .modules.joy import Joyplot

    viz = Joyplot(
        src_dem=dem,
        step=step,
        scale=scale,
        overlap=overlap,
        line_color=color,
        outfile=output
    )
    viz.run()

# --- HISTOGRAM COMMAND ---
@main.command()
@click.argument('dem', type=click.Path(exists=True))
@click.option('--bins', '-b', default=100, help='Data Bins.')
@click.option('--plot_type', '-p', default='pdf', help='Plot type.')
@click.option('--color', '-c', default='gray', help='Bar color.')
@click.option('--dpi', '-d', default=300, help='Dots Per Inch..')
@click.option('--stats', '-s', default=False, help='Display Statistics.')
@click.option('--title', '-t', default=None, help='Graph Title.')
@click.option('--output', '-out', default=None, help='Output filename.')
def hist(dem, bins, plot_type, color, dpi, stats, title,  output):
    """Histogram plot."""
    from .modules.histogram import Histogram

    viz = Histogram(
        src_dem=dem,
        bins=bins,
        plot_tpye=plot_type,
        color=color,
        dpi=dpi,
        stats=stats,
        title=title,
        outfile=output
    )
    viz.run()

# --- SHADE COMMAND ---
@main.command()
@click.argument('dem', type=click.Path(exists=True))
@click.option('--azimuth', '-az', default=315, help='Light azimuth (degrees).')
@click.option('--altitude', '-alt', default=45, help='Light altitude (degrees).')
@click.option('--exag', '-x', default=1.0, help='Vertical exaggeration.')
@click.option('--cmap', '-c', default=None, help='Colormap (e.g., terrain, viridis) or "gray".')
@click.option('--output', '-out', default=None, help='Output filename.')
def shade(dem, azimuth, altitude, exag, cmap, output):
    """Hillshade or Color Relief."""
    from .modules.shade import Hillshade

    viz = Hillshade(
        src_dem=dem,
        azimuth=azimuth,
        altitude=altitude,
        vert_exag=exag,
        cmap=cmap,
        outfile=output
    )
    viz.run()

# --- GEOSHADE COMMAND ---
@main.command()
@click.argument('dem', type=click.Path(exists=True))
@click.option('--azimuth', '-az', default=315, help='Light azimuth (degrees).')
@click.option('--altitude', '-alt', default=45, help='Light altitude (degrees).')
@click.option('--exag', '-x', default=1.0, help='Vertical exaggeration.')
@click.option('--cmap', '-c', default='gray', help='Colormap (e.g., terrain, viridis) or "gray".')
@click.option('--output', '-out', default=None, help='Output filename.')
def geoshade(dem, azimuth, altitude, exag, cmap, output):
    """Georeferenced Hillshade or Color Relief with Rasterio."""
    from .modules.geoshade import GeoHillshade

    viz = GeoHillshade(
        src_dem=dem,
        azimuth=azimuth,
        altitude=altitude,
        vert_exag=exag,
        cmap=cmap,
        outfile=output
    )
    viz.run()

# --- GDAL_HILLSHADE COMMAND ---
@main.command()
@click.argument('dem', type=click.Path(exists=True))
@click.option('--azimuth', '-az', default=315, help='Light azimuth (degrees).')
@click.option('--altitude', '-alt', default=45, help='Light altitude (degrees).')
@click.option('--exag', '-x', default=1.0, help='Vertical exaggeration.')
@click.option('--cpt', '-c', default=None, help='Colormap (e.g., terrain, viridis) or "gray".')
@click.option('--output', '-out', default=None, help='Output filename.')
def gdal_hillshade(dem, azimuth, altitude, exag, cpt, output):
    """Georeferenced Hillshade or Color Relief with GDAL."""
    from .modules.gdal_hillshade import GDALHillshade

    viz = GDALHillshade(
        src_dem=dem,
        vertical_exaggeration=exag,
        projection=4326,
        azimuth=azimuth,
        altitude=altitude,
        modulate=125,
        alpha=False,
        mode='multiply',
        gamma=0.5,
        cpt=cpt,
        outfile=output
    )
    viz.run()

# --- COLORBAR COMMAND ---
@main.command()
@click.argument('dem', type=click.Path(exists=True))
@click.option('--label', '-l', default='Elevation (m)', help='Label for the colorbar.')
@click.option('--cmap', '-c', default='terrain', help='Colormap name or CPT file.')
@click.option('--orient', '-o', type=click.Choice(['horizontal', 'vertical']), default='horizontal', help='Orientation.')
@click.option('--width', '-w', default=6.0, help='Width in inches.')
@click.option('--height', '-h', default=1.0, help='Height in inches.')
@click.option('--min-z', default=None, type=float, help='Force min value.')
@click.option('--max-z', default=None, type=float, help='Force max value.')
@click.option('--engine', '-e', type=click.Choice(['matplotlib', 'pygmt']), default='matplotlib', help='Rendering engine.')
@click.option('--output', '-out', default=None, help='Output filename.')
def colorbar(dem, label, cmap, orient, width, height, min_z, max_z, engine, output):
    """Generate a standalone colorbar image."""
    from .modules.colorbar import Colorbar

    if output is None:
        # Auto-name: input_cbar.png
        base = os.path.splitext(os.path.basename(dem))[0]
        output = f"{base}_cbar.png"

    viz = Colorbar(
        src_dem=dem,
        cmap=cmap,
        label=label,
        orientation=orient,
        width=width,
        height=height,
        engine=engine,
        min_z=min_z,
        max_z=max_z,
        outfile=output
    )
    viz.run()

if __name__ == '__main__':
    main()

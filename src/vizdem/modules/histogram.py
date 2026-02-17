#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.modules.histogram
~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
from ..base import VizdemBase

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Histogram(VizdemBase):
    """Generate a Hypsometric Histogram or CDF from a DEM.

    Visualizes the distribution of elevation values within the dataset.

    Args:
      bins (int) : Number of histogram bins. Default: 100.
      type (str) : Plot type ['pdf', 'cdf']. Default: 'pdf'.
                   - pdf: Probability Density Function (Standard Histogram)
                   - cdf: Cumulative Distribution Function
      color (str) : Bar color. Default: 'gray'.
      stats (bool) : Overlay mean/median/std_dev lines. Default: False.
      title (str) : Custom title for the plot.
      dpi (int) : Output resolution. Default: 300.

    < histogram:bins=100:type=pdf:stats=True:color=steelblue >
    """

    def __init__(self, bins=100, plot_type='pdf', color='gray', show_stats=False, title=None, dpi=300, **kwargs):
        super().__init__(mod='histogram', **kwargs)
        self.bins = int(bins)
        self.plot_type = plot_type
        self.color = color
        self.show_stats = show_stats
        self.title = title
        self.dpi = int(dpi)

    def run(self):
        logger.info("Reading raster data...")
        elev = self.load_dem(decimation=1)
        elev = elev.flatten()
        mean_val = np.mean(elev)
        median_val = np.median(elev)
        std_val = np.std(elev)
        min_val = np.min(elev)
        max_val = np.max(elev)

        fig, ax = plt.subplots(figsize=(10, 6))

        cumulative = (self.plot_type == 'cdf')
        density = True

        n, bins_out, patches = ax.hist(
            elev,
            bins=self.bins,
            density=density,
            cumulative=cumulative,
            color=self.color, # [self.color] * len(elev),
            alpha=0.75,
            edgecolor='black',
            linewidth=0.5
        )

        if self.title is None:
            self.title = f"Hypsometry: {os.path.basename(self.src_dem)}"

        ax.set_title(self.title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Elevation (z)", fontsize=12)
        ax.set_ylabel("Frequency / Probability", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        if self.show_stats:
            stats_text = (
                f"Min: {min_val:.2f}\n"
                f"Max: {max_val:.2f}\n"
                f"Mean: {mean_val:.2f}\n"
                f"Median: {median_val:.2f}\n"
                f"Std Dev: {std_val:.2f}"
            )

            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean ({mean_val:.1f})')
            ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Median ({median_val:.1f})')

            if not cumulative:
                ax.axvline(mean_val + std_val, color='orange', linestyle='dotted', label='1 Std Dev')
                ax.axvline(mean_val - std_val, color='orange', linestyle='dotted')

            ax.legend(loc='upper right')

            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        if self.verbose:
            logger.info(f"Saving histogram to {self.outfile}...")

        plt.savefig(self.outfile, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return self.outfile

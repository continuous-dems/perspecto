#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.modules.colorbar
~~~~~~~~~~~~~~~~~~~~~~~

Generate a color bar.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from ..base import VizdemBase
from .. import cpt as cpt_utils

logger = logging.getLogger(__name__)

class Colorbar(VizdemBase):
    """Generate a standalone colorbar image."""

    def __init__(self, cmap='terrain', label='Elevation (m)',
                 orientation='horizontal', width=6, height=1, dpi=300,
                 engine='matplotlib', min_z=None, max_z=None, **kwargs):
        super().__init__(**kwargs)
        self.cmap_name = cmap
        self.label = label
        self.orientation = orientation
        self.width = float(width)
        self.height = float(height)
        self.dpi = int(dpi)
        self.engine = engine.lower()
        self.min_z = min_z
        self.max_z = max_z

    def _get_z_range(self):
        """Determine min/max from args or DEM stats."""

        if self.min_z is not None and self.max_z is not None:
            return float(self.min_z), float(self.max_z)

        logger.info(f"Reading DEM statistics from {self.src_dem}...")
        z_min, z_max = self.dem_minmax(decimation=10)

        final_min = float(self.min_z) if self.min_z is not None else z_min
        final_max = float(self.max_z) if self.max_z is not None else z_max
        return final_min, final_max

    def run_matplotlib(self):
        """Generate colorbar using Matplotlib."""

        vmin, vmax = self._get_z_range()

        try:
            cmap = plt.get_cmap(self.cmap_name)
        except ValueError:
            if os.path.exists(self.cmap_name):
                cmap = cpt_utils.load_cmap(self.cmap_name)
            else:
                logger.info(f"Colormap '{self.cmap_name}' not found. Using 'terrain'.")
                cmap = plt.get_cmap('terrain')

        # Create Figure
        fig = plt.figure(figsize=(self.width, self.height))

        if self.orientation == 'horizontal':
            ax = fig.add_axes([0.05, 0.5, 0.9, 0.15])
        else:
            ax = fig.add_axes([0.5, 0.05, 0.15, 0.9])

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        cb = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation=self.orientation,
            label=self.label
        )

        logger.info(f"Saving Matplotlib colorbar to {self.outfile}")
        fig.savefig(self.outfile, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    def run_pygmt(self):
        """Generate colorbar using PyGMT."""

        try:
            import pygmt
        except ImportError:
            logger.error("Error: PyGMT not installed. Install it or use --engine matplotlib")
            return

        vmin, vmax = self._get_z_range()
        fig = pygmt.Figure()
        # pygmt.makecpt(cmap=self.cmap_name, series=[vmin, vmax])
        pos_str = f"JTC+w{self.width}i/{self.height}i+h" if self.orientation=='horizontal' else f"JML+w{self.width}i/{self.height}i"

        fig.colorbar(
            cmap=self.cmap_name,
            position=pos_str,
            frame=[f"x+l{self.label}", "y+1m"],
            region=[0, 10, 0, 10],
            projection="X1i/1i"
        )

        logger.info(f"Saving PyGMT colorbar to {self.outfile}")
        fig.savefig(self.outfile)

    def run(self):
        if self.engine == 'pygmt':
            self.run_pygmt()
        else:
            self.run_matplotlib()
        return self.outfile

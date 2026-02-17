#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.modules.shade
~~~~~~~~~~~~~~~~~~~~~~~

Generate a Shaded Relief image.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from ..base import VizdemBase
from .. import cpt as cpt_utils

logger = logging.getLogger(__name__)

class Hillshade(VizdemBase):
    """Generate Hillshades and Color Relief maps."""

    def __init__(self, azimuth=315, altitude=45, vert_exag=1,
                 blend_mode='soft', cmap=None, **kwargs):
        super().__init__(**kwargs)
        self.azimuth = azimuth
        self.altitude = altitude
        self.vert_exag = vert_exag
        self.blend_mode = blend_mode # 'soft', 'overlay', 'hsv'
        self.cmap = cmap

    def run(self):
        elev = self.load_dem(decimation=1)

        dx = self.res[0]
        dy = self.res[1]

        logger.info(f"Computing hillshade (Az: {self.azimuth}, Alt: {self.altitude})...")

        ls = LightSource(azdeg=self.azimuth, altdeg=self.altitude)

        dpi = 300
        height, width = elev.shape
        fig = plt.figure(frameon=False)
        fig.set_size_inches(width/dpi, height/dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        cm = cpt_utils.parse_cptmap(self.cmap)
        if cm is None or isinstance(cm, str):
            self.cpt = cm
            self.init_cpt()
            if os.path.exists(self.cpt):
                cm = cpt_utils.load_cmap(self.cpt)
            else:
                cm = plt.get_cmap(self.cmap)

        self.cmap = cm.name
        logger.info(f"Blending with colormap: {self.cmap} using mode: {self.blend_mode}")

        rgb = ls.shade(
            elev,
            cmap=cm,
            vert_exag=self.vert_exag,
            dx=dx,
            dy=dy,
            blend_mode=self.blend_mode
        )
        ax.imshow(rgb)

        logger.info(f"Saving to {self.outfile}")
        fig.savefig(self.outfile, dpi=dpi)
        plt.close(fig)

        return self.outfile

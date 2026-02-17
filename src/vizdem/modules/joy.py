#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.modules.shade
~~~~~~~~~~~~~~~~~~~~~~~

Generate a Shaded Relief image.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from ..base import VizdemBase

logger = logging.getLogger(__name__)

class Joyplot(VizdemBase):
    """Generate 'Joy Division' style ridgeline plots."""

    def __init__(self, step=10, scale=0.1, overlap=1.0,
                 line_color='black', face_color='white', dpi=300, **kwargs):
        super().__init__(**kwargs)
        self.step = int(step)
        self.scale = float(scale)
        self.overlap = float(overlap)
        self.line_color = line_color
        self.face_color = face_color
        self.dpi = int(dpi)

    def run(self):
        arr = self.load_dem(decimation=self.step)
        arr = np.flipud(arr)

        rows, cols = arr.shape
        x = np.arange(cols)

        aspect = cols / rows
        fig_width = 10
        fig_height = fig_width / aspect if aspect > 0 else 10

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if self.face_color.lower() in ['none', 'transparent']:
            ax.patch.set_alpha(0.0)
        else:
            ax.set_facecolor(self.face_color)

        z_min, z_max = np.nanmin(arr), np.nanmax(arr)
        offset_step = (z_max - z_min) * 0.1 * (1 / self.overlap)
        if offset_step == 0: offset_step = 1

        logger.info(f"Plotting {rows} lines...")

        for i, row in enumerate(arr):
            y_base = i * offset_step

            row_clean = np.nan_to_num(row, nan=z_min)
            y_signal = y_base + (row_clean * self.scale)

            ax.fill_between(
                x,
                y_signal,
                y_base - (z_max * self.scale * 2), # Fill deep enough to cover
                facecolor=self.face_color,
                edgecolor='none',
                zorder=i
            )

            ax.plot(
                x,
                y_signal,
                color=self.line_color,
                linewidth=0.5,
                zorder=i
            )

        ax.set_axis_off()

        logger.info(f"Saving to {self.outfile}")
        plt.savefig(
            self.outfile,
            bbox_inches='tight',
            pad_inches=0,
            dpi=self.dpi,
            transparent=(self.face_color.lower() in ['none', 'transparent'])
        )
        plt.close(fig)
        return self.outfile

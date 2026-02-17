#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.modules.geoshade
~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from ..base import VizdemBase
from .. import cpt as cpt_utils

logger = logging.getLogger(__name__)

class GeoHillshade(VizdemBase):
    """Generate a Georeferenced Hillshade/Relief using Rasterio and Numpy.

    This preserves the spatial metadata of the input (unlike standard image plotting)
    and supports complex blending modes.

    BETA
    """

    def __init__(
        self,
        azimuth=315,
        altitude=45,
        vert_exag=1,
        cmap='terrain',
        blend_mode='multiply',
        alpha=False,
        gamma=None,
        chunk_size=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.azimuth = azimuth
        self.altitude = altitude
        self.vert_exag = vert_exag
        self.cmap = cmap
        self.blend_mode = blend_mode
        self.alpha = alpha
        self.gamma = float(gamma) if gamma else None
        self.chunk_size = chunk_size

        if self.outfile.endswith('.png'):
            base = os.path.splitext(os.path.basename(self.src_dem))[0]
            self.outfile = f"{base}_hillshade.tif"

        self.cpt = cmap
        # self.init_cpt()
        # self.cpt_no_slash()


    def _apply_gamma(self, arr):
        """Apply gamma correction to a normalized (0-1) or uint8 array."""

        if self.gamma is None:
            return arr

        if np.issubdtype(arr.dtype, np.integer):
            arr = arr / 255.0

        corrected = arr ** (1 / self.gamma)
        return corrected

    def _blend_arrays(self, hs_norm, rgb_norm):
        """Blend normalized hillshade (H) and color (C) arrays (0.0 - 1.0).
        Returns blended array (0.0 - 1.0).
        """

        # H is (Rows, Cols), C is (Rows, Cols, Bands)
        H = hs_norm[..., np.newaxis]
        C = rgb_norm

        if self.blend_mode == 'multiply':
            return H * C

        elif self.blend_mode == 'screen':
            return 1 - (1 - H) * (1 - C)

        elif self.blend_mode == 'overlay':
            # Overlay: Multiply if H < 0.5, Screen if H >= 0.5
            mask = H < 0.5
            out = np.empty_like(C)
            out[mask] = 2 * H[mask] * C[mask]
            out[~mask] = 1 - 2 * (1 - H[~mask]) * (1 - C[~mask])
            return out

        elif self.blend_mode == 'soft_light':
            # Pegtop's Soft Light formula
            return (1 - 2 * H) * (C ** 2) + 2 * H * C

        elif self.blend_mode == 'hard_light':
            # Hard Light is Overlay with H and C swapped
            mask = C < 0.5
            out = np.empty_like(C)
            out[mask] = 2 * C[mask] * H[mask]
            out[~mask] = 1 - 2 * (1 - C[~mask]) * (1 - H[~mask])
            return out

        # Default
        return H * C

    def run(self):
        logger.info(f"Starting GeoHillshade (Mode: {self.blend_mode})...")

        profile, res = self.dem_infos()
        count = 4 if self.alpha else 3
        profile.update(
            dtype=rasterio.uint8,
            count=count,
            nodata=None,
            driver='GTiff',
            photometric='RGB'
        )
        dx, dy = res

        ls = LightSource(azdeg=self.azimuth, altdeg=self.altitude)

        cm = cpt_utils.parse_cptmap(self.cmap)
        if cm is None:
            self.init_cpt()
            if os.path.exists(self.cpt):
                cm = cpt_utils.load_cmap(self.cpt)
            else:
                cm = plt.get_cmap(self.cmap)

        self.cmap = cm.name

        # cm = cpt_utils.parse_cptmap(self.cmap)
        # if cm is None:
        #     self.init_cpt()
        #     if os.path.exists(self.cpt):
        #         cm = cpt_utils.load_cmap(self.cpt)
        #     else:
        #         cm = plt.get_cmap(self.cmap)
        # # Get Colormap
        # try:
        #     # If fetchez integration is ready, we would fetch CPT here
        #     cm = plt.get_cmap(self.cmap)
        #     #cm = self.cpt
        # except ValueError:
        #     logger.info(f"Colormap {self.cmap} not found, using 'terrain'")
        #     cm = plt.get_cmap('terrain')


        elev = self.load_dem()
        logger.info(f"Writing to {self.outfile}...")
        with rasterio.open(self.outfile, 'w', **profile) as dst:

            # Calculate Hillshade on padded chunk
            # Returns (Rows, Cols) float 0-1
            hs = ls.hillshade(elev, vert_exag=self.vert_exag, dx=dx, dy=dy)

            # Calculate Color Relief on padded chunk
            # Returns (Rows, Cols, 4) float 0-1 (RGBA)
            rgb = ls.shade(elev, cmap=cm, vert_exag=self.vert_exag, dx=dx, dy=dy)

            # Discard alpha from cmap if not requested, or keep it
            # We separate RGB from A for blending
            rgb_colors = rgb[..., :3]

            # Gamma Correction
            if self.gamma:
                hs = self._apply_gamma(hs)
                rgb_colors = self._apply_gamma(rgb_colors)

            # Blend
            blended = self._blend_arrays(hs, rgb_colors)

            # Prepare for Write (Convert to uint8 0-255)
            blended_uint8 = (blended * 255).astype(np.uint8)

            # Transpose
            write_data = np.transpose(blended_uint8, (2, 0, 1))

            # Handle Transparency (Alpha)
            if count == 4:
                if np.ma.is_masked(elev):
                    # mask = elev.mask[crop_r_start:crop_r_end, crop_c_start:crop_c_end]
                    alpha_band = (elev * 255).astype(np.uint8)
                else:
                    alpha_band = np.full((height, width), 255, dtype=np.uint8)

                write_data = np.concatenate([write_data, alpha_band[np.newaxis, ...]])

            dst.write(write_data)

        return self.outfile

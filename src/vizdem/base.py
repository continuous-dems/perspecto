#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.base
~~~~~~~~~~~~~~~~~~~~~~~

Base Class for vizdem

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import rasterio
from rasterio.enums import Resampling
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VizdemBase:
    """Base class for Vizdem modules.

    Handles DEM loading via Rasterio, metadata parsing, and standard output naming.
    """

    def __init__(self, src_dem, outfile=None, verbose=True, **kwargs):
        self.src_dem = src_dem
        self.verbose = verbose
        self.kwargs = kwargs
        self.outfile = outfile

        if not os.path.exists(self.src_dem):
            raise FileNotFoundError(f"Source DEM not found: {self.src_dem}")

        try:
            with rasterio.open(self.src_dem) as src:
                self.profile = src.profile
                self.bounds = src.bounds
                self.res = src.res
                self.width = src.width
                self.height = src.height
                self.crs = src.crs
                self.nodata = src.nodata
        except Exception as e:
            raise ValueError(f"Could not open source DEM '{src_dem}': {e}")

        if self.outfile is None:
            base = os.path.splitext(os.path.basename(self.src_dem))[0]
            self.outfile = f"{base}_viz.png"

        self.split_cpt = False
        self.cpt = None

    def cpt_no_slash(self):
        """Removes backslashes from the CPT file."""

        try:
            with open(self.cpt, 'r') as file:
                filedata = file.read()

            filedata = filedata.replace('\\', ' ')

            with open(self.cpt, 'w') as file:
                file.write(filedata)
        except OSError as e:
            logger.error(f"Error processing CPT file: {e}")

    def init_cpt(self, min_z=None, max_z=None, want_gdal=False):
        """Initialize and process the CPT file."""

        from . import cpt
        import shutil

        def _move_cpt(src):
            """Helper to rename the generic temporary CPT to a unique specific name."""

            if src and os.path.exists(src):
                dst = f"{os.path.splitext(src)[0]}.cpt"
                shutil.move(src, dst)
                return dst
            return src

        ## Determine Z-Range
        z_range = self.dem_minmax()
        min_z = min_z if min_z is not None else z_range[0]
        max_z = max_z if max_z is not None else z_range[1]

        logger.info(
            f"Initializing CPT: Range [{min_z}, {max_z}], GDAL={want_gdal}, Split={self.split_cpt}]"
        )

        ## No CPT provided: Generate ETOPO
        if self.cpt is None:
            tmp_cpt = cpt.generate_etopo_cpt(min_z, max_z)
            self.cpt = _move_cpt(tmp_cpt)

        ## Local File provided: Process it
        elif os.path.exists(self.cpt):
            tmp_cpt = cpt.process_cpt(
                self.cpt, min_z, max_z, gdal=want_gdal, split_cpt=self.split_cpt
            )
            self.cpt = _move_cpt(tmp_cpt)

        ## Remote/City CPT requested
        else:
            ## fetch_cpt_city returns a path to a downloaded file
            city_cpt = cpt.fetch_cpt_city(query=self.cpt)
            logger.info(city_cpt)
            if city_cpt:
                tmp_cpt = cpt.process_cpt(
                    city_cpt, min_z, max_z, gdal=want_gdal, split_cpt=self.split_cpt
                )
                self.cpt = _move_cpt(tmp_cpt)
            else:
                ## Fallback
                logger.warning(f"Could not find CPT {self.cpt}, falling back to ETOPO.")
                tmp_cpt = cpt.generate_etopo_cpt(min_z, max_z)
                self.cpt = _move_cpt(tmp_cpt)

    def dem_infos(self):
        profile = None
        res = None
        with rasterio.open(self.src_dem) as src:
            profile = src.profile.copy()
            res = src.res

        return profile, res

    def dem_minmax(self, decimation=1):
        try:
            data = self.load_dem(decimation=decimation)
            z_min = np.nanmin(data)
            z_max = np.nanmax(data)

            return z_min, z_max
        except Exception as e:
            raise ValueError(f"Could not open source DEM '{self.src_dem}': {e}")

    def load_dem(self, decimation=1):
        """Loads the DEM into a numpy array.

        Args:
            decimation (int): Factor to downsample data (1 = full res).
        """

        logger.info(f"Loading DEM: {self.src_dem} (Decimation: {decimation})")

        with rasterio.open(self.src_dem) as src:
            if decimation > 1:
                new_h = src.height // decimation
                new_w = src.width // decimation
                data = src.read(
                    1,
                    out_shape=(new_h, new_w),
                    resampling=Resampling.bilinear
                )
            else:
                data = src.read(1)

            if self.nodata is not None:
                data = data.astype('float32')
                data[data == self.nodata] = np.nan
            else:
                data = data.astype('float32')

        return data

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")

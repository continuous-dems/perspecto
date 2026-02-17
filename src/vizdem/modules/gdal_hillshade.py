#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.modules.gdal_hillshade
~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import shutil
import logging
import numpy as np

from fetchez.utils import str_or, float_or, make_temp_fn, remove_glob2, fn_url_p

from ..base import VizdemBase
from ..utils import run_cmd, yield_srcwin

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

logger = logging.getLogger(__name__)


class gdal_datasource:
    """Context manager for GDAL datasets.

    Usage:
        with gdal_datasource('input.tif') as src_ds:
            # do something
    """

    def __init__(self, src_gdal=None, update=False):
        self.src_gdal = src_gdal
        self.update = update
        self.src_ds = None

    def __enter__(self):
        if isinstance(self.src_gdal, gdal.Dataset):
            self.src_ds = self.src_gdal
            return self.src_ds

        if str_or(self.src_gdal) is not None and (
            os.path.exists(self.src_gdal) or
            fn_url_p(self.src_gdal) or
            ':' in self.src_gdal
        ):
            try:
                if self.update:
                    self.src_ds = gdal.OpenEx(
                        self.src_gdal,
                        gdal.OF_RASTER | gdal.OF_UPDATE,
                        # open_options=['IGNORE_COG_LAYOUT_BREAK=YES']
                    )
                else:
                    self.src_ds = gdal.Open(self.src_gdal, gdal.GA_ReadOnly)
            except Exception:
                self.src_ds = None

        if self.src_ds is not None and self.update:
             self.src_ds.BuildOverviews('AVERAGE', [])

        return self.src_ds

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not isinstance(self.src_gdal, gdal.Dataset) and self.src_ds:
            if self.update:
                try:
                    self.src_ds.FlushCache()
                except Exception:
                    pass
            self.src_ds = None


def gdal_infos(src_ds, scan=False, band=1):
    """Gather information from a GDAL dataset.
    Optimized for /vsicurl/ to avoid full-file reads.

    Returns dictionary with:
        nx, ny, nb (dims), geoT, proj, ndv, zr (min/max), fmt
    """

    ds = None
    close_ds = False
    is_remote = False

    if isinstance(src_ds, str):
        is_remote = src_ds.startswith(('/vsicurl/', 'http', 'https', 'ftp'))

        if is_remote:
            gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')

        try:
            ds = gdal.Open(src_ds, gdal.GA_ReadOnly)
            close_ds = True
        except Exception:
            pass
    else:
        ds = src_ds
        if hasattr(ds, 'GetDescription'):
             desc = ds.GetDescription()
             is_remote = desc.startswith(('/vsicurl/', 'http', 'https', 'ftp'))

    if ds is None:
        return None

    nx = ds.RasterXSize
    ny = ds.RasterYSize
    nb = nx * ny
    geoT = ds.GetGeoTransform()
    proj = ds.GetProjection()
    driver = ds.GetDriver()
    fmt = driver.ShortName if driver else 'Unknown'
    raster_count = ds.RasterCount
    metadata = ds.GetMetadata()

    ndv = None
    zr = [None, None]
    dtn = None
    dt = None

    if nb > 0:
        try:
            target_band = ds.GetRasterBand(band)
            ndv = target_band.GetNoDataValue()
            dtn = gdal.GetDataTypeName(target_band.DataType)
            dt = target_band.DataType
            force_scan = 1 if (scan and not is_remote) else 0

            stats = target_band.GetStatistics(1, force_scan)

            if stats:
                if stats[0] != stats[1]:
                    zr = [stats[0], stats[1]]
                else:
                    if scan and not is_remote:
                        zr = target_band.ComputeRasterMinMax(1)

        except Exception as e:
            if scan:
                logger.warning(f"Could not retrieve stats: {e}")

    if close_ds:
        ds = None

    return {
        'nx': nx,
        'ny': ny,
        'nb': nb,
        'geoT': geoT,
        'proj': proj,
        'ndv': ndv,
        'dt': dt,
        'dtn': dtn,
        'zr': zr,
        'fmt': fmt,
        'metadata': metadata,
        'raster_count': raster_count
    }


class GDALHillshade(VizdemBase):
    """Generate a Hillshade Image using GDAL and Numpy blending.

    Supports various blending modes:
    https://en.wikipedia.org/wiki/Blend_modes#Overlay

    Configuration Example:
    < hillshade:vertical_exaggeration=1:projection=4326:azimuth=315:altitude=45 >
    """

    def __init__(
            self,
            vertical_exaggeration=1,
            projection=4326,
            azimuth=315,
            altitude=45,
            modulate=125,
            alpha=False,
            mode='multiply',
            gamma=0.5,
            cpt=None,
            split_cpt=None,
            **kwargs,
    ):
        super().__init__(mod='hillshade', **kwargs)
        self.vertical_exaggeration = vertical_exaggeration
        self.projection = projection
        self.azimuth = azimuth
        self.altitude = altitude
        self.modulate = float_or(modulate, 115)
        self.alpha = alpha
        self.mode = mode
        self.gamma = float_or(gamma)
        self.cpt = cpt
        self.split_cpt = split_cpt

        self.init_cpt()
        self.cpt_no_slash()

        if self.outfile.endswith('.png'):
            base = os.path.splitext(os.path.basename(self.src_dem))[0]
            self.outfile = f"{base}_hillshade.tif"


    def _modulate(self, gdal_fn):
        """Adjusts brightness/saturation using ImageMagick (External utility)."""

        run_cmd(
            f"mogrify -modulate {self.modulate} -depth 8 {gdal_fn}",
        )


    def gamma_correction_calc(self, hs_fn=None, outfile='gdaldem_gamma.tif'):
        """Apply gamma correction using gdal_calc (External utility)."""

        from osgeo_utils import gdal_calc

        gdal_calc.Calc(
            "uint8(((A / 255.)**(1/0.5)) * 255)",
            A=hs_fn,
            outfile=outfile,
            quiet=True
        )
        return outfile


    def gamma_correction(self, arr):
        """Apply gamma correction to a numpy array."""

        if self.gamma is not None:
            return (((arr / 255.)**(1 / self.gamma)) * 255.).astype(np.uint8)
        return arr


    def blend(self, hs_file, rgb_file, mode='multiply', gamma=None, outfile='gdaldem_multiply.tif'):
        """Blend a Hillshade (hs_file) and Color Relief (rgb_file) using the specified mode."""

        valid_modes = ['multiply', 'screen', 'overlay', 'hard_light', 'soft_light']
        if mode not in valid_modes:
            mode = 'multiply'

        with gdal_datasource(rgb_file, update=True) as rgb_ds:
            with gdal_datasource(hs_file) as cr_ds:
                ds_config = gdal_infos(cr_ds)
                n_chunk = int(ds_config['nx'] * 0.1)

                # Iterate over the image in chunks to manage memory
                for srcwin in yield_srcwin(
                    (ds_config['ny'], ds_config['nx']),
                    n_chunk=n_chunk,
                    msg=f'Blending rgb and hillshade using {mode}',
                    end_msg='Generated color hillshade',
                    verbose=self.verbose
                ):
                    cr_band = cr_ds.GetRasterBand(1)
                    cr_arr = cr_band.ReadAsArray(*srcwin)
                    cr_arr = self.gamma_correction(cr_arr)

                    # Normalize hillshade for calculations
                    cr_norm = cr_arr / 255.0

                    for band_no in [1, 2, 3]:
                        band = rgb_ds.GetRasterBand(band_no)
                        band_arr = band.ReadAsArray(*srcwin)
                        band_norm = band_arr / 255.0

                        out_arr = np.copy(band_arr)

                        if mode == 'multiply':
                            out_arr = (cr_norm * band_norm * 255.).astype(np.uint8)

                        elif mode == 'screen':
                            out_arr = ((1 - ((1 - cr_norm) * (1 - band_norm))) * 255.).astype(np.uint8)

                        elif mode == 'overlay':
                            # Contrast mode: Multiplies or Screens based on bottom layer
                            mask_low = cr_arr < 128
                            mask_high = cr_arr >= 128

                            out_arr[mask_low] = (2 * cr_norm[mask_low] * band_norm[mask_low] * 255.).astype(np.uint8)
                            out_arr[mask_high] = ((1 - (2 * (1 - cr_norm[mask_high]) * (1 - band_norm[mask_high]))) * 255.).astype(np.uint8)

                        elif mode == 'hard_light':
                            # Like Overlay, but based on top layer
                            mask_low = band_arr < 128
                            mask_high = band_arr >= 128

                            out_arr[mask_low] = (2 * cr_norm[mask_low] * band_norm[mask_low] * 255.).astype(np.uint8)
                            out_arr[mask_high] = ((1 - (2 * (1 - cr_norm[mask_high]) * (1 - band_norm[mask_high]))) * 255.).astype(np.uint8)

                        elif mode == 'soft_light':
                            mask_low = band_arr < 128
                            mask_high = band_arr >= 128

                            # Soft light formula for darker pixels
                            term1_low = 2 * cr_norm[mask_low] * band_norm[mask_low]
                            term2_low = (cr_norm[mask_low]**2) * (1 - 2 * band_norm[mask_low])
                            out_arr[mask_low] = ((term1_low + term2_low) * 255.).astype(np.uint8)

                            # Soft light formula for lighter pixels
                            term1_high = 2 * cr_norm[mask_high] * (1 - band_norm[mask_high])
                            term2_high = np.sqrt(cr_norm[mask_high]) * (2 * band_norm[mask_high] - 1)
                            out_arr[mask_high] = ((term1_high + term2_high) * 255.).astype(np.uint8)

                        band.WriteArray(out_arr, srcwin[0], srcwin[1])

        os.rename(rgb_file, outfile)
        return outfile

    def run(self):
        hs_fn = make_temp_fn('gdaldem_hs.tif', os.path.dirname(self.outfile))
        gdal.DEMProcessing(
            hs_fn,
            self.src_dem,
            'hillshade',
            computeEdges=True,
            scale=111120,
            azimuth=self.azimuth,
            altitude=self.altitude,
            zFactor=self.vertical_exaggeration
        )

        cr_fn = make_temp_fn('gdaldem_cr.tif', os.path.dirname(self.outfile))
        gdal.DEMProcessing(
            cr_fn,
            self.src_dem,
            'color-relief',
            colorFilename=self.cpt,
            computeEdges=True,
            addAlpha=self.alpha
        )

        cr_hs_fn = make_temp_fn('gdaldem_cr_hs.tif', os.path.dirname(self.outfile))
        self.blend(hs_fn, cr_fn, gamma=self.gamma, mode=self.mode, outfile=cr_hs_fn)

        remove_glob2(hs_fn, cr_fn)
        os.rename(cr_hs_fn, self.outfile)

        # self._modulate(self.outfile) # Optional ImageMagick post-processing
        # self.gmt_figure()            # Optional GMT figure generation
        logger.info(f"Saving to {self.outfile}")
        return self.outfile


class Hillshade_cmd(VizdemBase):
    """
    Generate a Hillshade Image using command-line tools.
    Requires GDAL and ImageMagick installed in the system path.
    """

    def __init__(
        self,
        vertical_exaggeration=1,
        projection=None,
        azimuth=315,
        altitude=45,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vertical_exaggeration = vertical_exaggeration
        self.projection = projection
        self.azimuth = azimuth
        self.altitude = altitude
        self.cpt_no_slash()


    def run(self):
        basename = os.path.splitext(self.src_dem)[0]

        # Generate Hillshade
        run_cmd(
            f'gdaldem hillshade -compute_edges -s 111120 '
            f'-z {self.vertical_exaggeration} -az {self.azimuth} -alt {self.altitude} '
            f'{self.src_dem} hillshade.tif',
            verbose=self.verbose
        )

        # Generate Color Relief
        run_cmd(
            f'gdaldem color-relief {self.src_dem} {self.cpt} colors.tif',
            verbose=self.verbose
        )

        # Composite (ImageMagick)
        run_cmd(
            'composite -compose multiply -depth 8 colors.tif hillshade.tif output.tif',
            verbose=self.verbose
        )
        run_cmd(
            'mogrify -modulate 115 -depth 8 output.tif',
            verbose=self.verbose
        )

        # Restore Georeferencing
        run_cmd(
            f'gdal_translate -co "TFW=YES" {self.src_dem} temp.tif',
            verbose=self.verbose
        )
        run_cmd('mv temp.tfw output.tfw')

        # Translate back to GeoTiff
        srs_cmd = f'-a_srs epsg:{self.projection}' if self.projection else ''
        run_cmd(
            f'gdal_translate {srs_cmd} output.tif temp2.tif'
        )

        #  Cleanup
        remove_glob2(
            'output.tif*', 'temp.tif*', 'hillshade.tif*', 'colors.tif*', 'output.tfw*'
        )

        # Move
        final_output = f'{basename}_hs.tif'
        run_cmd(f'gdal_translate temp2.tif {final_output}')
        remove_glob2('temp2.tif')

        return final_output

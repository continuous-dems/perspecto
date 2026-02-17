import os
import shutil
import logging
from fetchez import core
from fetchez.utils import echo_msg, echo_error_msg
from fetchez.spatial import buffer_region
from fetchez.registry import FetchezRegistry
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
from .modules.shade import Hillshade
from .modules.joy import Joyplot
from .modules.colorbar import Colorbar

logger = logging.getLogger(__name__)

class QuickViz(core.FetchModule):
    """Fetchez module to auto-generate visualizations from the best available DEM data.

    Usage:
      fetchez quickviz --region "Boulder, CO" --style "poster"
    """

    PRIORITY = {
        'tnm': 1,        # USGS 3DEP (US Only)
        'copernicus': 2, # Global 30m
        'nasadem': 3,    # Global 30m (Void filled)
        #'aw3d30': 4,     # ALOS
        #'fabdem': 5,     # Forest removed
        'gebco': 10,     # Global 450m (Background)
        'etopo': 11,     # Global (Background)
    }

    def __init__(self, style='shade', output=None,
                 sources=None, **kwargs):
        super().__init__(**kwargs)
        self.style = style
        self.output = output
        self.sources = sources if sources else ['etopo']
        self.dem_path = None

    def fetch_and_merge_data(self):
        """Fetches data from multiple sources and merges them based on priority.
        High priority datasets (e.g., Copernicus) will overwrite lower priority
        datasets (e.g., GEBCO) where they overlap.
        """

        collected_files = []

        for source_name in self.sources:
            logger.info(f"Fetching coverage from {source_name}...")

            module_cls = FetchezRegistry.load_module(source_name)
            if not module_cls:
                logger.error(f"Module {source_name} not found in registry.")
                continue

            self.region = buffer_region(self.region, 200)
            fetcher = module_cls(src_region=self.region)
            print(fetcher.region)
            #outdir=os.path.join(self._outdir, '_raw_data', source_name),
            #)

            fetcher.run()

            if fetcher.results:
                core.run_fetchez([fetcher])

                found_files = [
                    r['dst_fn'] for r in fetcher.results
                    if 'dst_fn' in r and os.path.exists(r['dst_fn'])
                ]

                for f in found_files:
                    collected_files.append({
                        'path': f,
                        'source': source_name,
                        'priority': self.PRIORITY.get(source_name, 99)
                    })

        if not collected_files:
            return None

        collected_files.sort(key=lambda x: x['priority'])

        file_paths = [f['path'] for f in collected_files]
        logger.info(f"Merging {len(file_paths)} tiles (Priority Order: {[f['source'] for f in collected_files]})...")

        return self._dem_merge(file_paths)

    def _dem_merge(self, file_list):
        """Merges tiles using Rasterio and clips to the requested region."""

        temp_mosaic = os.path.join(self._outdir, '_temp_mosaic.tif')

        final_dem = os.path.join(self._outdir, 'vizdem_merged.tif')

        datasets_to_close = []
        datasets_to_merge = []

        try:
            first_src = rasterio.open(file_list[0])
            datasets_to_close.append(first_src)
            datasets_to_merge.append(first_src)
            dst_crs = first_src.crs

            for fn in file_list[1:]:
                src = rasterio.open(fn)
                datasets_to_close.append(src)

                if src.crs != dst_crs:
                    vrt = WarpedVRT(src, crs=dst_crs)
                    datasets_to_close.append(vrt)
                    datasets_to_merge.append(vrt)
                else:
                    datasets_to_merge.append(src)

            mosaic, out_trans = merge(datasets_to_merge, method='first')

            out_meta = datasets_to_merge[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": dst_crs
            })

            with rasterio.open(temp_mosaic, "w", **out_meta) as dest:
                dest.write(mosaic)

            w, e, s, n = self.region
            bounds = (w, s, e, n)
            if bounds:
                logger.info(f"Clipping output to region {bounds}...")
                with rasterio.open(temp_mosaic) as src:
                    win = from_bounds(*bounds, transform=src.transform)
                    clip_transform = src.window_transform(win)
                    data = src.read(window=win, boundless=False)
                    clip_meta = src.meta.copy()
                    clip_meta.update({
                        "height": data.shape[1],
                        "width": data.shape[2],
                        "transform": clip_transform
                    })

                    with rasterio.open(final_dem, "w", **clip_meta) as dst:
                        dst.write(data)
            else:
                shutil.move(temp_mosaic, final_dem)

        except Exception as e:
            logger.error(f"Merge/Clip failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            for ds in reversed(datasets_to_close):
                ds.close()
            if os.path.exists(temp_mosaic):
                os.remove(temp_mosaic)

        return final_dem

    def run(self):
        self.dem_path = self.fetch_and_merge_data()

        if not self.dem_path:
            logger.error("QuickViz: Could not obtain DEM data.")
            return

        if self.output is None:
            self.output = f"viz_{self.style}.png"
        self.output = os.path.join(self._outdir, self.output)

        logger.info(f"Generating '{self.style}' visualization...")

        viz = None

        if self.style == 'joy':
            viz = Joyplot(
                src_dem=self.dem_path,
                outfile=self.output,
                step=20,
                overlap=2.0,
                line_color='white',
                face_color='black'
            )

        elif self.style == 'poster':
            # Soft lighting, terrain color, vertical exaggeration
            viz = Hillshade(
                src_dem=self.dem_path,
                outfile=self.output,
                azimuth=315,
                altitude=45,
                vert_exag=3.0,
                blend_mode='soft',
                cmap='terrain'
            )

        elif self.style == 'scientific':
            viz = Hillshade(
                src_dem=self.dem_path,
                outfile=self.output,
                blend_mode='overlay', # Sharper than soft
                cmap='viridis'
            )
            # colorbar as well

        else:
            viz = Hillshade(src_dem=self.dem_path, outfile=self.output)

        if viz:
            result = viz.run()

            self.results.append({
                "url": f"file://{os.path.abspath(result)}",
                "dst_fn": result,
                "status": "generated"
            })

# --- Fetchez Hook ---
def setup_fetchez(registry):
    registry.register_module(
        mod_key="quickviz",
        mod_cls=QuickViz,
        metadata={
            "category": "Visualization",
            "desc": "Auto-fetch (Copernicus+GEBCO) and visualize DEMs.",
            "tags": ["viz", "dem", "pipeline", "merge"],
            "agency": "Vizdem",
            "license": "MIT"
        }
    )

import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import FullLoader, load

import numpy as np
import pandas as pd
import rasterio as rio
from glob import glob
from rasterio.crs import CRS
from rasterio.warp import reproject

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)

# Functions ------------------------------------------


def pct_to_rgb(input_dir, output_dir='outputs/rgb_images', plan_scales_path=None, nodata_key=255, tile_suffix='.tif', overwrite=False):

    os.makedirs(output_dir, exist_ok=True)

    tiles_list = glob(os.path.join(input_dir, '*.tif'))
    if len(tiles_list) == 0:
        logger.critical('No tile found in the input folder. Please control the path.')
        sys.exit(1)

    if plan_scales_path:
            plan_scales = pd.read_excel(plan_scales_path)
    else:
        logger.info('No info on the scale of each tile, setting the scale to 0.')

    for tile_path in tqdm(tiles_list, desc='Convert images from colormap to RGB'):
        tile_name = os.path.basename(tile_path).rstrip(tile_suffix)

        # Get the scale for the new tile name
        if plan_scales_path and (tile_name in plan_scales.Num_plan.unique()):
            tile_scale = plan_scales.loc[plan_scales.Num_plan==tile_name, 'Echelle'].iloc[0]
        else:
            tile_scale = 0

        out_path = os.path.join(output_dir, f"{tile_scale}_{tile_name[:6]}_{tile_name[6:]}.tif")
        if not overwrite and os.path.isfile(out_path):
            continue

        with rio.open(tile_path) as src:
            image = src.read()
            bounds = src.bounds
            meta = src.meta
            colormap = src.colormap(1)

        nodata_value = colormap[nodata_key][0]
        if nodata_value != 0:
            print()
            logger.warning(f'The nodata value for the plan {tile_name} is {nodata_value} and not 0.')
        converted_image = np.empty((3, meta['height'], meta['width']))
        # Efficient mapping: https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
        mapping_key = np.array(list(colormap.keys()))
        for band_nbr in range(3):
            # Get colormap corresponding to band 
            mapping_values = np.array([mapping_list[band_nbr] for mapping_list in colormap.values()])
            mapping_array = np.zeros(mapping_key.max()+1, dtype=mapping_values.dtype)
            mapping_array[mapping_key] = mapping_values
            
            # Translate colormap into corresponding band
            new_band = mapping_array[image]
            converted_image[band_nbr, :, :] = new_band

        if not meta['crs']:
            print()
            logger.warning(f'No crs for the tile {tile_name}. Setting it to EPSG:2056.')
            meta.update(crs=CRS.from_epsg(2056))
        elif meta['crs'] != CRS.from_epsg(2056):
            print()
            logger.warning(f'Wrong crs for the tile {tile_name}: {meta['crs']}, tile will be reprojected.')

        if (meta['crs'] != CRS.from_epsg(2056)) or (meta['transform'][1] != 0):
            new_transform, new_width, new_height = rio.warp.calculate_default_transform(meta['crs'], rio.CRS.from_epsg(2056), meta['width'], meta['height'],
                                                                    left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3])
            converted_image, new_transform = reproject(
                converted_image, np.empty((3, new_height, new_width)), src_transform=meta['transform'],
                src_crs=meta['crs'], dst_transform=new_transform, dst_crs=CRS.from_epsg(2056)
            )
            meta.update(transform=new_transform, width=new_width, height=new_height)

        meta.update(count=3, nodata=nodata_value)
        with rio.open(out_path, 'w', **meta) as dst:
            dst.write(converted_image)


# ------------------------------------------

if __name__ == "__main__":

    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = ArgumentParser(description="The script prepares the initial files for the use of the OD in the detection of border points.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    with open(args.config_file) as fp:
        cfg_globals = load(fp, Loader=FullLoader)['globals']

    WORKING_DIR = cfg['working_dir']
    INPUT_DIR = cfg['input_dir']
    OUTPUT_DIR = cfg['output_dir']

    PLAN_SCALES = cfg['plan_scales']
    NODATA_KEY = cfg['nodata_key'] if 'nodata_key' in cfg.keys() else 255

    TILE_SUFFIX = cfg_globals['original_tile_suffix']
    OVERWRITE = cfg_globals['overwrite']

    os.chdir(WORKING_DIR)

    pct_to_rgb(INPUT_DIR, OUTPUT_DIR, PLAN_SCALES, NODATA_KEY, TILE_SUFFIX, OVERWRITE)

    print()
    logger.success(f"The files were written in the folder {OUTPUT_DIR}. Let's check them out!")
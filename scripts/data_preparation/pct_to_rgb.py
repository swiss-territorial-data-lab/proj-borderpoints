import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import FullLoader, load

import numpy as np
import rasterio
from glob import glob
from rasterio.crs import CRS

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)

# Functions ------------------------------------------


def pct_to_rgb(input_dir, output_dir='outputs/rgb_images', nodata_key=255, overwrite=False):

    os.makedirs(output_dir, exist_ok=True)

    tiles_list = glob(os.path.join(input_dir, '*.tif'))
    if len(tiles_list) == 0:
        logger.critical('No tile found in the input folder. Please control the path.')
        sys.exit(1)

    for tile_path in tqdm(tiles_list, desc='Convert images from colormap to RGB'):
        tile_name = os.path.basename(tile_path)
        out_path = os.path.join(output_dir, tile_name)

        if not overwrite and os.path.isfile(out_path):
            continue

        with rasterio.open(tile_path) as src:
            image = src.read()
            meta = src.meta
            colormap = src.colormap(1)

        nodata_value = colormap[nodata_key][0]
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

        if meta['crs'] != CRS.from_epsg(2056):
            logger.warning(f'Wrong crs for the tile {tile_name}: {meta["crs"]}.')

        meta.update(count=3, nodata=nodata_value)
        with rasterio.open(out_path, 'w', **meta) as dst:
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

    NODATA_KEY = cfg['nodata_key'] if 'nodata_key' in cfg.keys() else 255

    OVERWRITE = cfg_globals['overwrite']

    os.chdir(WORKING_DIR)

    pct_to_rgb(INPUT_DIR, OUTPUT_DIR, NODATA_KEY, OVERWRITE)

    print()
    logger.success(f"The files were written in the folder {OUTPUT_DIR}. Let's check them out!")
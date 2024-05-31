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

from joblib import Parallel, delayed

sys.path.insert(1, 'scripts')
import constants as cst
from functions.fct_misc import format_logger

logger = format_logger(logger)

# Functions ------------------------------------------

def format_tiles(tile_path, output_dir, plan_scales_path=None, plan_scales_df=pd.DataFrame(columns=['Num_plan', 'Echelle']), nodata_key=255, tile_suffix='.tif'):
    """
    Formats a tiff image by renaming it, converting the color map, and reprojecting it if necessary.
    
    Args:
        tile_path (str): The path to the image file.
        output_dir (str): The directory where the formatted image will be saved.
        plan_scales_path (str, optional): The path to the plan scales file. Defaults to None.
        plan_scales_df (pandas.DataFrame, optional): The plan scales DataFrame. Defaults to a DataFrame with columns 'Num_plan' and 'Echelle'.
        nodata_key (int, optional): The key value for the nodata pixel. Defaults to 255.
        tile_suffix (str, optional): The suffix of the tile image file. Defaults to '.tif'.
    
    Returns:
        tuple: A tuple containing the original tile name and the new tile name (without the file extension).
    
    Raises:
        None
    """
    tile_name = os.path.basename(tile_path).rstrip(tile_suffix)

    # Get the scale for the new tile name
    if plan_scales_path and (tile_name in plan_scales_df.Num_plan.unique()):
        tile_scale = plan_scales_df.loc[plan_scales_df.Num_plan==tile_name, 'Echelle'].iloc[0]
    else:
        tile_scale = 0

    tile_new_name = f"{tile_scale}_{tile_name[:6]}_{tile_name[6:]}.tif"
    out_path = os.path.join(output_dir, tile_new_name)
    if not cst.OVERWRITE and os.path.isfile(out_path):
        return False
    
    name_correspondance = (tile_name, tile_new_name.rstrip('.tif'))

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
        logger.warning(f'Wrong crs for the tile {tile_name}: {meta["crs"]}, tile will be reprojected.')

    if (meta['crs'] != CRS.from_epsg(2056)) or (meta['transform'][1] != 0):
        converted_image, new_transform = reproject(
            converted_image, src_transform=meta['transform'],
            src_crs=meta['crs'], dst_crs=CRS.from_epsg(2056)
        )
        meta.update(transform=new_transform, height=converted_image.shape[1], width=converted_image.shape[2])

    meta.update(count=3, nodata=nodata_value)
    with rio.open(out_path, 'w', **meta) as dst:
        dst.write(converted_image)

    return name_correspondance

def pct_to_rgb(input_dir, output_dir='outputs/rgb_images', plan_scales_path=None, nodata_key=255, tile_suffix='.tif'):

    os.makedirs(output_dir, exist_ok=True)
    njobs=5

    tiles_list = glob(os.path.join(input_dir, '*.tif'))
    if len(tiles_list) == 0:
        logger.critical('No tile found in the input folder. Please control the path.')
        sys.exit(1)

    if plan_scales_path:
            plan_scales = pd.read_excel(plan_scales_path)
    else:
        logger.info('No info on the scale of each tile, setting the scale to 0.')
        plan_scales = pd.DataFrame(columns=['Num_plan', 'Echelle'])

    name_correspondance_list = Parallel(n_jobs=njobs, backend="loky")(
        delayed(format_tiles)(tile_path, output_dir, plan_scales_path, plan_scales, nodata_key, tile_suffix) 
        for tile_path in tqdm(tiles_list, desc=f'Convert images from colormap to RGB, {njobs} at a time')
    )

    if any(name_correspondance for name_correspondance in name_correspondance_list):
        name_correspondance_df = pd.DataFrame.from_records(name_correspondance_list, columns=['original_name', 'rgb_name'])
        name_correspondance_df.to_csv(os.path.join(output_dir, 'name_correspondance.csv'))

    logger.success(f"The files were written in the folder {output_dir}. Let's check them out!")


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
        cfg = load(fp, Loader=FullLoader)['prepare_data.py']

    with open(args.config_file) as fp:
        cfg_globals = load(fp, Loader=FullLoader)['globals']

    WORKING_DIR = cfg['working_dir']
    INPUT_DIR = cfg['input_dir']
    OUTPUT_DIR = cfg['tile_dir']

    PLAN_SCALES = cfg['plan_scales']
    NODATA_KEY = cfg['nodata_key'] if 'nodata_key' in cfg.keys() else 255

    TILE_SUFFIX = cfg['tile_suffix']

    os.chdir(WORKING_DIR)

    pct_to_rgb(INPUT_DIR, OUTPUT_DIR, PLAN_SCALES, NODATA_KEY, TILE_SUFFIX)
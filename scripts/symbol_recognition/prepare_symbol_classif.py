import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd

sys.path.insert(1,'scripts')
from constants import OVERWRITE
from data_preparation import format_surveying_data, get_delimitation_tiles, pct_to_rgb, tiles_to_box
from sandbox import get_point_bbox_size
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script prepares the initial files for the use of the classification of the symbol images.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR_VECT= cfg['output_dir']['vectors']

INITIAL_IMAGE_DIR = cfg['initial_image_dir']
TILE_DIR = cfg['tile_dir']

BORDER_POINTS_POLY = cfg['border_points_poly']
CADASTRAL_SURVEYING = cfg['cadastral_surveying']

CONVERT_IMAGES = cfg['convert_images']
TILE_SUFFIX = cfg['tile_suffix']

os.chdir(WORKING_DIR)

if CONVERT_IMAGES:
    pct_to_rgb.pct_to_rgb(INITIAL_IMAGE_DIR, TILE_DIR, tile_suffix=TILE_SUFFIX)

logger.info('Get the maximum size of border points by scale...')
pt_sizes_gdf, written_files = get_point_bbox_size.get_point_bbox_size(BORDER_POINTS_POLY, OUTPUT_DIR_VECT)

tiles_gdf, nodata_gdf, _, tmp_written_files = get_delimitation_tiles.get_delimitation_tiles(TILE_DIR, output_dir=OUTPUT_DIR_VECT, subtiles=False)
written_files.extend(tmp_written_files)

logger.info('Format cadastral surveying data...')
filepath = os.path.join(OUTPUT_DIR_VECT, 'MO_pt_polys.gpkg')
if not os.path.isfile(filepath) or OVERWRITE:
    nodata_gdf = nodata_gdf.dissolve(by='tile_name', as_index=False)
    # logger.info('Buffer nodata areas...')
    # nodata_gdf = misc.buffer_by_max_size(nodata_gdf, pt_sizes_gdf, factor=0.25)
    cs_points_gdf, tmp_written_files = format_surveying_data.format_surveying_data(CADASTRAL_SURVEYING, tiles_gdf, nodata_gdf, remove_duplicates=False, output_dir=OUTPUT_DIR_VECT)
    written_files.extend(tmp_written_files)

    logger.info('Transform points to polygons...')
    cs_points_poly_gdf = misc.buffer_by_max_size(cs_points_gdf, pt_sizes_gdf, factor=0.5, cap_style=3)
    # Save image name on tile
    cs_points_poly_gdf['image_name'] = (cs_points_poly_gdf.apply(lambda x: misc.get_tile_name(x.initial_tile.split('_')[0], x.geometry), axis=1)).str.rstrip('.tif')
    
    cs_points_poly_gdf.to_file(filepath)
    written_files.append(filepath)

else:
    logger.info(f'File {filepath} already exists. Read from disk...')
    cs_points_poly_gdf = gpd.read_file(filepath)

logger.info('Clip image for each border point...')
SYMBOL_IM_DIR = os.path.join(TILE_DIR, 'symbol_images')
tiles_to_box.tiles_to_box(TILE_DIR, cs_points_poly_gdf, SYMBOL_IM_DIR)

logger.info('Test if images intersect...')
_, tmp_written_files = misc.find_intersecting_polygons(cs_points_poly_gdf, OUTPUT_DIR_VECT)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
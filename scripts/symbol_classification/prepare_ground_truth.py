import os
import sys
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

sys.path.insert(1,'scripts')
from constants import OVERWRITE
from data_preparation import format_labels, get_delimitation_tiles, pct_to_rgb, tiles_to_box, get_point_bbox_size
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), "The script prepares the initial files for the classification of symbol images.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR_VECT = cfg['output_dir']['vectors']

INITIAL_IMAGE_DIR = cfg['initial_image_dir']
TILE_DIR = cfg['tile_dir']

BORDER_POINTS_PT = cfg['border_points_pt']
BORDER_POINTS_POLY = cfg['border_points_poly']

CONVERT_IMAGES = cfg['convert_images']
TILE_SUFFIX = cfg['tile_suffix']

os.chdir(WORKING_DIR)

if CONVERT_IMAGES:
    pct_to_rgb.main(INITIAL_IMAGE_DIR, TILE_DIR, tile_suffix=TILE_SUFFIX)

pts_gdf, written_files = format_labels.main(BORDER_POINTS_PT, os.path.join(OUTPUT_DIR_VECT, 'GT'))
pts_gdf['combo_id'] = pts_gdf['pt_id'] + ' - ' + pts_gdf['Num_plan']

logger.info('Get the maximum size of border points according to scale...')

pt_sizes_gdf, written_files = get_point_bbox_size.main(BORDER_POINTS_POLY, OUTPUT_DIR_VECT)

tiles_gdf, _, _, tmp_written_files = get_delimitation_tiles.main(TILE_DIR, output_dir=OUTPUT_DIR_VECT, subtiles=False)
written_files.extend(tmp_written_files)

logger.info('Format cadastral surveying data...')
filepath = os.path.join(OUTPUT_DIR_VECT, 'GT', 'GT_pt_polys.gpkg')
if not os.path.isfile(filepath) or OVERWRITE:
    
    logger.info('Transform points to polygons...')
    pts_gdf.rename(columns={'Echelle': 'scale'}, inplace=True)

    cs_points_poly_gdf = misc.buffer_by_max_size(pts_gdf, pt_sizes_gdf, factor=0.5, cap_style=3)

    # Get info on tile and image names
    tiles_gdf['initial_tile'] = [x + y for z, x, y in tiles_gdf.name.str.split('_')]
    cs_points_poly_gdf = pd.merge(
        cs_points_poly_gdf, tiles_gdf[['initial_tile', 'name']], left_on='Num_plan', right_on='initial_tile'
    ).drop(columns='initial_tile').rename(columns={'name': 'initial_tile'})
    cs_points_poly_gdf['image_name'] = (cs_points_poly_gdf.apply(lambda x: misc.get_tile_name(x.initial_tile.split('_')[0], x.geometry), axis=1)).str.rstrip('.tif')

    # Get combo id based on point and tile id
    cs_points_poly_gdf['combo_id'] = cs_points_poly_gdf['pt_id'] + ' - ' + cs_points_poly_gdf['initial_tile']

    cs_points_poly_gdf[['pt_id', 'combo_id', 'scale', 'initial_tile', 'image_name', 'CATEGORY', 'SUPERCATEGORY', 'geometry']].to_file(filepath)
    written_files.append(filepath)

else:
    logger.info(f'File {filepath} already exists. Read from disk...')
    cs_points_poly_gdf = gpd.read_file(filepath)

logger.info('Clip image for each border point...')
SYMBOL_IM_DIR = os.path.join(TILE_DIR, 'symbol_images_GT')
tiles_to_box.main(TILE_DIR, cs_points_poly_gdf, SYMBOL_IM_DIR)

logger.info('Test if images intersect...')
_, tmp_written_files = misc.find_intersecting_polygons(cs_points_poly_gdf, os.path.join(OUTPUT_DIR_VECT, 'GT'))

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
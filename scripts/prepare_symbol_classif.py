import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

from constants import OVERWRITE
from data_preparation import format_labels, format_surveying_data, get_delimitation_tiles, pct_to_rgb, tiles_to_box
from sandbox import get_point_bbox_size
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Functions ---------------------------------------

def buffer_by_max_size(gdf, pt_sizes_gdf, factor=1, cap_style=1):
    """
    Generate a buffer around each geometry of the passed Geodataframe depending on the scale with the size indicated in the second dataframe
    and multiplied by the factor (default is 1).
    """
     
    gdf['buffer_size'] = [pt_sizes_gdf.loc[pt_sizes_gdf['scale'] == scale, 'max_dx'].iloc[0] for scale in gdf['scale'].to_numpy()]
    gdf.loc[:, 'geometry'] = gdf.buffer(gdf['buffer_size']*factor, cap_style=cap_style)

    return gdf

# Processing ---------------------------------------

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

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR_VECT= cfg['output_dir']['vectors']

INITIAL_IMAGE_DIR = cfg['initial_image_dir']
TILE_DIR = cfg['tile_dir']

BORDER_POINTS_PT = cfg['border_points_pt']
BORDER_POINTS_POLY = cfg['border_points_poly']
CADASTRAL_SURVEYING = cfg['cadastral_surveying']

CONVERT_IMAGES = cfg['convert_images']
TILE_SUFFIX = cfg['tile_suffix']

os.chdir(WORKING_DIR)

if CONVERT_IMAGES:
    pct_to_rgb.pct_to_rgb(INITIAL_IMAGE_DIR, TILE_DIR, tile_suffix=TILE_SUFFIX)

pts_gdf, written_files = format_labels.format_labels(BORDER_POINTS_PT, OUTPUT_DIR_VECT)

logger.info('Get the maximum size of border points by scale...')
pt_sizes_gdf, tmp_written_files = get_point_bbox_size.get_point_bbox_size(BORDER_POINTS_POLY, OUTPUT_DIR_VECT)
written_files.extend(tmp_written_files)

tiles_gdf, nodata_gdf, _, written_files = get_delimitation_tiles.get_delimitation_tiles(TILE_DIR, output_dir=OUTPUT_DIR_VECT, subtiles=False)

logger.info('Format cadastral surveying data...')
filepath = os.path.join(OUTPUT_DIR_VECT, 'MO_pt_polys.gpkg')
if not os.path.isfile(filepath) or OVERWRITE:
    nodata_gdf = buffer_by_max_size(nodata_gdf, pt_sizes_gdf, factor=0.25)
    cs_points_gdf, tmp_written_files = format_surveying_data.format_surveying_data(CADASTRAL_SURVEYING, tiles_gdf, nodata_gdf, output_dir=OUTPUT_DIR_VECT)
    written_files.extend(tmp_written_files)

    logger.info('Transform points to polygons...')
    cs_points_gdf = buffer_by_max_size(cs_points_gdf, pt_sizes_gdf, factor=0.5, cap_style=3)
    cs_points_gdf.to_file(filepath)
    written_files.append(filepath)

else:
    logger.info(f'File {filepath} already exists. Read from disk...')
    cs_points_gdf = gpd.read_file(filepath)

logger.info('Clip image for each border point...')
SYMBOL_IM_DIR = os.path.join(TILE_DIR, 'symbol_images')
tiles_to_box.tiles_to_box(TILE_DIR, cs_points_gdf, SYMBOL_IM_DIR)

cs_points_gdf.to_file(os.path.join(OUTPUT_DIR_VECT, 'test.gpkg'))

logger.info('Test if images intersect...')
cs_points_gdf['ini_geom'] = cs_points_gdf['geometry']
joined_gdf = gpd.sjoin(cs_points_gdf[['pt_id', 'geometry']], cs_points_gdf[['pt_id', 'geometry', 'ini_geom']])
joined_gdf = joined_gdf[joined_gdf.pt_id_left > joined_gdf.pt_id_right].copy()
joined_gdf['iou'] = joined_gdf.apply(lambda x: misc.intersection_over_union(x['geometry'], x['ini_geom']), axis=1)
intersecting_gdf = joined_gdf[joined_gdf['iou'] > 0.5].copy()

intersecting_gdf.sort_values(by=['pt_id_left', 'pt_id_right'], inplace=True)
filepath = os.path.join(OUTPUT_DIR_VECT, 'overlapping_images.gpkg')
intersecting_gdf[['pt_id_left', 'pt_id_right', 'geometry', 'iou']].to_file(filepath)
written_files.append(filepath)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
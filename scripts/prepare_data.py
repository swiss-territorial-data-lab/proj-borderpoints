import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd
import rasterio as rio
from glob import glob
from shapely.geometry import box

import functions.fct_misc as misc

logger = misc.format_logger(logger)

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
OUTPUT_DIR = cfg['output_dir']

BORDER_POINTS = cfg['border_points']
TILE_DIR = cfg['tile_dir']
PLAN_SCALES = cfg['plan_scales']

os.chdir(WORKING_DIR)
written_files = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Format the labels...')
pts_gdf = gpd.read_file(BORDER_POINTS)

pts_gdf.drop(columns=['Shape_Leng', 'Shape_Area'], inplace=True)
pts_gdf['CATEGORY'] = str(pts_gdf.Code_type_) + pts_gdf.Couleur
pts_gdf['SUPERCATEGORY'] = 'border points'

filepath = os.path.join(OUTPUT_DIR, 'ground_truth_labels.gpkg')
pts_gdf.to_file(filepath)
written_files.append(filepath)

logger.info('Get the delimitation of tiles...')
tile_list = glob(os.path.join(TILE_DIR, '*.tif'))
plan_scales = pd.read_excel(PLAN_SCALES)

tiles_dict = {'id': [], 'name': [], 'scale': [], 'geometry': []}
strip_str = '_georeferenced.tif'
for tile in tile_list:
    tile_name = os.path.basename(tile).rstrip(strip_str)
    tile_scale = plan_scales.loc[plan_scales.Num_plan==tile_name, 'Echelle'].iloc[0]

    tiles_dict['name'].append(tile_name)
    tiles_dict['id'].append(f"({tile_name[:6]}, {tile_name[6:]}, {tile_scale})")
    tiles_dict['scale'].append(tile_scale)

    
    with rio.open(tile) as src:
        bounds = src.bounds
    
    geom = box(*bounds)
    tiles_dict['geometry'].append(geom)

tiles_gdf = gpd.GeoDataFrame(tiles_dict, crs='EPSG:2056')

filepath = os.path.join(OUTPUT_DIR, 'tiles.gpkg')
tiles_gdf.to_file(filepath)
written_files.append(filepath)

logger.info('Make hard link for tiles with name in format (x, y, z)...')

for tile in tiles_gdf.itertuples():
    old_path = os.path.join(TILE_DIR, tile.name + '_georeferenced.tif')
    new_path = os.path.join(TILE_DIR, 'renamed', f"{tile.scale}_{tile.name[:6]}_{tile.name[6:]}.tif")

    _ = misc.make_hard_link(old_path, new_path)

print()
logger.info("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

# Stop chronometer
toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
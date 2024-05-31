import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)

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
OUTPUT_DIR = cfg['output_dir']

TILES = cfg['tiles']
PLAN_SCALES = cfg['plan_scales']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

tiles_gdf = gpd.read_file(TILES)
plan_scales_df = pd.read_excel(PLAN_SCALES)

tiles_gdf['Num_plan'] = [name.split('_')[1] + name.split('_')[2] for name in tiles_gdf.name]
tiles_gdf['area'] = tiles_gdf.area
tiles_gdf['perimeter'] = tiles_gdf.length

tiles_w_scale = pd.merge(tiles_gdf[['Num_plan', 'area', 'perimeter']], plan_scales_df, on='Num_plan')

tiles_w_scale.to_excel(os.path.join(OUTPUT_DIR, 'plan_geometrical_values.xlsx'))
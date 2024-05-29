import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

from functions.fct_misc import format_logger

logger = format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script determines the class of missed points with TLM data.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

MATCHED_POINTS = cfg['matched_points']
TLM_FILE = cfg['TLM_data']['file']
FLOWING_WATER = cfg['TLM_data']['layer_flowing_water']
WATER_BODIES = cfg['TLM_data']['layer_water_bodies']
BUILDING_FOOTPRINT = cfg['TLM_data']['layer_building_footprint']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Read data...')

matched_points_gdf = gpd.read_file(MATCHED_POINTS)
flowing_water_gdf = gpd.read_file(TLM_FILE, layer=FLOWING_WATER)
water_bodies_gdf = gpd.read_file(TLM_FILE, layer=WATER_BODIES)
building_footprint_gdf = gpd.read_file(TLM_FILE, layer=BUILDING_FOOTPRINT)

logger.info('Format info for flowing water...')
rivers_gdf = flowing_water_gdf.loc[(flowing_water_gdf.OBJEKTART==4) & (flowing_water_gdf.STUFE==0), ['TLM_GEWAESSER_NAME_UUID', 'geometry']]
# Transform line to polygon
rivers_gdf.loc[:, 'geometry'] = rivers_gdf.buffer(1)

logger.info('Set class for undetermined points in water to "non-materilazied points"...')
water_gdf = pd.concat([rivers_gdf, water_bodies_gdf[['TLM_GEWAESSER_NAME_UUID', 'geometry']]])
tmp_affected_points = gpd.sjoin(matched_points_gdf, water_gdf)
matched_points_gdf.loc[
    (matched_points_gdf.det_category=='undetermined') & matched_points_gdf.pt_id.isin(tmp_affected_points.pt_id.unique()),
    ['det_class', 'det_category']
] = (7, '5n')

logger.info('Set class for undetermined points near buildings to "non-materilazied points"...')
building_footprint_gdf.loc[:, 'geometry'] = building_footprint_gdf.buffer(0.2)
tmp_affected_points = gpd.sjoin(matched_points_gdf, building_footprint_gdf)
matched_points_gdf.loc[
    (matched_points_gdf.det_category=='undetermined') & matched_points_gdf.pt_id.isin(tmp_affected_points.pt_id.unique()),
    ['det_class', 'det_category']
] = (7, '5n')

logger.info('Save result...')
filepath = os.path.join(OUTPUT_DIR, 'matched_points_2.gpkg')
matched_points_gdf.to_file(filepath)

logger.success(f'Done! The output was saved in {filepath}.')

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
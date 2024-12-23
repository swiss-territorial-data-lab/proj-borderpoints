import os
import sys
from loguru import logger
from time import time

import geopandas as gpd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = get_config('check_w_land_cover.py', 'The script determines the class of missed points with TLM data.')

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

MATCHED_POINTS = cfg['matched_points']
LAND_COVER = cfg['land_cover']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Read data...')

matched_points_gdf = gpd.read_file(MATCHED_POINTS)
land_cover_gdf = gpd.read_file(LAND_COVER)

logger.info('Set class for undetermined points near buildings and stagnant water to "non-materializied points"...')
object_footprint_gdf = land_cover_gdf[land_cover_gdf.CH_DESCR_F.isin(['batiment', 'eau_stagnante'])].copy()
object_footprint_gdf.loc[:, 'geometry'] = object_footprint_gdf.buffer(0.2)
tmp_affected_points = gpd.sjoin(matched_points_gdf, object_footprint_gdf)
matched_points_gdf.loc[
    (matched_points_gdf.det_category=='undetermined') & matched_points_gdf.pt_id.isin(tmp_affected_points.pt_id.unique()),
    ['det_class', 'det_category']
] = (7, '5n')

logger.info('Save result...')
filepath = os.path.join(OUTPUT_DIR, 'final_matched_points.gpkg')
matched_points_gdf.to_file(filepath)

logger.success(f'Done! The output was saved in {filepath}.')

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
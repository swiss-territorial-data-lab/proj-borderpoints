import os
import sys
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config
from point_matching import test_intersection

logger = format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = get_config(os.path.basename(__file__), 'The script classifies undetermines borderpoints with FP points based on the heatmap and built-up areas.')

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

FINAL_POINTS = cfg['final_points']
BUILT_AREAS = cfg['built_areas']
HEATMAP = cfg['heatmap']

PT_NEIGBORHOOD = {500: 2.5, 1000: 4.5, 2000: 7.0, 4000: 17.5}


# Processing  ---------------------------------------

os.chdir(WORKING_DIR)

logger.info('Read data...')

final_points_gdf = gpd.read_file(FINAL_POINTS)
built_areas_gdf = gpd.read_file(BUILT_AREAS)
heatmap_gdf = gpd.read_file(HEATMAP)

logger.info('Format points...')
border_pts_gdf = final_points_gdf[final_points_gdf.det_category=='undetermined'].copy()
detections_gdf = final_points_gdf[final_points_gdf.pt_id.isna()].copy()

if not border_pts_gdf[border_pts_gdf.duplicated('pt_id')].empty:
    logger.critical('Some border points have a duplicated id!')
    sys.exit(1)

# Transform the FP detections back to polygons based on the neighborhood
detections_gdf.loc[:, 'buffer_size'] = [PT_NEIGBORHOOD[scale] for scale in detections_gdf['scale']]
detections_gdf.loc[:, 'geometry'] = detections_gdf.buffer(detections_gdf.buffer_size)

logger.info('Format heatmap...')
# Do not consider FP in "large" built-up areas
large_built_areas_gdf = built_areas_gdf[built_areas_gdf.area > 1600].copy()
logger.info('     - Exclude built-up areas...')
heatmap_countryside_gdf = heatmap_gdf[~heatmap_gdf.intersects(large_built_areas_gdf.union_all())].reset_index(drop=True)

logger.info('     - Only keep elements with a high count or part of a cluster...')
heatmap_countryside_gdf.loc[:, 'geometry'] = heatmap_countryside_gdf.buffer(1)
self_intersect_heatmap = heatmap_countryside_gdf.sjoin(heatmap_countryside_gdf, how='inner')
self_intersect_heatmap = self_intersect_heatmap[self_intersect_heatmap.index != self_intersect_heatmap.index_right].copy()

heatmap_countryside_gdf.loc[:, 'geometry'] = heatmap_countryside_gdf.buffer(-1)
final_heatmap = heatmap_countryside_gdf[
    heatmap_countryside_gdf.index.isin(self_intersect_heatmap.index.unique())
    | (heatmap_countryside_gdf['count'] > 1)
].copy()

logger.info('Only keep detections overlapping the formatted heatmap...')
filtered_detections_gdf = gpd.sjoin(detections_gdf, final_heatmap, how='inner')
filtered_detections_gdf.drop_duplicates('det_id', inplace=True)

logger.info('Test intersection between the border points and detections...')
lonely_points_gdf, pts_w_cat_gdf = test_intersection(border_pts_gdf[['pt_id', 'geometry']], filtered_detections_gdf[['det_id', 'det_category', 'det_class', 'score', 'geometry']])

logger.info('Add results to the original points...')
pts_w_cat_gdf.sort_values('pt_id', inplace=True)
final_points_gdf.sort_values('pt_id', inplace=True)

classified_pts_list = pts_w_cat_gdf.pt_id.unique().tolist()
assigned_fp_list = pts_w_cat_gdf.det_id.unique().tolist()

final_points_gdf.loc[final_points_gdf.pt_id.isin(classified_pts_list), 'det_category'] = pts_w_cat_gdf.det_category.tolist()
final_points_gdf.loc[final_points_gdf.pt_id.isin(classified_pts_list), 'det_class'] = pts_w_cat_gdf.det_class.tolist()
final_points_gdf = final_points_gdf[~final_points_gdf.det_id.isin(assigned_fp_list)].copy()

final_points_gdf['comment'] = ['reclassified shifted point' if pt_id in classified_pts_list else None for pt_id in final_points_gdf.pt_id]

logger.info('Save results...')
filepath = os.path.join(OUTPUT_DIR, 'fully_classified_points.gpkg')
final_points_gdf.to_file(filepath)

logger.success(f'Done! One file was written: {filepath}.')
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(time()-tic):.2f} seconds")
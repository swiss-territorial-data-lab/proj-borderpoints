import os
import sys
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# Function definition ---------------------------------------

def resolve_multiple_matches(multiple_matches_gdf, detections_gdf):
    """
    Resolves multiple matches in the given GeoDataFrame of multiple matches and detections by spatial intersection.
    Ambiguous cases ares resolved based on the distance between centroids and the score.
    
    Args:
        multiple_matches_gdf (GeoDataFrame): A GeoDataFrame containing multiple matches.
        detections_gdf (GeoDataFrame): A GeoDataFrame containing detections.
    
    Returns:
        GeoDataFrame: A GeoDataFrame with the best match for each point in regards to the distance and score.
    """

    if any(multiple_matches_gdf.duplicated(['pt_id', 'det_id'])):
        logger.critical('Duplicates are present in the point-det matches.')
        sys.exit(1)

    # Bring back the geometries of the points
    multi_pts_df = pd.merge(
        multiple_matches_gdf.reset_index(), detections_gdf[['det_id', 'geometry']], 
        on='det_id', suffixes=('_pt', '_det')
    ).set_index('index')
    # Get a weighted score made of the distance and the initial score
    multi_pts_df.loc[:, 'geometry_det'] = multi_pts_df.geometry_det.centroid
    multi_pts_df['distance'] = multi_pts_df.geometry_pt.distance(multi_pts_df.geometry_det)
    multi_pts_df['weighted_score'] = multi_pts_df['distance']/multi_pts_df['score']

    # Keep best match in the gdf in regards to the distance and score
    multi_pts_df.sort_values('weighted_score', ascending=True, inplace=True)
    favorite_matches = multi_pts_df.duplicated('pt_id')
    favorite_matches_gdf = multiple_matches_gdf[favorite_matches.sort_index()]

    return favorite_matches_gdf


def test_intersection(border_pts_gdf, detections_gdf):
    """
    Test the intersection between the given border points and detections.
    
    Parameters:
    - border_pts_gdf (GeoDataFrame): The GeoDataFrame containing the border points.
    - detections_gdf (GeoDataFrame): The GeoDataFrame containing the detections.
    
    Returns:
    - lonely_points_gdf (GeoDataFrame): The GeoDataFrame containing the points without any intersection to the detections
    - pts_w_cat_gdf (GeoDataFrame): The GeoDataFrame containing the points with a category deducted from the intersecting detections.
    """
    
    intersected_pts_gdf = gpd.sjoin(detections_gdf, border_pts_gdf, how='right')
    intersected_pts_gdf.drop(columns=['index_left'], inplace=True)
    intersected_pts_gdf.sort_values('score', inplace=True)

    intersection_count_df = intersected_pts_gdf.groupby(['pt_id', 'det_category'], as_index=False).size()

    logger.info("   Isolate points properly intersected...")
    multiple_intersections = intersection_count_df.duplicated('pt_id')
    intersected_id = intersection_count_df.loc[~multiple_intersections, 'pt_id']
    pts_w_cat_gdf = intersected_pts_gdf[
        intersected_pts_gdf.pt_id.isin(intersected_id) & ~intersected_pts_gdf.det_category.isna() 
    ].copy()
    lonely_points_gdf = intersected_pts_gdf.loc[intersected_pts_gdf.det_category.isna(), border_pts_gdf.columns]

    logger.info("   Isolate points with multiple intersections")
    multiplied_id = intersection_count_df.loc[multiple_intersections, 'pt_id']
    multiple_matches_gdf = intersected_pts_gdf[intersected_pts_gdf.pt_id.isin(multiplied_id)].reset_index(drop=True)

    logger.info('Deal with points intersecting multiple detections...')
    favorite_matches_gdf = resolve_multiple_matches(multiple_matches_gdf, detections_gdf)

    pts_w_cat_gdf = pd.concat([pts_w_cat_gdf, favorite_matches_gdf], ignore_index=True)

    return lonely_points_gdf, pts_w_cat_gdf


# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = get_config(os.path.basename(__file__), 'The script matches the known border points with the segmented instances.')

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

DETECTIONS = cfg['detections']
BORDER_POINTS = cfg['border_points']

# Point neighborhood = max size of the point bbox depending on scale
PT_NEIGBORHOOD = {500: 2.4, 1000: 4.5, 2000: 6.8, 4000: 17.3}
# Based on visual assessment, buffer distance = 1/8 max size of the point bbox depending on scale
BUFFER_DISTANCE = {key: value/8 for key, value in PT_NEIGBORHOOD.items()}


# Processing  ---------------------------------------

os.chdir(WORKING_DIR)

logger.info('Read data...')

detections_gdf = gpd.read_file(DETECTIONS)
detections_gdf = detections_gdf[detections_gdf.det_category!='0s'].copy()
detections_gdf = detections_gdf[['det_id', 'det_category', 'score', 'scale', 'geometry']].copy()

border_pts_gdf = gpd.read_file(BORDER_POINTS)

if not border_pts_gdf[border_pts_gdf.duplicated('pt_id')].empty:
    logger.critical('Some border points have a duplicated id!')
    sys.exit(1)

logger.info('Test intersection between the border points and detections...')
lonely_points_gdf, pts_w_cat_gdf = test_intersection(border_pts_gdf, detections_gdf)

logger.info('Try again with a buffer for points with no intersection...')
lonely_dets_gdf = detections_gdf[
    ~detections_gdf.det_id.isin(pts_w_cat_gdf.det_id.unique().tolist())
].copy()
lonely_dets_gdf['buffer_size'] = [BUFFER_DISTANCE[scale] for scale in lonely_dets_gdf['scale']]
lonely_dets_gdf.loc[:, 'geometry'] = lonely_dets_gdf.buffer(lonely_dets_gdf.buffer_size)

if lonely_points_gdf.empty:
    tmp_pts_w_cat_gdf = gpd.GeoDataFrame(crs="EPSG:2056", columns=pts_w_cat_gdf.columns)
else:
    lonely_points_gdf, tmp_pts_w_cat_gdf = test_intersection(lonely_points_gdf, lonely_dets_gdf)

logger.info('Merge results...')
final_pts_w_cat_gdf = pd.concat([pts_w_cat_gdf, tmp_pts_w_cat_gdf], ignore_index=True)
new_border_pts_gdf = pd.merge(border_pts_gdf, final_pts_w_cat_gdf[['det_id', 'det_category', 'score', 'pt_id']], how='left', on='pt_id')

# Check if pts without category are matching across methods
lonely_ids = lonely_points_gdf.loc[:, 'pt_id']

assert (new_border_pts_gdf.loc[new_border_pts_gdf.det_category.isna(), 'pt_id'].to_list() == lonely_ids).all(), 'Ids for undetermined points not matching!'
new_border_pts_gdf.loc[new_border_pts_gdf.det_category.isna(), 'det_category'] = 'undetermined'

logger.info('Check if the remaining alone points are in the neighborhood of undetermined ones...')
lonely_dets_gdf = detections_gdf[~detections_gdf.det_id.isin(new_border_pts_gdf.det_id.unique())].copy()
lonely_dets_gdf.loc[:, 'buffer_size'] = [PT_NEIGBORHOOD[scale] for scale in lonely_dets_gdf['scale']]
lonely_dets_gdf.loc[:, 'geometry'] = lonely_dets_gdf.buffer(lonely_dets_gdf.buffer_size)

potential_miss_gdf = lonely_dets_gdf.sjoin(new_border_pts_gdf.loc[new_border_pts_gdf.det_category == 'undetermined', ['pt_id', 'geometry']])
potential_miss_gdf.drop_duplicates('det_id', inplace=True)
potential_miss_gdf.loc[:, 'geometry'] = potential_miss_gdf.geometry.centroid

all_pts_gdf = pd.concat([new_border_pts_gdf, potential_miss_gdf[detections_gdf.columns]], ignore_index=True)
all_pts_gdf.drop(columns=['scale'], inplace=True)

logger.info('Save result...')
filepath = os.path.join(OUTPUT_DIR, 'matched_points.gpkg')
all_pts_gdf.to_file(filepath)

logger.success(f'Done! The output was saved in {filepath}.')

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
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

# Function definition ---------------------------------------

def test_intersection(border_pts_gdf, detections_gdf):

    intersected_pts_gdf = gpd.sjoin(detections_gdf[['det_id', 'det_category', 'score', 'geometry']], border_pts_gdf, how='right')
    intersected_pts_gdf.drop(columns=['index_left'], inplace=True)

    intersection_count_df = intersected_pts_gdf.groupby(['pt_id', 'det_category'], as_index=False).size()

    logger.info("   Isolate points properly intersected...")
    intersected_id = intersection_count_df.loc[intersection_count_df['size']==1, 'pt_id']
    pts_w_cat_gdf = intersected_pts_gdf[
        intersected_pts_gdf.pt_id.isin(intersected_id) & ~intersected_pts_gdf.det_category.isna() 
    ].copy()

    logger.info("   Isolate points with multiple intersections")
    multiplied_id = intersection_count_df.loc[intersection_count_df['size']>1, 'pt_id']
    multiple_matches_gdf = intersected_pts_gdf[intersected_pts_gdf.pt_id.isin(multiplied_id)].reset_index(drop=True)

    return intersected_pts_gdf, pts_w_cat_gdf, multiple_matches_gdf


# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script performs the post-processing on the detections of border points.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

DETECTIONS = cfg['detections']
BORDER_POINTS = cfg['border_points']

# Processing  ---------------------------------------

os.chdir(WORKING_DIR)

logger.info('Read data...')

detections_gdf = gpd.read_file(DETECTIONS)
border_pts_gdf = gpd.read_file(BORDER_POINTS)

if not border_pts_gdf[border_pts_gdf.duplicated('pt_id')].empty:
    logger.critical('Some border points have a duplicated id!')
    sys.exit(1)

logger.info('Test intersection between the border points and detections')
intersected_pts_gdf, pts_w_cat_gdf, multiple_matches_gdf = test_intersection(border_pts_gdf, detections_gdf)

logger.info('Try again with a buffer for points with no intersection if they exists...')
lonely_points_gdf = intersected_pts_gdf[intersected_pts_gdf.det_category.isna()].copy()
lonely_points_gdf.loc[:, 'geometry'] = lonely_points_gdf.buffer(0.5)

if lonely_points_gdf.empty:
    tmp_pts_w_cat_gdf = gpd.GeoDataFrame(crs="EPSG:2056", columns=pts_w_cat_gdf.columns)
else:
    intersected_pts_gdf, tmp_pts_w_cat_gdf, tmp_multiple_matches_gdf = test_intersection(lonely_points_gdf[border_pts_gdf.columns], detections_gdf)
    multiple_matches_gdf = pd.concat([multiple_matches_gdf, tmp_multiple_matches_gdf], ignore_index=True)

logger.info('Deal with points intersecting multiple detections..')
# Bring back the geometries of the points
multi_pts_df = pd.merge(
    multiple_matches_gdf, detections_gdf[['det_id', 'geometry']], 
    on='det_id', suffixes=('_pt', '_det')
)
# Get a weighted score made of the distance and the initial score
multi_pts_df.loc[:, 'geometry_det'] = multi_pts_df.geometry_det.centroid
multi_pts_df['distance'] = multi_pts_df.geometry_pt.distance(multi_pts_df.geometry_det)
multi_pts_df['weighted_score'] = multi_pts_df['distance']/multi_pts_df['score']

# Keep best match in the gdf in regards to the distance and score
multi_pts_df.sort_values('weighted_score', ascending=False, inplace=True)
favorit_matches = multi_pts_df.duplicated('pt_id')
multiple_matches_gdf = multiple_matches_gdf[~favorit_matches.sort_index()]

logger.info('Attribute a final category to points...')
final_pts_w_cat_gdf = pd.concat([pts_w_cat_gdf, tmp_pts_w_cat_gdf, multiple_matches_gdf], ignore_index=True)
new_border_pts_gdf = pd.merge(border_pts_gdf, final_pts_w_cat_gdf.drop(columns=['geometry']), how='left', on='pt_id')

# Check if pts without category are matching across methods
lonely_ids = intersected_pts_gdf.loc[intersected_pts_gdf.det_category.isna(), 'pt_id']

assert (new_border_pts_gdf.loc[new_border_pts_gdf.det_category.isna(), 'pt_id'].to_list() == lonely_ids).all(), 'Ids for undetermined points not matching!'
new_border_pts_gdf.loc[new_border_pts_gdf.det_category.isna(), 'det_category'] = 'undetermined'

logger.info('Add centroid of excess detections to the final result...')
lonely_dets_gdf = detections_gdf[~detections_gdf.det_id.isin(new_border_pts_gdf.det_id.unique())].copy()
lonely_dets_gdf.loc[:, 'geometry'] = lonely_dets_gdf.geometry.centroid

all_pts_gdf = pd.concat([new_border_pts_gdf, lonely_dets_gdf], ignore_index=True)

logger.info('Save result...')
filepath = os.path.join(OUTPUT_DIR, 'matched_points.gpkg')
all_pts_gdf.to_file(filepath)

logger.success(f'Done! The output was saved in {filepath}.')

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
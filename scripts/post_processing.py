import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

import json

from functions.fct_metrics import intersection_over_union
from functions.fct_misc import format_logger

logger = format_logger(logger)

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
INPUT_DIR = cfg['input_dir']
OUTPUT_DIR = cfg['output_dir']

SUBTILES = cfg['subtiles']
DETECTIONS = cfg['detections']
CATEGORY_IDS_JSON = cfg['category_ids_json']
SCORE = cfg['score']
KEEP_DATASETS = cfg['keep_datasets']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Processing  ---------------------------------------

logger.info('Read files...')
subtiles_gdf = gpd.read_file(SUBTILES)
subtiles_gdf['geometry'] = subtiles_gdf.buffer(-0.03)

detections_gdf = gpd.GeoDataFrame()
for dataset_acronym in DETECTIONS.keys():
    dataset = gpd.read_file(os.path.join(INPUT_DIR, DETECTIONS[dataset_acronym]))
    dataset['dataset'] = dataset_acronym
    detections_gdf = pd.concat([detections_gdf, dataset], ignore_index=True)

detections_gdf['det_id'] = detections_gdf.index

filepath = open(os.path.join(INPUT_DIR, CATEGORY_IDS_JSON))
categories_json = json.load(filepath)
filepath.close()


logger.info('Filter dataframe by score value...')
detections_gdf = detections_gdf[detections_gdf['score'] > SCORE].copy()

dets_on_tile_gdf = gpd.sjoin(detections_gdf, subtiles_gdf[['id', 'geometry']].rename(columns={'id': 'subtile_id'}), lsuffix='det', rsuffix='tile')

logger.info('Find pairs of matching detections across tiles...')
dets_on_tile_gdf['original_geom'] = dets_on_tile_gdf.geometry
dets_on_tile_gdf['geometry'] = dets_on_tile_gdf.buffer(0.1)
joined_detections_gdf = gpd.sjoin(dets_on_tile_gdf, dets_on_tile_gdf).sort_values(['det_id_right', 'det_id_left'], ignore_index=True)
# Remove duplicates of the same tuple and self-intersections
joined_detections_gdf = joined_detections_gdf[joined_detections_gdf.det_id_left > joined_detections_gdf.det_id_right]
dets_one_obj_gdf = joined_detections_gdf[
    (joined_detections_gdf.subtile_id_right != joined_detections_gdf.subtile_id_left)
    & (joined_detections_gdf.det_class_right == joined_detections_gdf.det_class_left)
].copy()

logger.info('Attribute a cluster id...')
clustered_dets = dets_one_obj_gdf.det_id_left.unique()
detections_gdf['cluster_id'] = None
cluster_id = 0
overlap_clusters_dict = {}
for det_id in clustered_dets:
    concerned_dets = [det_id] + dets_one_obj_gdf.loc[dets_one_obj_gdf.det_id_left == det_id, 'det_id_right'].tolist()
    cluster_id += 1
    for det in concerned_dets:
        current_cluster = detections_gdf.loc[detections_gdf.det_id == det, 'cluster_id'].iloc[0]
        if not current_cluster:
            detections_gdf.loc[detections_gdf.det_id == det, 'cluster_id'] = cluster_id
        else:
            # Save number of overlapping clusters
            if not current_cluster in overlap_clusters_dict.keys():
                overlap_clusters_dict[current_cluster] = []
            overlap_clusters_dict[current_cluster].append(cluster_id)

# Give one id only to overlapping clusters
for actual_cluster in dict(sorted(overlap_clusters_dict.items(), reverse=True)):
    for duplicated_cluster in set(overlap_clusters_dict[actual_cluster]):
            detections_gdf.loc[detections_gdf.cluster_id == duplicated_cluster, 'cluster_id'] = actual_cluster

logger.info('Remove det pair with IoU > 0.5 and not in cluster...')
intersect_detections_gdf  = joined_detections_gdf[joined_detections_gdf.subtile_id_right == joined_detections_gdf.subtile_id_left].copy()
intersect_detections_gdf['iou'] = [
    intersection_over_union(geom1, geom2) for geom1, geom2 in zip(intersect_detections_gdf.geometry, intersect_detections_gdf.original_geom_right)
]
duplicated_detections_gdf = intersect_detections_gdf[intersect_detections_gdf.iou > 0.75].copy()

duplicated_det_ids = []
if not duplicated_detections_gdf.empty:
    for duplicate in duplicated_detections_gdf.itertuples():
        id_highest_score = duplicate.det_id_right if duplicate.score_right < duplicate.score_left else duplicate.det_id_left
        if (duplicate.det_id_right in clustered_dets) & (duplicate.det_id_left in clustered_dets):
            det_to_remove = id_highest_score
        elif duplicate.det_id_right in clustered_dets:
            det_to_remove = duplicate.det_id_left
        elif duplicate.det_id_left in clustered_dets:
            det_to_remove = duplicate.det_id_right
        else:
            det_to_remove = id_highest_score
        duplicated_det_ids.append(det_to_remove)

logger.info(f'{len(duplicated_det_ids)} detections will be removed because they are duplicates of a same object.')
detections_gdf = detections_gdf[~detections_gdf.det_id.isin(duplicated_det_ids)].copy()

del dets_one_obj_gdf, dets_on_tile_gdf, duplicated_detections_gdf, intersect_detections_gdf, joined_detections_gdf

logger.info('Dissolve dets in clusters....')
clustered_dets_gdf = detections_gdf[~detections_gdf.cluster_id.isnull()].copy()
clustered_dets_gdf.loc[:, 'geometry'] = clustered_dets_gdf.buffer(0.1)
if KEEP_DATASETS:
    dissolved_dets_gdf = clustered_dets_gdf[['score', 'dataset', 'cluster_id', 'det_class', 'geometry']].dissolve(
        ['cluster_id', 'dataset'], 'median', as_index=False
    )
else:
    dissolved_dets_gdf = clustered_dets_gdf[['score', 'cluster_id', 'det_class', 'geometry']].dissolve('cluster_id', 'median', as_index=False)
dissolved_dets_gdf = dissolved_dets_gdf.assign(det_id=dissolved_dets_gdf.cluster_id + detections_gdf.det_id.max(), geometry=dissolved_dets_gdf.buffer(-0.1))

detections_gdf = pd.concat(
    [detections_gdf[detections_gdf.cluster_id.isnull()], dissolved_dets_gdf],
    ignore_index=True
)


logger.info('Get the category name of each detection...')
# get corresponding class ids
categories_info_df = pd.DataFrame()

for key in categories_json.keys():
    categories_tmp={sub_key: [value] for sub_key, value in categories_json[key].items()}
    categories_info_df = pd.concat([categories_info_df, pd.DataFrame(categories_tmp)], ignore_index=True)

# Attribute a category to each row depending on the class id
detections_gdf['det_category'] = [
    categories_info_df.loc[categories_info_df['id']==det_class+1, 'name'].iloc[0]
    for det_class in detections_gdf.det_class.to_numpy()
] 


logger.info('Save file...')
if KEEP_DATASETS:
    detections_gdf.to_file(os.path.join(OUTPUT_DIR, 'dst_detected_points.gpkg'))
else:
    detections_gdf.to_file(os.path.join(OUTPUT_DIR, 'detected_points.gpkg'))

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
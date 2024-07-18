import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from joblib import load

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
import hog, color_treatment

logger = misc.format_logger(logger)

# ----- Get config -----

tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script apply the chose model to the whole maps.")

# load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
TILE_DIR = cfg['tile_dir']

IMAGE_INFO_GPKG = cfg['image_info_gpkg']
SCALER = cfg['scaler']
MODEL = cfg['model']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

# ----- Main -----

logger.info('Read model...')
scaler = load(SCALER, 'r')
model = load(MODEL, 'r')
image_info_gdf = gpd.read_file(IMAGE_INFO_GPKG)

logger.info('Extract HOG features...')
hog_features_df, written_files_hog = hog.main(TILE_DIR, output_dir=OUTPUT_DIR)
hog_features_df = misc.format_hog_info(hog_features_df)

logger.info('Extract color features...')
color_features_df, written_files_color = color_treatment.main(TILE_DIR, output_dir=OUTPUT_DIR)
images_w_stats_gdf, id_images_wo_info = misc.format_color_info(image_info_gdf[['image_name', 'geometry']], color_features_df)
logger.warning('Images without color info will be classified as undetermined.')

logger.info('Merge and scale data...')
features_gdf = images_w_stats_gdf.merge(hog_features_df, how='inner', on='image_name')
features_list = [col for col in features_gdf.columns if col.split('_')[0] in ['min', 'median', 'std', 'max', 'hog']]
features_arr = features_gdf[features_list].to_numpy()

scaled_features_arr = scaler.transform(features_arr)

logger.info('Apply classification model...')
pred_list = model.predict(scaled_features_arr)

pts_gdf = image_info_gdf.copy()
pts_gdf.loc[:, 'geometry'] = image_info_gdf.geometry.centroid

pts_gdf['method'] = 'symbol classification'
pts_gdf['class'] = None
pts_gdf.loc[~pts_gdf['image_name'].isin(id_images_wo_info), 'class'] = pred_list
pts_gdf.loc[pts_gdf['image_name'].isin(id_images_wo_info), ['class', 'method']] = ('undetermined', 'default, because no color info')

if pts_gdf['class'].isna().any():
    logger.critical('Some points were not classified!')
    sys.exit()

logger.info('Deal with points present on several maps with different classes other than "undetermined"...')
determined_pts_gdf = pts_gdf[pts_gdf['class'] != 'undetermined'].copy()
category_count_df = determined_pts_gdf.groupby(['pt_id', 'class'], as_index=False).size()

multiple_classes_ids = category_count_df.loc[category_count_df.duplicated('pt_id'), 'pt_id'].tolist()
pts_gdf.loc[pts_gdf['pt_id'].isin(multiple_classes_ids), 'class'] = 'undetermined'
logger.info(f'{len(multiple_classes_ids)} points are classified as undetermined because multiple classes were detected.')

pts_gdf.drop_duplicates(subset=['pt_id'], inplace=True)

logger.info('Save results...')
filepath = os.path.join(OUTPUT_DIR, 'classified_points.gpkg')
pts_gdf.to_file(filepath)
written_files.append(filepath)

logger.info(f'Done! The following file was written: {filepath}')
logger.info(f'Elapsed time: {(time() - tic):.2f} seconds')
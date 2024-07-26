import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import rasterio as rio

from joblib import load

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
import hog, color_treatment
from constants import MODEL

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
VARIANCE_FILTER = cfg['variance_filter']
MODEL_DIR = cfg['model_dir']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

# ----- Main -----

logger.info('Read data...')
image_info_gdf = gpd.read_file(IMAGE_INFO_GPKG)

tile_list = glob(os.path.join(TILE_DIR, '*.tif'))
image_data = {}
meta_data = {}
for tile_path in tqdm(tile_list, desc='Read images'):
    with rio.open(tile_path) as src:
        tile_name = os.path.basename(tile_path)
        image_data[tile_name] = src.read().transpose(1, 2, 0)
        meta_data[tile_name] = src.meta

logger.info('Extract HOG features...')
hog_features_df, written_files_hog = hog.main(image_data, fit_filter=False, filter_path=VARIANCE_FILTER, output_dir=OUTPUT_DIR)
hog_features_df = misc.format_hog_info(hog_features_df)

logger.info('Extract color features...')
color_features_df, written_files_color = color_treatment.main((image_data, meta_data), output_dir=OUTPUT_DIR)
images_w_stats_gdf, id_images_wo_info = misc.format_color_info(image_info_gdf[['image_name', 'geometry']], color_features_df)
logger.warning('Images without color info will be classified as undetermined.')

logger.info('Merge and scale data...')
model_dir = MODEL_DIR if MODEL_DIR.endswith(MODEL) or MODEL_DIR.endswith(MODEL + '/') else os.path.join(MODEL_DIR, MODEL)
# TODO: faire un pipeline et sauver celui-ci uniquement
with open(os.path.join(model_dir, f'scaler_{MODEL}.pkl'), 'rb') as f:
    scaler = load(f)
with open(os.path.join(model_dir, f'model_{MODEL}.pkl'), 'rb') as f:
    model = load(f)

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
    sys.exit(1)

logger.info('Deal with points present on several maps with different classes other than "undetermined"...')
determined_pts_gdf = pts_gdf[pts_gdf['class'] != 'undetermined'].copy()
category_count_df = determined_pts_gdf.groupby(['pt_id', 'class'], as_index=False).size()

multiple_classes_ids = category_count_df.loc[category_count_df.duplicated('pt_id'), 'pt_id'].tolist()
pts_gdf.loc[pts_gdf['pt_id'].isin(multiple_classes_ids), ['class', 'method']] = ('undetermined', 'default, because multiple classes were detected')
logger.info(f'{len(multiple_classes_ids)} points are classified as undetermined because multiple classes were detected.')

pts_gdf.drop_duplicates(subset=['pt_id'], inplace=True)

logger.info('Save results...')
filepath = os.path.join(OUTPUT_DIR, 'classified_points.gpkg')
pts_gdf.to_file(filepath)
written_files.append(filepath)

logger.info(f'Done! The following file was written: {filepath}')
logger.info(f'Elapsed time: {(time() - tic):.2f} seconds')
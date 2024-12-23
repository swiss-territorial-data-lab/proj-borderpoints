import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import rasterio as rio

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
import classify_images_in_folder,  color_treatment, hog, train_separated_models
from constants import AUGMENTATION, MODEL

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
output_dir = OUTPUT_DIR if MODEL.lower() in OUTPUT_DIR.lower() else os.path.join(OUTPUT_DIR, MODEL)
os.makedirs(output_dir, exist_ok=True)
written_files = []

if AUGMENTATION and ('augmented_images' not in MODEL_DIR.lower()):
    logger.error('The parameter for data augmentation is set to true, but no augmentation indicated in the model path.')
    sys.exit(1)
elif not AUGMENTATION and ('augmented_images' in MODEL_DIR.lower()):
    logger.error('The parameter for data augmentation is set to false, but augmentation indicated in the model path.')
    sys.exit(1)

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
hog_features_df, written_files_hog = hog.main(
    image_data,
    fit_filter=False, filter_path=VARIANCE_FILTER, output_dir=output_dir
)
hog_features_df = misc.format_hog_info(hog_features_df)

logger.info('Extract color features...')
color_features_df, written_files_color = color_treatment.main((image_data, meta_data), output_dir=output_dir)
images_w_stats_gdf, id_images_wo_info = misc.format_color_info(image_info_gdf[['image_name', 'geometry']], color_features_df)
logger.warning('Images without color info will be classified as undetermined.')

pt_colors_gdf = classify_images_in_folder.classify_points(images_w_stats_gdf, image_info_gdf, id_images_wo_info, MODEL_DIR, model_desc='color')
pt_shapes_gdf = classify_images_in_folder.classify_points(hog_features_df, image_info_gdf, id_images_wo_info, MODEL_DIR, model_desc='shape')
pts_gdf = train_separated_models.merge_color_and_shape(pt_colors_gdf, pt_shapes_gdf, ['1b', '2b', '3b', '1n', '5n', '1r', '3r', 'undetermined'])
pts_gdf.rename(columns={'pred': 'class'}, inplace=True)

logger.info('Deal with points present on several maps...')
determined_pts_gdf = pts_gdf[pts_gdf['class'] != 'undetermined'].copy()
category_count_df = determined_pts_gdf.groupby(['pt_id', 'class'], as_index=False).size()
multiple_classes_ids = category_count_df.loc[category_count_df.duplicated('pt_id'), 'pt_id'].tolist()

if MODEL == 'SVM':
    # Set points with different detected classes that are not "undetermined" to "undetermined"
    pts_gdf.loc[pts_gdf['pt_id'].isin(multiple_classes_ids), ['class', 'method']] = ('undetermined', 'default, because multiple classes were detected')
    logger.info(f'{len(multiple_classes_ids)} points are classified as undetermined because multiple classes were detected.')

elif MODEL in ['RF', 'HGBC']:
    pts_gdf['score'] = pts_gdf[['shape_score', 'color_score']].min(axis=1)

    # Determine best class based on score
    determined_pts_gdf['score'] = determined_pts_gdf[['shape_score', 'color_score']].min(axis=1)
    determined_pts_gdf.sort_values(by='score', ascending=False, inplace=True)
    determined_pts_gdf.drop_duplicates(subset=['pt_id'], inplace=True)
    best_score_duo = determined_pts_gdf.combo_id.unique()

    len_before = len(pts_gdf)
    pts_gdf = pts_gdf[~pts_gdf.pt_id.isin(multiple_classes_ids) | pts_gdf.combo_id.isin(best_score_duo)].copy()
    logger.info(f'{len_before - len(pts_gdf)} points were classified based on score because several classes were detected.')

    # Sort by score for the next duplicate drop
    pts_gdf.sort_values(by='score', ascending=False, inplace=True)

# Keep only the points with a determined class in case of several apparitions
determined_pts_id = pts_gdf.loc[pts_gdf['class'] != 'undetermined', 'pt_id'].unique().tolist()
len_before = len(pts_gdf)
pts_gdf = pts_gdf[~pts_gdf['pt_id'].isin(determined_pts_id) | (pts_gdf['class'] != 'undetermined')].copy()
logger.info(f'{len_before - len(pts_gdf)} undetermined points were removed because they had a known class on some other maps.')

pts_gdf.drop_duplicates(subset=['pt_id'], inplace=True)

# Indicate points classified as undetermined, because the detected class is not a valid combination of shape and color
pts_gdf.loc[
    (pts_gdf['class'] == 'undetermined') & pts_gdf.symbol_shape.isin(['1', '2', '3', '5']) & pts_gdf.color.isin(['b', 'n', 'r']), 
    'method'
] = 'invalid color and shape combination'

logger.info('Save results...')
filepath = os.path.join(output_dir, 'classified_points.gpkg')
pts_gdf.to_file(filepath)
written_files.append(filepath)

logger.info(f'Done! The following file was written: {filepath}')
logger.info(f'Elapsed time: {(time() - tic):.2f} seconds')
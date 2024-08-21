import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from joblib import load

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
import hog, color_treatment
from constants import MODEL

logger = misc.format_logger(logger)


def classify_points(features_gdf, image_info_gdf, id_images_wo_info, original_model_dir, model_desc=''):
    logger.info('Merge and scale data...')
    model_dir = original_model_dir if original_model_dir.endswith(MODEL) or original_model_dir.endswith(MODEL + '/') \
        else os.path.join(original_model_dir, MODEL)
    # TODO: faire un pipeline et sauver celui-ci uniquement
    with open(os.path.join(model_dir, f'scaler_{MODEL}{'_' + model_desc if model_desc != '' else ''}.pkl'), 'rb') as f:
        scaler = load(f)
    with open(os.path.join(model_dir, f'model_{MODEL}{'_' + model_desc if model_desc != '' else ''}.pkl'), 'rb') as f:
        model = load(f)

    features_list = [col for col in features_gdf.columns if col.split('_')[0] in ['min', 'median', 'mean', 'std', 'max', 'hog']]
    features_arr = features_gdf[features_list].to_numpy()

    scaled_features_arr = scaler.transform(features_arr)

    logger.info('Apply classification model...')
    pred_list = model.predict(scaled_features_arr)
    preds_df = pd.DataFrame({'image_name': features_gdf.image_name, 'pred': pred_list})
    if MODEL == 'RF':
        preds_df['score'] = model.predict_proba(scaled_features_arr).max(axis=1).round(3)

    pts_gdf = image_info_gdf.copy()
    pts_gdf.loc[:, 'geometry'] = image_info_gdf.geometry.centroid

    pts_gdf['method'] = 'symbol classification'
    classified_pts_gdf = pts_gdf.merge(preds_df, how='left', on='image_name')
    if model_desc != 'shape':
        classified_pts_gdf.loc[classified_pts_gdf.image_name.isin(id_images_wo_info), ['pred', 'method']] = ('undetermined', 'default, because no color info')

    multiple_pts_images = classified_pts_gdf.loc[classified_pts_gdf.image_name.duplicated(), 'image_name'].unique().tolist()
    logger.warning(f'{len(multiple_pts_images)} points share a same image because they are too close.')
    classified_pts_gdf.loc[classified_pts_gdf.image_name.isin(multiple_pts_images), 'method'] = 'classification on one image for multiple points'

    if classified_pts_gdf['pred'].isna().any():
        logger.critical('Some points were not classified!')
        sys.exit(1)

    return classified_pts_gdf


if __name__ == '__main__':

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
    output_dir = OUTPUT_DIR if OUTPUT_DIR.endswith(MODEL) or OUTPUT_DIR.endswith(MODEL + '/') else os.path.join(OUTPUT_DIR, MODEL)
    os.makedirs(output_dir, exist_ok=True)
    written_files = []

    # ----- data processing -----

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
        image_size=80, ppc=33, cpb=2, orientations=9, variance_threshold=0.0005,
        fit_filter=False, filter_path=VARIANCE_FILTER, output_dir=output_dir
    )
    hog_features_df = misc.format_hog_info(hog_features_df)

    logger.info('Extract color features...')
    color_features_df, written_files_color = color_treatment.main((image_data, meta_data), output_dir=output_dir)
    images_w_stats_gdf, id_images_wo_info = misc.format_color_info(image_info_gdf[['image_name', 'geometry']], color_features_df)
    logger.warning('Images without color info will be classified as undetermined.')

    features_gdf = images_w_stats_gdf.merge(hog_features_df, how='inner', on='image_name')
    pts_gdf = classify_points(features_gdf, image_info_gdf, id_images_wo_info, MODEL_DIR)
    pts_gdf.rename(columns={'pred': 'class'}, inplace=True)

    logger.info('Deal with points present on several maps...')
    determined_pts_gdf = pts_gdf[pts_gdf['class'] != 'undetermined'].copy()
    category_count_df = determined_pts_gdf.groupby(['pt_id', 'class'], as_index=False).size()
    multiple_classes_ids = category_count_df.loc[category_count_df.duplicated('pt_id'), 'pt_id'].tolist()

    if MODEL == 'SVM':
        # Set points with different detected classes that are not "undetermined" to "undetermined"
        pts_gdf.loc[pts_gdf['pt_id'].isin(multiple_classes_ids), ['class', 'method']] = ('undetermined', 'default, because multiple classes were detected')
        logger.info(f'{len(multiple_classes_ids)} points are classified as undetermined because multiple classes were detected.')

    elif MODEL == 'RF':
        # Determine best class based on score
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

    logger.info('Save results...')
    filepath = os.path.join(output_dir, 'classified_points.gpkg')
    pts_gdf.to_file(filepath)
    written_files.append(filepath)

    logger.info(f'Done! The following file was written: {filepath}')
    logger.info(f'Elapsed time: {(time() - tic):.2f} seconds')
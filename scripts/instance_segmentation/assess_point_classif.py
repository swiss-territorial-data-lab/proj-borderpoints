import os
import sys
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# ----- Get config -----

tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script assess the result of the point classification.")

# load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

CLASSIFIED_POINTS = cfg['classified_points']
SPLIT_AOI_TILES = cfg['split_aoi_tiles']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

# ----- Processing -----

logger.info('Read data...')
points_gdf = gpd.read_file(CLASSIFIED_POINTS)
# Remove FP points to assess only wanted border points
classified_points_gdf = points_gdf[~points_gdf.pt_id.isna()].copy()
classified_points_gdf['CATEGORY'] = classified_points_gdf.Code_type_.astype('Int64').astype('str') + classified_points_gdf.Couleur
classified_points_gdf.loc[classified_points_gdf.CATEGORY.isna(), 'CATEGORY'] = 'undetermined'
classified_points_gdf = classified_points_gdf[classified_points_gdf.CATEGORY != '3n'].copy()
split_aoi_tiles_gdf = gpd.read_file(SPLIT_AOI_TILES)
split_aoi_tiles_2056 = split_aoi_tiles_gdf.to_crs(classified_points_gdf.crs)

logger.info('Merge ground truth and classified points...')
comparison_gdf = gpd.sjoin(classified_points_gdf, split_aoi_tiles_2056[['dataset', 'geometry']], lsuffix='pt', rsuffix='tile')
if classified_points_gdf.shape[0] != comparison_gdf.shape[0]:
    logger.warning(f'Classified points have been lost in the join with the tiles: {classified_points_gdf.shape[0] - comparison_gdf.shape[0]}.')

comparison_gdf = comparison_gdf[comparison_gdf.dataset.isin(['trn', 'val', 'tst'])].copy()
classes = comparison_gdf.CATEGORY.unique()
classes.sort()

logger.info('Save confusion matrix and classification report...')
for dst in ['trn', 'val', 'tst']:
    labels = comparison_gdf.loc[comparison_gdf.dataset == dst, 'CATEGORY']
    predictions = comparison_gdf.loc[comparison_gdf.dataset == dst, 'det_category']

    confusion_matrix_df = pd.DataFrame(confusion_matrix(labels, predictions), columns=classes, index=classes)
    filepath = os.path.join(OUTPUT_DIR, f'{dst}_pt_confusion_matrix.csv')
    confusion_matrix_df.to_csv(filepath)
    written_files.append(filepath)

    cl_report = classification_report(labels, predictions, output_dict=True)
    filepath = os.path.join(OUTPUT_DIR, f'{dst}_pt_classification_report.csv')
    pd.DataFrame(cl_report).transpose().to_csv(filepath)
    written_files.append(filepath)

    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1_score = 2 * (precision * recall) / (precision + recall)
    logger.info(f'{dst} dataset - f1 score = {f1_score:.3f}')

logger.success('Done! The following files were written:')
for file in written_files:
    logger.success(file)

logger.info(f'Elapsed time: {time() - tic:.2f} seconds')
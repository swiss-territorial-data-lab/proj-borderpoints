import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from rasterio import open

import optuna
from functools import partial
from joblib import dump, load
from math import floor

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_optimization as opti
import hog, train_model


logger = misc.format_logger(logger)

# ----- Define functions -----

def objective(trial, tiles_dict, images_gdf, stat_features_df):

    # Suggest value range to test
    min_image_size, max_image_size = (75, 124)
    min_ppc, max_ppc = (8, 25)
    image_size = trial.suggest_int('image_size', min_image_size, max_image_size)
    ppc = trial.suggest_int('ppc', min_ppc, max_ppc)
    cells_per_block = trial.suggest_int('cpb', 2, floor(max_image_size/min_ppc))
    orientations = trial.suggest_int('orientations', 4, 9)
    variance_threshold = trial.suggest_float('variance_threshold', 0.0001, 0.01, step = 0.0005)

    dict_param = {
        'image_size': image_size,
        'ppc': ppc,
        'cpb': cells_per_block,
        'orientations': orientations,
        'variance_threshold': variance_threshold
    }
    print('params:', dict_param)

    if ppc*cells_per_block > image_size:
        return 0

    hog_features_df, _ = hog.main(tiles_dict, output_dir=OUTPUT_DIR, **dict_param)
    if hog_features_df.empty:
        return 0

    f1_score, _ = train_model.main(images_gdf, hog_features_df, stat_features_df, output_dir=OUTPUT_DIR)

    return f1_score

def callback(study, trial):
   # cf. https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601
    if (trial.number%5) == 0:
        study_path=os.path.join(OUTPUT_DIR, 'study.pkl')
        dump(study, study_path)
 

# ----- Main -----

tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script optimizes the HOG parameters and tests the impact on the SVM.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
TILE_DIR = cfg['tile_dir']

IMAGE_FILE = cfg['image_gpkg']
BAND_STATS = cfg['band_stats']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

logger.info('Read unchanged data...')
images_gdf = gpd.read_file(IMAGE_FILE)
stat_features_df = pd.read_csv(BAND_STATS)
tile_list = glob(os.path.join(TILE_DIR, '*.tif'))
image_data = {}
for tile_path in tqdm(tile_list, 'Read tiles'):
    with open(tile_path) as src:
        image_data[os.path.basename(tile_path)] = src.read().transpose(1, 2, 0)

logger.info('Optimize HOG parameters...')
study_path = os.path.join(OUTPUT_DIR, 'study.pkl')

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), study_name='Optimization of the HOG parameters')
# study = load(study_path, 'r')
objective = partial(objective, tiles_dict=image_data, images_gdf=images_gdf, stat_features_df=stat_features_df)
study.optimize(objective, n_trials=100, callbacks=[callback])

dump(study, study_path)
written_files.append(study_path)

logger.info('Save the best hyperparameters')
targets = {0: 'f1 score'}
written_files.append(opti.save_best_hyperparameters(study, targets, output_dir=OUTPUT_DIR))

logger.info('Plot results...')
output_plots = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(output_plots, exist_ok=True)

written_files.extend(opti.plot_optimization_results(study, targets, output_path=output_plots))

logger.info('Produce results for the best hyperparameters')
hog_features_df, written_files_hog = hog.main(image_data, output_dir=OUTPUT_DIR, **study.best_params)
_, written_files_svm = train_model.main(images_gdf, hog_features_df, stat_features_df, save_extra=True, output_dir=OUTPUT_DIR)

written_files.extend(written_files_hog)
written_files.extend(written_files_svm)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer  
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from joblib import dump

sys.path.insert(1,'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

def main(images, features_hog, features_stats, save_extra=False, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    written_files = []

    logger.info('Read data...')
    if isinstance(images, gpd.GeoDataFrame):
        images_gdf = images.copy()
    else:
        images_gdf = gpd.read_file(images)
    if isinstance(features_stats, pd.DataFrame):
        band_stats_df = features_stats.copy()
    else:
        band_stats_df = pd.read_csv(features_stats)
    if isinstance(features_hog, pd.DataFrame):
        hog_features_df = features_hog.copy()
        hog_features_df['image_name'] = hog_features_df.index.str.rstrip('.tif')
        hog_features_df.reset_index(drop=True, inplace=True)
    else:
        hog_features_df = pd.read_csv(features_hog)
        hog_features_df['image_name'] = hog_features_df['Unnamed: 0'].str.rstrip('.tif')
        hog_features_df.drop(columns=['Unnamed: 0'], inplace=True)


    images_w_stats_gdf, _ = misc.format_color_info(images_gdf, band_stats_df)
    hog_features_df = misc.format_hog_info(hog_features_df)
    
    # Get final features
    features_gdf = images_w_stats_gdf.merge(hog_features_df, how='inner', on='image_name')
    features_list = [col for col in features_gdf.columns if col.split('_')[0] in ['min', 'median', 'std', 'max', 'hog']]

    logger.info('Prepare SVM...')
    data_trn, data_tst, label_trn, label_tst, image_trn, image_tst = train_test_split(
        features_gdf[features_list].to_numpy(), features_gdf.CATEGORY, features_gdf.image_name, test_size=0.2, random_state=42
    )

    # https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    # Scale data
    scaler = StandardScaler()
    data_trn_scaled = scaler.fit_transform(data_trn)
    data_tst_scaled = scaler.transform(data_tst)

    filepath = os.path.join(output_dir, 'scaler_SVM.pkl')
    with open(filepath, 'wb') as f:
        dump(scaler, f, protocol=5)
    written_files.append(filepath)

    # Prepare SVM model
    svc_model = svm.SVC(random_state=42, cache_size=1000)
    parameters = {
        'C': [i/10 for i in range(5, 75, 1)],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    clf = GridSearchCV(svc_model, parameters, n_jobs=10, verbose=1, scoring='f1_weighted')

    logger.info('Train model with CV...')
    clf.fit(data_trn_scaled, label_trn)

    logger.info('Test model...')
    pred_tst = clf.predict(data_tst_scaled)
    metric = f1_score(label_tst, pred_tst, average='weighted')
    logger.success(f'Weighted f1 score: {round(metric, 2)}')

    if save_extra:
        logger.info('Save model...')
        filepath = os.path.join(output_dir, 'model_SVM.pkl')
        with open(filepath, 'wb') as f:
            dump(clf, f, protocol=5)
        written_files.append(filepath)

        logger.info('Save confusion matrix and classification report...')
        confusion_matrix_df = pd.DataFrame(confusion_matrix(label_tst, pred_tst), columns=clf.classes_, index=clf.classes_)
        filepath = os.path.join(output_dir, 'confusion_matrix.csv')
        confusion_matrix_df.to_csv(filepath)
        written_files.append(filepath)

        cl_report = classification_report(label_tst, pred_tst, output_dict=True)
        filepath = os.path.join(output_dir, 'classification_report.csv')
        pd.DataFrame(cl_report).transpose().to_csv(filepath)
        written_files.append(filepath)

    return metric, written_files



if __name__ == "__main__":

    tic = time()
    logger.info("Starting...")

    cfg = misc.get_config(os.path.basename(__file__), "The script trains the SVM and tests it.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']

    IMAGES_FILE = cfg['image_gpkg']
    HOG_FEATURES = cfg['hog_features']
    BAND_STATS = cfg['band_stats']

    os.chdir(WORKING_DIR)

    _, written_files = main(IMAGES_FILE, HOG_FEATURES, BAND_STATS, save_extra=True, output_dir=OUTPUT_DIR)

    logger.success("Done! The following files were written:")
    for written_file in written_files:
        logger.success(written_file)

    logger.info(f"Elapsed time: {time() - tic:.2f} seconds")
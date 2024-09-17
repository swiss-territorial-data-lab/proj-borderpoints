import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from numpy import nan, std
from sklearn import svm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from joblib import dump
import matplotlib.pyplot as plt

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
from constants import AUGMENTATION, MODEL

pd.options.plotting.backend = "plotly"
logger = misc.format_logger(logger)


def train_model(features_gdf, features_list, label_name='CATEGORY', test_ids=None):
    logger.info('Prepare data...')

    if isinstance(test_ids, pd.Series):
        test_mask = features_gdf.image_name.isin(test_ids)
        data_trn = features_gdf.loc[~test_mask, features_list].to_numpy()
        data_tst = features_gdf.loc[test_mask, features_list].to_numpy()
        labels_trn = features_gdf.loc[~test_mask, label_name].to_numpy()
        labels_tst = features_gdf.loc[test_mask, label_name].to_numpy()
        geometries_tst = features_gdf.loc[test_mask, 'geometry'].to_numpy()
        image_names_tst = features_gdf.loc[test_mask, 'image_name'].to_numpy()

    else:
        data_trn, data_tst, labels_trn, labels_tst, _, geometries_tst, _, image_names_tst = train_test_split(
            features_gdf[features_list].to_numpy(), features_gdf[label_name], features_gdf.geometry, features_gdf.image_name, test_size=0.2, stratify=features_gdf[label_name].to_numpy(),
            random_state=42
        )

    # Scale data 
    scaler = StandardScaler()
    data_trn_scaled = scaler.fit_transform(data_trn)
    data_tst_scaled = scaler.transform(data_tst)

    # TODO: setup pipelines
    if MODEL == 'SVM':
        logger.info('Prepare SVM model...')
        # https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
        svc_model = svm.SVC(random_state=42, cache_size=1000)
        parameters = {
            'C': [i/10 for i in range(5, 75, 1)],
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
        clf = GridSearchCV(svc_model, parameters, n_jobs=10, verbose=1, scoring='recall_macro')

    elif MODEL == 'RF':
        logger.info('Prepare RF model...')
        rf_model = RandomForestClassifier(random_state=42)
        parameters = {
            'n_estimators': range(120, 200, 5),
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        clf = GridSearchCV(rf_model, parameters, n_jobs=10, verbose=1, scoring='recall_macro')

    elif MODEL == 'HGBC':
        logger.info('Prepare HGBC model...')
        hgcb_model = HistGradientBoostingClassifier(random_state=42, early_stopping=True, validation_fraction=0.2, class_weight='balanced')
        parameters = {
            'learning_rate': [0.01, 0.1, 0.5],
            'max_iter': [25, 50, 75],
            'max_leaf_nodes': [15, 31, 63, None],
            'max_features': [0.15, 0.25, 0.33],

        }
        clf = GridSearchCV(hgcb_model, parameters, n_jobs=10, verbose=1, scoring='recall_macro')

    else:
        logger.critical(f'Model {MODEL} not implemented')
        sys.exit()

    logger.info('Train model with CV...')
    clf.fit(data_trn_scaled, labels_trn)
    logger.success(f"Best score: {clf.best_score_:.3f}")
    logger.success(f"Best parameters: {clf.best_params_}")

    logger.info('Test model...')
    pred_tst = clf.predict(data_tst_scaled)
    metric = balanced_accuracy_score(labels_tst, pred_tst)

    logger.info('Save a geodataframe with the test features...')
    if MODEL in ['RF', 'HGBC']:
        proba_pred_tst = clf.predict_proba(data_tst_scaled)
        classified_pts_tst_gdf = gpd.GeoDataFrame(
            {'image_name': image_names_tst, 'label': labels_tst, 'pred': pred_tst, 'score': proba_pred_tst.max(axis=1).round(3)}, 
            geometry=geometries_tst
        )
    else:
        classified_pts_tst_gdf = gpd.GeoDataFrame({'image_name': image_names_tst, 'label': labels_tst, 'pred': pred_tst}, geometry=geometries_tst)
    
    classified_pts_tst_gdf['correct'] = [True if row.label == row.pred else False for row in classified_pts_tst_gdf.itertuples()]

    tst_data_df = pd.DataFrame(data_tst_scaled, columns=features_list, index= image_names_tst).reset_index().rename(columns={'index': 'image_name'})
    classified_pts_tst_gdf = classified_pts_tst_gdf.merge(tst_data_df, how='inner', on='image_name')

    classified_pts_tst_gdf['method'] = 'test for the model training'

    return scaler, clf, metric, classified_pts_tst_gdf


def main(images, features_hog, features_stats, save_extra=False, output_dir='outputs'):
    output_dir_model = output_dir if MODEL.lower() in output_dir.lower() else os.path.join(output_dir, MODEL)
    os.makedirs(output_dir_model, exist_ok=True)
    written_files = []

    logger.info('Read data...')
    if isinstance(images, gpd.GeoDataFrame):
        images_gdf = images.copy()
    else:
        images_gdf = gpd.read_file(images)
        if AUGMENTATION:
            output_dir_model = os.path.join(output_dir_model, 'augmented_images')
            os.makedirs(output_dir_model, exist_ok=True)
            augmented_images_gdf = images_gdf.copy()
            augmented_images_gdf['image_name'] = augmented_images_gdf['image_name'].apply(lambda x: 'aug_' + x)
            images_gdf = pd.concat([images_gdf, augmented_images_gdf], ignore_index=True)
    if isinstance(features_stats, pd.DataFrame):
        band_stats_df = features_stats.copy()
    else:
        band_stats_df = pd.read_csv(features_stats)
    if isinstance(features_hog, pd.DataFrame):
        hog_features_df = features_hog.copy()
    else:
        hog_features_df = pd.read_csv(features_hog)


    images_w_stats_gdf, _ = misc.format_color_info(images_gdf, band_stats_df)
    hog_features_df = misc.format_hog_info(hog_features_df)
    
    # Get final features
    features_gdf = images_w_stats_gdf.merge(hog_features_df, how='inner', on='image_name')
    features_list = [col for col in features_gdf.columns if col.split('_')[0] in ['min', 'median', 'mean', 'std', 'max', 'hog']]

    scaler, clf, metric, classified_pts_tst_gdf = train_model(features_gdf, features_list)

    global_metric = balanced_accuracy_score(classified_pts_tst_gdf.label, classified_pts_tst_gdf.pred)
    logger.info(f'Balanced accuracy: {round(global_metric, 2)}')

    if save_extra:
        labels_list = classified_pts_tst_gdf.label.sort_values().unique().tolist()

        logger.info('Save scaler...')
        filepath = os.path.join(output_dir_model, f'scaler_{MODEL}.pkl')
        with open(filepath, 'wb') as f:
            dump(scaler, f, protocol=5)
        written_files.append(filepath)

        logger.info('Save model...')
        filepath = os.path.join(output_dir_model, f'model_{MODEL}.pkl')
        with open(filepath, 'wb') as f:
            dump(clf, f, protocol=5)
        written_files.append(filepath)

        logger.info('Save confusion matrix and classification report...')
        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(classified_pts_tst_gdf.label, classified_pts_tst_gdf.pred), columns=labels_list, index=labels_list
        )
        filepath = os.path.join(output_dir_model, 'confusion_matrix.csv')
        confusion_matrix_df.to_csv(filepath)
        written_files.append(filepath)

        cl_report = classification_report(classified_pts_tst_gdf.label, classified_pts_tst_gdf.pred, output_dict=True)
        filepath = os.path.join(output_dir_model, 'classification_report.csv')
        pd.DataFrame(cl_report).transpose().to_csv(filepath)
        written_files.append(filepath)

        filepath = os.path.join(output_dir_model, 'classified_pts_tst.gpkg')
        classified_pts_tst_gdf.to_file(filepath)
        written_files.append(filepath)

        if MODEL in ['RF', 'HGBC']:
            thresholds_bins = [i/100 for i in range(0, 100, 5)]
            weights_dict = {
                gt_class: classified_pts_tst_gdf[classified_pts_tst_gdf.label == gt_class].shape[0]/classified_pts_tst_gdf.shape[0] 
                for gt_class in labels_list
            }
            classified_pts_tst_gdf['weight'] = classified_pts_tst_gdf.label.map(weights_dict)

            logger.info('Get the accuracy at each score...')
            metric_dict = {'balanced accuracy': [], 'weighted accuracy': [], 'raw accuracy': [], 'dropped_fraction': []}
            for threshold in thresholds_bins:
                preds_above_score_gdf = classified_pts_tst_gdf[classified_pts_tst_gdf.score >= threshold].copy()
                if preds_above_score_gdf.empty:
                    for key in ['balanced accuracy', 'weighted accuracy', 'raw accuracy']:
                        metric_dict[key].append(nan)
                    continue
                
                metric_dict['dropped_fraction'].append(1 - preds_above_score_gdf.shape[0]/classified_pts_tst_gdf.shape[0])

                # cf. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
                # It is defined as the average of recall obtained on each class.
                metric_dict['balanced accuracy'].append(balanced_accuracy_score(preds_above_score_gdf.label, preds_above_score_gdf.pred))
                # cf. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
                # It computes subset accuracy: the set of labels predicted for a sample (y_pred) must exactly match the corresponding set of labels in y_true.
                metric_dict['weighted accuracy'].append(accuracy_score(preds_above_score_gdf.label, preds_above_score_gdf.pred, sample_weight=preds_above_score_gdf.weight))
                metric_dict['raw accuracy'].append(accuracy_score(preds_above_score_gdf.label, preds_above_score_gdf.pred))


            # Make figure
            fig=go.Figure()
            for metric in metric_dict.keys():
                fig.add_trace(
                    go.Scatter(
                        x=thresholds_bins,
                        y=metric_dict[metric],
                        mode='lines',
                        name=metric
                    )
                )

            fig.update_layout(
                xaxis_title="confidance threshold", yaxis_title="metric",
                title="metrics for each lower threshold on the confidence score"
            )

            file_to_write = os.path.join(output_dir_model, f'metrics_per_threshold.html')
            fig.write_html(file_to_write)
            written_files.append(file_to_write)

            logger.info('Calculate the bin accuracy to estimate the calibration...')
            accuracy_tables=[]

            for gt_class in labels_list + ['global']:
                if gt_class == 'global':
                    determined_types_gdf = classified_pts_tst_gdf.copy()
                else:
                    determined_types_gdf = classified_pts_tst_gdf[classified_pts_tst_gdf.label==gt_class].copy()


                bin_values=[]
                threshold_values=[]
                for threshold in thresholds_bins:
                    preds_in_bin = determined_types_gdf[
                                                (determined_types_gdf.score > threshold-0.05)
                                                & (determined_types_gdf.score <= threshold) 
                                                ].copy()

                    if not preds_in_bin.empty:
                        if gt_class == 'global':
                            bin_values.append(preds_in_bin[preds_in_bin.pred == preds_in_bin.label].shape[0]/preds_in_bin.shape[0])
                        else:
                            bin_values.append(preds_in_bin[preds_in_bin.pred == gt_class].shape[0]/preds_in_bin.shape[0])
                        threshold_values.append(threshold)

                df = pd.DataFrame({'threshold': threshold_values, 'accuracy': bin_values})
                df['name'] = gt_class
                accuracy_tables.append(df)

            # Make the calibration curve
            fig=go.Figure()

            for trace in accuracy_tables:
                fig.add_trace(
                    go.Scatter(
                        x=trace.threshold,
                        y=trace.accuracy,
                        mode='markers+lines',
                        name=trace.loc[0, 'name'],
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=thresholds_bins,
                    y=thresholds_bins,
                    mode='lines',
                    name='reference',
                )
            )

            fig.update_layout(xaxis_title="confidance threshold", yaxis_title="bin accuracy", title="Reliability diagram")

            file_to_write = os.path.join(output_dir_model, f'reliability_diagram.html')
            fig.write_html(file_to_write)
            written_files.append(file_to_write)

            logger.info('Get the feature importance...')

            if MODEL == 'RF':
                # Method of the mean decrease in impurity
                importances = clf.best_estimator_.feature_importances_
                std_feat_importance = std([tree.feature_importances_ for tree in clf.best_estimator_.estimators_], axis=0)

                forest_importances = pd.Series(importances, index=features_list)

                fig = forest_importances.plot.bar(error_y=std_feat_importance)
                fig.update_layout(xaxis_title="Feature", yaxis_title="Mean decrease in impurity", title="Feature importances using MDI")

                file_to_write = os.path.join(output_dir_model, f'feature_importance_MDI.html')
                fig.write_html(file_to_write)
                written_files.append(file_to_write)

            # Based on feature permutation
            data_tst_scaled = classified_pts_tst_gdf[features_list].to_numpy()

            result = permutation_importance(
                clf.best_estimator_, data_tst_scaled, classified_pts_tst_gdf.label.to_numpy(), n_repeats=10, random_state=42, n_jobs=2
            )
            forest_importances = pd.Series(result.importances_mean, index=features_list)

            fig = forest_importances.plot.bar(error_y=result.importances_std)
            fig.update_layout(xaxis_title="Feature", yaxis_title="Mean accuracy decrease", title="Feature importances using permutation on scaled variables")

            file_to_write = os.path.join(output_dir_model, f'feature_importance_permutations.html')
            fig.write_html(file_to_write)
            written_files.append(file_to_write)


    return global_metric, written_files



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
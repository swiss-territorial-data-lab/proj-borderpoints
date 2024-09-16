import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from numpy import std, nan
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

import plotly.graph_objects as go
from joblib import dump
import matplotlib.pyplot as plt

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
from constants import AUGMENTATION, MODEL
from train_model import train_model

logger = misc.format_logger(logger)


def merge_color_and_shape(colors_gdf, shapes_gdf, possible_classes, category_df=None):
    logger.info('Merge color and shape...')

    shapes_gdf.rename(columns={'pred': 'symbol_shape', 'score': 'shape_score', 'method':  'shape_method'}, inplace=True)
    colors_gdf.rename(columns={'pred': 'color', 'score': 'color_score', 'method':  'color_method'}, inplace=True)

    if 'combo_id' in colors_gdf.columns:
        classified_pts_gdf = shapes_gdf[['pt_id', 'combo_id', 'image_name', 'symbol_shape', 'shape_score', 'shape_method', 'geometry']].merge(
            colors_gdf[['combo_id', 'color', 'color_score', 'color_method']], how='left', on='combo_id'
        )
    else:
        classified_pts_gdf = shapes_gdf[['image_name', 'symbol_shape', 'shape_score', 'shape_method', 'geometry']].merge(
            colors_gdf[['image_name', 'color', 'color_score', 'color_method']], how='left', on='image_name'
        )
    if isinstance(category_df, pd.DataFrame):
        classified_pts_gdf = classified_pts_gdf.merge(category_df, how='inner', on='image_name')
        classified_pts_gdf.rename(columns={'CATEGORY': 'label'}, inplace=True)

    detected_categories_list = []
    methods_list = []
    for pt in classified_pts_gdf.itertuples():
        # Determine final category based on the combination of the shape and color
        if (pt.symbol_shape == 'undetermined') | (pt.color == 'undetermined'):
            detected_categories_list.append('undetermined')
        elif pt.symbol_shape == '5':
            detected_categories_list.append('5n')
        elif pt.color is nan:
            detected_categories_list.append('undetermined')
        else:
            pt_pred = pt.symbol_shape + pt.color
            if pt_pred == '2r':
                detected_categories_list.append('5n')
            elif pt_pred == '2n':
                detected_categories_list.append('1n')
            elif pt_pred in possible_classes:
                detected_categories_list.append(pt_pred)
            # elif pt_pred == '3n':
            #     detected_categories_list.append('3n')
            else:
                detected_categories_list.append('undetermined')

        # Determine the used method for the results on shape and color
        if pt.shape_method == pt.color_method:
            methods_list.append(pt.shape_method)
        else:
            methods_list.append('mixed')
        
    classified_pts_gdf['pred'] = detected_categories_list
    classified_pts_gdf['method'] = methods_list

    return classified_pts_gdf


def split_label_info(labels, info_type):
    new_labels_list = []
    for label in labels:
        if label == 'undetermined':
            new_labels_list.append('undetermined')
        elif info_type =='color':
            if label == '5n':
                new_labels_list.append('various')
            else:
                new_labels_list.append(label[1])
        elif info_type == 'shape':
            new_labels_list.append(label[0])
        else:
            logger.critical('Wrong info type')
            sys.exit(1)

    return new_labels_list


def main(images, features_hog, features_stats, save_extra=False, do_plot=False, output_dir='outputs'):
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
    categories_list = features_gdf.CATEGORY
    features_gdf['color'] = split_label_info(categories_list, 'color')
    features_gdf['symbol_shape'] = split_label_info(categories_list, 'shape')
    features_list = [col for col in features_gdf.columns if col.split('_')[0] in ['min', 'median', 'min', 'std', 'max', 'hog']]

    
    scaler_shapes, clf_shapes, metric_shapes, classified_shapes_tst_gdf = train_model(features_gdf, features_list[7:], label_name='symbol_shape')
    test_image_names = classified_shapes_tst_gdf.image_name
    scaler_colors, clf_colors, metric_colors, classified_colors_tst_gdf = train_model(
        features_gdf[features_gdf.CATEGORY != '5n'], features_list[:7], label_name='color', test_ids=test_image_names
    )
    
    classified_pts_tst_gdf = merge_color_and_shape(
        classified_colors_tst_gdf, classified_shapes_tst_gdf, categories_list.unique().tolist(), features_gdf[['image_name', 'CATEGORY']]
    )

    classified_pts_tst_gdf['correct'] = [True if row.label == row.pred else False for row in classified_pts_tst_gdf.itertuples()]
    global_metric = balanced_accuracy_score(classified_pts_tst_gdf.label, classified_pts_tst_gdf.pred)
    logger.info('Balanced accuracy:')
    logger.info(f'- for colors: {round(metric_colors, 2)}')
    logger.info(f'- for shapes: {round(metric_shapes, 2)}')
    logger.info(f'- for categories: {round(global_metric, 2)}')

    if save_extra:
        labels_list = classified_pts_tst_gdf.pred.sort_values().unique().tolist()
        shape_dict = {'desc': 'shape', 'model': clf_shapes, 'scaler': scaler_shapes, 'tst_results': classified_shapes_tst_gdf}
        color_dict = {'desc': 'color', 'model': clf_colors, 'scaler': scaler_colors, 'tst_results': classified_colors_tst_gdf}

        for model in [shape_dict, color_dict]:
            logger.info(f'Save scaler for {model["desc"]}...')
            filepath = os.path.join(output_dir_model, f'scaler_{MODEL}_{model["desc"]}.pkl')
            with open(filepath, 'wb') as f:
                dump(model['scaler'], f, protocol=5)
            written_files.append(filepath)

            logger.info(f'Save model for {model["desc"]}...')
            filepath = os.path.join(output_dir_model, f'model_{MODEL}_{model["desc"]}.pkl')
            with open(filepath, 'wb') as f:
                dump(model['model'], f, protocol=5)
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
            classified_pts_tst_gdf['score'] = classified_pts_tst_gdf[['shape_score', 'color_score']].min(axis=1)

            logger.info('Get the accuracy at each score...')
            metric_dict = {'balanced accuracy': [], 'weighted accuracy': [], 'raw accuracy': [], 'dropped_fraction': []}
            for threshold in thresholds_bins:
                preds_above_score_gdf = classified_pts_tst_gdf[classified_pts_tst_gdf.score >= threshold].copy()
                if preds_above_score_gdf.empty:
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
                    else:
                        continue

                df = pd.DataFrame({'threshold': threshold_values, 'accuracy': bin_values})
                df['name'] = gt_class
                accuracy_tables.append(df)

            # Make the calibration curve
            fig=go.Figure()

            for trace in accuracy_tables:
                if trace.empty:
                    continue
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

            for model in [shape_dict, color_dict]:
                logger.info(f'Get the feature importance in the determination of {model["desc"]}...')
                features_list = [col for col in model['tst_results'].columns if col.split('_')[0] in ['min', 'median', 'mean', 'std', 'max', 'hog']]

                if MODEL == 'RF':
                    # Method of the mean decrease in impurity
                    importances = model['model'].best_estimator_.feature_importances_
                    std_feat_importance = std([tree.feature_importances_ for tree in model['model'].best_estimator_.estimators_], axis=0)

                    forest_importances = pd.Series(importances, index=features_list)

                    fig, ax = plt.subplots(figsize=(9, 5))
                    forest_importances.plot.bar(yerr=std_feat_importance, ax=ax)
                    ax.set_title("Feature importances using MDI")
                    ax.set_ylabel("Mean decrease in impurity")
                    fig.tight_layout()

                    file_to_write = os.path.join(output_dir_model, f'feature_importance_{model["desc"]}s_MDI.jpeg')
                    fig.savefig(file_to_write)
                    written_files.append(file_to_write)

                # Based on feature permutation
                data_tst_scaled = model['tst_results'][features_list].to_numpy()

                result = permutation_importance(
                    model['model'].best_estimator_, data_tst_scaled, model['tst_results'].label.to_numpy(), n_repeats=10, random_state=42, n_jobs=2
                )
                forest_importances = pd.Series(result.importances_mean, index=features_list).sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(9, 5))
                forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on scaled variables")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()

                file_to_write = os.path.join(output_dir_model, f'feature_importance_{model["desc"]}_permutations.jpeg')
                fig.savefig(file_to_write)
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

    DO_PLOT = False

    os.chdir(WORKING_DIR)

    _, written_files = main(IMAGES_FILE, HOG_FEATURES, BAND_STATS, save_extra=True, do_plot=DO_PLOT, output_dir=OUTPUT_DIR)

    logger.success("Done! The following files were written:")
    for written_file in written_files:
        logger.success(written_file)

    logger.info(f"Elapsed time: {time() - tic:.2f} seconds")
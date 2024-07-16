import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import rasterio as rio
from skimage import color
from skimage.feature import hog
from skimage.transform import resize
import sklearn.feature_selection as sfse

from math import floor

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
from functions.fct_rasters import remove_black_border

logger = misc.format_logger(logger)

def im_list_to_hog(im_list, channel_axis=None):
    hog_images = {}
    hog_features = {}
    for name, image in tqdm(im_list.items()):
        ppc = floor(min(image.shape)/6)
        fd, hog_image = hog(image, orientations=4, pixels_per_cell=(ppc,ppc), cells_per_block=(4, 4), block_norm= 'L2', visualize=True, channel_axis=channel_axis)
        hog_images[name] = hog_image
        hog_features[name] = fd

    return hog_images, hog_features

def main(tile_dir, variance_threshold, output_dir='outputs'):

    os.makedirs(output_dir, exist_ok=True)

    tile_list = glob(os.path.join(tile_dir, '*.tif'))
    image_data = {}
    for tile_path in tqdm(tile_list, 'Read data'):
        with rio.open(tile_path) as src:
            image_data[os.path.basename(tile_path)] = src.read().transpose(1, 2, 0)

    logger.info('Format images...')
    data_gray = {key: color.rgb2gray(i) for key, i in image_data.items()}
    cropped_images = {k: remove_black_border(v) for k, v in data_gray.items()}
    resized_images = {}

    # Get the small size of the small side of the images
    min_size_images = {k: min(v.shape) for k, v in cropped_images.items()}
    min_array_values = np.array(list(min_size_images.values()))

    # Resize images to median value of the small side
    for name, image in cropped_images.items():
        new_size = np.median(min_array_values)
        if max(image.shape) <= new_size:
            resized_images[name] = resize(image, (new_size, new_size))
        else:
            resized_images[name] = resize(image, (new_size, new_size), anti_aliasing=True)

    logger.info('Apply HOG...')
    _, hog_gray_features = im_list_to_hog(resized_images)

    hog_features_df = pd.DataFrame(hog_gray_features)
    hog_features_df = hog_features_df.transpose()

    logger.info('Select features based on variance...')
    variance_filter = sfse.VarianceThreshold(threshold=variance_threshold)
    filtered_var_features = variance_filter.fit_transform(hog_features_df.to_numpy())
    filtered_hog_features_df = pd.DataFrame(filtered_var_features, index=hog_features_df.index)

    logger.info('Save file...')
    filepath = os.path.join(output_dir, 'hog_features.csv')
    filtered_hog_features_df.to_csv(filepath)
    written_files = [filepath]

    return filtered_hog_features_df, written_files

    

if __name__ == '__main__':
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    cfg = misc.get_config(os.path.basename(__file__), desc="The script calculate the histograms of oriented gradients for each image.")

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    TILE_DIR = cfg['tile_dir']

    os.chdir(WORKING_DIR)

    _ = written_files = main(TILE_DIR, 0.009, output_dir=OUTPUT_DIR)

    logger.success(f'All done! Time elapsed: {round(time() - tic, 1)} s')
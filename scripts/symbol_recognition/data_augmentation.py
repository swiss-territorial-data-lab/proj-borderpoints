import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import numpy as np
import rasterio as rio
from numpy.random import default_rng
from skimage.exposure import adjust_gamma, equalize_adapthist

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
from constants import OVERWRITE

rng = default_rng(42)
logger = misc.format_logger(logger)

def clear_w_gamma(name, image, meta, output_dir='outputs'):
    image = adjust_gamma(image, 0.75)
    image = np.flip(image, axis=rng.integers(2))

    with rio.open(os.path.join(output_dir, 'aug_' + name), 'w', **meta[name]) as dst:
        dst.write(image.transpose(2, 0, 1))

def darken_w_gamma(name, image, meta, output_dir='outputs'):
    image = adjust_gamma(image, 1.5)
    image = np.flip(image, axis=rng.integers(2))

    with rio.open(os.path.join(output_dir, 'aug_' + name), 'w', **meta[name]) as dst:
        dst.write(image.transpose(2, 0, 1))

def equalize_w_hist(name, image, meta, output_dir='outputs'):
    image = equalize_adapthist(image)
    image = np.flip(image, axis=rng.integers(2))

    new_image = image * 254
    new_image = new_image.astype(np.uint8)

    with rio.open(os.path.join(output_dir, 'aug_' + name), 'w', **meta[name]) as dst:
        dst.write(new_image.transpose(2, 0, 1))

def main(tile_dir, output_dir='outputs'):

    os.makedirs(output_dir, exist_ok=True)

    tile_list = glob(os.path.join(tile_dir, '*.tif'))
    image_data = {}
    meta_data = {}
    for tile_path in tqdm(tile_list, 'Read data'):
        with rio.open(tile_path) as src:
            image_data[os.path.basename(tile_path)] = src.read().transpose(1, 2, 0)
            meta_data[os.path.basename(tile_path)] = src.meta

    if len(image_data.values()) == 0:
        logger.critical('No image found')
        sys.exit(1)

    transformation_choices = [clear_w_gamma, darken_w_gamma, equalize_w_hist, make_red]

    for name, image in tqdm(image_data.items(), desc='Perform data augmentation'):
        if not OVERWRITE and os.path.exists(os.path.join(output_dir, 'aug_' + name)):
            continue
        random_transformation = rng.choice(transformation_choices)
        random_transformation(name, image, meta_data, output_dir=output_dir)

    logger.success(f'Done! The output was saved in {output_dir}.')


def make_red(name, image, meta, output_dir='outputs'):
    image[:,:, 1:3] = np.where(image[:, :, :1] > 245, image[:, :, 1:3]-20, image[:, :, 1:3])
    image = np.flip(image, axis=rng.integers(2))

    with rio.open(os.path.join(output_dir, 'aug_' + name), 'w', **meta[name]) as dst:
        dst.write(image.transpose(2, 0, 1))

def make_blue(name, image, meta, output_dir='outputs'):
    image[:,:, 0:2] = np.where((image[:, :, :1] > 225) & (image[:, :, 1:2] > 225) & (image[:, :, 2:3] < 255-20), image[:, :, 0:2]-50, image[:, :, 0:2])
    image[:, :, 2:3] = np.where((image[:, :, :1] > 225) & (image[:, :, 1:2] > 225) & (image[:, :, 2:3] < 255-20), image[:, :, 2:3]+20, image[:, :, 2:3])
    image = np.flip(image, axis=rng.integers(2))

    with rio.open(os.path.join(output_dir, 'aug_' + name), 'w', **meta[name]) as dst:
        dst.write(image.transpose(2, 0, 1))


if __name__ == '__main__':

    cfg = misc.get_config(os.path.basename(__file__), 'The script performs data augmentation of the images.')

    WORKING_DIR = cfg['working_dir']
    TILE_DIR = cfg['tile_dir']
    OUTPUT_DIR = cfg['output_dir']

    os.chdir(WORKING_DIR)

    main(tile_dir=TILE_DIR, output_dir=OUTPUT_DIR)
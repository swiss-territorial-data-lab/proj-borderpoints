import os
import sys
from argparse import ArgumentParser
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import numpy as np # linear algebra
import rasterio as rio
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

sys.path.insert(1,'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script makes boxplots with the pixel values of the image for each GT symbol.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
TILE_DIR = cfg['tile_dir']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Read data...')
tile_list = glob(os.path.join(TILE_DIR, '*.tif'))
image_data = []
for tile_path in tile_list:
    with rio.open(tile_path) as src:
        image_data.append(src.read())


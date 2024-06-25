import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.mask import mask

from matplotlib import pyplot as plt
import plotly.express as px

sys.path.insert(1,'scripts')
from constants import OVERWRITE
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script prepares the initial files for the use of the OD in the detection of border points.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
TILE_DIR = cfg['tile_dir']

IMAGE_DELIMITATION = cfg['image_delimitation']
TILES = cfg['tiles']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Read data...')

smbl_delimitation_gdf = gpd.read_file(IMAGE_DELIMITATION)
tiles_gdf = gpd.read_file(TILES)
tiles_gdf['Num_plan'] = [str(x) + str(y) for z, x, y in tiles_gdf.name.str.split('_')]

smbl_delimitation_gdf = pd.merge(smbl_delimitation_gdf, tiles_gdf[['Num_plan', 'name']], on='Num_plan')
smbl_delimitation_gdf.loc[smbl_delimitation_gdf.CATEGORY == 'undetermined', 'CATEGORY'] = 'undet'
smbl_delimitation_gdf.sort_values(by=['CATEGORY'], inplace=True, ignore_index=True)

pxl_values_dict = {
    0: {cat: [] for cat in smbl_delimitation_gdf.CATEGORY.unique()}, 
    1: {cat: [] for cat in smbl_delimitation_gdf.CATEGORY.unique()}, 
    2: {cat: [] for cat in smbl_delimitation_gdf.CATEGORY.unique()}
}
for tile in tqdm(smbl_delimitation_gdf.name.unique(), desc="Extract pixel values from tiles"):
    symbols_on_tile = smbl_delimitation_gdf[smbl_delimitation_gdf.name==tile].copy()
    
    with rio.open(os.path.join(TILE_DIR, tile + '.tif')) as src:
        for category in symbols_on_tile.CATEGORY.unique():
            tmp_pxl_values, _ = mask(src, symbols_on_tile.loc[symbols_on_tile.CATEGORY == category, 'geometry'], crop=True, filled=False)
            for band in range(3):
                pxl_values_dict[band][category].extend(tmp_pxl_values[band].data[tmp_pxl_values[band].mask].tolist())
                
for band in tqdm(range(3), desc='Produce boxplots for each band'):
    labels, data = [*zip(*pxl_values_dict[band].items())]

    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_pixels_{band}.png'))
    plt.close()
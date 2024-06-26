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
import rasterstats as rst
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

BAND_CORRESPONDENCE = {0: 'R', 1: 'G', 2: 'B'}
STAT_LIST = ['min', 'std', 'mean', 'median']

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
stats_df_dict = {band: pd.DataFrame() for band in BAND_CORRESPONDENCE.keys()}
for tile in tqdm(smbl_delimitation_gdf.name.unique(), desc="Extract pixel values from tiles"):
    symbols_on_tile = smbl_delimitation_gdf[smbl_delimitation_gdf.name==tile].copy()
    
    with rio.open(os.path.join(TILE_DIR, tile + '.tif')) as src:
        for category in symbols_on_tile.CATEGORY.unique():
            tmp_pxl_values, _ = mask(src, symbols_on_tile.loc[symbols_on_tile.CATEGORY == category, 'geometry'], crop=True, filled=False)
            for band in BAND_CORRESPONDENCE.keys():
                pxl_values_dict[band][category].extend(tmp_pxl_values[band].data[tmp_pxl_values[band].mask].tolist())

    for band in BAND_CORRESPONDENCE.keys():
        tmp_stats_list = rst.zonal_stats(
            symbols_on_tile, os.path.join(TILE_DIR, tile + '.tif'), stats=STAT_LIST, band_num=band
        )
        tmp_stats_df = pd.DataFrame.from_records(tmp_stats_list)
        tmp_stats_df['CATEGORY'] = symbols_on_tile.CATEGORY.tolist()
    
        stats_df_dict[band] = pd.concat([stats_df_dict[band], tmp_stats_df], ignore_index=True)
                
for band in tqdm(BAND_CORRESPONDENCE.keys(), desc='Produce boxplots for each band'):
    labels, data = [*zip(*pxl_values_dict[band].items())]

    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title(f'Pixel values on the {BAND_CORRESPONDENCE[band]} band')
    plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_pixels_{BAND_CORRESPONDENCE[band]}.png'), bbox_inches='tight')
    plt.close()

    for stat in STAT_LIST:
        stats_df = stats_df_dict[band].loc[: , ['CATEGORY', stat]].copy()
        stats_df.plot.box(by='CATEGORY')
        plt.title(f'{stat.title()} on the {BAND_CORRESPONDENCE[band]} band')
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_stats_{BAND_CORRESPONDENCE[band]}_{stat}.png'), bbox_inches='tight')
        plt.close()
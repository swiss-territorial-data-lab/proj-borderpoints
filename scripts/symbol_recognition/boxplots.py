import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterstats as rst
from rasterio.mask import mask

from matplotlib import pyplot as plt
import plotly.express as px

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
ratio_stats_df = pd.DataFrame()
for tile in tqdm(smbl_delimitation_gdf.name.unique(), desc="Extract pixel values from tiles"):
    symbols_on_tile = smbl_delimitation_gdf[smbl_delimitation_gdf.name==tile].copy()
    
    with rio.open(os.path.join(TILE_DIR, tile + '.tif')) as src:
        for category in symbols_on_tile.CATEGORY.unique():
            tmp_pxl_values, _ = mask(src, symbols_on_tile.loc[symbols_on_tile.CATEGORY == category, 'geometry'], crop=True, filled=False)
            for band in BAND_CORRESPONDENCE.keys():
                pxl_values_dict[band][category].extend(tmp_pxl_values[band].data[tmp_pxl_values[band].mask].tolist())

        br_ratio_arr = np.divide(src.read(3), src.read(1), out=src.read(1).astype(np.float64), where=src.read(1)!=0)
        br_ratio_stats_list = rst.zonal_stats(symbols_on_tile, br_ratio_arr, affine=src.transform, nodata=src.nodata, stats=STAT_LIST + ['max'])
        tmp_df = pd.DataFrame.from_records(br_ratio_stats_list)
        tmp_df['CATEGORY'] = symbols_on_tile.CATEGORY.tolist()
        ratio_stats_df = pd.concat([ratio_stats_df, tmp_df], ignore_index=True)

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
#         plt.close()

pxl_values_dict[4] = {}
pxl_values_dict[5] = {}
for category in tqdm(pxl_values_dict[0].keys(), desc='Calculate B/R ratio of pixels in each category'):
    pxl_values_dict[4][category] = [b_px/r_px if r_px != 0 else b_px/1 for b_px, r_px in zip(pxl_values_dict[2][category], pxl_values_dict[0][category])]
    pxl_values_dict[5][category] = [px for px in pxl_values_dict[4][category] if px < 2]
labels, data = [*zip(*pxl_values_dict[4].items())]

logger.info('Make some plots...')
plt.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.title(f'Pixel values on the BR ratio')
plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_pixels_BR_ratio.png'), bbox_inches='tight')
plt.close()

labels, data = [*zip(*pxl_values_dict[5].items())]

plt.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.title(f'Pixel values on the BR ratio for pixels with value lower than 2')
plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_pixels_BR_ratio_small_values.png'), bbox_inches='tight')
plt.close()

for stat in STAT_LIST + ['max']:
    ratio_stats_df.loc[: , ['CATEGORY', stat]].plot.box(by='CATEGORY')
    plt.title(f'{stat} on the BR ratio')
    plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_BR_ratio_{stat}.png'), bbox_inches='tight')
    plt.close()

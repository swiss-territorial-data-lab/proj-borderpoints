import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib import pyplot as plt
from rasterstats import zonal_stats
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.color import rgb2hsv

sys.path.insert(1,'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


def main(tile_dir, image_desc_gpkg=None, save_extra=False, output_dir='outputs'):

    os.makedirs(output_dir, exist_ok=True)
    written_files = []

    logger.info('Read data...')
    tile_list = glob(os.path.join(tile_dir, '*.tif'))
    if image_desc_gpkg:     # Without the images description based on the GT, don't do the parts about the category.
        images_gdf = gpd.read_file(image_desc_gpkg)
        images_gdf.loc[images_gdf.CATEGORY == 'undetermined', 'CATEGORY'] = 'undet'

    image_data = {}
    meta_data = {}
    for tile_path in tqdm(tile_list, desc='Read images'):
        with rio.open(tile_path) as src:
            tile_name = os.path.basename(tile_path)
            image_data[tile_name] = src.read().transpose(1, 2, 0)
            meta_data[tile_name] = src.meta

    logger.info('Produce HSV images...')
    data_hsv = {key: rgb2hsv(i) for key, i in image_data.items()}

    logger.info('Produce mask to filter symbols...')
    binary_list_final = {}
    for name, i in data_hsv.items():
        h, s, v = [i[:, :, band] for band in range(3)]
        condition_red = (s > 0.15) & (v > 0.8) & (h < 0.05)
        condition_black_blue = (v < 0.9) & ((h < 0.2) | (h > 0.45))

        binary_list_final[name] = np.where(condition_black_blue | condition_red, 1, 0, )

    logger.info('Extract pixel under mask')
    filtered_tile_dir = os.path.join(os.path.dirname(tile_dir), 'filtered_symbols')
    filtered_images = {}
    os.makedirs(filtered_tile_dir, exist_ok=True)
    for name, image in tqdm(image_data.items()):
        mask = np.repeat(binary_list_final[name][..., np.newaxis], repeats=3, axis=2)
        filtered_images[name] = np.where(mask, image, 0)
        with rio.open(os.path.join(filtered_tile_dir, name), 'w', **meta_data[name]) as src:
            src.write(filtered_images[name].transpose(2, 0, 1))

    # Define parameters
    BAND_CORRESPONDENCE = {0: 'R', 1: 'G', 2: 'B'}
    STAT_LIST = ['min', 'max', 'std', 'mean', 'median']
    stats_df_dict = {band: pd.DataFrame() for band in BAND_CORRESPONDENCE.keys()}

    for name, image in tqdm(image_data.items(), desc="Get statistics for each mask"):
        if image_desc_gpkg:
            category = images_gdf.loc[images_gdf.image_name == name.rstrip('.tif'), 'CATEGORY'].iloc[0]

        mask = binary_list_final[name]
        if (mask==0).all():
            continue

        # Polygonize mask
        geoms = ((shape(s), v) for s, v in shapes(mask.astype('uint8'), transform = meta_data[name]['transform']) if v == 1)
        mask_gdf = gpd.GeoDataFrame(geoms, columns=['geometry', 'class'], crs = meta_data[name]['crs'])
        mask_gdf = gpd.GeoDataFrame([name], geometry = [mask_gdf.unary_union], columns=['geometry'], crs = meta_data[name]['crs'])  

        for band in BAND_CORRESPONDENCE.keys():

            # Get stats on each image
            tmp_stats = zonal_stats(mask_gdf, os.path.join(filtered_tile_dir, name), stats=STAT_LIST, band_num=band+1)
            tmp_stats_df = pd.DataFrame.from_records(tmp_stats)
            if image_desc_gpkg:
                tmp_stats_df['CATEGORY'] = category
            tmp_stats_df['image_name'] = name.rstrip('.tif')
            if not tmp_stats_df[tmp_stats_df['median'].notna()].empty:
                stats_df_dict[band] = pd.concat([stats_df_dict[band], tmp_stats_df[tmp_stats_df['median'].notna()]], ignore_index=True)

        stats_df = pd.DataFrame()
        for band_nbr, band_letter in BAND_CORRESPONDENCE.items():
            tmp_df = stats_df_dict[band_nbr].copy()
            tmp_df['band'] = band_letter
            stats_df = pd.concat([stats_df, tmp_df], ignore_index=True)

        filepath = os.path.join(output_dir, 'stats_on_filtered_bands.csv')
        stats_df.to_csv(filepath, index=False)
        written_files.append(filepath)


    if image_desc_gpkg and save_extra:
        for band in tqdm(BAND_CORRESPONDENCE.keys(), desc='Produce boxplots for each band'):
            for stat in STAT_LIST:

                stats_df = stats_df_dict[band].loc[: , ['CATEGORY', stat]].copy()
                stats_df.plot.box(by='CATEGORY')
                plt.title(f'{stat.title()} on the {BAND_CORRESPONDENCE[band]} band')
                filepath = os.path.join(output_dir, f'boxplot_filtered_stats_{BAND_CORRESPONDENCE[band]}_{stat}.png')
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                written_files.append(filepath)


    return stats_df, written_files


if __name__ == "__main__":

    tic = time()
    logger.info("Starting...")

    cfg = misc.get_config(os.path.basename(__file__), "The script extracts the symbol colors.")

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    TILE_DIR = cfg['tile_dir']

    image_desc_gpkg = cfg['image_gpkg']

    SAVE_EXTRA = cfg['save_extra']

    os.chdir(WORKING_DIR)

    _, written_files = main(TILE_DIR, image_desc_gpkg, save_extra=SAVE_EXTRA, output_dir=OUTPUT_DIR)

    logger.success("Done! The following files were written:")
    for written_file in written_files:
        logger.success(written_file)

    logger.info(f"Elapsed time: {time() - tic:.2f} seconds")
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

from joblib import Parallel, delayed

sys.path.insert(1,'scripts')
import functions.fct_misc as misc
from constants import AUGMENTATION, OVERWRITE

logger = misc.format_logger(logger)

# ----- Define functions -----

def get_stats_under_mask(image_name, meta_data, binary_list, images_gdf, band_correspondance, stats_list, output_path):
        
    if not images_gdf.empty:
        try:
            category = images_gdf.loc[images_gdf.image_name == image_name.rstrip('.tif'), 'CATEGORY'].iloc[0]
        except IndexError:
            try:    # Check if we are looking for the augmented image
                category = images_gdf.loc[images_gdf.image_name == image_name.rstrip('.tif').lstrip('aug_'), 'CATEGORY'].iloc[0]
            except IndexError:
                logger.info(f'No image found for {image_name}.')
                return [{}, {}, {}]

    mask = binary_list[image_name]
    if (mask==0).all():
        return [{}, {}, {}]

    # Polygonize mask into one polygon
    geoms = ((shape(s), v) for s, v in shapes(mask.astype('uint8'), transform = meta_data[image_name]['transform']) if v == 1)
    mask_gdf = gpd.GeoDataFrame(geoms, columns=['geometry', 'class'], crs = meta_data[image_name]['crs'])
    mask_gdf = gpd.GeoDataFrame([image_name], geometry = [mask_gdf.union_all()], columns=['geometry'], crs = meta_data[image_name]['crs'])  

    stat_values_list = []
    for band in band_correspondance.keys():

        # Get stats on each image
        stats_dict = zonal_stats(mask_gdf, os.path.join(output_path, image_name), stats=stats_list, band_num=band+1)[0]  # zonal stats returns a list of dicts, but there is only one polygon
        stats_dict['image_name'] = image_name.rstrip('.tif')

        if not images_gdf.empty:
            stats_dict['CATEGORY'] = category

        if stats_dict['median'] is None:
            stat_values_list.append({})
        else:
            stats_dict['band'] = band
            stat_values_list.append(stats_dict)        
    
    return stat_values_list
    

def main(tiles, image_desc_gpkg=None, save_extra=False, output_dir='outputs'):

    os.makedirs(output_dir, exist_ok=True)
    written_files = []
        
    logger.info('Read data...')
    if image_desc_gpkg:     # Without the images description based on the GT, don't do the parts about the category.
        images_gdf = gpd.read_file(image_desc_gpkg)
        images_gdf.loc[images_gdf.CATEGORY == 'undetermined', 'CATEGORY'] = 'undet'
    else:
        images_gdf = pd.DataFrame()

    if isinstance(tiles, tuple):
        image_data = tiles[0]
        meta_data = tiles[1]
    else:
        tile_dir = tiles
        tile_list = glob(os.path.join(tile_dir, '*.tif'))
        image_data = {}
        meta_data = {}
        for tile_path in tqdm(tile_list, desc='Read images'):
            if os.path.basename(tile_path).startswith('aug_') and not AUGMENTATION:
                continue
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

    # Filter images to keep only the symbol pixels
    filtered_tile_dir = os.path.join(output_dir, 'filtered_symbols')
    os.makedirs(filtered_tile_dir, exist_ok=True)
    filtered_images = {}
    for name, image in tqdm(image_data.items(), desc='Save pixels under mask in a new image'):

        mask = np.repeat(binary_list_final[name][..., np.newaxis], repeats=3, axis=2)
        filtered_images[name] = np.where(mask, image, 0)
        
        filepath = os.path.join(filtered_tile_dir, name)
        if os.path.isfile(filepath) and not OVERWRITE:
            continue

        with rio.open(filepath, 'w', **meta_data[name]) as src:
            src.write(filtered_images[name].transpose(2, 0, 1))

        with rio.open(filepath, 'r') as src:
            test_image = src.read()
        comp_image = np.abs(filtered_images[name].transpose(2, 0, 1) - test_image)
        if (comp_image!=0).any():
            new_meta = meta_data.copy()
            new_meta.update({'dtype': np.int8})
            with rio.open(os.path.join(OUTPUT_DIR, 'comp_images', name), 'w', **new_meta[name]) as src:
                src.write(comp_image)

        mask = binary_list_final[name]
        if (mask==0).all():
            continue

        # Polygonize mask into one polygon
        geoms = ((shape(s), v) for s, v in shapes(mask.astype('uint8'), transform = meta_data[name]['transform']) if v == 1)
        mask_gdf = gpd.GeoDataFrame(geoms, columns=['geometry', 'class'], crs = meta_data[name]['crs'])
        mask_gdf = gpd.GeoDataFrame([name], geometry = [mask_gdf.unary_union], columns=['geometry'], crs = meta_data[name]['crs'])

    # Define parameters
    BAND_CORRESPONDENCE = {0: 'R', 1: 'G', 2: 'B'}
    STAT_LIST = ['min', 'max', 'std', 'mean', 'median']
    stats_values_list = []
    param_dict = {'meta_data': meta_data, 'binary_list': binary_list_final, 'images_gdf': images_gdf, 
                  'band_correspondance': BAND_CORRESPONDENCE, 'stats_list': STAT_LIST, 'output_path': filtered_tile_dir}
    image_data_keys = list(image_data.keys())
    del image_data
    
    stats_values_list = Parallel(n_jobs=5, backend='threading')(
        delayed(get_stats_under_mask)(name, **param_dict)
        for name in tqdm(image_data_keys, desc="Get statistics for each image")
    )

    stats_df_dict = {band: pd.DataFrame() for band in BAND_CORRESPONDENCE.keys()}
    count_no_symbol = 0
    for values_per_images in tqdm(stats_values_list, desc='Concatenate result'):
        for values_per_band in values_per_images:
            if values_per_band:
                band = values_per_band['band']
                stats_df_dict[band] = pd.concat([stats_df_dict[band], pd.DataFrame.from_records([values_per_band]).drop(columns='band')], ignore_index=True)
            else:
                count_no_symbol += 1

    del stats_values_list
    logger.info(f'{int(count_no_symbol/3)} images were only filled with background. No statistic was produced.')

    # Save all results in one dataframe
    stats_df = pd.DataFrame()
    for band_nbr, band_letter in BAND_CORRESPONDENCE.items():
        tmp_df = stats_df_dict[band_nbr].copy()
        tmp_df['band'] = band_letter
        stats_df = pd.concat([stats_df, tmp_df], ignore_index=True)

    filepath = os.path.join(output_dir, 'stats_on_filtered_bands.csv')
    stats_df.to_csv(filepath, index=False)
    written_files.append(filepath)


    if not images_gdf.empty and save_extra:
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
import os
import sys
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

sys.path.insert(1, 'scripts')
import constants as cst
import functions.fct_misc as misc
import functions.fct_rasters as rasters

logger = misc.format_logger(logger)


def tiles_to_box(tile_dir, bboxes, output_dir='outputs', tile_suffix='.tif'):

    written_files = []
    PAD_TILES = False

    logger.info('Read bounding boxes...')
    if isinstance(bboxes, str):
        bboxes_gdf = gpd.read_file(bboxes)
        bboxes_gdf['tilepath'] = [os.path.join(tile_dir, f'{bbox.Echelle}_{bbox.Num_plan[:6]}_{bbox.Num_plan[6:]}' + tile_suffix) for bbox in bboxes_gdf.itertuples()]
    elif isinstance(bboxes, gpd.GeoDataFrame):
        bboxes_gdf = bboxes.copy()
        bboxes_gdf['tilepath'] = [os.path.join(tile_dir, f'{initial_tile}.tif') for initial_tile in bboxes_gdf.initial_tile.to_numpy()]
        bboxes_gdf['Echelle'] = [initial_tile.split('_')[0] for initial_tile in bboxes_gdf.initial_tile.to_numpy()]
        if cst.CLIP_OR_PAD_SUBTILES == 'pad':
            logger.info('Results not entirely covered by the tile will be padded.')
            PAD_TILES = True
    else:
        logger.critical(f'Only the paths and the GeoDataFrames are accepted for the bbox parameter. Passed type: {type(bboxes)}.')
        sys.exit(1)

    name_correspondance_list = []
    for bbox in tqdm(bboxes_gdf.itertuples(), desc='Clip tiles to the AOI of the bbox', total=bboxes_gdf.shape[0]):

        tilepath = bbox.tilepath
        if os.path.exists(tilepath):
            with rasterio.open(tilepath) as src:
                out_image, out_transform, = mask(src, [bbox.geometry], crop=True)
                out_meta = src.meta

            height = out_image.shape[1]
            width = out_image.shape[2]
            side_diff = abs(height-width)

            if PAD_TILES and (side_diff > 1):
                pad_size = side_diff
                pad_side = ((0, 0), (pad_size, 0), (0, 0)) if height < width else ((0, 0), (0, 0), (0, pad_size))
                out_image = np.pad(out_image, pad_width=pad_side, constant_values=out_meta['nodata'])
                height = out_image.shape[1]
                width = out_image.shape[2]

            out_meta.update({"driver": "GTiff",
                 "height": height,
                 "width": width,
                 "transform": out_transform})
            
            (min_x, min_y) = rasters.get_bbox_origin(bbox.geometry)
            new_name = f"{bbox.Echelle}_{round(min_x)}_{round(min_y)}.tif"
            output_path = os.path.join(output_dir, new_name)

            if not cst.OVERWRITE and os.path.exists(output_path):
                continue

            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(out_image)

            name_correspondance_list.append((os.path.basename(tilepath), new_name))

        else:
            print()
            try:
                logger.warning(f"No tile correponding to plan {bbox.id}")
            except AttributeError:
                logger.warning(f"No tile correponding to plan {bbox.Num_plan}")

    second_name_corresp_df = pd.DataFrame.from_records(name_correspondance_list, columns=['rgb_name', 'bbox_name'])
    supposed_path = os.path.join(os.path.dirname(tilepath), 'name_correspondance.csv')
    if os.path.exists(supposed_path):
        name_correspondance_df = pd.read_csv(supposed_path)
        name_correspondance_df = name_correspondance_df.merge(second_name_corresp_df, left_on='new_name', right_on='rgb_name')
        name_correspondance_df.to_csv(supposed_path)
    else:
        second_name_corresp_df.to_csv(supposed_path)

    logger.success(f"The files were written in the folder {output_dir}. Let's check them out!")

# ------------------------------------------

if __name__ == "__main__":

    # Argument and parameter specification
    parser = ArgumentParser(description="The script formats the labels for the use of the OD in the detection of border points.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)['prepare_data.py']

    with open(args.config_file) as fp:
        cfg_globals = load(fp, Loader=FullLoader)['globals']

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR_TILES = cfg['output_dir_tiles']

    TILE_DIR = cfg['tile_dir']
    BBOX_PATH = cfg['bbox_path']

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR_TILES, exist_ok=True)

    tiles_to_box(TILE_DIR, BBOX_PATH, OUTPUT_DIR_TILES)


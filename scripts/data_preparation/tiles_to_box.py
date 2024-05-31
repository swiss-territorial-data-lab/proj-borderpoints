import os
import sys
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import rasterio
from glob import glob
from rasterio.mask import mask

sys.path.insert(1, 'scripts')
import constants as cst
import functions.fct_rasters as rasters
from functions.fct_misc import format_logger, save_name_correspondence

logger = format_logger(logger)


def tiles_to_box(tile_dir, bboxes, output_dir='outputs', tile_suffix='.tif'):
    """Clip the tiles in a directory to the box geometries from a GeoDataFrame

    Args:
        tile_dir (str): path to the directory containing the tiles
        bboxes (str or GeoDataFrame): path to the GeoDataFrame or GeoDataFrame with the definition of the boxes
        output_dir (str, optional): path to the output directory. Defaults to 'outputs'.
        tile_suffix (str, optional): suffix of the filename, which is the part coming after the tile number or id. Defaults to '.tif'.
    """

    os.makedirs(output_dir, exist_ok=True)
    pad_tiles = False

    logger.info('Read bounding boxes...')
    if isinstance(bboxes, str):
        bboxes_gdf = gpd.read_file(bboxes)
        # Find tilepath matching initial plan number
        bboxes_gdf['tilepath'] = [
            tilepath 
            for num_plan in bboxes_gdf.Num_plan
            for tilepath in glob(os.path.join(tile_dir, '*' + tile_suffix)) 
            if tilepath.endswith(f'{num_plan[:6]}_{num_plan[6:]}' + tile_suffix)
        ]
    elif isinstance(bboxes, gpd.GeoDataFrame):
        bboxes_gdf = bboxes.copy()
        bboxes_gdf['tilepath'] = [os.path.join(tile_dir, f'{initial_tile}.tif') for initial_tile in bboxes_gdf.initial_tile.to_numpy()]
        bboxes_gdf['Echelle'] = [initial_tile.split('_')[0] for initial_tile in bboxes_gdf.initial_tile.to_numpy()]
        if cst.CLIP_OR_PAD_SUBTILES == 'pad':
            logger.info('Subtiles not entirely covered by the tile will be padded.')
            pad_tiles = True
    else:
        logger.critical(f'Only the paths and the GeoDataFrames are accepted for the bbox parameter. Passed type: {type(bboxes)}.')
        sys.exit(1)

    name_correspondence_list = []
    for bbox in tqdm(bboxes_gdf.itertuples(), desc='Clip tiles to the AOI of the bbox', total=bboxes_gdf.shape[0]):

        tilepath = bbox.tilepath
        if os.path.exists(tilepath):
            with rasterio.open(tilepath) as src:
                out_image, out_transform, = mask(src, [bbox.geometry], crop=True)
                out_meta = src.meta

            height = out_image.shape[1]
            width = out_image.shape[2]
            side_diff = abs(height-width)

            if pad_tiles and (side_diff > 1):
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
            tile_nbr = int(os.path.basename(bbox.tilepath).split('_')[0])
            new_name = f"{tile_nbr}_{round(min_x)}_{round(min_y)}.tif"
            output_path = os.path.join(output_dir, new_name)

            if not cst.OVERWRITE and os.path.exists(output_path):
                continue

            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(out_image)

            name_correspondence_list.append((os.path.basename(tilepath).rstrip(tile_suffix), new_name.rstrip('.tif')))

        else:
            print()
            try:
                logger.warning(f"No tile correponding to plan {bbox.id}")
            except AttributeError:
                logger.warning(f"No tile correponding to plan {bbox.Num_plan}")

    if len(name_correspondence_list) > 0 & (not output_dir.endswith('subtiles')):
        save_name_correspondence(name_correspondence_list, tile_dir, 'rgb_name', 'bbox_name')

    if len(name_correspondence_list) > 0:
        logger.success(f"The files were written in the folder {output_dir}. Let's check them out!")
    else:
        logger.info(f"All files were already present in folder. Nothing done.")
        

# ------------------------------------------

if __name__ == "__main__":

    # Argument and parameter specification
    parser = ArgumentParser(description="The script formats the labels for the use of the OD in the detection of border points.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)['prepare_data.py']

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR_TILES = cfg['output_dir']['tiles']

    TILE_DIR = cfg['tile_dir']
    BBOX_PATH = cfg['bbox']

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR_TILES, exist_ok=True)

    tiles_to_box(TILE_DIR, BBOX_PATH, OUTPUT_DIR_TILES)


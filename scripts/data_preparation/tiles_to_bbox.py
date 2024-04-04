import os
import sys
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import rasterio
from rasterio.mask import mask

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_rasters as rasters

logger = misc.format_logger(logger)


def tiles_to_bbox(tile_dir, bbox_path, output_dir='outputs', overwrite=False, tile_suffix='.tif'):

    written_files = []

    logger.info('Read bounding boxes...')
    bbox_gdf = gpd.read_file(bbox_path)

    for bbox in tqdm(bbox_gdf.itertuples(), desc='Clip tiles to the AOI of the bbox'):
        tilepath = os.path.join(tile_dir, bbox.Num_plan + tile_suffix)

        if os.path.exists(tilepath):
            with rasterio.open(tilepath) as src:
                out_image, out_transform, = mask(src, [bbox.geometry], crop=True)
                out_meta = src.meta

            out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
            
            (min_x, min_y) = rasters.get_bbox_origin(bbox.geometry)
            output_path = os.path.join(output_dir, f"{bbox.Echelle}_{round(min_x)}_{round(min_y)}.tif")

            if not overwrite and os.path.exists(output_path):
                continue

            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(out_image)
            written_files.append(output_path)

        else:
            logger.warning(f"No tile correponding to plan {bbox.Num_plan}")

    return written_files

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

    TILE_SUFFIX = cfg_globals['original_tile_suffix']
    OVERWRITE = cfg_globals['overwrite']

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR_TILES, exist_ok=True)

    written_files = tiles_to_bbox(TILE_DIR, BBOX_PATH, OUTPUT_DIR_TILES, OVERWRITE, TILE_SUFFIX)

    print()
    logger.success("Done! The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)
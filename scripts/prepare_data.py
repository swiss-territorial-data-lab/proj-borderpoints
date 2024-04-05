import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from data_preparation import format_labels, tiles_to_bbox, get_delimitation_tiles, get_point_bbox_size
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

with open(args.config_file) as fp:
    cfg_globals = load(fp, Loader=FullLoader)['globals']

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR_VECTORS = cfg['output_dir_vectors']
OUTPUT_DIR_TILES = cfg['output_dir_tiles']

BORDER_POINTS = cfg['border_points']
BBOX = cfg['bbox']
TILE_DIR = cfg['tile_dir']
PLAN_SCALES = cfg['plan_scales']
OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None

TILE_SUFFIX = cfg_globals['original_tile_suffix']

OVERLAP_LARGE_TILES = cfg_globals['thresholds']['max_nodata_large_tiles']
OVERLAP_SMALL_TILES = cfg_globals['thresholds']['max_nodata_small_tiles']
GRID_LARGE_TILES = cfg_globals['grid_width_large']
GRID_SMALL_TILES = cfg_globals['grid_width_small']

OVERWRITE = cfg_globals['overwrite']

os.chdir(WORKING_DIR)

pts_gdf, written_files = format_labels.format_labels(BORDER_POINTS, OUTPUT_DIR_VECTORS)

size_per_scale_df, tmp_written_files = get_point_bbox_size.get_point_bbox_size(BORDER_POINTS, OUTPUT_DIR_VECTORS)
written_files.extend(tmp_written_files)

tmp_written_files = tiles_to_bbox.tiles_to_bbox(TILE_DIR, BBOX, OUTPUT_DIR_TILES, overwrite=OVERWRITE, tile_suffix=TILE_SUFFIX)
written_files.extend(tmp_written_files)

tiles_gdf, subtiles_gdf, tmp_written_files = get_delimitation_tiles.get_delimitation_tiles(OUTPUT_DIR_TILES, PLAN_SCALES,
                                                                                            GRID_LARGE_TILES, GRID_SMALL_TILES, OVERLAP_LARGE_TILES, OVERLAP_SMALL_TILES,
                                                                                            OVERLAP_INFO, TILE_SUFFIX,
                                                                                            OUTPUT_DIR_VECTORS, overwrite_tiles=OVERWRITE, subtiles=True)
written_files.extend(tmp_written_files)

subtiles_dir = os.path.join(OUTPUT_DIR_TILES, 'subtiles')
os.makedirs(subtiles_dir, exist_ok=True)
tmp_written_files = tiles_to_bbox.tiles_to_bbox(OUTPUT_DIR_TILES, subtiles_gdf, subtiles_dir, overwrite=OVERWRITE)
written_files.extend(tmp_written_files)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
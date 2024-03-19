import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from data_preparation import format_labels, get_delimitation_tiles  , rename_with_hard_link
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
OUTPUT_DIR = cfg['output_dir']

BORDER_POINTS = cfg['border_points']
TILE_DIR = cfg['tile_dir']
PLAN_SCALES = cfg['plan_scales']

OVERLAP_LARGE_TILES = cfg['thresholds']['overlap_large_tiles']
OVERLAP_SMALL_TILES = cfg['thresholds']['overlap_small_tiles']
GRID_LARGE_TILES = cfg_globals['grid_width_large']
GRID_SMALL_TILES = cfg_globals['grid_width_large']

os.chdir(WORKING_DIR)

pts_gdf, written_files = format_labels(BORDER_POINTS, OUTPUT_DIR)

tiles_gdf, subtiles_gdf, tmp_written_files = get_delimitation_tiles(TILE_DIR, PLAN_SCALES, GRID_LARGE_TILES, OUTPUT_DIR, subtiles=True)
written_files.extend(tmp_written_files)

rename_with_hard_link(tiles_gdf, TILE_DIR)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
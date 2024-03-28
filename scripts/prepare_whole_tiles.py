import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from data_preparation import get_delimitation_tiles , rename_with_hard_link
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

TILE_DIR = cfg['tile_dir']
PLAN_SCALES = cfg['plan_scales']
OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None

OVERLAP_LARGE_TILES = cfg_globals['thresholds']['max_nodata_large_tiles']
OVERLAP_SMALL_TILES = cfg_globals['thresholds']['max_nodata_small_tiles']
GRID_LARGE_TILES = cfg_globals['grid_width_large']
GRID_SMALL_TILES = cfg_globals['grid_width_large']

OVERWRITE = cfg_globals['overwrite']

os.chdir(WORKING_DIR)

tiles_gdf, subtiles_gdf, written_files = get_delimitation_tiles.get_delimitation_tiles(TILE_DIR, PLAN_SCALES,
                                                                            GRID_LARGE_TILES, GRID_SMALL_TILES, OVERLAP_LARGE_TILES, OVERLAP_SMALL_TILES,
                                                                            OVERLAP_INFO,
                                                                            OUTPUT_DIR, overwrite_tiles=OVERWRITE, subtiles=True)

rename_with_hard_link.rename_with_hard_link(tiles_gdf, TILE_DIR, OVERWRITE)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
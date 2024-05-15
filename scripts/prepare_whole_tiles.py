import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from data_preparation import get_delimitation_tiles, pct_to_rgb, tiles_to_box
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
OUTPUT_DIR_VECT= cfg['output_dir']['vectors']

INITIAL_IMAGE_DIR = cfg['initial_image_dir']
TILE_DIR = cfg['tile_dir']

CADASTRAL_SURVEYING = cfg['cadastral_surveying'] if 'cadastral_surveying' in cfg.keys() else None
OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None
PLAN_SCALES = cfg['plan_scales'] if 'plan_scales' in cfg.keys() else None

TILE_SUFFIX = cfg['tile_suffix'] if 'tile_suffix' in cfg.keys() else '.tif'
CONVERT_IMAGES = cfg['convert_images']

os.chdir(WORKING_DIR)

if CONVERT_IMAGES:
    pct_to_rgb.pct_to_rgb(INITIAL_IMAGE_DIR, TILE_DIR, PLAN_SCALES, tile_suffix=TILE_SUFFIX)

tiles_gdf, subtiles_gdf, written_files = get_delimitation_tiles.get_delimitation_tiles(TILE_DIR, cadastral_surveying=CADASTRAL_SURVEYING, 
                                                                                       overlap_info=OVERLAP_INFO, output_dir=OUTPUT_DIR_VECT, subtiles=True)

SUBTILE_DIR = os.path.join(TILE_DIR, 'subtiles')
os.makedirs(SUBTILE_DIR, exist_ok=True)
tiles_to_box.tiles_to_box(TILE_DIR, subtiles_gdf, SUBTILE_DIR)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
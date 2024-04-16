import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

from data_preparation import get_delimitation_tiles , tiles_to_box
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
OUTPUT_DIR = cfg['output_dir']

TILE_DIR = cfg['tile_dir']
OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None

os.chdir(WORKING_DIR)

tiles_gdf, subtiles_gdf, written_files = get_delimitation_tiles.get_delimitation_tiles(TILE_DIR, OVERLAP_INFO, OUTPUT_DIR, subtiles=True)

subtiles_dir = os.path.join(TILE_DIR, 'subtiles')
os.makedirs(subtiles_dir, exist_ok=True)
tmp_written_files = tiles_to_box.tiles_to_box(TILE_DIR, subtiles_gdf, subtiles_dir)
written_files.extend(tmp_written_files)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
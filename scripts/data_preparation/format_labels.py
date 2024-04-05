import os
import sys
from argparse import ArgumentParser
from loguru import logger
from yaml import load, FullLoader

import geopandas as gpd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)


def format_labels(path_points_poly, output_dir='outputs'):

    written_files =[] 

    os.makedirs(output_dir, exist_ok=True)

    logger.info('Format the labels...')
    pts_gdf = gpd.read_file(path_points_poly)

    pts_gdf.drop(columns=['Shape_Leng', 'Shape_Area'], inplace=True)
    pts_gdf['CATEGORY'] = [str(code) + color for code, color in zip(pts_gdf.Code_type_, pts_gdf.Couleur)] 
    pts_gdf['SUPERCATEGORY'] = 'border points'

    logger.info('Export the labels...')
    filepath = os.path.join(output_dir, 'ground_truth_labels.gpkg')
    pts_gdf.to_file(filepath)
    written_files.append(filepath)

    return pts_gdf, written_files


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
    OUTPUT_DIR = cfg['output_dir_vectors']

    BORDER_POINTS = cfg['border_points']

    os.chdir(WORKING_DIR)

    _, written_files = format_labels(BORDER_POINTS, OUTPUT_DIR)

    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)
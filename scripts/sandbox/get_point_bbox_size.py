import os
import sys
from argparse import ArgumentParser
from loguru import logger
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)

def get_point_bbox_size(border_pts_path, output_dir='outputs'):
    """Get the maximum bounding box size of the border points at each scale.

    Args:
        border_pts_path (str): path to the border points
        output_dir (str, optional): path to the output directory. Defaults to 'outputs'.

    Returns:
        size_per_scale_df: DataFrame with the maximum size at each scale
        filepath: list with the filepath of the saved file
    """

    logger.info('Read data...')
    pts_gdf = gpd.read_file(border_pts_path)

    logger.info('Iterate through scale to find the max point size...')
    size_per_scale_dict = {'scale': [], 'max_dx': [], 'max_dy': []}
    for scale in pts_gdf.Echelle.unique():
        pts_single_scale_gdf = pts_gdf[pts_gdf.Echelle==scale].copy()    
        max_dx = 0
        max_dy = 0

        for pt in pts_single_scale_gdf.itertuples():
            
            coords = pt.geometry.exterior.coords.xy
            min_x = min(coords[0])
            min_y = min(coords[1])
            max_x = max(coords[0])
            max_y = max(coords[1])

            dx = round(max_x - min_x, 2)
            dy = round(max_y - min_y, 2)

            if dx > max_dx:
                max_dx = dx
            if dy > max_dy:
                max_dy = dy
        
        size_per_scale_dict['scale'].append(scale)
        size_per_scale_dict['max_dx'].append(max_dx)
        size_per_scale_dict['max_dy'].append(max_dy)

    logger.info('Export results...')
    size_per_scale_df = pd.DataFrame(size_per_scale_dict)

    filepath = os.path.join(output_dir, 'max_point_size.csv')
    size_per_scale_df.to_csv(filepath, index=False)

    return size_per_scale_df, [filepath]

if __name__ == "__main__":

    # Argument and parameter specification
    parser = ArgumentParser(description="Get the max size of the label for each scale.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']['vectors']

    BORDER_POINTS = cfg['border_points']

    os.chdir(WORKING_DIR)

    _, written_files = get_point_bbox_size(BORDER_POINTS, OUTPUT_DIR)

    print()
    logger.success(f"The file {written_files[0]} was written. Let's check it out!")
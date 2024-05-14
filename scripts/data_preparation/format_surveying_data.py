import os
import sys
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)


def format_surveying_data(path_surveying, path_tiles, output_dir='outputs'):

    written_files =[] 

    os.makedirs(output_dir, exist_ok=True)

    logger.info('Read file...')
    survey_poly_gdf = gpd.read_file(path_surveying)
    tiles_gdf = gpd.read_file(path_tiles)

    logger.info('Get point coordinates...')
    pts_list = [Point(pt) for poly in survey_poly_gdf.geometry for geom in poly.geoms for pt in geom.exterior.coords[:-1]]
    coor_list = [str(pt.coords[0][0])[2:].replace('.', '')[:10] + str(pt.coords[0][1])[2:].replace('.', '')[:10] for pt in pts_list]

    logger.info('Get point ids...')
    pt_ids_list = [
        str(i) + '_' + str(pt.coords[0][0])[2:].replace('.', '')[:5] + str(pt.coords[0][1])[2:].replace('.', '')[:5]
        for i, pt in zip(range(len(pts_list)), pts_list)
    ]

    logger.info('Save infered points in a GeoDataFrame...')
    pts_gdf = gpd.GeoDataFrame({'pt_id': pt_ids_list, 'approx_coor': coor_list, 'geometry': pts_list}, crs='EPSG:2056')
    pts_gdf.drop_duplicates('approx_coor', inplace=True, ignore_index=True)
    pts_gdf.drop(columns=['approx_coor'], inplace=True)

    logger.info('Exclude points outside the dataset of tiles...')
    pts_in_tiles_gdf = gpd.overlay(pts_gdf, tiles_gdf[['id', 'geometry']], keep_geom_type=True)
    pts_in_tiles_gdf.drop_duplicates('pt_id', inplace=True)
    pts_in_tiles_gdf.drop(columns=['id'], inplace=True)

    filepath = os.path.join(output_dir, 'MO_points.gpkg')
    pts_in_tiles_gdf.to_file(filepath)
    written_files.append(filepath)

    return pts_gdf, written_files


# ------------------------------------------

if __name__ == "__main__":
    # Argument and parameter specification
    parser = ArgumentParser(description="The script formats the cadastral surveying to limit the produced tiles.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']['vectors']

    CADASTRAL_SURVEYING = cfg['cadastral_surveying']
    TILES = cfg['tiles']

    os.chdir(WORKING_DIR)

    _, written_files = format_surveying_data(CADASTRAL_SURVEYING, TILES, OUTPUT_DIR)

    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)
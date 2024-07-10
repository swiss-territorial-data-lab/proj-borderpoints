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
from constants import OVERWRITE

logger = format_logger(logger)


def format_surveying_data(path_surveying, tiles, nodata_gdf=None, remove_duplicates=True, output_dir='outputs'):
    """
    Formats the surveying data by extracting points from the survey polygons and saving them in a GeoDataFrame.
    
    Parameters:
        path_surveying (str): The path to the survey polygons file.
        tiles (str or GeoDataFrame): The path to the tiles file or the tiles GeoDataFrame.
        nodata_gdf (GeoDataFrame): The Geodataframe with the area not covered by tiles.
        remove_duplicates (bool, optional): Whether to remove duplicate points due to the intersection with several tiles. Defaults to False.
        output_dir (str, optional): The directory where the output file will be saved. Defaults to 'outputs'.
    
    Returns:
        tuple: A tuple containing the inferred points GeoDataFrame and a list of written files.
    """
    written_files =[] 

    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, 'MO_points.gpkg')
    if os.path.isfile(filepath) and (not OVERWRITE):
        logger.info('File already exists, reading from disk...')
        return gpd.read_file(filepath), []

    logger.info('Read file...')
    survey_poly_gdf = gpd.read_file(path_surveying)
    if isinstance(tiles, str):
        tiles_gdf = gpd.read_file(tiles)
    elif isinstance(tiles, gpd.GeoDataFrame):
        tiles_gdf = tiles.copy()

    logger.info('Get point coordinates...')
    if (survey_poly_gdf.geom_type == 'Polygon').all():
        pts_list = [
            Point(pt) for poly in survey_poly_gdf.geometry 
            for geom in poly.geoms 
            for pt in geom.exterior.coords[:-1]
        ]
    elif (survey_poly_gdf.geom_type == 'Point').all():
        pts_list = survey_poly_gdf.geometry
    else:
        raise ValueError('The geometry type of the survey data is not supported.')
    
    coor_list = [
            str(pt.coords[0][0])[2:].replace('.', '')[:10] + str(pt.coords[0][1])[2:].replace('.', '')[:10] 
            for pt in pts_list
        ]

    logger.info('Get point ids...')
    pt_ids_list = [
        str(i) + '_' + str(pt.coords[0][0])[2:].replace('.', '')[:5] + str(pt.coords[0][1])[2:].replace('.', '')[:5]
        for i, pt in zip(range(len(pts_list)), pts_list)
    ]

    logger.info('Put infered points in a GeoDataFrame...')
    pts_gdf = gpd.GeoDataFrame({'pt_id': pt_ids_list, 'approx_coor': coor_list, 'geometry': pts_list}, crs='EPSG:2056')
    pts_gdf.drop_duplicates('approx_coor', inplace=True, ignore_index=True)
    pts_gdf.drop(columns=['approx_coor'], inplace=True)

    logger.info('Exclude points outside the tiles...')
    try:                # Working with tile attributes
        pts_in_tiles_gdf = gpd.sjoin(pts_gdf, tiles_gdf[['name', 'geometry', 'scale']], how='inner').drop(columns=['index_right'])
        pts_in_tiles_gdf['combo_id'] = pts_in_tiles_gdf['pt_id'] + ' - ' + pts_in_tiles_gdf['name']
    except KeyError:    # Working with subtile attributes
        pts_in_tiles_gdf = gpd.overlay(pts_gdf, tiles_gdf[['id', 'initial_tile', 'geometry']], keep_geom_type=True)
        pts_in_tiles_gdf['combo_id'] = pts_in_tiles_gdf['pt_id'] + ' - ' + pts_in_tiles_gdf['initial_tile']
    if remove_duplicates:
        pts_in_tiles_gdf.drop_duplicates('pt_id', inplace=True)
 
    if isinstance(nodata_gdf, gpd.GeoDataFrame):
        pts_on_nodata_gdf = gpd.sjoin(pts_in_tiles_gdf, nodata_gdf)
        pts_to_remove = pts_on_nodata_gdf.loc[pts_on_nodata_gdf.name==pts_on_nodata_gdf.tile_name, 'combo_id'].tolist()
        pts_in_tiles_gdf = pts_in_tiles_gdf[~pts_in_tiles_gdf['combo_id'].isin(pts_to_remove)].copy()
        pts_in_tiles_gdf.rename(columns={'name': 'initial_tile'}, inplace=True)
    else:
        pts_in_tiles_gdf.drop(columns=['name'], inplace=True, errors='ignore')

    pts_in_tiles_gdf.to_file(filepath)
    written_files.append(filepath)

    logger.success('Done formatting surveying data!')
    return pts_in_tiles_gdf, written_files


# ------------------------------------------

if __name__ == "__main__":
    # Argument and parameter specification
    parser = ArgumentParser(description="The script formats the cadastral surveying to limit the produced tiles.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)['prepare_whole_tiles.py']

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']['vectors']

    CADASTRAL_SURVEYING = cfg['cadastral_surveying']
    TILES = os.path.join(OUTPUT_DIR, 'subtiles.gpkg')

    os.chdir(WORKING_DIR)

    _, written_files = format_surveying_data(CADASTRAL_SURVEYING, TILES, output_dir=OUTPUT_DIR)

    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)
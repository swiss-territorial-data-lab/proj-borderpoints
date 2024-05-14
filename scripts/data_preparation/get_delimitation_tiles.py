import os
import sys
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd
import rasterio as rio
from glob import glob
from shapely.geometry import box, Polygon

from joblib import Parallel, delayed

sys.path.insert(1, 'scripts')
import constants as cst
import functions.fct_misc as misc
import functions.fct_rasters as rasters

logger = misc.format_logger(logger)


def control_overlap(gdf1, gdf2, threshold=0.5, op='larger'):
    
    gdf1['total_area'] = gdf1.area

    intersection_gdf = gpd.overlay(gdf1, gdf2, how="difference", keep_geom_type=True)
    intersection_gdf = intersection_gdf.dissolve('id', as_index=False)
    intersection_gdf['percentage_area_left'] = intersection_gdf.area / intersection_gdf.total_area
    if op=='larger':
        id_to_keep = intersection_gdf.loc[intersection_gdf.percentage_area_left > threshold, 'id'].unique().tolist()
    elif op=='seq':
        id_to_keep = intersection_gdf.loc[intersection_gdf.percentage_area_left <= threshold, 'id'].unique().tolist()
    else:
        logger.critical('Passed operator is unknow. Please pass "larger" or "seq" (= smaller or equal).')
        sys.exit(1)

    return id_to_keep


def extract_tile_info(tile, overlap_info=None, tile_suffix='.tif'):
    tiles_dict = {'id': [], 'name': [], 'scale': [], 'geometry': [], 'pixel_size_x': [], 'pixel_size_y': [], 'dimension': [], 'origin': [], 'max_dx': [], 'max_dy': []}

    tile_name = os.path.basename(tile).rstrip(tile_suffix)
    try:
        tile_scale = int(tile_name.split('_')[0])
    except:
        logger.warning(f'Missing info corresponding to the tile {tile_name}. Skipping it.')
        return (None, None)

    # Set attribute of the tiles
    tiles_dict['name'].append(tile_name)
    if '_' in tile_name:
        tiles_dict['id'].append(f"({tile_name.split('_')[1]}, {tile_name.split('_')[2]}, {tile_scale})")
    else:
        tiles_dict['id'].append(f"({tile_name[:6]}, {tile_name[6:].split('.')[0]}, {tile_scale})")
    tiles_dict['scale'].append(tile_scale)

    with rio.open(tile) as src:
        bounds = src.bounds
        first_band = src.read(1)
        meta = src.meta

    # Set tile geometry
    geom = box(*bounds)
    tiles_dict['geometry'].append(geom)
    tiles_dict['origin'].append(str(rasters.get_bbox_origin(geom)))
    tile_size = (meta['width'], meta['height'])
    tiles_dict['dimension'].append(str(tile_size))
    
    # Set pixel size
    pixel_size_x = abs(meta['transform'][0])
    pixel_size_y = abs(meta['transform'][4])

    try:
        assert round(pixel_size_x, 5) == round(pixel_size_y, 5), f'The pixels are not square on tile {tile_name}: {round(pixel_size_x, 5)} x {round(pixel_size_y, 5)} m.'
    except AssertionError as e:
        print()
        logger.warning(e)

    tiles_dict['pixel_size_x'].append(pixel_size_x)
    tiles_dict['pixel_size_y'].append(pixel_size_y)

    # If no info on the plan scales, leave dx and dy to 0.
    if overlap_info:
        if isinstance(overlap_info, str):
            overlap_info_df = pd.read_csv(overlap_info)
        elif isinstance(overlap_info, pd.DataFrame):
            overlap_info_df = overlap_info
        else:
            logger.error('Unrecognized format for the overlap info!')
            sys.exit(1)
        max_dx = overlap_info_df.loc[overlap_info_df.scale==tile_scale, 'max_dx'].iloc[0]/pixel_size_x
        max_dy = overlap_info_df.loc[overlap_info_df.scale==tile_scale, 'max_dy'].iloc[0]/pixel_size_y
    else:
        max_dx = 0
        max_dy = 0
    tiles_dict['max_dx'].append(max_dx)
    tiles_dict['max_dy'].append(max_dy)

    # Transform nodata area into polygons
    nodata_gdf = rasters.no_data_to_polygons(first_band, meta['transform'], meta['nodata'])
    nodata_gdf = pad_geodataframe(nodata_gdf, bounds, tile_size, max(pixel_size_x, pixel_size_y), cst.GRID_LARGE_TILES, cst.GRID_LARGE_TILES, max_dx, max_dy)
    nodata_gdf['tile_name'] = tile_name

    return (tiles_dict, nodata_gdf)


def get_delimitation_tiles(tile_dir, cadastral_surveying=None, overlap_info=None, tile_suffix='.tif', output_dir='outputs', subtiles=False):

    os.makedirs(output_dir, exist_ok=True)
    written_files = [] 

    output_path_tiles = os.path.join(output_dir, 'tiles.gpkg')
    output_path_nodata = os.path.join(output_dir, 'nodata_areas.gpkg')

    if not cst.OVERWRITE and os.path.exists(output_path_tiles) and os.path.exists(output_path_nodata):
        tiles_gdf = gpd.read_file(output_path_tiles)
        nodata_gdf=gpd.read_file(output_path_nodata)

    else:
        
        logger.info('Read info for tiles...')
        tile_list = glob(os.path.join(tile_dir, '*.tif'))

        if len(tile_list) == 0:
            logger.critical('No tile in the tile directory.')
            sys.exit(1)

        logger.info('Create a geodataframe with tile info...')

        tile_tuples_list = Parallel(n_jobs=10, backend="loky")(
            delayed(extract_tile_info)(tile, overlap_info, tile_suffix) for tile in tqdm(tile_list, desc='Read tile info')
            )
        tiles_dict_list = [tile_tuple[0] for tile_tuple in tile_tuples_list if tile_tuple[0]]
        nodata_gdf_list = [tile_tuple[1] for tile_tuple in tile_tuples_list if tile_tuple[1]]

        tiles_gdf = gpd.GeoDataFrame.from_features(tiles_dict_list, crs='EPSG:2056')

        tiles_gdf.to_file(output_path_tiles)
        written_files.append(output_path_tiles)

        nodata_gdf = pd.concat(nodata_gdf_list, ignore_index=True)
        nodata_gdf.to_file(output_path_nodata)
        written_files.append(output_path_nodata)


    if subtiles:
       
        job_outcome = Parallel(n_jobs=7, backend="loky")(
            delayed(get_subtiles)(nodata_gdf, tile_row) for tile_row in tqdm(tiles_gdf.itertuples(), desc='Determine subtiles', total=len(tiles_gdf))
            )
        
        logger.info('Concatenate the result...')
        subtiles_gdf = pd.concat(job_outcome, ignore_index=True)
        
        if cadastral_surveying:
            logger.info('Only keep tiles that are intersecting survey points...')
            survey_pts_gdf = gpd.read_file(cadastral_surveying)
            subtiles_w_pts = gpd.sjoin(subtiles_gdf, survey_pts_gdf, how="inner").id
            subtiles_gdf = subtiles_gdf[subtiles_gdf.id.isin(subtiles_w_pts)].copy()

        if cst.CLIP_OR_PAD_SUBTILES == 'clip':
            logger.info('The tiles are clipped to the image border.')
            tiling_zone = tiles_gdf.unary_union
            subtiles_gdf = gpd.overlay(
                subtiles_gdf, gpd.GeoDataFrame({'tiling_id': [1], 'geometry': [tiling_zone]}, crs='EPSG:2056'), 
                how="intersection", keep_geom_type=True
            )

        filepath = os.path.join(output_dir, 'subtiles.gpkg')
        subtiles_gdf.to_file(filepath)
        written_files.append(filepath)

    else:
        subtiles_gdf = None
    
    logger.success('Done determining the tiling!')
    return tiles_gdf, subtiles_gdf, written_files


def get_subtiles(nodata_gdf, tile_row):
    tile_infos = {
        'tile_size': tuple(map(int, tile_row.dimension.strip('()').split(', '))), 
        'tile_origin': tuple(map(float, tile_row.origin.strip('()').split(', '))), 
        'pixel_size_x': tile_row.pixel_size_x,
        'pixel_size_y': tile_row.pixel_size_y,
        'max_dx': tile_row.max_dx,
        'max_dy': tile_row.max_dy
    }
    
    nodata_subset_gdf = nodata_gdf[nodata_gdf.tile_name==tile_row.name].copy()
    # Make a large tiling grid to cover the image
    temp_gdf = rasters.grid_over_tiles(grid_width=cst.GRID_LARGE_TILES, grid_height=cst.GRID_LARGE_TILES, **tile_infos)

    # Only keep tiles that do not overlap too much the nodata zone
    large_id_on_image = control_overlap(temp_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=cst.OVERLAP_LARGE_TILES)
    large_subtiles_gdf = temp_gdf[temp_gdf.id.isin(large_id_on_image)].copy()
    large_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile_row.scale)})' for subtile_id in large_subtiles_gdf.id] 
    large_subtiles_gdf['initial_tile'] = tile_row.name

    if (tile_row.max_dx == 0) and (tile_row.max_dy == 0):
        # Make a smaller tiling grid to not lose too much data
        temp_gdf = rasters.grid_over_tiles(grid_width=cst.GRID_SMALL_TILES, grid_height=cst.GRID_SMALL_TILES, **tile_infos)
        # Only keep smal subtiles not under a large one
        small_subtiles_gdf = gpd.overlay(temp_gdf, large_subtiles_gdf, how='difference', keep_geom_type=True)
        small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.area > 10].copy()
        
        if not small_subtiles_gdf.empty:
            # Only keep tiles that do not overlap too much the nodata zone
            small_id_on_image = control_overlap(small_subtiles_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=cst.OVERLAP_SMALL_TILES)
            small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.id.isin(small_id_on_image)].copy()
            small_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile_row.scale)})' for subtile_id in small_subtiles_gdf.id]
            small_subtiles_gdf['initial_tile'] = tile_row.name

            subtiles_gdf = pd.concat([large_subtiles_gdf, small_subtiles_gdf], ignore_index=True)
    
    subtiles_gdf = large_subtiles_gdf.copy()

    return subtiles_gdf

    
def pad_geodataframe(gdf, tile_bounds, tile_size, pixel_size, grid_width=256, grid_height=256, max_dx=0, max_dy=0):

    min_x, min_y, max_x, max_y = tile_bounds
    tile_width, tile_height = tile_size
    number_cells_x, number_cells_y = rasters.get_grid_size(tile_size, grid_width, grid_height, max_dx, max_dy)

    # Get difference between grid size and tile size
    pad_width_px_x = number_cells_x * (grid_width - max_dx) + max_dx - tile_width
    pad_width_px_y = number_cells_y * (grid_height - max_dy) + max_dy - tile_height

    # Convert dimensions from pixels to meters
    pad_width_m_x = pad_width_px_x * pixel_size
    pad_width_m_y = pad_width_px_y * pixel_size

    # Pad on the top
    vertices = [(min_x, max_y),
                (max_x + pad_width_m_x, max_y),
                (max_x + pad_width_m_x, max_y + pad_width_m_y),
                (min_x, max_y + pad_width_m_y)]
    polygon_top = Polygon(vertices)
    
    # Pad on the right
    vertices = [(max_x, min_y),
                (max_x + pad_width_m_x, min_y),
                (max_x + pad_width_m_x, max_y ),
                (max_x, max_y)]
    polygon_right = Polygon(vertices)

    gdf = pd.concat([gdf, gpd.GeoDataFrame({'id_nodata_poly': [10001, 10002], 'geometry': [polygon_top, polygon_right]}, crs="EPSG:2056")], ignore_index=True)

    return gdf


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
    OUTPUT_DIR = cfg['output_dir']['vectors']
    TILE_DIR = cfg['output_dir']['tiles']
    CADASTRAL_SURVEYING = cfg['cadastral_surveying'] if 'cadastral_surveying' in cfg.keys() else None
    OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None

    os.chdir(WORKING_DIR)

    _, _, written_files = get_delimitation_tiles(TILE_DIR, CADASTRAL_SURVEYING,  OVERLAP_INFO, output_dir=OUTPUT_DIR, subtiles=True)

    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)
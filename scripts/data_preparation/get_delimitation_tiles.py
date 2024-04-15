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

sys.path.insert(1, 'scripts')
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


def define_subtiles(tiles_gdf, nodata_gdf, grid_width_large, grid_width_small, max_nodata_large_tiles=0.5, max_nodata_small_tiles=0.5, output_dir='outputs'):

    logger.info('Determine subtiles...')
    subtiles_gdf = gpd.GeoDataFrame()
    for tile in tqdm(tiles_gdf.itertuples(), desc='Define a grid to subdivide tiles', total=tiles_gdf.shape[0]):
        tile_infos = {
            'tile_size': tuple(map(int, tile.dimension.strip('()').split(', '))), 
            'tile_origin': tuple(map(float, tile.origin.strip('()').split(', '))), 
            'pixel_size_x': tile.pixel_size_x,
            'pixel_size_y': tile.pixel_size_y,
            'max_dx': tile.max_dx,
            'max_dy': tile.max_dy
        }
        nodata_subset_gdf = nodata_gdf[nodata_gdf.tile_name==tile.name]

        # Make a large tiling grid to cover the image
        temp_gdf = rasters.grid_over_tiles(grid_width=grid_width_large, grid_height=grid_width_large, **tile_infos)

        # Only keep tiles that do not overlap too much the nodata zone
        large_id_on_image = control_overlap(temp_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=max_nodata_large_tiles)
        large_subtiles_gdf = temp_gdf[temp_gdf.id.isin(large_id_on_image)].copy()
        large_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile.scale)})' for subtile_id in large_subtiles_gdf.id] 
        large_subtiles_gdf['initial_tile'] = tile.name

        if (tile.max_dx == 0) and (tile.max_dy == 0):
            # Make a smaller tiling grid to not lose too much data
            temp_gdf = rasters.grid_over_tiles(grid_width=grid_width_small, grid_height=grid_width_small, **tile_infos)
            small_subtiles_gdf = gpd.overlay(temp_gdf, large_subtiles_gdf, how='difference', keep_geom_type=True)
            small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.area > 10].copy()
            
            if not small_subtiles_gdf.empty:
                # Only keep tiles that do not overlap too much the nodata zone
                small_id_on_image = control_overlap(small_subtiles_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=max_nodata_small_tiles)
                small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.id.isin(small_id_on_image)].copy()
                small_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile.scale)})' for subtile_id in small_subtiles_gdf.id]
                small_subtiles_gdf['initial_tile'] = tile.name

                subtiles_gdf = pd.concat([subtiles_gdf, small_subtiles_gdf], ignore_index=True)
        
        subtiles_gdf = pd.concat([subtiles_gdf, large_subtiles_gdf], ignore_index=True)


    filepath = os.path.join(output_dir, 'subtiles.gpkg')
    subtiles_gdf.to_file(filepath)

    return subtiles_gdf, filepath


def get_delimitation_tiles(tile_dir,
                           grid_width_large, grid_width_small, max_nodata_large_tiles=0.5, max_nodata_small_tiles=0.5, 
                           overlap_info=None, tile_suffix='.tif',
                           output_dir='outputs', overwrite_tiles=False, subtiles=False):

    os.makedirs(output_dir, exist_ok=True)
    written_files = [] 

    output_path_tiles = os.path.join(output_dir, 'tiles.gpkg')
    output_path_nodata = os.path.join(output_dir, 'nodata_areas.gpkg')

    if not overwrite_tiles and os.path.exists(output_path_tiles) and os.path.exists(output_path_nodata):
        tiles_gdf = gpd.read_file(output_path_tiles)
        nodata_gdf=gpd.read_file(output_path_nodata)

    else:
        
        logger.info('Read info for tiles...')
        tile_list = glob(os.path.join(tile_dir, '*.tif'))

        logger.info('Create a geodataframe with tile info...')
        tiles_dict = {'id': [], 'name': [], 'scale': [], 'geometry': [], 'pixel_size_x': [], 'pixel_size_y': [], 'dimension': [], 'origin': [], 'max_dx': [], 'max_dy': []}
        nodata_gdf = gpd.GeoDataFrame()
        for tile in tqdm(tile_list, desc='Read tile info'):

            tile_name = os.path.basename(tile).rstrip(tile_suffix)
            try:
                tile_scale = int(tile_name.split('_')[0])
            except:
                logger.warning(f'Missing info corresponding to the tile {tile_name}. Skipping it.')
                continue

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
            temp_gdf = rasters.no_data_to_polygons(first_band, meta['transform'], meta['nodata'])
            temp_gdf = pad_geodataframe(temp_gdf, bounds, tile_size, max(pixel_size_x, pixel_size_y), grid_width_large, grid_width_large, max_dx, max_dy)
            temp_gdf['tile_name'] = tile_name
            nodata_gdf = pd.concat([nodata_gdf, temp_gdf], ignore_index=True)

        tiles_gdf = gpd.GeoDataFrame(tiles_dict, crs='EPSG:2056')

        tiles_gdf.to_file(output_path_tiles)
        written_files.append(output_path_tiles)

        nodata_gdf.to_file(output_path_nodata)
        written_files.append(output_path_nodata)


    if subtiles:
       
        subtiles_gdf, filepath = define_subtiles(tiles_gdf, nodata_gdf,
                                                 grid_width_large, grid_width_small, max_nodata_large_tiles, max_nodata_small_tiles,
                                                 output_dir)
        written_files.append(filepath)

        return tiles_gdf, subtiles_gdf, written_files
    else:
        return tiles_gdf, None, written_files
    
    
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

    with open(args.config_file) as fp:
        cfg_globals = load(fp, Loader=FullLoader)['globals']

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir_vectors']

    TILE_DIR = cfg['output_dir_tiles']
    PLAN_SCALES = cfg['plan_scales']

    TILE_SUFFIX = cfg_globals['original_tile_suffix']

    MAX_NODATA_LARGE_TILES = cfg_globals['thresholds']['max_nodata_large_tiles']
    MAX_NODATA_SMALL_TILES = cfg_globals['thresholds']['max_nodata_small_tiles']
    GRID_LARGE_TILES = cfg_globals['grid_width_large']
    GRID_SMALL_TILES = cfg_globals['grid_width_small']

    OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None

    OVERWRITE = cfg_globals['overwrite']

    os.chdir(WORKING_DIR)

    _, _, written_files = get_delimitation_tiles(TILE_DIR, 
                                                GRID_LARGE_TILES, GRID_SMALL_TILES, MAX_NODATA_LARGE_TILES, MAX_NODATA_SMALL_TILES, 
                                                OVERLAP_INFO,
                                                OUTPUT_DIR, overwrite_tiles=OVERWRITE, subtiles=True)

    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)
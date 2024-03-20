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


def define_subtiles(tiles_gdf, nodata_gdf, grid_width_large, grid_width_small, overlap_large_tiles=0.5, overlap_small_tiles=0.5, output_dir='outputs'):

    subtiles_gdf = gpd.GeoDataFrame()
    for tile in tqdm(tiles_gdf.itertuples(), desc='Define a grid to subdivide tiles', total=tiles_gdf.shape[0]):
        tile_infos = {
            'tile_size': tuple(map(int, tile.dimension.strip('()').split(', '))), 
            'tile_origin': tuple(map(float, tile.origin.strip('()').split(', '))), 
            'pixel_size': tile.pixel_size
        }
        nodata_subset_gdf = nodata_gdf[nodata_gdf.tile_name==tile.name]

        # Make a large tiling grid to cover the image
        temp_gdf = rasters.grid_over_tiles(grid_width=grid_width_large, grid_height=grid_width_large, **tile_infos)

        # Only keep tiles that do not overlap too much the nodata zone
        large_id_on_image = control_overlap(temp_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=overlap_large_tiles)
        large_subtiles_gdf = temp_gdf[temp_gdf.id.isin(large_id_on_image)].copy()
        large_subtiles_gdf.loc[:, 'id'] = [subtile_id + '_' + str(tile.scale) for subtile_id in large_subtiles_gdf.id] 
        large_subtiles_gdf['initial_tile'] = tile.name

        # Make a smaller tiling grid to not lose too much data
        temp_gdf = rasters.grid_over_tiles(grid_width=grid_width_small, grid_height=grid_width_small, **tile_infos)
        small_subtiles_gdf = gpd.overlay(temp_gdf, large_subtiles_gdf, how='difference', keep_geom_type=True)
        small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.area < 1].copy()
        
        if not small_subtiles_gdf.empty:
            # Only keep tiles that do not overlap too much the nodata zone
            small_id_on_image = control_overlap(small_subtiles_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=overlap_small_tiles)
            small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.id.isin(small_id_on_image)].copy()
            small_subtiles_gdf.loc[:, 'id'] = [subtile_id + '_' + str(tile.scale) for subtile_id in small_subtiles_gdf.id] 

        subtiles_gdf = pd.concat([subtiles_gdf, large_subtiles_gdf, small_subtiles_gdf], ignore_index=True)
        large_subtiles_gdf['initial_tile'] = tile.name

    filepath = os.path.join(output_dir, 'subtiles.gpkg')
    subtiles_gdf.to_file(filepath)

    return subtiles_gdf, filepath


def get_delimitation_tiles(tile_dir, plan_scales_path, 
                           grid_width_large, grid_width_small, overlap_large_tiles=0.5, overlap_small_tiles=0.5, 
                           output_dir='outputs', overwrite_tiles=False, subtiles=False):

    os.makedirs(output_dir, exist_ok=True)
    written_files = [] 

    output_path_tiles = os.path.join(output_dir, 'tiles.gpkg')
    output_path_nodata = os.path.join(output_dir, 'nodata_areas.gpkg')

    if not overwrite_tiles and os.path.exists(output_path_tiles) and os.path.exists(output_path_nodata):
        tiles_gdf = gpd.read_file(output_path_tiles)
        nodata_gdf=gpd.read_file(output_path_nodata)

    else:
        logger.info('Get the delimitation and nodata area of tiles...')
        tile_list = glob(os.path.join(tile_dir, '*.tif'))
        plan_scales = pd.read_excel(plan_scales_path)

        tiles_dict = {'id': [], 'name': [], 'scale': [], 'geometry': [], 'pixel_size': [], 'dimension': [], 'origin': []}
        nodata_gdf = gpd.GeoDataFrame()
        strip_str = '_georeferenced.tif'
        for tile in tqdm(tile_list, desc='Read tile info'):
            tile_name = os.path.basename(tile).rstrip(strip_str)
            if tile_name in plan_scales.Num_plan.unique():
                tile_scale = plan_scales.loc[plan_scales.Num_plan==tile_name, 'Echelle'].iloc[0]
            else:
                try:
                    tile_scale = int(tile_name.split('_')[0])
                except:
                    logger.warning(f'Missing info corresponding to the tile {tile_name}. Skipping it.')
                    continue

            # Set attribute of the tiles
            tiles_dict['name'].append(tile_name)
            tiles_dict['id'].append(f"({tile_name[:6]}, {tile_name[6:]}, {tile_scale})")
            tiles_dict['scale'].append(tile_scale)

            with rio.open(tile) as src:
                bounds = src.bounds
                first_band = src.read(1)
                transform = src.transform
                nodata_value = 0 # src.nodata
                tile_size = (src.width, src.height)
                    
            # Set tile geometry
            geom = box(*bounds)
            tiles_dict['geometry'].append(geom)
            tiles_dict['dimension'].append(str(tile_size))
            tiles_dict['origin'].append(str(rasters.get_bbox_origin(geom)))

            # Set pixel size
            pixel_size = round(transform[0], 5)
            try:
                assert round(transform[0], 5) == round(-transform[4], 5), f'The pixels are not square on tile {tile_name}: {transform[0]}x{-transform[4]}.'
            except AssertionError as e:
                logger.error(e)
                logger.warning('Using the smaller dimension as the pixel size.')

            tiles_dict['pixel_size'].append(pixel_size)

            # Transform nodata area into polygons
            temp_gdf = rasters.no_data_to_polygons(first_band, transform, nodata_value)
            temp_gdf = pad_geodataframe(temp_gdf, bounds, tile_size, pixel_size, grid_width_large, grid_width_large)
            temp_gdf['tile_name'] = tile_name
            nodata_gdf = pd.concat([nodata_gdf, temp_gdf], ignore_index=True)

        tiles_gdf = gpd.GeoDataFrame(tiles_dict, crs='EPSG:2056')

        tiles_gdf.to_file(output_path_tiles)
        written_files.append(output_path_tiles)

        nodata_gdf.to_file(output_path_nodata)
        written_files.append(output_path_nodata)


    if subtiles:
        subtiles_gdf, filepath = define_subtiles(tiles_gdf, nodata_gdf, 
                                                        grid_width_large, grid_width_small, overlap_large_tiles, overlap_small_tiles, 
                                                        output_dir)
        written_files.append(filepath)

        return tiles_gdf, subtiles_gdf, written_files
    else:
        return tiles_gdf, None, written_files
    
    
def pad_geodataframe(gdf, tile_bounds, tile_size, pixel_size, grid_width=256, grid_height=256):

    min_x, min_y, max_x, max_y = tile_bounds
    tile_width, tile_height = tile_size
    number_cells_x, number_cells_y = rasters.get_grid_size(tile_size, grid_width, grid_height)

    # Get difference between grid size and tile size
    pad_width_px_x = number_cells_x * grid_width - tile_width
    pad_width_px_y = number_cells_y * grid_height - tile_height

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
    OUTPUT_DIR = cfg['output_dir']

    TILE_DIR = cfg['tile_dir']
    PLAN_SCALES = cfg['plan_scales']

    OVERLAP_LARGE_TILES = cfg_globals['thresholds']['overlap_large_tiles']
    OVERLAP_SMALL_TILES = cfg_globals['thresholds']['overlap_small_tiles']
    GRID_LARGE_TILES = cfg_globals['grid_width_large']
    GRID_SMALL_TILES = cfg_globals['grid_width_large']

    OVERWRITE = cfg_globals['overwrite']

    os.chdir(WORKING_DIR)

    _, _, written_files = get_delimitation_tiles(TILE_DIR, PLAN_SCALES, 
                                                GRID_LARGE_TILES, GRID_SMALL_TILES, OVERLAP_LARGE_TILES, OVERLAP_SMALL_TILES, 
                                                OUTPUT_DIR, overwrite_tiles=OVERWRITE, subtiles=True)

    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)
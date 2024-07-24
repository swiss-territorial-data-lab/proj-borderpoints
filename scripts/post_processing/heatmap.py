import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from folium import Map, plugins

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config
from functions.fct_rasters import get_bbox_origin, get_east_north, grid_over_tile

logger = format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = get_config(os.path.basename, desc="The script creates a heatmap of the false positive points.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

MATCHED_POINTS = cfg['matched_points']
BDMO_POLY = cfg['bdmo_poly']

PIXEL_SIZE = 100
GRID_SIZE = 1

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Read data...')

matched_points_gdf = gpd.read_file(MATCHED_POINTS)
bdmo_poly_gdf = gpd.read_file(BDMO_POLY)

logger.info('Determine zones of cadastral surveys...')

cs_zones_gdf = bdmo_poly_gdf[['IDENTND_NU', 'geometry']].dissolve().explode(ignore_index=True)
cs_zones_gdf.loc[:, 'geometry'] = cs_zones_gdf.geometry.buffer(PIXEL_SIZE+1)
cs_zones_gdf['zone'] = cs_zones_gdf.index

logger.info('Intersect points with zones...')
# Join the points to their zones
fp_points_gdf = matched_points_gdf[matched_points_gdf.pt_id.isna()].copy()
fp_points_with_zones_gdf = fp_points_gdf.sjoin(cs_zones_gdf, predicate='within')

total_grid_gdf = gpd.GeoDataFrame()
for zone in tqdm(fp_points_with_zones_gdf.zone.unique(), desc='Create a grid to calculate the FP density'):
    zone_geom = cs_zones_gdf.loc[cs_zones_gdf.zone==zone, 'geometry'].iloc[0]
    tile_origin = get_bbox_origin(zone_geom)
    tile_n_e = get_east_north(zone_geom)
    tile_size = (
        ((tile_n_e[0] - tile_origin[0])/(GRID_SIZE*PIXEL_SIZE), 
         (tile_n_e[1] - tile_origin[1])/(GRID_SIZE*PIXEL_SIZE))
    )

    heatgrid = grid_over_tile(tile_size, tile_origin, pixel_size_x=PIXEL_SIZE, grid_width=GRID_SIZE, grid_height=GRID_SIZE, test_shape=zone_geom)
    total_grid_gdf = pd.concat([total_grid_gdf, heatgrid], ignore_index=True)

    # Get number of points in the grid
    fp_points_in_zone_gdf = fp_points_with_zones_gdf[fp_points_with_zones_gdf.zone==zone].copy()

# Plot heatmap
# https://geopandas.org/en/stable/gallery/plotting_with_folium.html#Folium-Heatmaps
map = Map(location=[46.7, 7.1], zoom_start=11, control_scale=True,)
heat_data = [[point.xy[1][0], point.xy[0][0]] for point in fp_points_gdf.geometry.to_crs(4326)]
plugins.HeatMap(heat_data).add_to(map)
map.save(os.path.join(OUTPUT_DIR, f'heatmap.html'))

# Get density of points in the grid
fp_points_on_grid_gdf = total_grid_gdf.sjoin(fp_points_gdf[['pt_id', 'geometry']])
point_count_df = fp_points_on_grid_gdf.groupby('id').size().reset_index(name='count')
point_count_gdf = total_grid_gdf.merge(point_count_df, how='right', on='id')

logger.info('Save result...')
filepath = os.path.join(OUTPUT_DIR, 'heatmap.gpkg')
point_count_gdf.to_file(filepath)

logger.success(f'Done! One file was written: {filepath}.')
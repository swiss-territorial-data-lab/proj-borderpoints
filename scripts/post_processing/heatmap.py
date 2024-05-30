import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import pandas as pd
from folium import Map, plugins
from matplotlib import pyplot as plt

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger
from functions.fct_rasters import get_bbox_origin, get_easting_northing, grid_over_tiles

logger = format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script creates a heatmap of the false positive points.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

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

logger.info('Create heatmaps...')
total_grid_gdf = gpd.GeoDataFrame()
for zone in tqdm(fp_points_with_zones_gdf.zone.unique(), desc='Produce heatmap for zones of missing cadastral survey'):
    # Determine a fictive grid of 10 x 10 m to calculate point density
    zone_geom = cs_zones_gdf.loc[cs_zones_gdf.zone==zone, 'geometry'].iloc[0]
    tile_origin = get_bbox_origin(zone_geom)
    tile_n_e = get_easting_northing(zone_geom)
    tile_size = (
        ((tile_n_e[0] - tile_origin[0])/(GRID_SIZE*PIXEL_SIZE), 
         (tile_n_e[1] - tile_origin[1])/(GRID_SIZE*PIXEL_SIZE))
    )

    heatgrid = grid_over_tiles(tile_size, tile_origin, pixel_size_x=PIXEL_SIZE, grid_width=GRID_SIZE, grid_height=GRID_SIZE, test_shape=zone_geom)
    total_grid_gdf = pd.concat([total_grid_gdf, heatgrid], ignore_index=True)

    # Get density of points in the grid
    fp_points_in_zone_gdf = fp_points_with_zones_gdf[fp_points_with_zones_gdf.zone==zone].copy()

    # Plot heatmap
    
    # https://residentmario.github.io/geoplot/plot_references/plot_reference.html#kdeplot
    reprojected_cs_zone = cs_zones_gdf[cs_zones_gdf.zone==zone].to_crs(4326)
    tile_origin_4326 = get_bbox_origin(reprojected_cs_zone.geometry.iloc[0])
    tile_n_e_4326 = get_easting_northing(reprojected_cs_zone.geometry.iloc[0])

    ax = gplt.polyplot(reprojected_cs_zone, projection=gcrs.Mercator())
    gplt.kdeplot(
        fp_points_in_zone_gdf.to_crs(4326), cmap='Reds', 
        extent=(tile_origin_4326[0]-0.01, tile_origin_4326[1]-0.01, tile_n_e_4326[0]+0.01, tile_n_e_4326[1]+0.01), ax=ax
    )
    plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_1_{zone}.png'))

# https://geopandas.org/en/stable/gallery/plotting_with_folium.html#Folium-Heatmaps
map = Map(location=[46.7, 7.1], zoom_start=11, control_scale=True,)
heat_data = [[point.xy[1][0], point.xy[0][0]] for point in fp_points_gdf.geometry.to_crs(4326)]
plugins.HeatMap(heat_data).add_to(map)
map.save(os.path.join(OUTPUT_DIR, f'heatmap.html'))

# Get density of points in the grid
fp_points_on_grid_gdf = total_grid_gdf.sjoin(fp_points_gdf[['pt_id', 'geometry']])
point_count_gdf = fp_points_on_grid_gdf.groupby(['id', 'geometry']).size().reset_index(name='count')
point_count_gdf = point_count_gdf[point_count_gdf['count']>0].copy()

logger.info('Save result...')
filepath = os.path.join(OUTPUT_DIR, 'heatmap.gpkg')
gpd.GeoDataFrame(point_count_gdf).to_file(filepath)

logger.success(f'Done! One file was written: {filepath}.')
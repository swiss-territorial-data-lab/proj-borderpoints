from tqdm import tqdm

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import box, Polygon, shape

from math import ceil

def get_grid_size(tile_size, grid_width=256, grid_height=256):

    tile_width, tile_height = tile_size
    number_cells_x = ceil(tile_width/grid_width)
    number_cells_y = ceil(tile_height/grid_height)

    return number_cells_x, number_cells_y


def get_tiles_origin(tile_geom):
    coords = tile_geom.exterior.coords.xy
    min_x = min(coords[0])
    min_y = min(coords[1])

    return (min_x, min_y)

def grid_over_tiles(tile_size, tile_origin, pixel_size, grid_width=256, grid_height=256, crs='EPSG:2056'):

    min_x, min_y = tile_origin

    number_cells_x, number_cells_y = get_grid_size(tile_size, grid_width, grid_height)

    # Convert dimensions from pixels to meters
    grid_x_dim = grid_width * pixel_size
    grid_y_dim = grid_height * pixel_size

    # Create grid polygons
    polygons = []
    for x in range(number_cells_x):
        for y in range(number_cells_y):
            # Define the coordinates of the polygon vertices
            vertices = [(min_x + x * grid_x_dim, min_y + y * grid_y_dim),
                        (min_x + (x + 1) * grid_x_dim, min_y + y * grid_y_dim),
                        (min_x + (x + 1) * grid_x_dim, min_y + (y + 1) * grid_y_dim),
                        (min_x + x * grid_x_dim, min_y + (y + 1) * grid_y_dim)]

            # Create a Polygon object
            polygon = Polygon(vertices)
            polygons.append(polygon)

    # Create a GeoDataFrame from the polygons
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    grid_gdf['id'] = [f'{round(min_x)}_{round(min_y)}' for min_x, min_y in [get_tiles_origin(poly) for poly in grid_gdf.geometry]]

    return grid_gdf


def no_data_to_polygons(image_band, transform, nodata_value, tile_name=None):
    """Convert nodata values in raster (numpy array) to polygons
    cf. https://gis.stackexchange.com/questions/295362/polygonize-raster-file-according-to-band-values

    Args:
        images (DataFrame): image dataframe with an attribute named path

    Returns:
        GeoDataFrame: the polygons of the area with nodata values on the read rasters.
    """

    nodata_polygons = []

    shapes = list(rasterio.features.shapes(image_band, transform=transform))
    nodata_polygons.extend([shape(geom) for geom, value in shapes if value == nodata_value])

    nodata_df = gpd.GeoDataFrame({'id_nodata_poly': [i for i in range(len(nodata_polygons))], 'geometry': nodata_polygons}, crs='EPSG:2056')
    if tile_name:
        nodata_df['tile_name'] = tile_name

    return nodata_df


def pad_band_with_nodata(tile, tile_size, nodata = 9999, grid_width=256, grid_height=256):     # TODO: nodata value to adapt

    tile_width, tile_height = tile_size
    number_cells_x, number_cells_y = get_grid_size(tile_size, grid_width, grid_height)

    # Get difference between grid size and tile size
    pad_width_x = number_cells_x * grid_width - tile_width
    pad_width_y = number_cells_y * grid_height - tile_height

    padded_tile = np.pad(tile, ((0, pad_width_x), (0, pad_width_y)), constant_values=nodata)

    return padded_tile
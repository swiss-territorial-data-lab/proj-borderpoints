from tqdm import tqdm

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import Polygon, shape

from math import ceil, floor

def get_grid_size(tile_size, grid_width=256, grid_height=256, max_dx=0, max_dy=0):

    tile_width, tile_height = tile_size
    number_cells_x = ceil((tile_width - max_dx)/(grid_width - max_dx))
    number_cells_y = ceil((tile_height - max_dy)/(grid_height - max_dy))

    return number_cells_x, number_cells_y


def get_bbox_origin(bbox_geom):
    coords = bbox_geom.exterior.coords.xy
    min_x = min(coords[0])
    min_y = min(coords[1])

    return (min_x, min_y)

def grid_over_tiles(tile_size, tile_origin, pixel_size_x, pixel_size_y=None, max_dx=0, max_dy=0, grid_width=256, grid_height=256, crs='EPSG:2056'):

    min_x, min_y = tile_origin

    number_cells_x, number_cells_y = get_grid_size(tile_size, grid_width, grid_height, max_dx, max_dy)

    # Convert dimensions from pixels to meters
    pixel_size_y = pixel_size_y if pixel_size_y else pixel_size_x
    grid_x_dim = grid_width * pixel_size_x
    grid_y_dim = grid_height * pixel_size_y
    max_dx_dim = max_dx * pixel_size_x
    max_dy_dim = max_dy * pixel_size_y

    # Create grid polygons
    polygons = []
    for x in range(number_cells_x):
        for y in range(number_cells_y):
            # Define the coordinates of the polygon vertices
            vertices = [(min_x + x * (grid_x_dim - max_dx_dim), min_y + y * (grid_y_dim - max_dy_dim)),
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + y * (grid_y_dim - max_dy_dim)),
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + (y + 1) * grid_y_dim - y * max_dy_dim),
                        (min_x + x * (grid_x_dim - max_dx_dim), min_y + (y + 1) * grid_y_dim - y * max_dy_dim)]

            # Create a Polygon object
            polygon = Polygon(vertices)
            polygons.append(polygon)

    # Create a GeoDataFrame from the polygons
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    grid_gdf['id'] = [f'{round(min_x)}, {round(min_y)}' for min_x, min_y in [get_bbox_origin(poly) for poly in grid_gdf.geometry]]

    return grid_gdf


def no_data_to_polygons(image_band, transform, nodata_value, crs="EPSG:2056"):
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

    nodata_df = gpd.GeoDataFrame({'id_nodata_poly': [i for i in range(len(nodata_polygons))], 'geometry': nodata_polygons}, crs=crs)

    return nodata_df
import os
import sys
from loguru import logger

import geopandas as gpd
import pandas as pd
from shapely.affinity import scale


def clip_labels(labels_gdf, tiles_gdf, fact=0.99):
    """
    Clips the labels in the `labels_gdf` GeoDataFrame to the tiles in the `tiles_gdf` GeoDataFrame.
    
    Parameters:
        labels_gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the labels to be clipped.
        tiles_gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the tiles to clip the labels to.
        fact (float, optional): The scaling factor to apply to the tiles when clipping the labels. Defaults to 0.99.
    
    Returns:
        geopandas.GeoDataFrame: The clipped labels GeoDataFrame.
        
    Raises:
        AssertionError: If the CRS of `labels_gdf` is not equal to the CRS of `tiles_gdf`.
    """

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    assert(labels_gdf.crs == tiles_gdf.crs)
    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='intersects')
    
    def clip_row(row, fact=fact):
        
        old_geo = row.geometry
        scaled_tile_geo = scale(row.tile_geometry, xfact=fact, yfact=fact)
        new_geo = old_geo.intersection(scaled_tile_geo)
        row['geometry'] = new_geo

        return row

    clipped_labels_gdf = labels_tiles_sjoined_gdf.apply(lambda row: clip_row(row, fact), axis=1)
    clipped_labels_gdf.crs = labels_gdf.crs

    clipped_labels_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    clipped_labels_gdf.rename(columns={'id': 'tile_id'}, inplace=True)

    return clipped_labels_gdf


def format_logger(logger):
    """
    Format the logger from loguru.

    Args:
        logger (loguru.Logger): The logger object from loguru.

    Returns:
        loguru.Logger: The formatted logger object.
    """

    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")

    return logger


def intersection_over_union(polygon1_shape, polygon2_shape):
    """
    Determine the intersection area over union area (IoU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IoU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    
    return polygon_intersection / polygon_union


def save_name_correspondence(features_list, output_dir, initial_name_column, new_name_column):
    """
    Save the name correspondence of tiles between transformations.
    If a file of name correspondences already exists in the output folder, the names for the converted tiles will be added. 

    Args:
        features_list (list): A list of features containing the initial name and new name.
        output_dir (str): The directory where the name correspondence file will be saved.
        initial_name_column (str): The name of the column containing the initial name.
        new_name_column (str): The name of the column containing the new name.

    Returns:
        None
    """

    name_correspondence_df = pd.DataFrame.from_records(features_list, columns=[initial_name_column, new_name_column])
    filepath = os.path.join(output_dir, 'name_correspondence.csv')

    if os.path.isfile(filepath):
        logger.warning("A file of name correspondences already exists in the output folder. The names of the converted tiles will be added.")
        existing_df = pd.read_csv(filepath)

        if len(existing_df.columns) > 2:
            existing_df = existing_df[['original_name', 'rgb_name']].copy()

        if (initial_name_column in existing_df.columns) and (new_name_column in existing_df.columns):
            name_correspondence_df = pd.concat([existing_df, name_correspondence_df], ignore_index=True)
        else:
            name_correspondence_df = pd.merge(existing_df, name_correspondence_df, on=initial_name_column)

    name_correspondence_df.to_csv(filepath, index=False)
    logger.success(f'The name correspondence of tiles across tranformations was saved in {filepath}.')
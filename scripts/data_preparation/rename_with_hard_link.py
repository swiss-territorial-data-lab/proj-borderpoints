import os
from loguru import logger

import functions.fct_misc as misc

logger = misc.format_logger(logger)

def rename_with_hard_link(tiles_gdf, tile_dir, overwrite=False):

    logger.info('Make hard link for tiles with name in format (x, y, z)...')

    for tile in tiles_gdf.itertuples():
        old_path = os.path.join(tile_dir, tile.name + '_georeferenced.tif')
        new_path = os.path.join(tile_dir, 'renamed', f"{tile.scale}_{tile.name[:6]}_{tile.name[6:]}.tif")

        if not os.path.exists(new_path) or overwrite:
            _ = misc.make_hard_link(old_path, new_path)

    logger.success(f'The images were renamed and copied with a hard link to {os.path.join(tile_dir, "renamed")} ')
import os
import sys
from loguru import logger


def format_logger(logger):
    """Format the logger from loguru

    Args:
        logger: logger object from loguru

    Returns:
        logger: formatted logger object
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


def make_hard_link(img_file, new_img_file):

        if not os.path.isfile(img_file):
                raise FileNotFoundError(img_file)

        src_file = img_file
        dst_file = new_img_file

        dirname = os.path.dirname(dst_file)

        if not os.path.exists(dirname):
                os.makedirs(dirname)

        if not os.path.exists(dst_file):
                os.link(src_file, dst_file)

        return None
"""This module aims to define utils function for the project."""


def find_input_size(cfg):
    """Find the input size of the image after preprocessing

    Args:
        cfg (Dict): config file

    Returns:
        int: input size of the image
    """
    if cfg["SQUARE_PADDING"]["ACTIVE"]:
        input_size = cfg["SQUARE_PADDING"]["INPUT_SIZE"]

    elif cfg["RESIZE_CROP"]["ACTIVE"]:
        input_size = cfg["RESIZE_CROP"]["INPUT_SIZE"]
    return input_size

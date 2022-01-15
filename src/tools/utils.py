"""This module aims to define utils function for the project."""
from models.ConvNet import ConvNet
from models.LinearNet import LinearNet
from models.HRNet import get_HRNet
from models.ResNet import ResNet


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

    elif cfg["RESIZE"]["ACTIVE"]:
        input_size = cfg["RESIZE"]["INPUT_SIZE"]
    return input_size


def load_model(cfg, input_size, num_classes):
    """This function aims to load the right model regarding the configuration file

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """
    if cfg["TRAIN"]["MODEL"] == "LinearNet":
        return LinearNet(input_size=1 * input_size ** 2, num_classes=num_classes)
    elif cfg["TRAIN"]["MODEL"] == "ConvNet":
        return ConvNet(
            input_size=1 * input_size,
            num_classes=num_classes,
            channels=cfg["DATASET"]['PREPROCESSING']["CHANNELS"],
        )
    elif cfg["TRAIN"]["MODEL"] == "ResNet":
        return ResNet(input_size=1 * input_size, num_classes=num_classes)
    elif cfg["TRAIN"]["MODEL"] == "HRNet":
        return get_HRNet(cfg)
    else:
        return LinearNet(input_size=1 * input_size ** 2, num_classes=num_classes)

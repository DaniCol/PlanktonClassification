import os
import argparse
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.LinearNet import LinearNet
from tools.trainer import train_one_epoch
from tools.valid import test_one_epoch
import data.loader as loader


def main(cfg):
    """Main pipeline to train a model

    Args:
        cfg (dict): config with all the necessary parameters
    """

    # Load data
    train_loader, valid_loader,_ = loader.main(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    input_size = cfg["DATASET"]["PREPROCESSING"]["SQUARE_PADDING"]["INPUT_SIZE"]
    model = LinearNet(
        1 * input_size ** 2, cfg["DATASET"]["NUM_CLASSES"]
    )  # 1*input because we only have one channel (gray scale)
    model = model.to(device)

    # Define the loss
    f_loss = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Tracking with tensorboard
    tensorboard_writer = SummaryWriter(log_dir=cfg["TRAIN"]["LOG_DIR"])

    # Launch training loop
    for t in range(cfg["TRAIN"]["EPOCH"]):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, f_loss, optimizer, device
        )
        val_loss, val_acc = test_one_epoch(model, valid_loader, f_loss, device)

        # Track performances with tensorboard
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_loss"), train_loss, t
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_acc"), train_acc, t
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_loss"), val_loss, t
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_acc"), val_acc, t
        )


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file",
    )
    args = parser.parse_args()

    with open(args.path_to_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    main(cfg)

"""This module aims to launch a training procedure."""
# pylint: disable=import-error, no-name-in-module
import os
import argparse
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.LinearNet import LinearNet
from tools.trainer import train_one_epoch
from tools.utils import find_input_size
from tools.valid import test_one_epoch, ModelCheckpoint
import data.loader as loader


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist

    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file

    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def main(cfg):  # pylint: disable=too-many-locals
    """Main pipeline to train a model

    Args:
        cfg (dict): config with all the necessary parameters
    """

    # Load data
    train_loader, valid_loader = loader.main(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    input_size = find_input_size(cfg=cfg["DATASET"]["PREPROCESSING"])
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

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, "linear")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Init Checkpoint class
    checkpoint = ModelCheckpoint(
        save_dir, model, cfg["TRAIN"]["EPOCH"], cfg["TRAIN"]["CHECKPOINT_STEP"]
    )

    # Launch training loop
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        print("EPOCH : {}".format(epoch))

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, f_loss, optimizer, device
        )
        val_loss, val_acc, val_f1 = test_one_epoch(model, valid_loader, f_loss, device)

        # Save best checkpoint
        checkpoint.update(val_loss, epoch)

        # Track performances with tensorboard
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_loss"), train_loss, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_acc"), train_acc, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_f1"), train_f1, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_loss"), val_loss, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_acc"), val_acc, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_f1"), val_f1, epoch
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
        config_file = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    main(cfg=config_file)

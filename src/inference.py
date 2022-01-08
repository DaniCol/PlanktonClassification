"""This module aims to load models for inference and try it on test data."""
# pylint: disable=import-error, no-name-in-module, unused-import
import argparse
import csv
import torch
import yaml

import data.loader as loader
from models.LinearNet import LinearNet
from models.ConvNet import ConvNet
from tools.utils import find_input_size


def inference(cfg):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
    """

    # Open the csv_file to save the results
    file = open(cfg["TEST"]["PATH_TO_CSV"], "w")

    # create the csv writer
    writer = csv.writer(file)

    # write the header of the csv file
    writer.writerow(["imgname", "label"])

    # Load test data
    test_dataloader = loader.main(cfg=cfg, only_test=True)

    # Define device for computational efficiency
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Load model for inference
    input_size = find_input_size(cfg=cfg["DATASET"]["PREPROCESSING"])
    model = LinearNet(1 * input_size ** 2, cfg["DATASET"]["NUM_CLASSES"])
    model.load_state_dict(torch.load(cfg["TEST"]["PATH_TO_MODEL"]))
    model.eval()

    # TODO_ renvoyer le nom de l'image dans le test_loader pour
    # pouvoir ecrire les bons trucs dans le csv file
    for images, names in test_dataloader:
        print(names)
        images = images.to(device)
        outputs = model(images)
        predicted_targets = outputs.argmax(dim=1).tolist()

        # Write the predictions of the batch on the csv file
        for i, name in enumerate(names):
            writer.writerow([name, predicted_targets[i]])

    # close the file
    file.close()


if __name__ == "__main__":
    # Init the parser;
    inference_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add path to the config file to the command line arguments;
    inference_parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file.",
    )
    args = inference_parser.parse_args()

    # Load config file
    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    # Run inference
    inference(cfg=config_file)
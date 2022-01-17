"""This module aims to load models for inference and try it on test data."""
# pylint: disable=import-error, no-name-in-module, unused-import
import argparse
import csv
import sys
import torch
import tqdm
import yaml

import numpy as np
import pandas as pd
import data.loader as loader
from tools.utils import find_input_size, load_model


def inference(cfg, model_path, index=0):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
        model_path (str): path to model
    """

    # Open the csv_file to save the results
    file = open(f"temp_model_{index}.csv", "w")

    # create the csv writer
    writer = csv.writer(file)

    # write the header of the csv file
    writer.writerow(
        ["imgname", *np.arange(int(cfg["DATASET"]["NUM_CLASSES"])).tolist()]
    )

    # Load test data
    test_dataloader = loader.main(cfg=cfg, only_test=True)

    # Define device for computational efficiency
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Load model for inference
    input_size = find_input_size(cfg=cfg["DATASET"]["PREPROCESSING"])
    with torch.no_grad():
        model = load_model(cfg, input_size, cfg["DATASET"]["NUM_CLASSES"])
        model = model.to(device)

        model.load_state_dict(torch.load(model_path))
        model.eval()

        for images, names in tqdm.tqdm(test_dataloader):
            # print(names)
            images = images.to(device)
            outputs = model(images)
            predicted_targets = outputs.cpu().numpy().tolist()

            # Write the predictions of the batch on the csv file
            for i, name in enumerate(names):
                writer.writerow([name, *predicted_targets[i]])

    # close the file
    file.close()


def model_average(cfg):
    """Compute the average prediction

    Args:
        cfg (dict): configuration
    """

    if not cfg["TEST"]["AVERAGE"]["ACTIVE"]:
        print("You should use inference.py !")
        sys.exit()

    # Compute probabilities for every models
    models_predictions = []
    for index, elem in enumerate(cfg["TEST"]["AVERAGE"]["PATH"]):
        with open(elem["CONFIG"], "r") as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.Loader)

        inference(cfg=config_file, model_path=elem["MODEL"], index=index)

        models_predictions.append(
            pd.read_csv(f"temp_model_{index}.csv", header=0, delimiter=",")
        )

    # Compute mean prediction
    size = len(models_predictions)
    predictions = np.argmax(
        sum(models_predictions[i].iloc[:, 1:].to_numpy() for i in range(size)) / size,
        axis=1,
    )
    name = models_predictions[0]["imgname"].to_numpy()

    # Output format
    results = np.concatenate((name.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)

    # Save csv file
    pd.DataFrame(results, columns=["imgname", "label"]).to_csv(
        cfg["TEST"]["PATH_TO_CSV"], index=False
    )


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
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    # Run model average
    model_average(cfg=config_file)

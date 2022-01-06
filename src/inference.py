import torch 
import yaml
import csv
import argparse

import data.loader as loader
from models.LinearNet import LinearNet
from models.ConvNet import ConvNet


def inference(cfg):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration 
    """

    # Open the csv_file to save the results
    f = open(cfg['TEST']['PATH_TO_CSV'], 'w')

    # create the csv writer
    writer = csv.writer(f)

    # write the header of the csv file
    writer.writerow(['imgname','label'])
   
    # Load test data
    _ ,_ , test_dataloader = loader.main(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    input_size = cfg["DATASET"]["PREPROCESSING"]["SQUARE_PADDING"]["INPUT_SIZE"]
    model = LinearNet(1 * input_size ** 2, cfg["DATASET"]["NUM_CLASSES"])
    model.load_state_dict(torch.load(cfg["TEST"]["PATH_TO_MODEL"]))
    model.eval()

    #TODO renvoyer le nom de l'image dans le test_loader pour pouvoir ecrire les bons trucs dans le csv file
    for images, names in test_dataloader:
        print(names)
        images = images.to(device)
        outputs = model(images)
        predicted_targets = outputs.argmax(dim=1).tolist()

        # Write the predictions of the batch on the csv file
        for i,name in enumerate(names):
            writer.writerow([name,predicted_targets[i]])

    # close the file
    f.close()


if __name__=="__main__":
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

    # Load config
    with open(args.path_to_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    # Run inference
    inference(cfg)




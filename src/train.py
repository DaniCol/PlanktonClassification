import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

from models.LinearNet import LinearNet
from models.trainer import train_one_epoch
from models.tester import test_one_epoch
import data.loader as loader 
import yaml 

import os 

def main(cfg):
    # Load data 
    train_loader, valid_loader, test_loader = loader.main(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Define the model 
    input_size = cfg['DATASET']['PREPROCESSING']['SQUARE_PADDING']['INPUT_SIZE']
    model = LinearNet(1*input_size**2, cfg['DATASET']['NUM_CLASSES']) # 1*input because we only have one channel (gray scale)
    model = model.to(device)

    # Define the loss 
    f_loss = torch.nn.CrossEntropyLoss()

    # Define the optimizer 
    optimizer = torch.optim.Adam(model.parameters())

    # Tracking with tensorboard 
    tensorboard_writer = SummaryWriter(log_dir = logdir)

    # Launch training loop 
    for t in range(cfg['TRAIN']['EPOCH']):

        train_loss, train_acc = train_one_epoch(model, train_loader, f_loss, optimizer, device)
        test_loss, test_acc = test_one_epoch(model, valid_loader, f_loss, device)

        # Track performances with tensorboard
        tensorboard_writer.add_scalar(os.path.join(cfg['TRAIN']['LOG_DIR'],'train_loss'), train_loss, t)
        tensorboard_writer.add_scalar(os.path.join(cfg['TRAIN']['LOG_DIR'],'train_acc'), train_acc, t)
        tensorboard_writer.add_scalar(os.path.join(cfg['TRAIN']['LOG_DIR'],'val_loss'), val_loss, t)
        tensorboard_writer.add_scalar(os.path.join(cfg['TRAIN']['LOG_DIR'],'val_acc'),  val_acc, t)



if _name_ == "_main_":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the train folder to the command line arguments
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="path to config file",
    )
    args = parser.parse_args()
    
    with open(args.path_to_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    main(cfg)




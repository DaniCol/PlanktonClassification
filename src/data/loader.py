"""This module aims to load and process the data."""

import argparse  # pylint: disable=import-error
import torch  # pylint: disable=import-error
import torchvision.datasets as datasets  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error

from torch.utils.data import DataLoader  # pylint: disable=import-error

from utils import DatasetTransformer, SquarePad


def main(
    path_to_train,
    path_to_test,
    valid_ratio=0.2,
    batch_size=256,
    num_threads=4,
    verbosity=False,
):  #  pylint: disable=too-many-arguments, too-many-locals
    """Main function to call to load and process data

    Args:
        path_to_train (str): path to the folder contraining the train images
        path_to_test (str): path to the folder contraining the train images
        valid_ratio (float, optional): ratio of data for validation dataset. Defaults to 0.2.
        batch_size (int, optional): DataLoarder batch size. Defaults to 256.
        num_threads (int, optional): Number of cpu to use for DataLoarder. Defaults to 4.
        verbosity (bool, optional): Print the size of the different dataset. Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train, validation and test DataLoader
    """
    new_width, new_height = 300, 300

    # Load the dataset for the training/validation sets
    train_valid_dataset = datasets.ImageFolder(path_to_train)

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid = len(train_valid_dataset) - nb_train
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
        dataset=train_valid_dataset, lengths=[nb_train, nb_valid]
    )

    # Load the test set
    test_dataset = datasets.ImageFolder(path_to_test)

    # DatasetTransformer
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            SquarePad(new_height=new_height, new_width=new_width),
        ]
    )

    train_dataset = DatasetTransformer(train_dataset, data_transforms)
    valid_dataset = DatasetTransformer(valid_dataset, data_transforms)
    test_dataset = DatasetTransformer(test_dataset, data_transforms)

    # Dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_threads,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
    )

    if verbosity:
        print(
            f"The train set contains {len(train_loader.dataset)} images,"
            f" in {len(train_loader)} batches"
        )
        print(
            f"The validation set contains {len(valid_loader.dataset)} images,"
            f" in {len(valid_loader)} batches"
        )
        print(
            f"The test set contains {len(test_loader.dataset)} images,"
            f" in {len(test_loader)} batches"
        )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the train folder to the command line arguments
    parser.add_argument(
        "--path_to_train",
        type=str,
        required=True,
        help="path to the folder containing the train dataset",
    )

    # Add path to the test folder to the command line arguments
    parser.add_argument(
        "--path_to_test",
        type=str,
        required=True,
        help="path to the folder containing the test dataset",
    )

    # Add validation ratio to the command line arguments
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.2,
        help="ratio of data for validation dataset",
    )

    # Add batch size to the command line arguments
    parser.add_argument(
        "--batch_size", type=int, default=256, help="DataLoarder batch size"
    )

    # Add number of cpu to use to the command line arguments
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of cpu to use for DataLoarder",
    )

    # Add verbosity to use to the command line arguments
    parser.add_argument(
        "--verbosity",
        type=bool,
        default=False,
        help="Print the size of the different dataset",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        path_to_train=args.path_to_train,
        path_to_test=args.path_to_test,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        verbosity=args.verbosity,
    )

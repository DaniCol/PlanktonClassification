"""This module aims to load and process the data."""
# pylint: disable=import-error, no-name-in-module
import os
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from data.preprocessing import DatasetTransformer, apply_preprocessing
from data.dataset_utils import (
    random_split_for_unbalanced_dataset,
    basic_random_split,
    TestLoader,
)


def main(cfg, only_test=False):
    """Main function to call to load and process data

    Args:
        cfg (dict): configuration file

    Returns:
        tuple[DataLoader, DataLoader]: train and validation DataLoader
        DataLoader: test DataLoader
    """
    # DatasetTransformer
    data_transforms = apply_preprocessing(cfg=cfg["DATASET"]["PREPROCESSING"])

    # Load test if necessary
    if only_test:
        # Set test path
        path_to_test = os.path.join(cfg["DATA_DIR"], "test/")

        # Load the test set
        test_dataset = TestLoader(path_to_test)

        test_dataset = DatasetTransformer(
            test_dataset, transforms.Compose(data_transforms["test"])
        )

        # Dataloaders
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg["DATASET"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=cfg["DATASET"]["NUM_THREADS"],
        )

        if cfg["DATASET"]["VERBOSITY"]:
            print(
                f"The test set contains {len(test_loader.dataset)} images,"
                f" in {len(test_loader)} batches"
            )

        return test_loader

    # Set train path
    path_to_train = os.path.join(cfg["DATA_DIR"], "train/")

    # Load the dataset for the training/validation sets
    if cfg["DATASET"]["SMART_SPLIT"]:
        train_dataset, valid_dataset = random_split_for_unbalanced_dataset(
            path_to_train=path_to_train, valid_ratio=cfg["DATASET"]["VALID_RATIO"]
        )
    else:
        train_dataset, valid_dataset = basic_random_split(
            path_to_train=path_to_train, valid_ratio=cfg["DATASET"]["VALID_RATIO"]
        )

    train_dataset = DatasetTransformer(
        train_dataset, transforms.Compose(data_transforms["train"])
    )
    valid_dataset = DatasetTransformer(
        valid_dataset, transforms.Compose(data_transforms["valid"])
    )

    # Dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )

    if cfg["DATASET"]["VERBOSITY"]:
        print(
            f"The train set contains {len(train_loader.dataset)} images,"
            f" in {len(train_loader)} batches"
        )
        print(
            f"The validation set contains {len(valid_loader.dataset)} images,"
            f" in {len(valid_loader)} batches"
        )
    return train_loader, valid_loader

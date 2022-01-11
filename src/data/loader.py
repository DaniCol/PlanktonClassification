"""This module aims to load and process the data."""
# pylint: disable=import-error, no-name-in-module
import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from data.preprocessing import (
    DatasetTransformer,
    apply_preprocessing,
    compute_mean_std,
    mean_std_image_path,
)
from data.dataset_utils import (
    random_split_for_unbalanced_dataset,
    basic_random_split,
    TestLoader,
    create_weighted_sampler,
)


def main(cfg):  # pylint: disable=too-many-locals
    """Main function to call to load and process data

    Args:
        cfg (dict): configuration file

    Returns:
        tuple[DataLoader, DataLoader]: train and validation DataLoader
        DataLoader: test DataLoader
    """

    # Set test path
    path_to_train = os.path.join(cfg["DATA_DIR"], "train/")
    path_to_test = os.path.join(cfg["DATA_DIR"], "test/")

    # Load the dataset for the training/validation sets
    if cfg["DATASET"]["SMART_SPLIT"]:
        train_dataset, valid_dataset, targets = random_split_for_unbalanced_dataset(
            path_to_train=path_to_train, valid_ratio=cfg["DATASET"]["VALID_RATIO"]
        )
        # Create a weighted sampler
        train_sampler = create_weighted_sampler(targets=targets)
    else:
        train_dataset, valid_dataset = basic_random_split(
            path_to_train=path_to_train, valid_ratio=cfg["DATASET"]["VALID_RATIO"]
        )
    # Load the test set
    test_dataset = TestLoader(path_to_test)

    # Compute mean and std images of the training dataset and save them
    if cfg["DATASET"]["PREPROCESSING"]["NORMALIZE"]["ACTIVE"]:
        name_mean, name_std = mean_std_image_path(cfg=cfg["DATASET"]["PREPROCESSING"])

        if not os.path.isfile(os.path.join("./data/normalized_images", name_mean)):
            data_transforms_for_norm = apply_preprocessing(
                cfg=cfg["DATASET"]["PREPROCESSING"], for_norm=True
            )

            normalizing_dataset = DatasetTransformer(
                train_dataset, transforms.Compose(data_transforms_for_norm["train"])
            )
            normalizing_loader = DataLoader(
                dataset=normalizing_dataset,
                batch_size=cfg["DATASET"]["BATCH_SIZE"],
                num_workers=cfg["DATASET"]["NUM_THREADS"],
            )

            # Compute mean and variance from the training set
            mean_img, std_img = compute_mean_std(normalizing_loader)

            # Save images
            torch.save(mean_img, os.path.join("./data/normalized_images", name_mean))
            torch.save(std_img, os.path.join("./data/normalized_images", name_std))

            # Clear memory
            del normalizing_dataset
            del normalizing_loader
            del data_transforms_for_norm

    # DatasetTransformer
    data_transforms = apply_preprocessing(cfg=cfg["DATASET"]["PREPROCESSING"])

    train_dataset = DatasetTransformer(
        train_dataset, transforms.Compose(data_transforms["train"])
    )
    valid_dataset = DatasetTransformer(
        valid_dataset, transforms.Compose(data_transforms["valid"])
    )
    test_dataset = DatasetTransformer(
        test_dataset, transforms.Compose(data_transforms["test"])
    )

    # Dataloaders
    if cfg["DATASET"]["SMART_SPLIT"]:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg["DATASET"]["BATCH_SIZE"],
            num_workers=cfg["DATASET"]["NUM_THREADS"],
            sampler=train_sampler,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg["DATASET"]["BATCH_SIZE"],
            num_workers=cfg["DATASET"]["NUM_THREADS"],
            shuffle=True,
        )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg["TEST"]["BATCH_SIZE"],
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
        print(
            f"The test set contains {len(test_loader.dataset)} images,"
            f" in {len(test_loader)} batches"
        )

    return train_loader, valid_loader, test_loader

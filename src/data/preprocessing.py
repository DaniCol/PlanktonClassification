"""This file contains all functions related to preprocessing."""
# pylint: disable=import-error
import torch
import torchvision.transforms as transforms

# Data transformation


class DatasetTransformer(torch.utils.data.Dataset):
    """Apply transformation to a torch Dataset
    """

    def __init__(self, base_dataset, transform):
        """Initialize DatasetTransformer class

        Args:
            base_dataset (torchvision.datasets.folder.ImageFolder): Image dataset
            transform (torchvision.transforms.Compose): List of transformation to apply
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


class SquarePad:  # pylint: disable=too-few-public-methods
    """This class aims to resize images with zero padding by centering the images
    """

    def __init__(self, new_height, new_width, value=1.0) -> None:
        """Initialize the SquarePad class

        Args:
            new_height (int): Height of the output image
            new_width (int): Width of the output image
            value (float): Fill value for 'constant' padding
        """
        self.new_height = new_height
        self.new_width = new_width
        self.value = value

    def __call__(self, image):
        """Return the reshaped image

        Args:
            image (torch.Tensor): Image to resize

        Returns:
            torch.Tensor: Reshaped image (... x new_height x new_width)
        """
        height, width = image.shape[1:]

        left_pad = int((self.new_width - width) / 2)
        rigth_pad = self.new_width - (left_pad + width)

        top_pad = int((self.new_height - height) / 2)
        bottom_pad = self.new_height - (top_pad + height)

        padding = (left_pad, top_pad, rigth_pad, bottom_pad)
        return transforms.functional.pad(image, padding, self.value, "constant")


def resizing_strategy(cfg, data_transforms):
    """Choose the method to resize all images

    Args:
        cfg (dict): preprocessing configuration file
        data_transforms (dict): data transformation for the different dataset

    Returns:
        dict: data transformation for the different dataset
    """
    if cfg["SQUARE_PADDING"]["ACTIVE"]:
        input_size = cfg["SQUARE_PADDING"]["INPUT_SIZE"]
        padding_value = 0.0 if cfg["REVERSE_COLOR"] else 1.0
        for set_ in data_transforms:
            data_transforms[set_].append(
                SquarePad(
                    new_height=input_size, new_width=input_size, value=padding_value
                )
            )

    elif cfg["RESIZE_CROP"]["ACTIVE"]:
        input_size = cfg["RESIZE_CROP"]["INPUT_SIZE"]
        for set_ in data_transforms:
            data_transforms[set_].append(transforms.Resize(size=input_size))
            data_transforms[set_].append(transforms.CenterCrop(size=input_size))

    elif cfg["RESIZE"]["ACTIVE"]:
        input_size = cfg["RESIZE"]["INPUT_SIZE"]
        for set_ in data_transforms:
            data_transforms[set_].append(
                transforms.Resize(size=(input_size, input_size))
            )

    return data_transforms


def data_augmentation_strategy(cfg, data_transforms):
    """Do we apply data augmentation methods.

    Args:
        cfg (dict): preprocessing configuration file
        data_transforms (dict): data transformation for the different dataset

    Returns:
        dict: data transformation for the different dataset
    """
    if cfg["FLIP"]["HORIZONTAL"]["ACTIVE"]:
        data_transforms["train"].append(
            transforms.RandomHorizontalFlip(p=cfg["FLIP"]["HORIZONTAL"]["VALUE"])
        )

    if cfg["FLIP"]["VERTICAL"]["ACTIVE"]:
        data_transforms["train"].append(
            transforms.RandomVerticalFlip(p=cfg["FLIP"]["HORIZONTAL"]["VALUE"])
        )

    if cfg["AFFINE"]["ACTIVE"]:
        data_transforms["train"].append(
            transforms.RandomAffine(
                degrees=cfg["AFFINE"]["DEGREES"],
                translate=tuple(cfg["AFFINE"]["TRANSLATE"]),
            )
        )
    return data_transforms


def apply_preprocessing(cfg):
    """This function sets all the transformation to apply to the different dataset.

    Args:
        cfg (dict): Preprocessing config dict

    Returns:
        dict: data transformation for train, valid and test dataset
    """
    data_transforms = {
        "train": [transforms.Grayscale(num_output_channels=1)],
        "valid": [transforms.Grayscale(num_output_channels=1)],
        "test": [transforms.Grayscale(num_output_channels=1)],
    }

    for set_ in data_transforms:
        data_transforms[set_].append(transforms.ToTensor())

    # White or black background (if true => black background)
    if cfg["REVERSE_COLOR"]:
        for set_ in data_transforms:
            data_transforms[set_].append(transforms.Lambda(lambda x: 1 - x))

    # Resize the images
    data_transforms = resizing_strategy(cfg=cfg, data_transforms=data_transforms)

    # Data augmentation
    data_transforms = data_augmentation_strategy(
        cfg=cfg, data_transforms=data_transforms
    )

    return data_transforms

"""Define useful functions and classes for the loader module."""
import torch  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error


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

    def __init__(self, new_height, new_width) -> None:
        """Initialize the SquarePad class

        Args:
            new_height (int): Height of the output image
            new_width (int): Width of the output image
        """
        self.new_height = new_height
        self.new_width = new_width

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
        return transforms.functional.pad(image, padding, 1.0, "constant")

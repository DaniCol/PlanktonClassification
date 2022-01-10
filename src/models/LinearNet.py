# pylint: disable=invalid-name
"""Create a Linear NN."""
import torch
import torch.nn as nn


class LinearNet(nn.Module):  # pylint: disable=too-few-public-methods
    """Define our Linear model"""

    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()  # pylint: disable=super-with-arguments
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):  # pylint : disable=invalid-name, redefined-outer-name
        """Define the forward method

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: labels
        """
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


if __name__ == "__main__":
    x = torch.randn(2, 1, 300, 300)
    model = LinearNet(300 * 300, 10)
    output = model(x)

    # Should return torch.Size([2, 10])
    print(output.shape)

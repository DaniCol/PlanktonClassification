# pylint: disable=invalid-name
"""Create a Conv NN."""
import torch
import torch.nn as nn


class ConvNet(nn.Module):  # pylint: disable=too-few-public-methods
    """Define our Convolutional model"""

    def __init__(self, input_size, num_classes):
        super(ConvNet, self).__init__()  # pylint: disable=super-with-arguments
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )
        self.classifier = nn.Linear(
            (input_size // 4) * (input_size // 4) * 32, num_classes
        )

    def forward(self, x):  # pylint : disable=invalid-name, redefined-outer-name
        """Define the forward method

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: labels
        """
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 1, 300, 300)
    model = ConvNet(300, 10)
    output = model(x)
    # Should return torch.Size([2, 10])
    print(output.shape)

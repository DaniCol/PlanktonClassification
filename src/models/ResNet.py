# pylint: disable=invalid-name
"""Create a Residual NN."""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):  # pylint: disable=too-few-public-methods
    """Define our Residual Block"""

    def __init__(
        self, in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False
    ):  # pylint: disable=too-many-arguments
        super(ResidualBlock, self).__init__()  # pylint: disable=super-with-arguments
        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        """Define the forward method

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: labels
        """
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x


class ResNet(
    nn.Module
):  # pylint: disable=too-few-public-methods disable=too-many-instance-attributes
    """Define our Residual Network model"""

    def __init__(self, input_size, num_classes):
        super(ResNet, self).__init__()  # pylint: disable=super-with-arguments

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(1, 1), ResidualBlock(64, 64), ResidualBlock(64, 64, 2)
        )

        self.block3 = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 128, 2))

        self.block4 = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256, 2))
        self.block5 = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512, 2))

        self.avgpool = nn.AvgPool2d(2)
        # vowel_diacritic
        self.fc1 = nn.Linear(512, 11)
        # grapheme_root
        self.fc2 = nn.Linear(512, 168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512, 7)

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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3


if __name__ == "__main__":
    x = torch.randn(2, 1, 300, 300)
    model = ResNet(300, 10)
    output = model(x)
    # Should return torch.Size([2, 10])
    print(output.shape)

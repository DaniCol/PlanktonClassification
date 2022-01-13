# pylint: disable=invalid-name
"""Create a Residual NN."""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):  # pylint: disable=too-few-public-methods
    """Define our Residual model Block"""

    def __init__(
        self, in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False
    ):  # pylint: disable=too-many-arguments
        super(ResidualBlock, self).__init__()  # pylint: disable=super-with-arguments
        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
        x = nn.ReLU(inplace=True)(x)
        return x


class ResNet(nn.Module):  # pylint: disable=too-few-public-methods
    """Define our Residual model Network"""

    def __init__(self, input_size, num_classes):  # pylint: disable=unused-argument
        super(ResNet, self).__init__()  # pylint: disable=super-with-arguments

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(1, 1), ResidualBlock(64, 64), ResidualBlock(64, 64, 2)
        )

        self.block3 = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 128, 2))

        self.block4 = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256, 2))
        self.block5 = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512, 2))

        self.avgpool = nn.AvgPool2d(2)

        self.classifier = nn.Linear(12800, num_classes)

    def forward(self, x):
        """Define the forward method

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: labels
        """
        # print("I started my adventure as:")
        # print(x.shape)
        x = self.block1(x)
        # print("I survived block 1 and left as")
        # print(x.shape)
        x = self.block2(x)
        # print("I survived block 2 and left as")
        # print(x.shape)
        x = self.block3(x)
        # print("I survived block 3 and left as")
        # print(x.shape)
        x = self.block4(x)
        # print("I survived block 4 and left as")
        # print(x.shape)
        x = self.block5(x)
        # print("I survived block 5 and left as")
        # print(x.shape)
        x = self.avgpool(x)
        # print("I survived the avg pooling and left as")
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print("I changed using the view to:")
        # print(x.shape)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 1, 300, 300)
    model = ResNet(input_size=300, num_classes=10)
    output = model(x)
    # Should return torch.Size([2, 10])
    print(output.shape)

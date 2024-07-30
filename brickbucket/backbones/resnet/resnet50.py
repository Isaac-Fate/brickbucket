import torch
from torch import Tensor
from torch import nn

from .bottleneck import Bottleneck


class ResNet50(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            Bottleneck(
                in_channels=64,
                middle_channels=64,
            ),
            *(
                Bottleneck(
                    in_channels=256,
                    middle_channels=64,
                )
                for _ in range(2)
            ),
        )

        self.layer2 = nn.Sequential(
            Bottleneck(
                in_channels=256,
                middle_channels=128,
                apply_downsample=True,
            ),
            *(
                Bottleneck(
                    in_channels=512,
                    middle_channels=128,
                )
                for _ in range(3)
            ),
        )

        self.layer3 = nn.Sequential(
            Bottleneck(
                in_channels=512,
                middle_channels=256,
                apply_downsample=True,
            ),
            *(
                Bottleneck(
                    in_channels=1024,
                    middle_channels=256,
                )
                for _ in range(5)
            ),
        )

        self.layer4 = nn.Sequential(
            Bottleneck(
                in_channels=1024,
                middle_channels=512,
                apply_downsample=True,
            ),
            *(
                Bottleneck(
                    in_channels=2048,
                    middle_channels=512,
                )
                for _ in range(2)
            ),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2048, 1000)

    def forward(self, x: Tensor) -> Tensor:

        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification layer

        # Adaptive average pooling
        x = self.avg_pool(x)

        # Flatten the output starting from the 2nd dimension (channels)
        # 1st dimension is the batch size
        x = torch.flatten(x, start_dim=1)

        # Fullly connected layer to produce probabilities for each class
        x = self.fc(x)

        return x

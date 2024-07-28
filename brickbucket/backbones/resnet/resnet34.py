import torch
from torch import Tensor
from torch import nn

from .basic_block import BasicBlock


class ResNet34(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        # First convolutional block

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

        # Layers consisting of residual blocks

        self.layer1 = nn.Sequential(
            *(BasicBlock(64) for _ in range(3)),
        )

        self.layer2 = nn.Sequential(
            BasicBlock(
                64,
                apply_downsample=True,
            ),
            *(BasicBlock(128) for _ in range(3)),
        )

        self.layer3 = nn.Sequential(
            BasicBlock(
                128,
                apply_downsample=True,
            ),
            *(BasicBlock(256) for _ in range(5)),
        )

        self.layer4 = nn.Sequential(
            BasicBlock(
                256,
                apply_downsample=True,
            ),
            *(BasicBlock(512) for _ in range(2)),
        )

        # Classification layer

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 1000 is the number of classes to predict
        self.fc = nn.Linear(512, 1000)

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
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

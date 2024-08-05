import torch
from torch import Tensor
from torch import nn

from .basic_block_stack import BasicBlockStack


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

        self.layer1 = BasicBlockStack(
            64,
            num_blocks=3,
        )

        self.layer2 = BasicBlockStack(
            64,
            num_blocks=4,
            apply_downsample_in_first_block=True,
        )

        self.layer3 = BasicBlockStack(
            128,
            num_blocks=6,
            apply_downsample_in_first_block=True,
        )

        self.layer4 = BasicBlockStack(
            256,
            num_blocks=3,
            apply_downsample_in_first_block=True,
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

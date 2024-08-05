import torch
from torch import Tensor
from torch import nn

from .bottleneck_stack import BottleneckStack


class ResNet50(nn.Module):

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

        # Layers consisting of bottlenecks

        self.layer1 = BottleneckStack(
            64,
            256,
            num_blocks=3,
        )

        self.layer2 = BottleneckStack(
            256,
            512,
            num_blocks=4,
            apply_downsample_in_first_block=True,
        )

        self.layer3 = BottleneckStack(
            512,
            1024,
            num_blocks=6,
            apply_downsample_in_first_block=True,
        )

        self.layer4 = BottleneckStack(
            1024,
            2048,
            num_blocks=3,
            apply_downsample_in_first_block=True,
        )

        # Classification layer

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

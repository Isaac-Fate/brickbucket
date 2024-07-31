from typing import Optional

from torch import Tensor
from torch import nn


class Bottleneck(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        middle_channels: Optional[int] = None,
        apply_downsample: bool = False,
    ) -> None:

        super().__init__()

        # Attributes

        # Number of input and output channels
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Number of input and output channels of the middle convolutional layer
        self._middle_channels = middle_channels

        if middle_channels is not None:
            self._middle_channels = middle_channels

        # Number of middle channels is not specified
        else:
            # Number of middle channels is 1/4 of the number of output channels
            self._middle_channels, remainder = divmod(out_channels, 4)

            # Number of output channels must be a multiple of 4
            if remainder != 0:
                raise ValueError(
                    "the number of output channels must be a multiple of 4 since the number of middle channels is not specified"
                )

        # Ensure that the number of middle channels is less than or equal to the number of input and output channels
        if (
            self.middle_channels > self.in_channels
            or self.middle_channels > self.out_channels
        ):
            raise ValueError(
                "the number of middle channels must be less than or equal to the number of input and output channels, otherwise the structure is not a bottleneck"
            )

        # Whether to apply downsampling
        self._apply_downsample = apply_downsample

        # Stride of the middle convolutional layer
        conv2_stride = 2 if apply_downsample else 1

        # Layers

        # 1-by-1 convolution to reduce dimensionality

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.middle_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(self.middle_channels)

        # 3-by-3 convolution for extracting features
        # from smaller spatial dimensions

        self.conv2 = nn.Conv2d(
            self.middle_channels,
            self.middle_channels,
            kernel_size=3,
            stride=conv2_stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(self.middle_channels)

        # 1-by-1 convolution to restore dimensionality

        self.conv3 = nn.Conv2d(
            self.middle_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(self.out_channels)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

        if self.in_channels != self.out_channels or self.apply_downsample:

            # Projection shortcut
            #
            # If downsampling is not required
            # and the numbers of input and output channels are not the same,
            # then this layer should really be named as `shortcut` (projection shortcut)
            # instead of `downsample`
            #
            # However, the pretrained weights from torchvision use `downsample`,
            # so we leave it as is
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=conv2_stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
            )

    @property
    def in_channels(self) -> int:
        """
        The number of input channels.
        """

        return self._in_channels

    @property
    def middle_channels(self) -> int:
        """
        The number of input and output channels of the middle convolutional layer.
        If not specified, the number of middle channels is 1/4 of the number of output channels.
        """

        return self._middle_channels

    @property
    def out_channels(self) -> int:
        """
        The number of output channels.
        It must be a multiple of 4 if the number of middle channels is not specified.
        """

        return self._out_channels

    @property
    def apply_downsample(self) -> bool:
        """
        Whether to apply downsampling.
        """

        return self._apply_downsample

    @property
    def shortcut(self) -> nn.Module:
        """
        The projection shortcut layer.
        - If downsampling is required, then stride of the convolutional layer is 2, otherwise, stride is 1
        - If downsampling is not required, and the numbers of input and output channels are the same, then this is an identity map
        """

        if hasattr(self, "downsample"):
            return self.downsample

        return nn.Identity()

    def forward(self, x: Tensor) -> Tensor:

        # Store the input for the shortcut connection
        x0 = x

        # 1-by-1 convolution to reduce dimensionality
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 3-by-3 convolution for extracting features
        # from smaller spatial dimensions
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 1-by-1 convolution to restore dimensionality
        x = self.conv3(x)
        x = self.bn3(x)

        # Add the shortcut
        x += self.shortcut(x0)

        # ReLU
        x = self.relu(x)

        return x

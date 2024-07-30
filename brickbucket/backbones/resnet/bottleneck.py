from torch import Tensor
from torch import nn


class Bottleneck(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        middle_channels: int,
        apply_downsample: bool = False,
    ) -> None:

        super().__init__()

        # Attributes

        # Number of input channels
        self._in_channels = in_channels

        # Number of input and output channels of the middle convolutional layer
        self._middle_channels = middle_channels

        # Number of output channels is 4 times the number of middle channels
        self._out_channels = 4 * middle_channels

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
        """

        return self._middle_channels

    @property
    def out_channels(self) -> int:
        """
        The number of output channels.
        It is 4 times the number of middle channels
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

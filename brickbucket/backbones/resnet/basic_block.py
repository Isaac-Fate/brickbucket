from torch import Tensor
from torch import nn


class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        *,
        apply_downsample: bool = False,
    ) -> None:

        super().__init__()

        # Attributes

        self._in_channels = in_channels
        self._apply_downsample = apply_downsample

        # Downsampling is required
        if apply_downsample:
            # The first convolutional layer should have stride 2
            first_conv_stride = 2

            # The number of output channels should be
            # double the number of input channels
            self._out_channels = in_channels * 2

        # No downsampling is required
        else:
            # The first convolutional layer should have stride 1
            first_conv_stride = 1

            # The number of output channels should be
            # equal to the number of input channels
            self._out_channels = in_channels

        # Layers

        self.conv1 = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=3,
            stride=first_conv_stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(self.out_channels)

        # Reusable inplace ReLU layer
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(self.out_channels)

        # The 1x1 convolutional layer is used to match the dimensions
        # of the shortcut connection
        # This layer is present only if downsampling is required
        if apply_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=first_conv_stride,
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
    def out_channels(self) -> int:
        """
        The number of output channels.
        """

        return self._out_channels

    @property
    def apply_downsample(self) -> bool:
        """
        Whether to apply downsampling to the input feature map.
        If true, a `downsample` layer is present in this module
        to match the dimensions of the shortcut connection.
        And the `out_channels` will be twice the `in_channels`.
        """

        return self._apply_downsample

    def forward(self, x: Tensor) -> Tensor:

        # Store the input for the shortcut connection
        x0 = x

        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x)

        # Apply the projection to match the dimensions
        # Otherwise, apply the identity map (by doing nothing)
        if self.apply_downsample:
            x0 = self.downsample(x0)

        # Shortcut connection
        # It is applied before the activation layer
        x += x0

        # ReLU activation
        x = self.relu(x)

        return x

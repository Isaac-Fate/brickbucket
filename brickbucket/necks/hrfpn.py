import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from ..types import PoolingType
from ..basics import ConvBlock


class HRFPN(nn.Module):

    def __init__(
        self,
        *,
        in_channels_tuple: tuple[int],
        out_channels: int,
        num_pyramid_levels: int = 5,
        pooling_type: PoolingType = PoolingType.AVG,
    ) -> None:

        super().__init__()

        self._in_channels_tuple = in_channels_tuple
        self._out_channels = out_channels
        self._num_pyramid_levels = num_pyramid_levels

        # The number of channels of the base pyramid level is
        # the sum of the numbers of all input channels
        base_pyramid_lavel_channels = sum(in_channels_tuple)

        # Convolutional layer to reduce the number of channels of the base pyramid level
        self.reduction_conv = ConvBlock(
            base_pyramid_lavel_channels,
            self.out_channels,
            kernel_size=1,
            with_batch_norm=False,
            with_activation=False,
        )

        # Functional pooling layer
        match pooling_type:

            case PoolingType.MAX:
                self.pool = F.max_pool2d

            case PoolingType.AVG:
                self.pool = F.avg_pool2d

        # Convolutional layers to apply to each pyramid level
        self.fpn_convs = nn.ModuleList()

        for _ in range(self.num_pyramid_levels):

            # Keep the spatial dimensions unchanged
            self.fpn_convs.append(
                ConvBlock(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                    with_batch_norm=False,
                    with_activation=False,
                )
            )

    @property
    def in_channels_tuple(self) -> tuple[int]:
        """
        A tuple consisting of the number of input channels for each branch.
        """

        return self._in_channels_tuple

    @property
    def out_channels(self) -> int:
        """
        The number of output channels.
        """

        return self._out_channels

    @property
    def num_pyramid_levels(self) -> int:
        """
        The number of pyramid levels.
        """

        return self._num_pyramid_levels

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor]:

        # Ouputs to form the base pyramid level
        outputs: list[Tensor] = []

        for index, input in enumerate(inputs):

            # Append the first (largest) input feature maps without upsampling
            if index == 0:
                outputs.append(input)
                continue

            # Upsample the input feature maps
            output = F.interpolate(
                input,
                scale_factor=2**index,
                mode="bilinear",
            )

            outputs.append(output)

        # Make the base pyramid level
        base_pyramid_level = torch.cat(outputs, dim=1)

        # Reduce the number of channels of the base pyramid level
        base_pyramid_level = self.reduction_conv(base_pyramid_level)

        # Feature pyramid levels associated with different resolutions
        pyramid_levels: list[Tensor] = [base_pyramid_level]

        for index in range(1, self.num_pyramid_levels):

            # Downsample the base pyramid level to
            # make a smaller pyramid level
            pyramid_level = self.pool(
                base_pyramid_level,
                kernel_size=2**index,
                stride=2**index,
            )

            # Append the pyramid level
            pyramid_levels.append(pyramid_level)

        # Apply the convolutional layers to each pyramid level
        for index in range(len(pyramid_levels)):

            # Get the pyramid level
            pyramid_level = pyramid_levels[index]

            # Get the convolutional layer
            fpn_conv = self.fpn_convs[index]

            # Apply the convolutional layer, and
            # set the output to the pyramid level
            pyramid_levels[index] = fpn_conv(pyramid_level)

        return tuple(pyramid_levels)

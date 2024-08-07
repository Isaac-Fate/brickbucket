from typing import Optional
import warnings

from torch import Tensor
from torch import nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        with_batch_norm: bool = True,
        with_activation: bool = True,
    ) -> None:

        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

        # Set default value for bias
        if bias is None:

            if with_batch_norm:

                self._bias = False

            else:
                self._bias = True

        elif bias is True and with_batch_norm is True:

            # Set bias to False
            self._bias = False

            # Display warning
            warnings.warn(
                "the bias of the convolution is set to False even its provided value is True because there is a BatchNorm layer after it",
            )

        else:
            self._bias = bias

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

        if with_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        if with_activation:
            self.activate = nn.ReLU(inplace=True)

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
    def kernel_size(self) -> int:
        """
        The kernel size.
        """

        return self._kernel_size

    @property
    def stride(self) -> int:
        """
        The stride.
        """

        return self._stride

    @property
    def padding(self) -> int:
        """
        The padding.
        """

        return self._padding

    @property
    def dilation(self) -> int:
        """
        The dilation.
        """

        return self._dilation

    @property
    def groups(self) -> int:
        """
        The groups.
        """

        return self._groups

    @property
    def bias(self) -> bool:
        """
        The bias.
        """

        return self._bias

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv(x)

        if hasattr(self, "bn"):
            x = self.bn(x)

        if hasattr(self, "activate"):
            x = self.activate(x)

        return x

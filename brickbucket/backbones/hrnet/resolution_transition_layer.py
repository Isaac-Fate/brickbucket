from torch import nn


class ResolutionTransitionLayer(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        log_scale_factor: int = 0,
    ) -> None:

        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        # The spatial dimensions remain the same
        if log_scale_factor == 0:

            # The channels are also the same
            if self.in_channels == self.out_channels:

                # Just apply an identity map to the input feature maps
                self.append(nn.Identity())

            # Only the number of channels are different
            else:

                # Apply a stride 1 convolution
                self.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.in_channels,
                            self.out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.out_channels),
                        nn.ReLU(inplace=True),
                    )
                )

        # The spatial dimensions need to increase
        elif log_scale_factor > 0:

            # Calcuate the scale factor
            scale_factor = 2**log_scale_factor

            # 1-by-1 convolution to match the number of channels
            self.append(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=False,
                )
            )

            self.append(nn.BatchNorm2d(self.out_channels))

            # Apply upsampling
            self.append(
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode="nearest",
                )
            )

        # The spatial dimensions need to reduce
        # The only case here is when log_scale_factor < 0
        else:

            # Input and output channels of each downsampling convolution
            in_channels = self.in_channels
            out_channels = self.out_channels

            # Number of downsampling stages to apply
            num_downsample_stages = -log_scale_factor

            for index in range(num_downsample_stages):

                # Downsampling layer
                downsample = nn.Sequential()

                # Apply a stride 2 3-by-3 convolution for downsampling
                downsample.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )

                # Normalization
                downsample.append(nn.BatchNorm2d(out_channels))

                # Apply ReLU for intermediate downsampling stages
                if index < num_downsample_stages - 1:

                    # ReLU
                    downsample.append(nn.ReLU(inplace=True))

                # Append the downsampling layer
                self.append(downsample)

                # Update number of input channels
                in_channels = out_channels

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

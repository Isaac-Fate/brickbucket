import torch
from torch import Tensor
from torch import nn
from functools import reduce

from .resolution_transition_layer import ResolutionTransitionLayer


class FuseLayer(nn.ModuleList):

    def __init__(
        self,
        *,
        in_channels_tuple: tuple[int],
        out_channels: int,
        log_scale_factors: tuple[int],
    ) -> None:

        super().__init__()

        # Number of input channels for each branch
        self._in_channels_tuple = in_channels_tuple

        # Number of output channels
        self._out_channels = out_channels

        # Log scale factors
        # Each log scale factor indicates how the input branch is transformed to the output resolution
        self._log_scale_factors = log_scale_factors

        # Ensure that the number of input channels is equal to the number of log scale factors
        if len(in_channels_tuple) != len(log_scale_factors):
            raise ValueError(
                "the number of input channels must be equal to the number of log scale factors"
            )

        # Make the transition layers
        for index, in_channels in enumerate(self.in_channels_tuple):

            # Get the log scale factor
            log_scale_factor = self.log_scale_factors[index]

            # Add the resolution transition layer
            self.append(
                ResolutionTransitionLayer(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    log_scale_factor=log_scale_factor,
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
    def log_scale_factors(self) -> tuple[int]:
        """
        A tuple consisting of the log scale factors.
        """

        return self._log_scale_factors

    @property
    def num_branches(self) -> int:
        """
        The number of input branches.
        Each branch handles a certain resolution level of feature maps.
        """

        return len(self.in_channels_tuple)

    def forward(self, inputs: list[Tensor]) -> Tensor:

        # Initialize an empty list to store the multiscale outputs from each branch
        outputs = []

        for index, input in enumerate(inputs):

            # Get the transition layer
            transition_layer: ResolutionTransitionLayer = self[index]

            # Forward the input through the transition layer
            output = transition_layer.forward(input)

            # Collect the output
            outputs.append(output)

        # Fuse the outputs
        output = reduce(torch.add, outputs)

        # Apply activation
        output = nn.ReLU(inplace=True)(output)

        return output

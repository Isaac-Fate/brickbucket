from torch import Tensor
from torch import nn

from .resolution_transition_layer import ResolutionTransitionLayer


class ResolutionSplittingLayer(nn.ModuleList):

    def __init__(
        self,
        in_channels,
        *,
        out_channels_tuple: tuple[int],
        log_scale_factors: tuple[int],
    ) -> None:

        super().__init__()

        self._in_channels = in_channels
        self._out_channels_tuple = out_channels_tuple

        for index, log_scale_factor in enumerate(log_scale_factors):

            # Get the number of output channels
            out_channels = out_channels_tuple[index]

            # Add the resolution transition layer
            self.append(
                ResolutionTransitionLayer(
                    in_channels,
                    out_channels,
                    log_scale_factor=log_scale_factor,
                )
            )

    @property
    def in_channels(self) -> int:
        """
        The number of input channels.
        """

        return self._in_channels

    @property
    def out_channels_tuple(self) -> tuple[int]:
        """
        A tuple consisting of the number of output channels associated with each resolution.
        """

        return self._out_channels_tuple

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Produces the output feature maps associated with each resolution.

        Parameters
        ----------
        x : Tensor
            Shape: (N, C, H, W)

        Returns
        -------
        list[Tensor]
            Each output feature maps is associated with a resolution.
            The number of channels is as specified.
            And the height and width may vary depending on the resolution.

            Shape:
            - (N, C1, H1, W1)
            - (N, C2, H2, W2)
            - ...
        """

        # Output feature maps associated with each resolution
        outputs: list[Tensor] = []

        resolution_transition_layer: ResolutionTransitionLayer
        for resolution_transition_layer in self:

            # Forward the input
            output = resolution_transition_layer(x)

            # Save the output
            outputs.append(output)

        return outputs

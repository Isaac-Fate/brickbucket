from typing import Optional
from torch import Tensor
from torch import nn

from .resolution_transition_layer import ResolutionTransitionLayer


class HRStageTransitionLayer(nn.ModuleList):

    def __init__(
        self,
        *,
        in_channels_tuple: tuple[int],
        out_channels_tuple: Optional[tuple[int]] = None,
    ) -> None:

        super().__init__()

        self._in_channels_tuple = in_channels_tuple

        # Tuple consisting of the number of output channels

        if out_channels_tuple is not None:

            # Check the number of output channels
            if len(out_channels_tuple) != len(in_channels_tuple) + 1:
                raise ValueError(
                    "the number of output channels must be equal to the number of input channels plus 1"
                )

            self._out_channels = out_channels_tuple

        # Create the tuple of output channels if not specified
        else:
            last_in_channels = in_channels_tuple[-1]
            last_out_channels = last_in_channels * 2
            self._out_channels = (*self.in_channels_tuple, last_out_channels)

        # In fact, all transition layers except the last one are just identity maps
        for index, in_channels in enumerate(self.in_channels_tuple):

            # Get the number of output channels
            out_channels = self.out_channels_tuple[index]

            self.append(
                ResolutionTransitionLayer(
                    in_channels,
                    out_channels,
                    apply_activation_to_output=True,
                )
            )

        # Last transition layer
        # The resolution will be reduced, and
        # the number of channels will be doubled
        self.append(
            ResolutionTransitionLayer(
                self.in_channels_tuple[-1],
                self.out_channels_tuple[-1],
                log_scale_factor=-1,
                apply_activation_to_output=True,
            )
        )

    @property
    def in_channels_tuple(self) -> tuple[int]:
        """
        A tuple consisting of the number of input channels associated with each resolution.
        """

        return self._in_channels_tuple

    @property
    def out_channels_tuple(self) -> tuple[int]:
        """
        A tuple consisting of the number of output channels associated with each resolution.
        If the `in_channels_tuple` is (C1, C2, ..., Cn), then the `out_channels_tuple` is (C1, C2, ..., Cn, 2 * Cn).
        """

        return self._out_channels

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:

        outputs: list[Tensor] = []

        for branch_index, input in enumerate(inputs):

            # Get the resolution transition layer
            resolution_transition_layer = self[branch_index]

            # Apply the resolution transition layer
            output = resolution_transition_layer(input)

            # Append the output
            outputs.append(output)

        # Get the last input
        last_input = inputs[-1]

        # Get the last resolution transition layer
        resolution_transition_layer = self[-1]

        # Apply the resolution transition layer
        output = resolution_transition_layer(last_input)

        # Append the output
        outputs.append(output)

        return outputs

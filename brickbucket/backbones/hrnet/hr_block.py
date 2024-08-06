from torch import Tensor
from torch import nn

from ...residual_blocks import BasicBlockStack
from .fuse_layer import FuseLayer


class HRBlock(nn.Module):

    def __init__(
        self,
        *,
        in_channels_tuple: tuple[int],
    ) -> None:

        super().__init__()

        # Number of input channels for each branch
        self._in_channels_tuple = in_channels_tuple

        # Each branch is a basic block stack
        self.branches: nn.ModuleList = nn.ModuleList()

        # Make each branch
        for in_channels in self.in_channels_tuple:
            self.branches.append(
                BasicBlockStack(
                    in_channels,
                    num_blocks=4,
                )
            )

        # Fuse layers

        self.fuse_layers = nn.ModuleList()

        # Each branch is a basic block stack
        branch: BasicBlockStack
        for output_branch_index, branch in enumerate(self.branches):

            # Find the number of output channels of the resolution transition layer
            out_channels = branch.in_channels

            # Number of output channels and log scale factor associated with each resolution
            in_channels_list: list[int] = []
            log_scale_factors: list[int] = []

            for input_branch_index, branch in enumerate(self.branches):

                # Find the number of input channels of the resolution transition layer
                in_channels = branch.out_channels

                # Append the number of input channels
                in_channels_list.append(in_channels)

                # Find the log scale factor of the resolution transition layer
                log_scale_factor = input_branch_index - output_branch_index

                # Append the log scale factor
                log_scale_factors.append(log_scale_factor)

            # Make the fuze layer
            fuze_layer = FuseLayer(
                in_channels_tuple=tuple(in_channels_list),
                out_channels=out_channels,
                log_scale_factors=tuple(log_scale_factors),
            )

            # Add the fuze layer
            self.fuse_layers.append(fuze_layer)

        # ReLU
        self.relu = nn.ReLU()

    @property
    def in_channels_tuple(self) -> tuple[int]:
        """
        A tuple consisting of the number of input channels for each branch.
        """

        return self._in_channels_tuple

    @property
    def out_channels_tuple(self) -> tuple[int]:
        """
        A tuple consisting of the number of output channels for each branch.
        It is the same as the `branch_in_channels_tuple`.
        """

        return self.in_channels_tuple

    @property
    def num_branches(self) -> int:
        """
        The number of branches.
        Each branch handles a certain resolution level of feature maps.
        """

        return len(self.in_channels_tuple)

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        """
        Forwards the inputs associated with each resolution separately, and then fuses the outputs.

        Parameters
        ----------
        inputs : list[Tensor]
            The inputs associated with each resolution.

        Returns
        -------
        list[Tensor]
            The outputs associated with each resolution.
        """

        # Ouputs from each branch
        branch_outputs: list[Tensor] = []

        # Forward the inputs through each branch separately
        for branch_index, input in enumerate(inputs):

            # Get the branch consisting of stack of basic blocks
            branch: BasicBlockStack = self.branches[branch_index]

            # Forward the input through the branch
            output = branch(input)

            # Append the output
            branch_outputs.append(output)

        # Outputs associated with each resolution
        outputs: list[Tensor] = []

        # Fuse the branch outputs
        for branch_index in range(self.num_branches):

            # Get the fuze layer
            fuze_layer: FuseLayer = self.fuse_layers[branch_index]

            # Fuse the branch outputs to produce a output associated with a resolution
            output = fuze_layer.forward(branch_outputs)

            # Append the output
            outputs.append(output)

        return outputs

from typing import Optional
from torch import nn

from .basic_block import BasicBlock


class BasicBlockStack(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        *,
        num_blocks: int = 1,
        apply_downsample_in_first_block: bool = False,
    ) -> None:
        """
        A sequential module made by stacking multiple basic blocks.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        num_blocks : int
            Number of basic blocks.

        apply_downsample_in_first_block : bool
            Whether to apply a downsampling operation in the first basic block.

        Raises
        ------
        ValueError
            If the number of blocks is less than 1.
        """

        if num_blocks < 1:
            raise ValueError("number of blocks must be >= 1")

        # Make the first basic block
        first_block = BasicBlock(
            in_channels,
            apply_downsample=apply_downsample_in_first_block,
        )

        super().__init__(
            first_block,
            *(BasicBlock(first_block.out_channels) for _ in range(num_blocks - 1)),
        )

    @property
    def in_channels(self) -> int:
        """
        The number of input channels.
        """

        # Get the first block
        first_block: BasicBlock = self[0]

        return first_block.in_channels

    @property
    def out_channels(self) -> int:
        """
        The number of output channels.
        """

        # Get the last block
        last_block: BasicBlock = self[-1]

        return last_block.out_channels

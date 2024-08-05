from typing import Optional
from torch import nn

from .bottleneck import Bottleneck


class BottleneckStack(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        middle_channels: Optional[int] = None,
        num_blocks: int = 1,
        apply_downsample_in_first_block: bool = False,
    ) -> None:
        """
        A sequential module made by stacking multiple bottleneck blocks.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        middle_channels : int
            Number of middle channels.

        num_blocks : int
            Number of bottleneck blocks.

        apply_downsample_in_first_block : bool
            Whether to apply a downsampling operation in the first bottleneck block.

        Raises
        ------
        ValueError
            If the number of blocks is less than 1.
        """

        if num_blocks < 1:
            raise ValueError("number of blocks must be >= 1")

        # Make the first bottleneck block
        first_block = Bottleneck(
            in_channels,
            out_channels,
            middle_channels=middle_channels,
            apply_downsample=apply_downsample_in_first_block,
        )

        super().__init__(
            first_block,
            *(
                Bottleneck(
                    out_channels,
                    out_channels,
                    middle_channels=middle_channels,
                )
                for _ in range(num_blocks - 1)
            ),
        )

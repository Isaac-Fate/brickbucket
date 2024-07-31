from typing import Optional
from torch import nn

from .bottleneck import Bottleneck


def make_bottleneck_stack(
    in_channels: int,
    out_channels: int,
    *,
    middle_channels: Optional[int] = None,
    num_blocks: int,
    apply_downsample_in_first_block: bool = False,
) -> nn.Sequential:
    """Makes a stack of bottleneck blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    num_blocks : int
        Number of bottleneck blocks in the stack.

    middle_channels : Optional[int], optional
        Number of input and output channels of the middle convolutional layer, by default None.

    apply_downsample_in_first_block : bool, optional
        Whether to apply downsample in the first bottleneck block, by default False.

    Returns
    -------
    nn.Sequential
        The stack of bottleneck blocks.

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

    return nn.Sequential(
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

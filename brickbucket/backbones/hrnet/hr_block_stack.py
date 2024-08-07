from torch import nn

from .hr_block import HRBlock


class HRBlockStack(nn.Sequential):

    def __init__(
        self,
        *,
        in_channels_tuple: tuple[int],
        num_blocks: int = 1,
    ) -> None:

        super().__init__()

        self._in_channels_tuple = in_channels_tuple
        self._num_blocks = num_blocks

        # Stack the blocks
        for _ in range(self.num_blocks):
            self.append(
                HRBlock(
                    in_channels_tuple=in_channels_tuple,
                )
            )

    @property
    def in_channels_tuple(self) -> tuple[int]:
        """
        A tuple consisting of the number of input channels for each branch.
        """

        return self._in_channels_tuple

    @property
    def num_blocks(self) -> int:
        """
        The number of high resolution blocks.
        """

        return self._num_blocks

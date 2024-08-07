from enum import StrEnum


class PoolingType(StrEnum):
    """
    The type of pooling.
    """

    MAX = "MAX"
    AVG = "AVG"

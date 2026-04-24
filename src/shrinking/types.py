"""
Type aliases used by the package.

:var MatrixLike: Concrete NumPy matrix container accepted internally.
:var MatrixInput: Public matrix input accepted by the API.
:var FixedBlockSizes: Fixed-block descriptor accepted by the API.
"""

from collections.abc import Sequence
from enum import IntEnum
from typing import TypeAlias

import numpy as np


class FixedBlockVariant(IntEnum):
    """
    Describe which diagonal blocks are preserved in the fixed-block
    correlation-matrix specializations.

    The preserved block is copied from ``matrix0`` into the target matrix. Any
    unpreserved diagonal block is replaced by an identity block in the target.
    In the reduced test matrices used by the specialized algorithms, the
    unfixed block still has its diagonal zeroed because the unit diagonal is
    handled separately by the correlation-matrix formulation.
    """

    IDENTITY = 0
    PRESERVE_TRAILING_BLOCK = 1
    PRESERVE_LEADING_BLOCK = 2
    PRESERVE_BOTH_BLOCKS = 3


MatrixLike: TypeAlias = np.ndarray | np.matrix
RealScalar: TypeAlias = int | float
MatrixInput: TypeAlias = MatrixLike | Sequence[Sequence[RealScalar]]
FixedBlockSizes: TypeAlias = int | Sequence[int]

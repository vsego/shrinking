"""
Public package interface for the shrinking algorithms.

:var __version__: Package version string.
"""

from . import backwards_compatibility
from .algorithms.bisection import bisection, bisection_meta
from .algorithms.bisection_fb import (
    bisection_with_fixed_block,
    bisection_with_fixed_block_meta,
)
from .algorithms.gep import gep, gep_meta
from .algorithms.gep_fb import gep_with_fixed_block, gep_with_fixed_block_meta
from .algorithms.newton import newton, newton_meta
from .exceptions import NotConvergingError
from .results import AlgorithmResult
from .s import (
    s, s_with_difference, s_with_fixed_blocks, s_with_identity, s_with_target,
)
from .types import FixedBlockVariant
from .utils import blocks_to_target, check_pos_def
from .version import __version__

__all__ = [
    "AlgorithmResult",
    "FixedBlockVariant",
    "NotConvergingError",
    "__version__",
    "backwards_compatibility",
    "bisection",
    "bisection_meta",
    "bisection_with_fixed_block",
    "bisection_with_fixed_block_meta",
    "blocks_to_target",
    "check_pos_def",
    "gep",
    "gep_meta",
    "gep_with_fixed_block",
    "gep_with_fixed_block_meta",
    "newton",
    "newton_meta",
    "s",
    "s_with_difference",
    "s_with_fixed_blocks",
    "s_with_identity",
    "s_with_target",
]

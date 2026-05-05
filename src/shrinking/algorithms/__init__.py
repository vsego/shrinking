"""
Algorithm implementations exposed by :mod:`shrinking`.
"""

from .bisection import bisection, bisection_meta
from .bisection_fb import (
    bisection_with_fixed_block, bisection_with_fixed_block_meta,
)
from .gep import gep, gep_meta
from .gep_fb import gep_with_fixed_block, gep_with_fixed_block_meta
from .newton import newton, newton_meta

__all__ = [
    "bisection",
    "bisection_meta",
    "bisection_with_fixed_block",
    "bisection_with_fixed_block_meta",
    "gep",
    "gep_meta",
    "gep_with_fixed_block",
    "gep_with_fixed_block_meta",
    "newton",
    "newton_meta",
]

"""
Structured metadata returned by the ``*_meta`` functions.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlgorithmResult:
    """
    Store metadata returned by algorithm variants that expose extra
    information.

    :ivar alpha: The computed shrinking parameter.
    :ivar iterations: Number of iterations used by the algorithm.
    """

    alpha: float
    iterations: int

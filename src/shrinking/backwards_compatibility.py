"""
Compatibility exports for the legacy public API.

This module exposes the original CamelCase and mixed-case function names from
the historical single-file implementation. New code should prefer the public
snake_case API from :mod:`shrinking`.
"""

from .algorithms.bisection import bisection_meta as _bisection_meta
from .algorithms.bisection_fb import (
    bisection_with_fixed_block_meta as _bisection_with_fixed_block_meta,
)
from .algorithms.gep import gep_meta as _gep_meta
from .algorithms.gep_fb import (
    gep_with_fixed_block_meta as _gep_with_fixed_block_meta,
)
from .algorithms.newton import newton_meta as _newton_meta
from .exceptions import NotConvergingError
from .problems import (
    ExplicitTargetProblem,
    FixedBlockProblem,
    initialize_problem as _initialize_problem,
)
from .results import AlgorithmResult
from .s import (
    s as _s,
    s_with_difference as _s_with_difference,
    s_with_fixed_blocks as _s_with_fixed_blocks,
    s_with_identity as _s_with_identity,
    s_with_target as _s_with_target,
)
from .types import FixedBlockSizes, FixedBlockVariant, MatrixInput, MatrixLike
from .utils import (
    blocks_to_target as _blocks_to_target,
    check_matrix0 as initCheckM0,
    check_pos_def as _check_pos_def,
    smallest_eigenvector as x,
)

lastIterations: int = 0

__all__ = [
    "GEP",
    "GEPFB",
    "NotConvergingError",
    "S",
    "SFB",
    "ScorrId",
    "Sdef",
    "Sdif",
    "bisection",
    "bisectionFB",
    "blocks2target",
    "checkPD",
    "initCheckM0",
    "initialize",
    "lastIterations",
    "newton",
    "x",
]

Sdef = _s_with_target
Sdif = _s_with_difference
SFB = _s_with_fixed_blocks
ScorrId = _s_with_identity
blocks2target = _blocks_to_target


def checkPD(matrix: MatrixInput, exception: bool = True) -> bool:
    """
    Backward-compatible wrapper for :func:`shrinking.check_pos_def`.

    :param matrix: Matrix to test.
    :param exception: If ``True``, raise ``ValueError`` for a non-positive-
        definite matrix.
    :return: ``True`` if the matrix is positive definite; otherwise ``False``
        when ``exception`` is ``False``.
    :raises ValueError: If the matrix is not positive definite and
        ``exception`` is ``True``.
    """
    return _check_pos_def(matrix, exception=exception)


def S(
    M0: MatrixInput,
    alpha: float,
    *,
    M1: MatrixInput | None = None,
    dM0M1: MatrixInput | None = None,
    fbs: FixedBlockSizes | None = None,
) -> MatrixLike:
    """
    Backward-compatible wrapper for :func:`shrinking.s`.

    :param M0: Starting matrix.
    :param alpha: Shrinking parameter.
    :param M1: Explicit target matrix.
    :param dM0M1: Precomputed difference ``M0 - M1``.
    :param fbs: Fixed-block descriptor.
    :return: Matrix ``S(alpha)``.
    :raises ValueError: If more than one target specification is provided.
    """
    return _s(
        M0,
        alpha,
        matrix1=M1,
        difference_matrix=dM0M1,
        fixed_block_sizes=fbs,
    )


def _set_last_iterations(result: AlgorithmResult) -> float:
    """
    Store the legacy iteration count and return the shrinking parameter
    ``alpha``.

    :param result: Metadata object returned by a modern ``*_meta`` function.
    :return: Shrinking parameter ``alpha`` extracted from ``result``.
    """
    global lastIterations
    lastIterations = result.iterations
    return result.alpha


def initialize(
    M0: MatrixInput,
    M1: MatrixInput | None,
    fbs: FixedBlockSizes | None,
    checkM0: bool = True,
    buildM1: bool = True,
) -> MatrixLike | bool | None:
    """
    Provide the backward-compatible initialization helper.

    :param M0: Starting matrix.
    :param M1: Explicit target matrix.
    :param fbs: Fixed-block descriptor.
    :param checkM0: Reserved compatibility argument. It is ignored.
    :param buildM1: Whether to build the target matrix when ``fbs``
        is supplied.
    :return: ``None``, ``True``, or the validated target matrix.
    :raises ValueError: If the input specification is inconsistent.
    """
    prepared_problem = _initialize_problem(
        M0,
        M1,
        fbs,
    )
    if prepared_problem is None:
        return None
    if isinstance(prepared_problem, ExplicitTargetProblem):
        return prepared_problem.target_matrix
    if not isinstance(prepared_problem, FixedBlockProblem):
        raise RuntimeError("internal error: unexpected problem type")
    return prepared_problem.target_matrix if buildM1 else True


def bisection(
    M0: MatrixInput,
    *,
    M1: MatrixInput | None = None,
    fbs: FixedBlockSizes | None = None,
    tol: float = 10 ** (-6),
    maxIterations: int | None = 53,
    checkM0: bool = True,
) -> float:
    """
    Backward-compatible wrapper for :func:`shrinking.bisection`.

    :param M0: Starting matrix.
    :param M1: Explicit target matrix.
    :param fbs: Fixed-block descriptor.
    :param tol: Bisection tolerance.
    :param maxIterations: Optional hard limit on the number of iterations.
    :param checkM0: Reserved compatibility argument. It is ignored.
    :return: Optimal shrinking parameter.
    :raises ValueError: If the inputs are invalid or convergence fails.
    """
    return _set_last_iterations(
        _bisection_meta(
            M0,
            matrix1=M1,
            fixed_block_sizes=fbs,
            tol=tol,
            max_iterations=maxIterations,
        ),
    )


def bisectionFB(
    M0: MatrixInput,
    fbSize: int | None = None,
    tol: float = 10 ** (-6),
    which: int | FixedBlockVariant = FixedBlockVariant.PRESERVE_LEADING_BLOCK,
    checkM0: bool = True,
) -> float:
    """
    Provide a backward-compatible wrapper for
    :func:`shrinking.bisection_with_fixed_block`.

    :param M0: Correlation matrix to shrink.
    :param fbSize: Size of the leading principal block.
    :param tol: Bisection tolerance.
    :param which: Fixed-block variant describing which diagonal blocks are
        preserved in the target.
    :param checkM0: Reserved compatibility argument. It is ignored.
    :return: Optimal shrinking parameter.
    :raises ValueError: If the inputs are invalid or convergence fails.
    """
    return _set_last_iterations(
        _bisection_with_fixed_block_meta(M0, fbSize, tol=tol, which=which),
    )


def newton(
    M0: MatrixInput,
    *,
    M1: MatrixInput | None = None,
    fbs: FixedBlockSizes | None = None,
    tol: float = 10 ** (-6),
    maxIterations: int | None = None,
    checkM0: bool = True,
) -> float:
    """
    Backward-compatible wrapper for :func:`shrinking.newton`.

    :param M0: Starting matrix.
    :param M1: Explicit target matrix.
    :param fbs: Fixed-block descriptor.
    :param tol: Newton stopping tolerance.
    :param maxIterations: Optional hard limit on the number of iterations.
    :param checkM0: Reserved compatibility argument. It is ignored.
    :return: Optimal shrinking parameter.
    :raises NotConvergingError: If ``maxIterations`` is exceeded.
    :raises ValueError: If the inputs are invalid.
    """
    return _set_last_iterations(
        _newton_meta(
            M0,
            matrix1=M1,
            fixed_block_sizes=fbs,
            tol=tol,
            max_iterations=maxIterations,
        ),
    )


def GEP(
    M0: MatrixInput,
    *,
    M1: MatrixInput | None = None,
    fbs: FixedBlockSizes | None = None,
    posdefM1: bool = True,
    checkM0: bool = True,
) -> float:
    """
    Backward-compatible wrapper for :func:`shrinking.gep`.

    :param M0: Starting matrix.
    :param M1: Explicit target matrix.
    :param fbs: Fixed-block descriptor.
    :param posdefM1: Whether to use the positive-definite generalized
        eigenvalue formulation. ``False`` is no longer supported.
    :param checkM0: Reserved compatibility argument. It is ignored.
    :return: Optimal shrinking parameter.
    :raises NotImplementedError: If ``posdefM1`` is ``False``.
    :raises ValueError: If the inputs are invalid.
    """
    if not posdefM1:
        raise NotImplementedError(
            "GEP with posdefM1=False is no longer supported, in line with"
            " other algorithms and the aim of keeping this package simple",
        )
    return _set_last_iterations(
        _gep_meta(
            M0,
            matrix1=M1,
            fixed_block_sizes=fbs,
        ),
    )


def GEPFB(
    M0: MatrixInput,
    fbSize: int | None = None,
    which: int | FixedBlockVariant = FixedBlockVariant.PRESERVE_LEADING_BLOCK,
    checkM0: bool = True,
) -> float:
    """
    Backward-compatible wrapper for :func:`shrinking.gep_with_fixed_block`.

    :param M0: Correlation matrix to shrink.
    :param fbSize: Size of the leading principal block.
    :param which: Fixed-block variant describing which diagonal blocks are
        preserved in the target.
    :param checkM0: Reserved compatibility argument. It is ignored.
    :return: Optimal shrinking parameter.
    :raises ValueError: If the inputs are invalid.
    """
    return _set_last_iterations(
        _gep_with_fixed_block_meta(M0, fbSize, which=which),
    )

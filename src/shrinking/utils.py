"""
Utility helpers shared by the public API and the algorithms.
"""

import math
from collections.abc import Sequence
from numbers import Integral
from typing import cast

import numpy as np
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs

from .types import (
    FixedBlockSizes, FixedBlockVariant, MatrixInput, MatrixLike,
)


def _square_matrix_error(argument_name: str) -> ValueError:
    """
    Build the standard validation error for non-square matrices.

    :param argument_name: Public argument name used in the error message.
    :return: Prepared ``ValueError`` instance.
    """
    return ValueError(f"{argument_name} must be a square matrix")


def _empty_matrix_error(argument_name: str) -> ValueError:
    """
    Build the standard validation error for empty matrices.

    :param argument_name: Public argument name used in the error message.
    :return: Prepared ``ValueError`` instance.
    """
    return ValueError(f"{argument_name} must be a non-empty square matrix")


def _symmetric_matrix_error(argument_name: str) -> ValueError:
    """
    Build the standard validation error for nonsymmetric matrices.

    :param argument_name: Public argument name used in the error message.
    :return: Prepared ``ValueError`` instance.
    """
    return ValueError(f"{argument_name} must be a symmetric matrix")


def _finite_matrix_error(argument_name: str) -> ValueError:
    """
    Build the standard validation error for non-finite matrices.

    :param argument_name: Public argument name used in the error message.
    :return: Prepared ``ValueError`` instance.
    """
    return ValueError(f"{argument_name} must contain only finite values")


def _matrix_atol(matrix: MatrixLike) -> float:
    """
    Return a small absolute tolerance for matrix equality checks.

    :param matrix: Matrix whose dtype and scale determine the tolerance.
    :return: Absolute tolerance suitable for roundoff-level comparisons.
    """
    float_matrix = np.asarray(matrix, dtype=float)
    scale = max(1.0, float(np.abs(float_matrix).max(initial=0.0)))
    return 10.0 * np.finfo(float_matrix.dtype).eps * scale


def require_square_matrix(
    matrix: MatrixInput,
    *,
    argument_name: str,
) -> MatrixLike:
    """
    Normalize ``matrix`` and require it to be square.

    :param matrix: Matrix-like value to validate.
    :param argument_name: Public argument name used in error messages.
    :return: Normalized non-empty square matrix.
    :raises ValueError: If ``matrix`` is empty, not two-dimensional and
        square, or contains non-finite entries.
    """
    norm_matrix = normalize_matrix(matrix)
    if len(norm_matrix.shape) != 2:
        raise _square_matrix_error(argument_name)
    rows, cols = norm_matrix.shape
    if rows == 0:
        raise _empty_matrix_error(argument_name)
    if rows != cols:
        raise _square_matrix_error(argument_name)
    if not np.all(np.isfinite(np.asarray(norm_matrix, dtype=float))):
        raise _finite_matrix_error(argument_name)
    return norm_matrix


def require_symmetric_matrix(
    matrix: MatrixInput,
    *,
    argument_name: str,
) -> MatrixLike:
    """
    Normalize ``matrix`` and require it to be square and symmetric.

    :param matrix: Matrix-like value to validate.
    :param argument_name: Public argument name used in error messages.
    :return: Normalized non-empty symmetric matrix.
    :raises ValueError: If ``matrix`` is empty, not square, or not symmetric.
    """
    norm_matrix = require_square_matrix(
        matrix,
        argument_name=argument_name,
    )
    if not np.allclose(
        norm_matrix.T,
        norm_matrix,
        rtol=0.0,
        atol=_matrix_atol(norm_matrix),
    ):
        raise _symmetric_matrix_error(argument_name)
    return norm_matrix


def normalize_matrix(matrix: MatrixInput) -> MatrixLike:
    """
    Normalize a matrix input to a supported concrete NumPy container.

    ``numpy.matrix`` inputs remain matrices. All other supported inputs,
    including ``numpy.ndarray`` and plain nested sequences, are converted to
    arrays. The numerical algorithms work in floating point, so integer inputs
    are promoted to ``float`` during normalization.

    :param matrix: Matrix-like value to normalize.
    :return: Normalized matrix as either ``numpy.ndarray`` or
        ``numpy.matrix``.
    """
    if isinstance(matrix, np.matrix):
        return np.asmatrix(matrix, dtype=float)
    return np.asarray(matrix, dtype=float)


def convert_like(reference: MatrixLike, value: MatrixInput) -> MatrixLike:
    """
    Convert ``value`` to the same broad matrix family as ``reference``.

    :param reference: Matrix whose container type should be mirrored.
    :param value: Matrix-like value to convert.
    :return: ``value`` converted to either ``numpy.ndarray``
        or ``numpy.matrix``.
    """
    if isinstance(reference, np.matrix):
        return np.asmatrix(value)
    return np.asarray(value)


def all_are_matrices(*matrices: MatrixLike) -> bool:
    """
    Return ``True`` when all matrices are ``numpy.matrix`` instances.

    :param matrices: Concrete normalized matrices to inspect.
    :return: ``True`` if every matrix is a ``numpy.matrix``.
    """
    return all(isinstance(matrix, np.matrix) for matrix in matrices)


def align_matrix_pair(
    matrix0: MatrixLike,
    matrix1: MatrixLike,
) -> tuple[MatrixLike, MatrixLike]:
    """
    Convert two normalized matrices to the same broad NumPy container family.

    If both inputs are ``numpy.matrix`` objects, both outputs remain matrices.
    Otherwise both outputs become arrays.

    :param matrix0: First normalized matrix.
    :param matrix1: Second normalized matrix.
    :return: Pair of normalized matrices in a common container family.
    """
    if all_are_matrices(matrix0, matrix1):
        return np.asmatrix(matrix0), np.asmatrix(matrix1)
    return np.asarray(matrix0), np.asarray(matrix1)


def normalize_fixed_block_sizes(
    fixed_block_sizes: FixedBlockSizes,
    *,
    matrix_size: int | None = None,
) -> list[int]:
    """
    Normalize fixed block sizes and validate basic invariants.

    :param fixed_block_sizes: One block size or a sequence of block sizes.
    :param matrix_size: Optional upper bound for the sum of block sizes.
    :return: Normalized block sizes as a list of positive integers.
    :raises ValueError: If any block size is invalid or the sum exceeds
        ``matrix_size``.
    """
    if isinstance(fixed_block_sizes, bool):
        sizes = list()
    elif isinstance(fixed_block_sizes, Integral):
        sizes = [fixed_block_sizes]
    else:
        sizes = list(cast(Sequence[int], fixed_block_sizes))
    if (
        not sizes
        or not all(
            (
                isinstance(size, Integral)
                and not isinstance(size, bool)
                and int(size) > 0
            )
            for size in sizes
        )
    ):
        raise ValueError("fixed block sizes must be positive integers")
    normalized_sizes = [int(size) for size in sizes]
    if matrix_size is not None and sum(normalized_sizes) > matrix_size:
        raise ValueError("fixed block sizes must not exceed the matrix order")
    return normalized_sizes


def is_pos_def(matrix: MatrixInput) -> bool:
    """
    Return ``True`` when ``matrix`` is positive definite.

    :param matrix: Matrix to test.
    :return: ``True`` if Cholesky factorization succeeds, ``False`` otherwise.
    """
    norm_matrix = normalize_matrix(matrix)
    try:
        potrf, = get_lapack_funcs(("potrf",), (norm_matrix,))
        info = potrf(
            norm_matrix,
            lower=False,
            overwrite_a=False,
            clean=False,
        )
    except Exception:
        return False
    else:
        return info[1] == 0


def check_pos_def(matrix: MatrixInput, *, exception: bool = True) -> bool:
    """
    Check whether ``matrix`` is positive definite.

    :param matrix: Matrix to test.
    :param exception: If ``True``, raise ``ValueError`` instead of returning
        ``False`` for a non-positive-definite matrix.
    :return: ``True`` if the matrix is positive definite; otherwise ``False``
        when ``exception`` is ``False``.
    :raises ValueError: If the matrix is not positive definite and
        ``exception`` is ``True``.
    """
    norm_matrix = require_symmetric_matrix(matrix, argument_name="matrix")
    if is_pos_def(norm_matrix):
        return True
    if exception:
        raise ValueError("The matrix is not positive definite")
    return False


def blocks_to_target(
    matrix0: MatrixInput,
    fixed_block_sizes: FixedBlockSizes,
) -> MatrixLike:
    """
    Build the target matrix described by ``fixed_block_sizes``.

    The leading diagonal blocks listed in ``fixed_block_sizes`` are copied from
    ``matrix0``. Any remaining trailing block is replaced by its diagonal part.

    :param matrix0: Starting matrix whose diagonal blocks define the target.
    :param fixed_block_sizes: One block size or a sequence of consecutive block
        sizes.
    :return: Target matrix in the same broad container family as ``matrix0``.
    :raises ValueError: If the block sizes are invalid.
    """
    norm_matrix0 = require_square_matrix(
        matrix0,
        argument_name="matrix0",
    )
    sizes = normalize_fixed_block_sizes(
        fixed_block_sizes,
        matrix_size=norm_matrix0.shape[0],
    )
    diagonal_blocks: list[MatrixLike] = list()
    start = 0
    for size in sizes:
        end = start + size
        diagonal_blocks.append(norm_matrix0[start:end, start:end])
        start = end
    if start < norm_matrix0.shape[0]:
        diagonal_blocks.append(
            np.diag(np.diag(norm_matrix0[start:, start:])),
        )
    return convert_like(
        norm_matrix0,
        np.asarray(scipy.linalg.block_diag(*diagonal_blocks)),
    )


def check_matrix0(matrix0: MatrixInput) -> bool:
    """
    Validate that ``matrix0`` is square and symmetric, and report whether it is
    already positive definite.

    :param matrix0: Candidate input matrix.
    :return: ``True`` if ``matrix0`` is already positive definite; ``False``
        otherwise.
    :raises ValueError: If ``matrix0`` is not square or not symmetric.
    """
    norm_matrix0 = require_symmetric_matrix(
        matrix0,
        argument_name="matrix0",
    )
    return is_pos_def(norm_matrix0)


def smallest_eigenvector(matrix: MatrixInput) -> np.matrix:
    """
    Return a unit eigenvector associated with the smallest eigenvalue.

    :param matrix: Symmetric matrix whose smallest eigenvector is required.
    :return: Column vector stored as ``numpy.matrix``.
    """
    norm_matrix = normalize_matrix(matrix)
    eigenvectors = scipy.linalg.eigh(
        norm_matrix,
        check_finite=False,
        subset_by_index=[0, 0],
    )[1]
    return np.asmatrix(eigenvectors)


def require_unit_diagonal(matrix0: MatrixInput) -> MatrixLike:
    """
    Validate that ``matrix0`` has unit diagonal and return it in normalized
    form.

    :param matrix0: Matrix to validate.
    :return: Normalized version of ``matrix0``.
    :raises ValueError: If the diagonal is not identically one.
    """
    norm_matrix0 = require_square_matrix(
        matrix0,
        argument_name="matrix0",
    )
    if not np.all(
        np.isclose(
            np.asarray(np.diag(norm_matrix0), dtype=float),
            np.ones(norm_matrix0.shape[0]),
            rtol=0.0,
            atol=_matrix_atol(norm_matrix0),
        ),
    ):
        raise ValueError("matrix0 must have unit diagonal")
    return norm_matrix0


def require_fixed_block_variant(
    which: int | FixedBlockVariant,
    fb_size: int | None,
    matrix0: MatrixLike,
) -> tuple[FixedBlockVariant, int]:
    """
    Validate the fixed-block variant and block size.

    :param which: Variant describing which diagonal blocks are preserved.
    :param fb_size: Size of the leading principal block for variants that use
        one.
    :param matrix0: Matrix whose order constrains the valid block size.
    :return: Validated variant and block size. The block size is ``0`` for the
        identity-target variant.
    :raises ValueError: If ``which`` or ``fb_size`` is invalid.
    """
    if isinstance(which, bool):
        raise ValueError(
            "The argument 'which' must be a FixedBlockVariant value",
        )
    try:
        variant = FixedBlockVariant(which)
    except ValueError as exc:
        raise ValueError(
            "The argument 'which' must be a FixedBlockVariant value",
        ) from exc

    if (
        fb_size is not None
        and (
            not isinstance(fb_size, Integral)
            or isinstance(fb_size, bool)
            or fb_size <= 0
            or fb_size >= matrix0.shape[0]
        )
    ):
        raise ValueError(
            "fb_size must be a positive integer smaller than the matrix order",
        )

    if (
        variant is not FixedBlockVariant.IDENTITY
        and fb_size is None
    ):
        raise ValueError(
            "fb_size must be a positive integer smaller than the matrix order",
        )
    return variant, 0 if fb_size is None else int(fb_size)


def require_positive_tolerance(tol: float) -> None:
    """
    Validate that an iterative tolerance is strictly positive.

    :param tol: Tolerance supplied by the caller.
    :return: ``None``.
    :raises ValueError: If ``tol`` is not strictly positive.
    """
    if (
        isinstance(tol, bool)
        or not math.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("tol must be a finite positive number")


def require_bisection_tolerance(tol: float) -> None:
    """
    Validate a bisection tolerance against the unit search interval.

    Bisection searches ``alpha`` on ``[0, 1]``, so tolerances greater than or
    equal to ``1`` would stop immediately and return a meaningless endpoint.

    :param tol: Bisection tolerance supplied by the caller.
    :return: ``None``.
    :raises ValueError: If ``tol`` is not strictly between ``0`` and ``1``.
    """
    require_positive_tolerance(tol)
    if tol >= 1:
        raise ValueError("tol must be smaller than 1 for bisection")


def require_positive_iteration_limit(
    max_iterations: int | None,
    *,
    argument_name: str = "max_iterations",
) -> None:
    """
    Validate an optional iteration cap.

    :param max_iterations: Optional iteration limit supplied by the caller.
    :param argument_name: Public argument name used in error messages.
    :return: ``None``.
    :raises ValueError: If the supplied limit is not a positive integer.
    """
    if max_iterations is None:
        return
    if (
        not isinstance(max_iterations, Integral)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError(f"{argument_name} must be a positive integer")

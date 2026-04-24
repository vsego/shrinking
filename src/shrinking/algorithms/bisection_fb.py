"""
Bisection specialized for correlation matrices with fixed blocks.
"""

import numpy as np
import scipy.linalg

from ..results import AlgorithmResult
from ..types import FixedBlockVariant, MatrixInput, MatrixLike
from ..utils import (
    check_matrix0, check_pos_def, require_bisection_tolerance,
    require_fixed_block_variant, require_unit_diagonal,
)


def bisection_with_fixed_block_meta(
    matrix0: MatrixInput,
    fb_size: int | None = None,
    *,
    tol: float = 10 ** (-6),
    which: int | FixedBlockVariant = FixedBlockVariant.PRESERVE_LEADING_BLOCK,
) -> AlgorithmResult:
    """
    Return the specialized bisection result and the iteration count.

    This specialization is intended for correlation matrices and targets of the
    form ``diag(A, I)``, ``diag(I, B)``, ``diag(A, B)``, or ``I``.

    :param matrix0: Correlation matrix to shrink.
    :param fb_size: Size of the leading principal block.
    :param tol: Bisection tolerance.
    :param which: Fixed-block variant describing which diagonal blocks are
        preserved in the target. Preserving a block means copying that block
        from ``matrix0`` into the target matrix. Any unpreserved diagonal
        block is
        replaced by identity in the target, so the reduced matrices used by the
        specialized test still have the corresponding diagonal removed.
    :return: ``AlgorithmResult`` containing the shrinking parameter and the
        number of iterations.
    :raises ValueError: If the input matrix, block size, variant selector, or
        fixed blocks are invalid.
    """
    require_bisection_tolerance(tol)
    norm_matrix0 = require_unit_diagonal(matrix0)
    which, fb_size = require_fixed_block_variant(which, fb_size, norm_matrix0)

    if check_matrix0(norm_matrix0):
        return AlgorithmResult(alpha=0.0, iterations=0)

    left = 0.0
    right = 1.0
    iterations = 0
    zeroed_matrix0: MatrixLike | None = None
    second_block: np.matrix | None = None
    z_matrix: np.matrix | None = None

    if which is FixedBlockVariant.IDENTITY:
        zeroed_matrix0 = norm_matrix0.copy()
        np.fill_diagonal(zeroed_matrix0, 0)
    else:
        first_block_size = fb_size
        second_block_size = norm_matrix0.shape[0] - first_block_size
        if (
            which is FixedBlockVariant.PRESERVE_BOTH_BLOCKS
            and first_block_size < second_block_size
        ):
            swap = True
            second_block = np.asmatrix(
                norm_matrix0[:first_block_size, :first_block_size],
            ).copy()
            first_block = np.asmatrix(
                norm_matrix0[first_block_size:, first_block_size:],
            ).copy()
        else:
            swap = False
            first_block = np.asmatrix(
                norm_matrix0[:first_block_size, :first_block_size],
            ).copy()
            second_block = np.asmatrix(
                norm_matrix0[first_block_size:, first_block_size:],
            ).copy()

        try:
            r11: np.ndarray | None = None
            r22: np.ndarray | None = None
            if which in {
                FixedBlockVariant.PRESERVE_LEADING_BLOCK,
                FixedBlockVariant.PRESERVE_BOTH_BLOCKS,
            }:
                r11 = scipy.linalg.cholesky(
                    first_block,
                    lower=False,
                    check_finite=False,
                )
            if which in {
                FixedBlockVariant.PRESERVE_TRAILING_BLOCK,
                FixedBlockVariant.PRESERVE_BOTH_BLOCKS,
            }:
                r22 = scipy.linalg.cholesky(
                    second_block,
                    lower=False,
                    check_finite=False,
                )
        except Exception as exc:
            raise ValueError(
                "The fixed blocks in matrix0 must be positive definite",
            ) from exc

        y_matrix = np.asmatrix(
            norm_matrix0[first_block_size:, :first_block_size]
            if swap
            else norm_matrix0[:first_block_size, first_block_size:]
        )
        if which in {
            FixedBlockVariant.PRESERVE_LEADING_BLOCK,
            FixedBlockVariant.PRESERVE_BOTH_BLOCKS,
        }:
            if r11 is None:
                raise RuntimeError("internal error: missing Cholesky factor")
            x_matrix = np.asmatrix(
                scipy.linalg.solve_triangular(
                    r11,
                    y_matrix,
                    lower=False,
                    trans=1,
                    check_finite=False,
                ),
            )
        else:
            if r22 is None:
                raise RuntimeError("internal error: missing Cholesky factor")
            x_matrix = np.asmatrix(
                scipy.linalg.solve_triangular(
                    r22,
                    y_matrix.T,
                    lower=False,
                    trans=1,
                    check_finite=False,
                ),
            )
        z_matrix = x_matrix.T * x_matrix

        if which in {
            FixedBlockVariant.PRESERVE_TRAILING_BLOCK,
            FixedBlockVariant.PRESERVE_LEADING_BLOCK,
        }:
            if which is FixedBlockVariant.PRESERVE_TRAILING_BLOCK:
                second_block = first_block
            np.fill_diagonal(second_block, 0)

    if which is FixedBlockVariant.IDENTITY:
        if zeroed_matrix0 is None:
            raise RuntimeError("internal error: missing zeroed matrix")
        while right - left >= tol:
            iterations += 1
            if iterations > 53:
                raise ValueError(
                    "Not converging, probably due to too small tolerance",
                )
            alpha = (left + right) / 2.0
            candidate = (1.0 - alpha) * zeroed_matrix0
            np.fill_diagonal(candidate, 1)
            if check_pos_def(candidate, exception=False):
                right = alpha
            else:
                left = alpha
    else:
        if second_block is None or z_matrix is None:
            raise RuntimeError("internal error: missing fixed-block data")
        while right - left >= tol:
            iterations += 1
            if iterations > 53:
                raise ValueError(
                    "Not converging, probably due to too small tolerance",
                )
            alpha = (left + right) / 2.0
            if which in {
                FixedBlockVariant.PRESERVE_TRAILING_BLOCK,
                FixedBlockVariant.PRESERVE_LEADING_BLOCK,
            }:
                candidate = (1.0 - alpha) * second_block
                np.fill_diagonal(candidate, 1)
                candidate = candidate - (1.0 - alpha) ** 2 * z_matrix
            else:
                candidate = second_block - (1.0 - alpha) ** 2 * z_matrix

            if check_pos_def(candidate, exception=False):
                right = alpha
            else:
                left = alpha

    return AlgorithmResult(alpha=right, iterations=iterations)


def bisection_with_fixed_block(
    matrix0: MatrixInput,
    fb_size: int | None = None,
    *,
    tol: float = 10 ** (-6),
    which: int | FixedBlockVariant = FixedBlockVariant.PRESERVE_LEADING_BLOCK,
) -> float:
    """
    Return only the specialized bisection result.

    :param matrix0: Correlation matrix to shrink.
    :param fb_size: Size of the leading principal block.
    :param tol: Bisection tolerance.
    :param which: Fixed-block variant describing which diagonal blocks are
        preserved in the target.
    :return: Optimal shrinking parameter.
    :raises ValueError: If the input matrix, block size, variant selector, or
        fixed blocks are invalid.
    """
    return bisection_with_fixed_block_meta(
        matrix0,
        fb_size,
        tol=tol,
        which=which,
    ).alpha

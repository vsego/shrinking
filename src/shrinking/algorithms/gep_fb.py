"""
Generalized eigenvalue algorithm specialized for fixed blocks.
"""

import numpy as np
import scipy.linalg

from ..results import AlgorithmResult
from ..types import FixedBlockVariant, MatrixInput
from ..utils import (
    check_matrix0, require_fixed_block_variant, require_unit_diagonal,
)


def gep_with_fixed_block_meta(
    matrix0: MatrixInput,
    fb_size: int | None = None,
    *,
    which: int | FixedBlockVariant = FixedBlockVariant.PRESERVE_LEADING_BLOCK,
) -> AlgorithmResult:
    """
    Return the specialized generalized-eigenvalue result and the iteration
    count.

    :param matrix0: Correlation matrix to shrink.
    :param fb_size: Size of the leading principal block.
    :param which: Fixed-block variant describing which diagonal blocks are
        preserved in the target. Preserving a block means copying that block
        from ``matrix0`` into the target matrix. Any unpreserved diagonal
        block is
        replaced by identity in the target, so the reduced matrices used by the
        specialized test still have the corresponding diagonal removed.
    :return: ``AlgorithmResult`` containing the shrinking parameter and the
        logical iteration count.
    :raises ValueError: If the input matrix, block size, or selector is
        invalid.
    """
    norm_matrix0 = require_unit_diagonal(matrix0)
    which, fb_size = require_fixed_block_variant(which, fb_size, norm_matrix0)
    if check_matrix0(norm_matrix0):
        return AlgorithmResult(alpha=0.0, iterations=0)

    if which is FixedBlockVariant.IDENTITY:
        zeroed_matrix = norm_matrix0.copy()
        np.fill_diagonal(zeroed_matrix, 0)
        eigenvalues = scipy.linalg.eigvalsh(
            zeroed_matrix,
            check_finite=False,
            subset_by_index=[0, 0],
        )
        eigenvalue = eigenvalues[0]
        alpha = 0.0 if eigenvalue > -1 else 1.0 + 1.0 / eigenvalue
        return AlgorithmResult(alpha=alpha, iterations=1)

    first_block = np.asmatrix(norm_matrix0[:fb_size, :fb_size]).copy()
    second_block = np.asmatrix(norm_matrix0[fb_size:, fb_size:]).copy()
    y_matrix = np.asmatrix(norm_matrix0[:fb_size, fb_size:]).copy()

    if which is FixedBlockVariant.PRESERVE_TRAILING_BLOCK:
        second_block_size = norm_matrix0.shape[0] - fb_size
        try:
            r22 = scipy.linalg.cholesky(
                second_block,
                lower=False,
                check_finite=False,
            )
        except Exception as exc:  # pylint: disable=W0718
            raise ValueError(
                "The fixed blocks in matrix0 must be positive definite",
            ) from exc
        x_matrix = np.asmatrix(
            scipy.linalg.solve_triangular(
                r22,
                y_matrix.T,
                lower=False,
                trans=1,
                check_finite=False,
            ),
        )
        np.fill_diagonal(first_block, 0)
        candidate = np.bmat(
            [
                [np.zeros((second_block_size, second_block_size)), x_matrix],
                [x_matrix.T, first_block],
            ],
        )
        eigenvalue = scipy.linalg.eigvalsh(
            candidate,
            check_finite=False,
            subset_by_index=[0, 0],
        )[0]
        alpha = 0.0 if eigenvalue > -1 else 1.0 + 1.0 / eigenvalue
        return AlgorithmResult(alpha=alpha, iterations=1)

    if which is FixedBlockVariant.PRESERVE_LEADING_BLOCK:
        try:
            r11 = scipy.linalg.cholesky(
                first_block,
                lower=False,
                check_finite=False,
            )
        except Exception as exc:  # pylint: disable=W0718
            raise ValueError(
                "The fixed blocks in matrix0 must be positive definite",
            ) from exc
        x_matrix = np.asmatrix(
            scipy.linalg.solve_triangular(
                r11,
                y_matrix,
                lower=False,
                trans=1,
                check_finite=False,
            ),
        )
        np.fill_diagonal(second_block, 0)
        candidate = np.bmat(
            [
                [np.zeros((fb_size, fb_size)), x_matrix],
                [x_matrix.T, second_block],
            ],
        )
        eigenvalue = scipy.linalg.eigvalsh(
            candidate,
            check_finite=False,
            subset_by_index=[0, 0],
        )[0]
        alpha = 0.0 if eigenvalue > -1 else 1.0 + 1.0 / eigenvalue
        return AlgorithmResult(alpha=alpha, iterations=1)

    if which is FixedBlockVariant.PRESERVE_BOTH_BLOCKS:
        try:
            r11 = scipy.linalg.cholesky(
                first_block,
                lower=False,
                check_finite=False,
            )
            r22 = scipy.linalg.cholesky(
                second_block,
                lower=False,
                check_finite=False,
            )
        except Exception as exc:  # pylint: disable=W0718
            raise ValueError(
                "The fixed blocks in matrix0 must be positive definite",
            ) from exc
        x_matrix = np.asmatrix(
            scipy.linalg.solve_triangular(
                r11,
                y_matrix,
                lower=False,
                trans=1,
                check_finite=False,
            ),
        )
        x_matrix = np.asmatrix(
            scipy.linalg.solve_triangular(
                r22,
                x_matrix.T,
                lower=False,
                trans=1,
                check_finite=False,
            ),
        )
        singular_value = scipy.linalg.svdvals(x_matrix, check_finite=False)[0]
        alpha = 0.0 if singular_value < 1 else 1.0 - 1.0 / singular_value
        return AlgorithmResult(alpha=alpha, iterations=1)

    raise RuntimeError("internal error: unexpected fixed-block variant")


def gep_with_fixed_block(
    matrix0: MatrixInput,
    fb_size: int | None = None,
    *,
    which: int | FixedBlockVariant = FixedBlockVariant.PRESERVE_LEADING_BLOCK,
) -> float:
    """
    Return only the specialized generalized-eigenvalue result.

    :param matrix0: Correlation matrix to shrink.
    :param fb_size: Size of the leading principal block.
    :param which: Fixed-block variant describing which diagonal blocks are
        preserved in the target.
    :return: Optimal shrinking parameter.
    :raises ValueError: If the input matrix, block size, or selector is
        invalid.
    """
    return gep_with_fixed_block_meta(
        matrix0,
        fb_size,
        which=which,
    ).alpha

"""
Helpers for evaluating the various ``S(alpha)`` forms.
"""

import numpy as np

from .types import FixedBlockSizes, MatrixInput, MatrixLike
from .utils import (
    align_matrix_pair, convert_like, normalize_fixed_block_sizes,
    require_square_matrix,
)


def s_with_target(
    matrix0: MatrixInput,
    matrix1: MatrixInput,
    alpha: float,
) -> MatrixLike:
    """
    Return ``(1 - alpha) * matrix0 + alpha * matrix1``.

    :param matrix0: Starting matrix.
    :param matrix1: Explicit target matrix.
    :param alpha: Shrinking parameter.
    :return: Convex combination of ``matrix0`` and ``matrix1``.
    :raises ValueError: If ``matrix0`` or ``matrix1`` is not square, or if
        their shapes differ.
    """
    norm_matrix0 = require_square_matrix(matrix0, argument_name="matrix0")
    norm_matrix1 = require_square_matrix(matrix1, argument_name="matrix1")
    norm_matrix0, norm_matrix1 = align_matrix_pair(norm_matrix0, norm_matrix1)
    if norm_matrix0.shape != norm_matrix1.shape:
        raise ValueError("matrix1 must have the same shape as matrix0")
    return (1.0 - alpha) * norm_matrix0 + alpha * norm_matrix1


def s_with_difference(
    matrix0: MatrixInput,
    difference_matrix: MatrixInput,
    alpha: float,
) -> MatrixLike:
    """
    Return ``matrix0 - alpha * (matrix0 - matrix1)``.

    :param matrix0: Starting matrix.
    :param difference_matrix: Precomputed difference ``matrix0 - matrix1``.
    :param alpha: Shrinking parameter.
    :return: Matrix ``S(alpha)`` reconstructed from the difference form.
    :raises ValueError: If ``matrix0`` or ``difference_matrix`` is not
        square, or if their shapes differ.
    """
    norm_matrix0 = require_square_matrix(matrix0, argument_name="matrix0")
    norm_difference = require_square_matrix(
        difference_matrix,
        argument_name="difference_matrix",
    )
    norm_matrix0, norm_difference = align_matrix_pair(
        norm_matrix0,
        norm_difference,
    )
    if norm_matrix0.shape != norm_difference.shape:
        raise ValueError(
            "difference_matrix must have the same shape as matrix0",
        )
    return norm_matrix0 - alpha * norm_difference


def s_with_fixed_blocks(
    matrix0: MatrixInput,
    fixed_block_sizes: FixedBlockSizes,
    alpha: float,
) -> MatrixLike:
    """
    Return ``S(alpha)`` for a target defined by fixed blocks.

    :param matrix0: Starting matrix.
    :param fixed_block_sizes: One block size or a sequence of consecutive fixed
        block sizes.
    :param alpha: Shrinking parameter.
    :return: Matrix ``S(alpha)`` built without explicitly forming the target.
    :raises ValueError: If the block sizes are invalid.
    """
    norm_matrix0 = require_square_matrix(matrix0, argument_name="matrix0")
    sizes = normalize_fixed_block_sizes(
        fixed_block_sizes,
        matrix_size=norm_matrix0.shape[0],
    )
    factor = 1.0 - alpha
    result = norm_matrix0.copy()
    fixed_block_index = 1
    processed_size = sizes[0]
    total_fixed_size = sum(sizes)

    if total_fixed_size < norm_matrix0.shape[0]:
        preserved_diagonal = np.diag(
            result[total_fixed_size:, total_fixed_size:],
        ).copy()
        np.fill_diagonal(result[total_fixed_size:, total_fixed_size:], 0)
    else:
        preserved_diagonal = None

    while fixed_block_index < len(sizes):
        block_size = sizes[fixed_block_index]
        next_processed_size = processed_size + block_size
        result[processed_size:next_processed_size, :processed_size] *= factor
        result[:processed_size, processed_size:next_processed_size] *= factor
        processed_size = next_processed_size
        fixed_block_index += 1

    if processed_size < norm_matrix0.shape[0]:
        if preserved_diagonal is None:
            raise RuntimeError(
                "internal error: missing preserved diagonal for trailing block",
            )
        result[processed_size:, :] *= factor
        result[:processed_size, processed_size:] *= factor
        result[total_fixed_size:, total_fixed_size:] += convert_like(
            norm_matrix0,
            np.diag(preserved_diagonal),
        )

    return result


def s_with_identity(matrix0: MatrixInput, alpha: float) -> MatrixLike:
    """
    Return the identity-target variant of ``S(alpha)``.

    :param matrix0: Starting matrix, typically a correlation matrix with unit
        diagonal.
    :param alpha: Shrinking parameter.
    :return: Matrix ``S(alpha)`` with the diagonal forced to one.
    :raises ValueError: If ``matrix0`` is not square.
    """
    norm_matrix0 = require_square_matrix(matrix0, argument_name="matrix0")
    result = (1.0 - alpha) * norm_matrix0
    np.fill_diagonal(result, 1)
    return result


def s(
    matrix0: MatrixInput,
    alpha: float,
    *,
    matrix1: MatrixInput | None = None,
    difference_matrix: MatrixInput | None = None,
    fixed_block_sizes: FixedBlockSizes | None = None,
) -> MatrixLike:
    """
    Evaluate the most suitable ``S(alpha)`` variant for the given inputs.

    Exactly one of ``matrix1``, ``difference_matrix``, and
    ``fixed_block_sizes`` may be provided. If none is provided, the
    identity-target variant is used.

    :param matrix0: Starting matrix.
    :param alpha: Shrinking parameter.
    :param matrix1: Explicit target matrix.
    :param difference_matrix: Precomputed difference ``matrix0 - matrix1``.
    :param fixed_block_sizes: Fixed-block descriptor for the implicit target.
    :return: Matrix ``S(alpha)``.
    :raises ValueError: If more than one target specification is provided.
    """
    cnt_arguments = sum(
        value is not None
        for value in (matrix1, difference_matrix, fixed_block_sizes)
    )
    if cnt_arguments > 1:
        raise ValueError(
            "Only one of matrix1, difference_matrix, and fixed_block_sizes may"
            " be given",
        )
    if fixed_block_sizes is not None:
        return s_with_fixed_blocks(matrix0, fixed_block_sizes, alpha)
    if difference_matrix is not None:
        return s_with_difference(matrix0, difference_matrix, alpha)
    if matrix1 is not None:
        return s_with_target(matrix0, matrix1, alpha)
    return s_with_identity(matrix0, alpha)

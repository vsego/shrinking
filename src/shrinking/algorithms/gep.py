"""
Generalized eigenvalue algorithm.
"""

import scipy.linalg

from ..problems import initialize_problem
from ..results import AlgorithmResult
from ..types import FixedBlockSizes, MatrixInput


def gep_meta(
    matrix0: MatrixInput,
    matrix1: MatrixInput | None = None,
    *,
    fixed_block_sizes: FixedBlockSizes | None = None,
) -> AlgorithmResult:
    """
    Return the optimal shrinking parameter and the iteration count.

    :param matrix0: Symmetric indefinite matrix to shrink.
    :param matrix1: Explicit positive definite target matrix.
    :param fixed_block_sizes: Fixed-block descriptor used to build the target.
    :return: ``AlgorithmResult`` containing the shrinking parameter and the
        logical iteration count.
    :raises ValueError: If the inputs are inconsistent.
    """
    problem = initialize_problem(
        matrix0,
        matrix1,
        fixed_block_sizes,
    )
    if problem is None:
        return AlgorithmResult(alpha=0.0, iterations=0)
    norm_matrix0 = problem.matrix0
    target_matrix = problem.target_matrix

    eigenvalues = scipy.linalg.eigvalsh(
        norm_matrix0,
        target_matrix,
        check_finite=False,
        subset_by_index=[0, 0],
    )
    alpha = 1.0 + 1.0 / (eigenvalues[0] - 1.0)

    return AlgorithmResult(alpha=alpha, iterations=1)


def gep(
    matrix0: MatrixInput,
    matrix1: MatrixInput | None = None,
    *,
    fixed_block_sizes: FixedBlockSizes | None = None,
) -> float:
    """
    Return only the shrinking parameter.

    :param matrix0: Symmetric indefinite matrix to shrink.
    :param matrix1: Explicit positive definite target matrix.
    :param fixed_block_sizes: Fixed-block descriptor used to build the target.
    :return: Optimal shrinking parameter.
    :raises ValueError: If the inputs are inconsistent.
    """
    return gep_meta(
        matrix0,
        matrix1=matrix1,
        fixed_block_sizes=fixed_block_sizes,
    ).alpha

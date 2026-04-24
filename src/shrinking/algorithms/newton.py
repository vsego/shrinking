"""
Newton's method.
"""

import numpy as np

from ..exceptions import NotConvergingError
from ..problems import initialize_problem
from ..results import AlgorithmResult
from ..types import FixedBlockSizes, MatrixInput
from ..utils import (
    require_positive_iteration_limit, require_positive_tolerance,
    smallest_eigenvector,
)


def newton_meta(
    matrix0: MatrixInput,
    matrix1: MatrixInput | None = None,
    *,
    fixed_block_sizes: FixedBlockSizes | None = None,
    tol: float = 10 ** (-6),
    max_iterations: int | None = None,
) -> AlgorithmResult:
    """
    Return the optimal shrinking parameter and the iteration count.

    :param matrix0: Symmetric indefinite matrix to shrink.
    :param matrix1: Explicit positive definite target matrix.
    :param fixed_block_sizes: Fixed-block descriptor used to build the target.
    :param tol: Stopping tolerance for successive iterates.
    :param max_iterations: Optional hard limit on the number of Newton steps.
    :return: ``AlgorithmResult`` containing the shrinking parameter and the
        number of iterations.
    :raises NotConvergingError: If ``max_iterations`` is exceeded.
    :raises ValueError: If the inputs are inconsistent.
    """
    require_positive_tolerance(tol)
    require_positive_iteration_limit(max_iterations)
    problem = initialize_problem(
        matrix0,
        matrix1,
        fixed_block_sizes,
    )
    if problem is None:
        return AlgorithmResult(alpha=0.0, iterations=0)
    norm_matrix0 = problem.matrix0
    target_matrix = problem.target_matrix

    matrix0_for_products = np.asmatrix(norm_matrix0)
    difference_matrix = matrix0_for_products - np.asmatrix(target_matrix)
    alpha = 0.0
    iterations = 0
    while True:
        iterations += 1
        if max_iterations is not None and iterations > max_iterations:
            raise NotConvergingError("Not converging", alpha)
        candidate = problem.s(alpha)
        vector_x = smallest_eigenvector(candidate)
        alpha_num = (vector_x.T * matrix0_for_products * vector_x)[0, 0]
        alpha_denom = (vector_x.T * difference_matrix * vector_x)[0, 0]
        new_alpha = float(alpha_num / alpha_denom)
        if abs(new_alpha - alpha) < tol:
            return AlgorithmResult(
                alpha=new_alpha,
                iterations=iterations,
            )
        alpha = new_alpha


def newton(
    matrix0: MatrixInput,
    matrix1: MatrixInput | None = None,
    *,
    fixed_block_sizes: FixedBlockSizes | None = None,
    tol: float = 10 ** (-6),
    max_iterations: int | None = None,
) -> float:
    """
    Return only the shrinking parameter.

    :param matrix0: Symmetric indefinite matrix to shrink.
    :param matrix1: Explicit positive definite target matrix.
    :param fixed_block_sizes: Fixed-block descriptor used to build the target.
    :param tol: Stopping tolerance for successive iterates.
    :param max_iterations: Optional hard limit on the number of Newton steps.
    :return: Optimal shrinking parameter.
    :raises NotConvergingError: If ``max_iterations`` is exceeded.
    :raises ValueError: If the inputs are inconsistent.
    """
    return newton_meta(
        matrix0,
        matrix1=matrix1,
        fixed_block_sizes=fixed_block_sizes,
        tol=tol,
        max_iterations=max_iterations,
    ).alpha

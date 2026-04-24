"""
Generic bisection algorithm.
"""

from ..problems import initialize_problem
from ..results import AlgorithmResult
from ..types import FixedBlockSizes, MatrixInput
from ..utils import (
    check_pos_def, require_bisection_tolerance,
    require_positive_iteration_limit,
)


def bisection_meta(
    matrix0: MatrixInput,
    matrix1: MatrixInput | None = None,
    *,
    fixed_block_sizes: FixedBlockSizes | None = None,
    tol: float = 10 ** (-6),
    max_iterations: int | None = 53,
) -> AlgorithmResult:
    """
    Return the optimal shrinking parameter and the iteration count.

    :param matrix0: Symmetric indefinite matrix to shrink.
    :param matrix1: Explicit positive definite target matrix.
    :param fixed_block_sizes: Fixed-block descriptor used to build the target.
    :param tol: Bisection tolerance.
    :param max_iterations: Optional hard limit on the number of iterations.
    :return: ``AlgorithmResult`` containing the shrinking parameter and the
        number of bisection steps.
    :raises ValueError: If the inputs are inconsistent, the tolerance is too
        small for reliable convergence, or ``max_iterations`` is exceeded.
    """
    require_bisection_tolerance(tol)
    require_positive_iteration_limit(max_iterations)
    problem = initialize_problem(
        matrix0,
        matrix1,
        fixed_block_sizes,
    )
    if problem is None:
        return AlgorithmResult(alpha=0.0, iterations=0)

    left = 0.0
    right = 1.0
    iterations = 0
    while right - left >= tol:
        iterations += 1
        if max_iterations is not None and iterations > max_iterations:
            raise ValueError(
                "Not converging, probably due to too small tolerance",
            )
        alpha = (left + right) / 2.0
        if check_pos_def(problem.s(alpha), exception=False):
            right = alpha
        else:
            left = alpha

    return AlgorithmResult(alpha=right, iterations=iterations)


def bisection(
    matrix0: MatrixInput,
    matrix1: MatrixInput | None = None,
    *,
    fixed_block_sizes: FixedBlockSizes | None = None,
    tol: float = 10 ** (-6),
    max_iterations: int | None = 53,
) -> float:
    """
    Return only the shrinking parameter.

    :param matrix0: Symmetric indefinite matrix to shrink.
    :param matrix1: Explicit positive definite target matrix.
    :param fixed_block_sizes: Fixed-block descriptor used to build the target.
    :param tol: Bisection tolerance.
    :param max_iterations: Optional hard limit on the number of iterations.
    :return: Optimal shrinking parameter.
    :raises ValueError: If the inputs are inconsistent, the tolerance is too
        small for reliable convergence, or ``max_iterations`` is exceeded.
    """
    return bisection_meta(
        matrix0,
        matrix1=matrix1,
        fixed_block_sizes=fixed_block_sizes,
        tol=tol,
        max_iterations=max_iterations,
    ).alpha

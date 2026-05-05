"""
Prepared problem representations for the shrinking algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .s import s_with_difference, s_with_fixed_blocks
from .types import FixedBlockSizes, MatrixInput, MatrixLike
from .utils import (
    align_matrix_pair, blocks_to_target, check_matrix0, check_pos_def,
    normalize_matrix, require_symmetric_matrix,
)


class Problem(ABC):
    """
    Represent a prepared shrinking problem.

    :ivar matrix0: Normalized starting matrix.
    """

    # Data attribute expected on all concrete problem types.
    matrix0: MatrixLike

    @property
    @abstractmethod
    def target_matrix(self) -> MatrixLike:
        """
        Return the normalized target matrix.

        :return: Normalized target matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def s(self, alpha: float) -> MatrixLike:
        """
        Return ``S(alpha)`` for this problem.

        :param alpha: Shrinking parameter.
        :return: Matrix ``S(alpha)``.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ExplicitTargetProblem(Problem):
    """
    Store a normalized problem with an explicit target matrix.

    :ivar matrix0: Normalized starting matrix.
    :ivar matrix1: Normalized explicit target matrix.
    """

    matrix0: MatrixLike
    matrix1: MatrixLike

    def s(self, alpha: float) -> MatrixLike:
        """
        Return ``S(alpha)`` for this explicit-target problem.

        :param alpha: Shrinking parameter.
        :return: Matrix ``S(alpha)``.
        """
        return s_with_difference(
            self.matrix0,
            self.matrix0 - self.matrix1,
            alpha,
        )

    @property
    def target_matrix(self) -> MatrixLike:
        """
        Return the explicit target matrix.

        :return: Normalized target matrix.
        """
        return self.matrix1


@dataclass(frozen=True, slots=True)
class FixedBlockProblem(Problem):
    """
    Store a normalized problem with a fixed-block target specification.

    :ivar matrix0: Normalized starting matrix.
    :ivar fixed_block_sizes: Validated fixed-block descriptor.
    """

    matrix0: MatrixLike
    fixed_block_sizes: FixedBlockSizes

    def s(self, alpha: float) -> MatrixLike:
        """
        Return ``S(alpha)`` for this fixed-block problem.

        :param alpha: Shrinking parameter.
        :return: Matrix ``S(alpha)``.
        """
        return s_with_fixed_blocks(
            self.matrix0,
            self.fixed_block_sizes,
            alpha,
        )

    @property
    def target_matrix(self) -> MatrixLike:
        """
        Build and return the fixed-block target matrix.

        :return: Normalized target matrix.
        """
        return blocks_to_target(self.matrix0, self.fixed_block_sizes)


def initialize_problem(
    matrix0: MatrixInput,
    matrix1: MatrixInput | None,
    fixed_block_sizes: FixedBlockSizes | None,
) -> Problem | None:
    """
    Normalize and validate a shrinking problem specification.

    Return ``None`` when ``matrix0`` is already positive definite, an
    ``ExplicitTargetProblem`` when an explicit target matrix is supplied, or a
    ``FixedBlockProblem`` when the target is described by fixed blocks.

    :param matrix0: Starting matrix to repair.
    :param matrix1: Explicit target matrix, if supplied.
    :param fixed_block_sizes: Fixed-block descriptor used to build the target.
    :return: ``None`` or a prepared problem object.
    :raises ValueError: If the target specification is inconsistent or the
        matrix dimensions are incompatible.
    """
    norm_matrix0 = require_symmetric_matrix(
        matrix0,
        argument_name="matrix0",
    )
    if matrix1 is None:
        norm_matrix1 = None
    else:
        norm_matrix1 = normalize_matrix(matrix1)
        norm_matrix0, norm_matrix1 = align_matrix_pair(
            norm_matrix0,
            norm_matrix1,
        )

    if (norm_matrix1 is None) == (fixed_block_sizes is None):
        if norm_matrix1 is None:
            raise ValueError(
                "Provide either matrix1 or fixed_block_sizes",
            )
        raise ValueError(
            "Provide only one of matrix1 and fixed_block_sizes",
        )

    if norm_matrix1 is None:
        if fixed_block_sizes is None:
            raise RuntimeError("internal error: missing fixed-block sizes")
        check_pos_def(
            blocks_to_target(norm_matrix0, fixed_block_sizes),
        )
        if check_matrix0(norm_matrix0):
            return None
        return FixedBlockProblem(norm_matrix0, fixed_block_sizes)

    if (
        len(norm_matrix0.shape) == len(norm_matrix1.shape) == 2
        and norm_matrix0.shape == norm_matrix1.shape
    ):
        check_pos_def(norm_matrix1)
        if check_matrix0(norm_matrix0):
            return None
        return ExplicitTargetProblem(norm_matrix0, norm_matrix1)

    raise ValueError(
        "matrix1 must be a positive definite matrix of the same order as"
        " matrix0",
    )

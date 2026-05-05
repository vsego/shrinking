"""
Regression tests for the refactored shrinking package.
"""

import unittest
import warnings
from typing import cast

import numpy as np

import shrinking
from shrinking import backwards_compatibility as compat
from shrinking.exceptions import NotConvergingError

from .utils import TestsBase

KNOWN_ALPHA = 1.0 / 6.0


def setUpModule() -> None:
    """
    Suppress known ``numpy.matrix`` pending-deprecation noise in tests.

    :return: ``None``.
    """
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def known_indefinite_matrix(as_matrix: bool) -> np.ndarray | np.matrix:
    """
    Return a simple 2x2 indefinite correlation matrix.

    :param as_matrix: If ``True``, return ``numpy.matrix``; otherwise return
        ``numpy.ndarray``.
    :return: Indefinite 2x2 test matrix.
    """
    data = [[1.0, 6.0 / 5.0], [6.0 / 5.0, 1.0]]
    return np.matrix(data) if as_matrix else np.array(data, dtype=float)


def known_identity(as_matrix: bool) -> np.ndarray | np.matrix:
    """
    Return the 2x2 identity in the requested container type.

    :param as_matrix: If ``True``, return ``numpy.matrix``; otherwise return
        ``numpy.ndarray``.
    :return: Identity matrix in the requested container type.
    """
    return np.matrix(np.identity(2)) if as_matrix else np.identity(2)


class PublicApiHelperTests(TestsBase):
    def test_plain_nested_sequences_normalize_to_arrays(self) -> None:
        result = shrinking.s_with_target(
            [[1.0, 1.2], [1.2, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            0.5,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertNotIsInstance(result, np.matrix)

    def test_mixed_matrix_inputs_normalize_to_arrays(self) -> None:
        matrix_value = np.matrix([[1.0, 2.0], [3.0, 4.0]])
        array_value = np.array([[5.0, 6.0], [7.0, 8.0]])
        mixed_target = shrinking.s_with_target(matrix_value, array_value, 0.5)
        mixed_target_reversed = shrinking.s_with_target(
            array_value,
            matrix_value,
            0.5,
        )
        self.assertIsInstance(mixed_target, np.ndarray)
        self.assertNotIsInstance(mixed_target, np.matrix)
        self.assertIsInstance(mixed_target_reversed, np.ndarray)
        self.assertNotIsInstance(mixed_target_reversed, np.matrix)

    def test_check_pos_def_accepts_arrays_and_matrices(self) -> None:
        for as_matrix in (False, True):
            matrix = known_identity(as_matrix)
            self.assertTrue(shrinking.check_pos_def(matrix))
            self.assertFalse(
                shrinking.check_pos_def(
                    known_indefinite_matrix(as_matrix),
                    exception=False,
                ),
            )

    def test_check_pos_def_rejects_nonsymmetric_matrices(self) -> None:
        with self.assertRaises(ValueError):
            shrinking.check_pos_def(
                np.array([[2.0, 0.0], [10.0, 2.0]], dtype=float),
            )

    def test_nonfinite_matrices_are_rejected_consistently(self) -> None:
        matrix0 = np.array([[1.0, float("inf")], [float("inf"), 1.0]])
        matrix1 = np.identity(2)
        with self.assertRaisesRegex(
            ValueError,
            "matrix must contain only finite values",
        ):
            shrinking.check_pos_def(matrix0)
        for algorithm in (
            shrinking.bisection,
            shrinking.newton,
            shrinking.gep,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "matrix0 must contain only finite values",
            ):
                algorithm(matrix0, matrix1=matrix1)
        for algorithm in (
            shrinking.bisection_with_fixed_block,
            shrinking.gep_with_fixed_block,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "matrix0 must contain only finite values",
            ):
                algorithm(matrix0, fb_size=1)

    def test_empty_matrices_are_rejected(self) -> None:
        empty = np.empty((0, 0))
        with self.assertRaisesRegex(
            ValueError,
            "matrix0 must be a non-empty square matrix",
        ):
            shrinking.bisection(empty, matrix1=np.identity(0))
        with self.assertRaisesRegex(
            ValueError,
            "matrix must be a non-empty square matrix",
        ):
            shrinking.check_pos_def(empty)

    def test_blocks_to_target_keeps_fixed_blocks(self) -> None:
        matrix0 = np.array(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            dtype=float,
        )
        target = shrinking.blocks_to_target(matrix0, 1)
        expected = np.array(
            [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]],
            dtype=float,
        )
        np.testing.assert_allclose(target, expected)

    def test_s_with_fixed_blocks_scales_only_nonfixed_entries(self) -> None:
        matrix0 = np.array(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            dtype=float,
        )
        result = shrinking.s_with_fixed_blocks(matrix0, 1, 0.25)
        expected = np.array(
            [[1.0, 1.5, 2.25], [1.5, 4.0, 3.75], [2.25, 3.75, 6.0]],
            dtype=float,
        )
        np.testing.assert_allclose(result, expected)

    def test_integer_nested_sequences_work_with_fixed_block_helpers(
        self,
    ) -> None:
        result = shrinking.s_with_fixed_blocks([[1, 2], [2, 1]], 1, 0.25)
        expected = np.array([[1.0, 1.5], [1.5, 1.0]], dtype=float)
        np.testing.assert_allclose(result, expected)

    def test_s_rejects_multiple_target_descriptions(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        with self.assertRaises(ValueError):
            shrinking.s(
                matrix0,
                0.1,
                matrix1=known_identity(False),
                difference_matrix=matrix0 - known_identity(False),
            )

    def test_s_uses_identity_when_target_arguments_are_none(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        result = shrinking.s(
            matrix0,
            KNOWN_ALPHA,
            matrix1=None,
            difference_matrix=None,
            fixed_block_sizes=None,
        )
        expected = shrinking.s_with_identity(matrix0, KNOWN_ALPHA)
        np.testing.assert_allclose(result, expected)

    def test_s_with_fixed_blocks_rejects_invalid_block_sizes(self) -> None:
        with self.assertRaises(ValueError):
            shrinking.s_with_fixed_blocks(
                known_indefinite_matrix(False),
                [0],
                0.1,
            )

    def test_s_helpers_reject_shape_mismatches(self) -> None:
        matrix0 = np.identity(2)
        with self.assertRaises(ValueError):
            shrinking.s_with_target(matrix0, np.array([1.0, 1.0]), 0.1)
        with self.assertRaises(ValueError):
            shrinking.s_with_difference(matrix0, np.array([1.0, 1.0]), 0.1)

    def test_roundoff_level_symmetry_is_accepted(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.standard_normal((5, 5))
        symmetric = (sample + sample.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        shifted = reconstructed + (
            abs(float(eigenvalues[0])) + 1.0
        ) * np.identity(5)
        self.assertLess(
            float(np.abs(reconstructed - reconstructed.T).max()),
            1e-12,
        )
        self.assertTrue(
            shrinking.check_pos_def(
                shifted,
            ),
        )

    def test_roundoff_level_unit_diagonal_is_accepted(self) -> None:
        matrix0 = np.identity(3)
        matrix0[0, 0] = 1.0 + 1e-15
        self.assertAlmostEqual(
            shrinking.gep_with_fixed_block(matrix0, fb_size=1),
            0.0,
        )

    def test_identity_variant_rejects_invalid_fb_size(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        with self.assertRaises(ValueError):
            shrinking.bisection_with_fixed_block(
                matrix0,
                fb_size=True,
                which=shrinking.FixedBlockVariant.IDENTITY,
            )
        with self.assertRaises(ValueError):
            shrinking.gep_with_fixed_block(
                matrix0,
                fb_size=cast(int | None, 1.5),
                which=shrinking.FixedBlockVariant.IDENTITY,
            )


class PublicApiAlgorithmTests(TestsBase):
    def test_algorithms_match_known_result_for_arrays_and_matrices(
        self,
    ) -> None:
        for as_matrix in (False, True):
            matrix0 = known_indefinite_matrix(as_matrix)
            matrix1 = known_identity(as_matrix)
            self.assertAlmostEqual(
                shrinking.bisection(matrix0, matrix1=matrix1),
                KNOWN_ALPHA,
                places=6,
            )
            self.assertAlmostEqual(
                shrinking.bisection(matrix0, fixed_block_sizes=1),
                KNOWN_ALPHA,
                places=6,
            )
            self.assertAlmostEqual(
                shrinking.bisection_with_fixed_block(matrix0, fb_size=1),
                KNOWN_ALPHA,
                places=6,
            )
            self.assertAlmostEqual(
                shrinking.newton(matrix0, matrix1=matrix1),
                KNOWN_ALPHA,
                places=6,
            )
            self.assertAlmostEqual(
                shrinking.gep(matrix0, matrix1=matrix1),
                KNOWN_ALPHA,
                places=6,
            )
            self.assertAlmostEqual(
                shrinking.gep_with_fixed_block(matrix0, fb_size=1),
                KNOWN_ALPHA,
                places=6,
            )

    def test_positive_definite_inputs_return_zero(self) -> None:
        matrix0 = np.identity(2)
        self.assertEqual(
            shrinking.bisection(matrix0, matrix1=np.identity(2)),
            0.0,
        )
        self.assertEqual(
            shrinking.newton(matrix0, matrix1=np.identity(2)),
            0.0,
        )
        self.assertEqual(shrinking.gep(matrix0, matrix1=np.identity(2)), 0.0)

    def test_invalid_targets_are_rejected_even_when_matrix0_is_pd(
        self,
    ) -> None:
        matrix0 = np.identity(2)
        bad_matrix1 = known_indefinite_matrix(False)
        for algorithm in (
            shrinking.bisection,
            shrinking.newton,
            shrinking.gep,
        ):
            with self.assertRaises(ValueError):
                algorithm(matrix0, matrix1=bad_matrix1)
        for algorithm in (
            shrinking.bisection,
            shrinking.newton,
            shrinking.gep,
        ):
            with self.assertRaises(ValueError):
                algorithm(matrix0, fixed_block_sizes=True)

    def test_newton_reports_non_convergence(self) -> None:
        with self.assertRaises(NotConvergingError) as ctx:
            shrinking.newton(
                known_indefinite_matrix(False),
                matrix1=known_identity(False),
                max_iterations=1,
            )
        self.assertIsInstance(ctx.exception.alpha, float)

    def test_meta_results_expose_iterations(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        matrix1 = known_identity(False)
        results = [
            shrinking.bisection_meta(matrix0, matrix1=matrix1),
            shrinking.bisection_with_fixed_block_meta(matrix0, fb_size=1),
            shrinking.newton_meta(matrix0, matrix1=matrix1),
            shrinking.gep_meta(matrix0, matrix1=matrix1),
            shrinking.gep_with_fixed_block_meta(matrix0, fb_size=1),
        ]
        for result in results:
            self.assertAlmostEqual(result.alpha, KNOWN_ALPHA, places=6)
            self.assertGreaterEqual(result.iterations, 1)

    def test_nonpositive_tolerances_are_rejected(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        matrix1 = known_identity(False)
        for tol in (0.0, -1e-6):
            with self.assertRaises(ValueError):
                shrinking.bisection(matrix0, matrix1=matrix1, tol=tol)
            with self.assertRaises(ValueError):
                shrinking.newton(matrix0, matrix1=matrix1, tol=tol)
            with self.assertRaises(ValueError):
                shrinking.bisection_with_fixed_block(
                    matrix0,
                    fb_size=1,
                    tol=tol,
                )

    def test_nonfinite_tolerances_are_rejected(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        matrix1 = known_identity(False)
        for tol in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                shrinking.bisection(matrix0, matrix1=matrix1, tol=tol)
            with self.assertRaises(ValueError):
                shrinking.newton(matrix0, matrix1=matrix1, tol=tol)
            with self.assertRaises(ValueError):
                shrinking.bisection_with_fixed_block(
                    matrix0,
                    fb_size=1,
                    tol=tol,
                )

    def test_bisection_tolerances_at_least_one_are_rejected(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        matrix1 = known_identity(False)
        for tol in (1.0, 2.0):
            with self.assertRaises(ValueError):
                shrinking.bisection(matrix0, matrix1=matrix1, tol=tol)
            with self.assertRaises(ValueError):
                shrinking.bisection_with_fixed_block(
                    matrix0,
                    fb_size=1,
                    tol=tol,
                )

    def test_fixed_block_bisection_uses_generic_stopping_rule(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        self.assertAlmostEqual(
            shrinking.bisection(matrix0, fixed_block_sizes=1, tol=0.5),
            shrinking.bisection_with_fixed_block(matrix0, fb_size=1, tol=0.5),
            places=12,
        )

    def test_numpy_integer_iteration_limits_are_accepted(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        matrix1 = known_identity(False)
        self.assertAlmostEqual(
            shrinking.bisection(
                matrix0,
                matrix1=matrix1,
                max_iterations=cast(int | None, np.int64(60)),
            ),
            KNOWN_ALPHA,
            places=6,
        )
        self.assertAlmostEqual(
            shrinking.newton(
                matrix0,
                matrix1=matrix1,
                max_iterations=cast(int | None, np.int64(10)),
            ),
            KNOWN_ALPHA,
            places=6,
        )

    def test_integer_nested_sequences_work_with_fixed_block_algorithms(
        self,
    ) -> None:
        matrix0 = [[1, 2], [2, 1]]
        self.assertAlmostEqual(
            shrinking.bisection(matrix0, fixed_block_sizes=1),
            0.5,
            places=5,
        )
        self.assertAlmostEqual(
            shrinking.newton(matrix0, fixed_block_sizes=1),
            0.5,
        )
        self.assertAlmostEqual(
            shrinking.gep(matrix0, fixed_block_sizes=1),
            0.5,
        )

    def test_s_dispatch_helper_works(self) -> None:
        result = shrinking.s(
            known_indefinite_matrix(False),
            KNOWN_ALPHA,
            matrix1=known_identity(False),
        )
        expected = np.identity(2)
        expected[0, 1] = 1.0
        expected[1, 0] = 1.0
        np.testing.assert_allclose(result, expected)

    def test_fixed_block_specializations_match_generic_algorithms(
        self,
    ) -> None:
        matrix0 = np.array(
            [[1.0, 0.95, 0.95], [0.95, 1.0, 0.1], [0.95, 0.1, 1.0]],
            dtype=float,
        )
        block_a = matrix0[:1, :1]
        block_b = matrix0[1:, 1:]
        identity_1 = np.identity(1)
        identity_2 = np.identity(2)
        targets = {
            0: np.identity(3),
            1: np.block(
                [[identity_1, np.zeros((1, 2))], [np.zeros((2, 1)), block_b]],
            ),
            2: np.block(
                [[block_a, np.zeros((1, 2))], [np.zeros((2, 1)), identity_2]],
            ),
            3: np.block(
                [[block_a, np.zeros((1, 2))], [np.zeros((2, 1)), block_b]],
            ),
        }

        for which, target in targets.items():
            self.assertAlmostEqual(
                shrinking.bisection(matrix0, matrix1=target),
                shrinking.bisection_with_fixed_block(
                    matrix0,
                    fb_size=1,
                    which=which,
                ),
                places=6,
            )
            self.assertAlmostEqual(
                shrinking.gep(matrix0, matrix1=target),
                shrinking.gep_with_fixed_block(
                    matrix0,
                    fb_size=1,
                    which=which,
                ),
                places=6,
            )

    def test_fixed_block_algorithms_validate_inputs(self) -> None:
        with self.assertRaises(ValueError):
            shrinking.bisection_with_fixed_block(
                np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float),
                fb_size=1,
            )
        with self.assertRaises(ValueError):
            shrinking.gep_with_fixed_block(
                np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
                fb_size=1,
                which=4,
            )
        with self.assertRaises(ValueError):
            shrinking.gep_with_fixed_block(
                np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
                fb_size=1,
                which=True,
            )
        with self.assertRaisesRegex(
            ValueError,
            "fixed blocks in matrix0 must be positive definite",
        ):
            shrinking.gep_with_fixed_block(
                np.array(
                    [[1.0, 1.2, 0.0], [1.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=float,
                ),
                fb_size=2,
            )

    def test_generic_algorithms_reject_non_positive_definite_target(
        self,
    ) -> None:
        matrix0 = known_indefinite_matrix(False)
        matrix1 = known_indefinite_matrix(False)
        for algorithm in (
            shrinking.bisection,
            shrinking.newton,
            shrinking.gep,
        ):
            with self.assertRaises(ValueError):
                algorithm(matrix0, matrix1=matrix1)

    def test_gep_rejects_semidefinite_target(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        semidefinite_target = np.array(
            [[1.0, 0.0], [0.0, 0.0]],
            dtype=float,
        )
        with self.assertRaises(ValueError):
            shrinking.gep(matrix0, matrix1=semidefinite_target)

    def test_generic_algorithms_reject_nonsymmetric_target(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        bad_matrix1 = np.array([[2.0, 0.0], [10.0, 2.0]], dtype=float)
        for algorithm in (
            shrinking.bisection,
            shrinking.newton,
            shrinking.gep,
        ):
            with self.assertRaises(ValueError):
                algorithm(matrix0, matrix1=bad_matrix1)

    def test_generic_algorithms_reject_non_positive_definite_fixed_block(
        self,
    ) -> None:
        matrix0 = np.array(
            [[1.0, 1.2, 0.0], [1.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        for algorithm in (
            shrinking.bisection,
            shrinking.newton,
            shrinking.gep,
        ):
            with self.assertRaises(ValueError):
                algorithm(matrix0, fixed_block_sizes=2)


class CompatibilityLayerTests(TestsBase):
    def test_package_root_exposes_compatibility_module(self) -> None:
        self.assertIs(shrinking.backwards_compatibility, compat)

    def test_legacy_names_match_new_api(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        matrix1 = known_identity(False)
        self.assertAlmostEqual(
            compat.bisection(matrix0, M1=matrix1),
            shrinking.bisection(matrix0, matrix1=matrix1),
        )
        self.assertAlmostEqual(
            compat.bisectionFB(matrix0, fbSize=1),
            shrinking.bisection_with_fixed_block(matrix0, fb_size=1),
        )
        self.assertAlmostEqual(
            compat.newton(matrix0, M1=matrix1),
            shrinking.newton(matrix0, matrix1=matrix1),
        )
        self.assertAlmostEqual(
            compat.GEP(matrix0, M1=matrix1),
            shrinking.gep(matrix0, matrix1=matrix1),
        )
        self.assertAlmostEqual(
            compat.GEPFB(matrix0, fbSize=1),
            shrinking.gep_with_fixed_block(matrix0, fb_size=1),
        )

    def test_legacy_last_iterations_tracks_latest_run(self) -> None:
        compat.lastIterations = 0
        compat.bisection(
            known_indefinite_matrix(False),
            M1=known_identity(False),
        )
        self.assertGreater(compat.lastIterations, 0)

    def test_legacy_star_import_exports_last_iterations(self) -> None:
        namespace: dict[str, object] = dict()
        exec(
            "from shrinking.backwards_compatibility import *",
            namespace,
        )
        self.assertIn("lastIterations", namespace)

    def test_legacy_checkm0_placeholders_are_ignored(self) -> None:
        matrix0 = np.identity(2)
        matrix1 = np.identity(2)
        self.assertEqual(
            compat.bisection(matrix0, M1=matrix1, checkM0=False),
            0.0,
        )
        self.assertEqual(
            compat.newton(matrix0, M1=matrix1, checkM0=False),
            0.0,
        )
        self.assertEqual(
            compat.GEP(matrix0, M1=matrix1, checkM0=False),
            0.0,
        )

    def test_legacy_gep_rejects_removed_semidefinite_mode(self) -> None:
        with self.assertRaises(NotImplementedError):
            compat.GEP(
                known_indefinite_matrix(False),
                M1=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float),
                posdefM1=False,
            )

    def test_legacy_initialize_validates_target_selection(self) -> None:
        with self.assertRaises(ValueError):
            compat.initialize(known_indefinite_matrix(False), None, None)

    def test_legacy_initialize_accepts_none_matrix1_for_fixed_blocks(
        self,
    ) -> None:
        result = compat.initialize(
            known_indefinite_matrix(False),
            None,
            1,
            buildM1=False,
        )
        self.assertTrue(result)

    def test_legacy_initialize_rejects_non_positive_definite_target(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            compat.initialize(
                known_indefinite_matrix(False),
                known_indefinite_matrix(False),
                None,
            )


class InputValidationTests(TestsBase):
    def test_bool_is_not_accepted_as_block_size(self) -> None:
        with self.assertRaises(ValueError):
            shrinking.blocks_to_target(np.identity(2), True)

    def test_numpy_integer_block_sizes_are_accepted(self) -> None:
        matrix0 = known_indefinite_matrix(False)
        numpy_block_size = cast(int, np.int64(1))
        target = shrinking.blocks_to_target(matrix0, numpy_block_size)
        np.testing.assert_allclose(target, np.identity(2))
        self.assertAlmostEqual(
            shrinking.bisection(
                matrix0,
                fixed_block_sizes=numpy_block_size,
            ),
            KNOWN_ALPHA,
            places=6,
        )
        self.assertAlmostEqual(
            shrinking.bisection_with_fixed_block(
                matrix0,
                fb_size=numpy_block_size,
            ),
            KNOWN_ALPHA,
            places=6,
        )

    def test_initialize_problem_reports_invalid_matrix0_first(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "matrix0 must be a square matrix",
        ):
            shrinking.bisection(
                np.ones((2, 3)),
                matrix1=np.identity(2),
            )


if __name__ == "__main__":
    unittest.main()

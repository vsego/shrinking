#!/usr/bin/env python3

"""
Small demonstration script for the shrinking package.
"""

import argparse
import sys
from time import perf_counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

if __package__ in {None, ""}:
    repo_src = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(repo_src))

from shrinking import (
    bisection,
    bisection_with_fixed_block,
    gep,
    gep_with_fixed_block,
    newton,
    s_with_fixed_blocks,
)
from shrinking.backwards_compatibility import checkPD


def randcorr(n: int, dof: int | None = None) -> np.matrix:
    """
    Return a random correlation matrix of order ``n`` and rank ``dof``.

    :param n: Matrix order.
    :param dof: Rank parameter. If omitted, ``n`` is used.
    :return: Random correlation matrix stored as ``numpy.matrix``.
    """
    if dof is None:
        dof = n
    vecs = np.random.randn(n, dof)
    vecs /= np.sqrt((vecs * vecs).sum(axis=-1))[:, None]
    result = np.matrix(np.dot(vecs, vecs.T))
    np.fill_diagonal(result, 1)
    return result


def get_test_matrix(m: int, n: int) -> np.matrix:
    """
    Build a random indefinite correlation matrix with a positive definite
    block.

    :param m: Size of the leading positive definite block.
    :param n: Size of the trailing block.
    :return: Indefinite correlation matrix suitable for the demo.
    """
    while True:
        first_block = randcorr(m)
        second_block = np.identity(n)
        cross_block = np.matrix(np.random.randn(m, n) / (m + n) ** 1.2)
        matrix0 = np.bmat(
            [[first_block, cross_block], [cross_block.T, second_block]],
        )
        if not checkPD(matrix0, False):
            return matrix0


def timed_run(name: str, func, matrix0: np.matrix, **kwargs: int) -> float:
    """
    Execute one algorithm and print its result with timing.

    :param name: Display name of the algorithm.
    :param func: Callable implementing the algorithm.
    :param matrix0: Matrix passed to the algorithm.
    :param kwargs: Additional keyword arguments forwarded to ``func``.
    :return: Computed shrinking parameter.
    """
    started_at = perf_counter()
    alpha = func(matrix0, **kwargs)
    elapsed = perf_counter() - started_at
    print(f"{name:<28} {alpha:.6f} ({elapsed:.5f} sec)")
    return alpha


def run(m: int = 10, n: int = 10, *, show_all_eigs: bool = False) -> None:
    """
    Run the demo workflow and display the corresponding plot.

    :param m: Size of the leading positive definite block.
    :param n: Size of the trailing block.
    :param show_all_eigs: If ``True``, plot all eigenvalue paths
        instead of only the smallest one.
    :return: ``None``.
    """
    matrix0 = get_test_matrix(m, n)

    timed_run("Bisection", bisection, matrix0, fixed_block_sizes=m)
    timed_run(
        "Bisection with fixed block",
        bisection_with_fixed_block,
        matrix0,
        fb_size=m,
    )
    timed_run("Newton", newton, matrix0, fixed_block_sizes=m)
    timed_run("GEP", gep, matrix0, fixed_block_sizes=m)
    alpha = timed_run(
        "GEP with fixed block",
        gep_with_fixed_block,
        matrix0,
        fb_size=m,
    )

    alphas = np.linspace(0, 1, 20)
    eigenvalue_paths = np.array(
        [
            scipy.linalg.eigvalsh(
                s_with_fixed_blocks(matrix0, m, alpha_value),
                check_finite=False,
            )
            for alpha_value in alphas
        ],
    ).transpose()
    plt.plot(alphas, eigenvalue_paths[0])
    if show_all_eigs:
        for index in range(1, m + n):
            plt.plot(alphas, eigenvalue_paths[index], color="0.71")
    plt.axhline(color="black")
    plt.plot([alpha], [0], "rD")
    plt.annotate(
        r"$\alpha_*$",
        xy=(alpha, 0),
        xytext=(-20, 20),
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffffb0", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    if show_all_eigs:
        plt.legend(
            [r"$\lambda_{\min}$", r"$\lambda > \lambda_{\min}$"],
            loc="lower right",
        )
        plt.suptitle(r"$\alpha \mapsto \lambda(S(\alpha))$", fontsize=23)
        ymax = max(
            map(abs, [eigenvalue_paths[0][0], 10 * eigenvalue_paths[0][-1]]),
        )
        xmin, xmax, _, _ = plt.axis()
        plt.axis((xmin, xmax, eigenvalue_paths[0][0], 5 * ymax))
        plt.ylabel(r"$\lambda(S(\alpha))$", fontsize=17)
    else:
        plt.legend([r"$\lambda_{\min}$"], loc="lower right")
        plt.suptitle(
            r"$\alpha \mapsto \lambda_{\min}(S(\alpha))$",
            fontsize=23,
        )
        plt.ylabel(r"$\lambda_{\min}(S(\alpha))$", fontsize=17)
    plt.xlabel(r"$\alpha$", fontsize=17)
    plt.subplots_adjust(left=0.17, right=0.97)
    plt.show()


def main() -> None:
    """
    Parse CLI arguments and launch the demo.

    :return: ``None``.
    """
    if __doc__:
        description = __doc__.splitlines()[0]
    else:
        # In case it was run with `python -OO`.
        description = "shrinking demo"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "m",
        type=int,
        help="size of the top-left positive definite block",
    )
    parser.add_argument("n", type=int, help="size of the bottom-right block")
    parser.add_argument(
        "--show-all-eigs",
        "-a",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="show all eigenvalues instead of only the smallest one",
    )
    args = parser.parse_args()
    run(args.m, args.n, show_all_eigs=args.show_all_eigs)


if __name__ == "__main__":
    main()

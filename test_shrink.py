#!/usr/bin/env python3

"""
A simple test script for the shrinking module.

Syntax: ./test.py m n

Each run creates an invalid correlation matrix of order m+n, and then
calls all five shrinking algorithms and times them. Finally, it draws
a plot showing how the minimal eigenvalue of `S(alpha)` changes as
`alpha` goes from 0 to 1. If not too big, other eigenvalues are shown
as well.

Created by Vedran Šego <vsego@vsego.org>
"""

import numpy as np
import scipy.linalg
from time import time
from sys import argv, exit
import matplotlib.pyplot as plt
import shrinking

# Set to False in order to display the other eigenvalues as well.
# This rarely looks good, because the smallest eigenvalue is usually
# of a too small order of magnitude in comparison with the other
# eigenvalues.
displayOnlyLambdaMin = True

def randcorr(n, dof=None):
    """
    Return a random correlation matrix of order `n` and rank `dof`.

    Written by Robert Kern and taken from 
    http://permalink.gmane.org/gmane.comp.python.scientific.devel/9657
    """

    if dof is None:
        dof = n

    vecs = np.random.randn(n, dof)
    vecs /= np.sqrt((vecs*vecs).sum(axis=-1))[:,None]
    M = np.matrix(np.dot(vecs, vecs.T))
    np.fill_diagonal(M, 1)
    return M

def mkmatrix(m, n):
    """
    Return a random invalid correlation matrix of order `m+n` with
    a positive definite top left block of order `m`.
    """

    while True:
        A = randcorr(m)
        #B = randcorr(n)
        B = np.identity(n)
        Y = np.matrix(np.random.randn(m, n) / (m+n)**1.2)
        M0 = np.bmat([[A, Y], [Y.T, B]])
        if not shrinking.checkPD(M0, False):
            return M0

def timing(st):
    return time() - st

def run(m=10, n=10):
    """
    Run a single test:
    - generate a random invalid correlation matrix of order m+n,
    - shrink it with all 5 methods,
    - draw a graph :math:`\\alpha \\mapsto \\lambda_{\\min}(S(\\alpha))`.
    """

    M0 = mkmatrix(m, n)

    st = time()
    print("Bisection:   %.6f (%.5f sec)" % (shrinking.bisection(M0, fbs=m), timing(st)))

    st = time()
    print("BisectionFB: %.6f (%.5f sec)" % (shrinking.bisectionFB(M0, fbSize=m), timing(st)))

    st = time()
    print("Newton:      %.6f (%.5f sec)" % (shrinking.newton(M0, fbs=m), timing(st)))

    st = time()
    print("GEP:         %.6f (%.5f sec)" % (shrinking.GEP(M0, fbs=m), timing(st)))

    st = time()
    print("GEPFB:       %.6f (%.5f sec)" % (shrinking.GEPFB(M0, fbSize=m), timing(st)))

    alphas = np.linspace(0, 1, 20)
    Svalues = np.array([
        list(scipy.linalg.eigvalsh(
            shrinking.SFB(M0, m, alpha), check_finite = False
        ) for alpha in alphas)
    ]).transpose()
    plt.plot(alphas, Svalues[0])
    if not displayOnlyLambdaMin:
        for p in range(1, n):
            plt.plot(alphas, Svalues[p], color="0.71")
    plt.axhline(color='black')

    alpha = shrinking.bisectionFB(M0, fbSize=m)
    plt.plot([alpha], [0], 'rD')
    # Parameters for plt.annotate taken from
    # http://stackoverflow.com/a/5147430/1667018
    plt.annotate(
        r"$\alpha_*$",
        xy=(alpha, 0), xytext=(-20, 20), textcoords='offset points',
        ha='right', va='bottom',
        bbox = dict(boxstyle='round,pad=0.5', fc='#ffffb0', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )

    if displayOnlyLambdaMin:
        plt.legend([r"$\lambda_{\min}$"], loc='lower right')
    else:
        plt.legend([r"$\lambda_{\min}$", r"$\lambda > \lambda_{\min}$"], loc='lower right')

    x1,x2,y1,y2 = plt.axis()
    plt.xlabel(r"$\alpha$", fontsize=17)
    if displayOnlyLambdaMin:
        plt.suptitle(r"$\alpha \mapsto \lambda_{\min}(S(\alpha))$", fontsize=23)
        plt.ylabel(r"$\lambda_{\min}(S(\alpha))$", fontsize=17)
    else:
        plt.suptitle(r"$\alpha \mapsto \lambda(S(\alpha))$", fontsize=23)
        ym = max(map(abs, [Svalues[0][0], 10 * Svalues[0][-1]]))
        plt.axis((x1, x2, Svalues[0][0], 5 * ym))
        plt.ylabel(r"$\lambda(S(\alpha))$", fontsize=17)
    plt.subplots_adjust(left=0.17, right=0.97)

    plt.show()


if __name__ == "__main__":

    if len(argv) != 3:
        print("Syntax: ./test.py m n")
        exit(1)

    run(int(argv[1]), int(argv[2]))


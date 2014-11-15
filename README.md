`shrinking` - a Python Module for Restoring Definiteness via Shrinking
===

About
---

`shrinking` is a Python module incorporating methods for repairing invalid (indefinite) covariance and correlation matrices, based on the paper Higham, Strabić, Šego, "[Restoring Definiteness via Shrinking, with an Application to Correlation Matrices with a Fixed Block](http://eprints.ma.man.ac.uk/2191/)"

There is one significant difference between the algorithms in the paper and in this module: the paper handles the cases of both positive definite and positive semidefinite targets, while this module is mostly restricted to the positive definite ones, because the semidefinite case is rarely an issue in practice, and it would significantly complicate the code.

The module incorporates the following methods:

* `bisection` -- the bisection method in its general form,
* `bisectionFB` -- a variant of the bisection optimised for a 2x2 block-diagonal target with the diagonal blocks either taken from the starting matrix or set to the identity matrix,
* `newton` -- Newton's method,
* `GEM` -- the generalized eigenvalue method,
* `GEMFB` -- a variant of the generalized eigenvalue method optimised for a 2x2 block-diagonal target with the diagonal blocks either taken from the starting matrix or set to the identity matrix.

Various other routines are available in suitably optimized versions:

* `checkPD` for checking if a given matrix is positive definite (can be used outside of the shrinking context),
* `S` for computing `S(alpha)` in a manner appropriate for various types of the input data (each of the variants can also be invoked directly via specialised functions),
* `x` for computing the eigenvector associated with the smallest eigenvalue (used by Newton's method),
* `blocks2target` for converting a starting matrix and a diagonal blocks descriptor to an appropriate target matrix.

It is possible to incorporate weights into the target matrix that reflect the confidence with which individual matrix entries are known. See the paper above for details on how to do this.

Requirements
---

The module requires `scipy.linalg` from [SciPy](http://www.scipy.org/) and a working LAPACK library. [OpenBLAS](http://www.openblas.net/) proved to be about 1.5-2.5 times faster than [ATLAS](http://math-atlas.sourceforge.net/) with this module (tested on Fedora 20 x86_64).

License
---

See `license.txt` for licensing information.
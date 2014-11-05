`shrinking` - a Python Module for Restoring Definiteness via Shrinking
===

`shrinking` is a Python module incorporating methods for repairing invalid (indefinite) covariance and correlation matrices, based on the paper  
Higham, Strabić, Šego, "[Restoring Definiteness via Shrinking, with an Application to Correlation Matrices with a Fixed Block](http://eprints.ma.man.ac.uk/2191/)"

The module requires `scipy.linalg` and it incorporates the following methods:

* `bisection` -- the bisection method in its general form,
* `bisectionFB` -- a variant of the bisection optimised for a 2x2 block-diagonal target with the diagonal blocks either taken from the starting matrix or set to the identity matrix,
* `newton` -- Newton's method,
* `GEM` -- the generalized eigenvalue method,
* `GEMFB` -- a variant of the generalized eigenvalue method optimised for a 2x2 block-diagonal target with the diagonal blocks either taken from the starting matrix or set to the identity matrix.

Various other routines are available in suitably optimized versions:

* `checkPD` for checking if a given matrix is positive definite (can be used outside of the shrinking context),
* `S` for computing `S(alpha)` in a manner appropriate for various types of the input data (each of the variants can also be invoked directly via specialised functions),
* `blocks2target` for converting a starting matrix and a diagonal blocks descriptor to an appropriate target matrix.

It is possible to incorporate weights into the target matrix that reflect the confidence with which individual matrix entries are known. See the paper above for details on how to do this.
#!/usr/bin/env python3

"""
A module with implementations of the methods in the paper
Higham, Strabic, Sego, "Restoring Definiteness via Shrinking,
with an Application to Correlation Matrices with a Fixed Block"

There is one significant difference between the algorithms
in the paper and in this module: the paper handles the cases
of both positive definite and positive semidefinite targets,
while this module is restricted only to the positive definite
ones, because the semidefinite case is rarely an issue in
practice, and it significantly complicates the code.

Created by Vedran Šego <vsego@vsego.org>
"""

import numpy as np
import scipy.linalg
import time

lastIterations = 0



### Custom exceptions ###

class NotConvergingError(Exception):
    pass



### Posdef checker ###

from scipy.linalg.lapack import get_lapack_funcs

def checkPD(A, exception = True):
    """
    Check if `A` is positive definite; return False or raise exception if not.

    This is cholesky() from scipy.linalg, stripped of all the irrelevant data
    processing (we just need to see if it finishes without raising an exception).
    """

    try:
        info = checkPD.potrf(A, lower = False, overwrite_a = False, clean = False)[1]
    except:
        info = 1

    if info:
        if exception:
            raise ValueError("The matrix is not positive definite")
        else:
            return False

    return True

checkPD.potrf, = get_lapack_funcs(('potrf',))



### S(\alpha) ###

def Sdef(M0, M1, alpha):
    """
    :math:`S(\\alpha)`, as defined in the paper.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    M1 : ndarray or matrix
        A positive definite target matrix.
    alpha : float
        A shrinking parameter.

    Returns
    -------
    S(alpha) : float
        The convex combination :math:`(1-\\alpha) M_0 + \\alpha M_1`.
    """
    return (1-alpha) * M0 + alpha * M1

def Sdif(M0, dM0M1, alpha):
    """
    :math:`S(\\alpha)`, as defined in the paper, computed using `M0`,
    `M0 - M1`, and `alpha`.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    dM0M1 : ndarray or matrix
        M0 - M1, where M1 is a positive definite target matrix.
    alpha : float
        A shrinking parameter.

    Returns
    -------
    S(alpha) : float
        The convex combination :math:`(1-\\alpha) M_0 + \\alpha M_1`.
    """
    return M0 - alpha * dM0M1

def SFB(M0, fbs, alpha):
    """
    :math:`S(\\alpha)`, as defined in the paper, computed using `M0`,
    `fbs`, and `alpha`. It runs fast because it only needs to
    multiply non-fixed elements by 1 - `alpha`, which is done by
    blocks.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    fbs : int or list
        A list of dimensions of the consecutive fixed blocks,
        as described in `bisectionFB`.
    alpha : float
        A shrinking parameter.

    Returns
    -------
    S(alpha) : float
        The convex combination :math:`(1-\\alpha) M_0 + \\alpha M_1`.
    """

    if type(fbs) == int:
        fbs = [ fbs ]
        cnt = 1
    else:
        cnt = len(fbs)

    n = M0.shape[0]
    factor = 1 - alpha
    S = np.asmatrix(M0.copy())
    fbi = 1 # current fb index
    fbsum = fbs[0] # fbs sum (of those that were processed)

    # Handling diagonal parts is slow, so we extract the diagonal
    # and put it back in when we're done
    sfb = sum(fbs)
    if sfb < n:
        D = np.diag(S[sfb:,sfb:])
        np.fill_diagonal(S[sfb:,sfb:], 0)

    # Modify elements in lower rows and
    while fbi < cnt:
        fb = fbs[fbi] # the size of the current fixed block
        nfbsum = fbsum + fb # new fbsum
        S[fbsum:nfbsum,:fbsum] *= factor
        S[:fbsum,fbsum:nfbsum] *= factor
        fbsum = nfbsum
        fbi += 1

    if fbsum < n:
        S[fbsum:,:] *= factor
        S[:fbsum,fbsum:] *= factor
        # Put the diagonal back in
        S[sfb:,sfb:] += np.diag(D)

    return S

def ScorrId(M0, alpha):
    """
    :math:`S(\\alpha)`, as defined in the paper, for target `M1 = I`,
    the identity matrix, under the (unchecked!) assumption that
    M0 has a unit diagonal (used in correlation problems).

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix (with a unit diagonal) to be shrunk.
    alpha : float
        A shrinking parameter.

    Returns
    -------
    S(alpha) : float
        The convex combination :math:`(1-\\alpha) M_0 + \\alpha I`.
    """

    res = (1 - alpha) * M0
    np.fill_diagonal(res, 1)
    return res

def S(M0, alpha, *, M1 = None, dM0M1 = None, fbs = None):
    """
    :math:`S(\\alpha)`, as defined in the paper, computed using the optimal
    method (with respect to the given input).

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    M1 : ndarray or matrix
        A positive definite target matrix.
    dM0M1 : ndarray or matrix
        M0 - M1, where M1 is a positive definite target matrix.
    fbs : int or list
        A list of dimensions of the consecutive fixed blocks,
        as described in `bisectionFB`.
    alpha : float
        A shrinking parameter.

    Returns
    -------
    S(alpha) : float
        The convex combination :math:`(1-\\alpha) M_0 + \\alpha M_1`.
    """
    if fbs:
        #return Sdef(M0, blocks2target(M0, fbs), alpha)
        return SFB(M0, fbs, alpha)
    if dM0M1 != None:
        return Sdif(M0, dM0M1, alpha)
    if M1 != None:
        return Sdef(M0, M1, alpha)
    return ScorrId(M0, alpha) # might not be the smartest default



### Create M1 from M0 and fixed blocks ###

def blocks2target(M0, fbs):
    """
    Create the target matrix defined by the parameter fbs, which consists of the dimensions of the consecutive blocks from :math:`M_0` to be kept as they are. The rest of the elements are set to zero.

    If `sum(fbs)` < `M0.shape[0]`, then the rest of the blocks are considered to be of size 1, i.e., all of the diagonal elements in `M0` are copied, regardless of `fbs`.
    
    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    fb : int or a list of ints
        Dimension(s) of fixed block(s) in M0, used to create M1 as
        M1 = diag(A11, A22,..., Akk, I),
        where Aii are the diagonal blocks of M0, k=len(fbs), and I is the identity matrix.

    Returns
    -------
    M1 : matrix
        Target matrix.
    """

    n = M0.shape[0]
    if type(fbs) == int:
        fbs = [fbs]
    diagBlocks = []
    start = 0
    for dim in fbs:
        end = start + dim
        diagBlocks.append(M0[start:end,start:end])
        start = end
    if start < n:
        diagBlocks.append(np.diag(np.diag(M0[start:, start:])))
    return scipy.linalg.block_diag(*diagBlocks)



### Initializers ###

def initCheckM0(M0):
    """
    Check if M0 is positive definite (return `True` if it is), and if it is symmetric
    (raise a ValueError exception if it is not).
    Intended to be used as a part of initialization.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.

    Returns
    -------
    isPosdef : bool
        `True` if `M0` is positive definite; `False` otherwise.

    Raises
    ------
    ValueError
        If `M0` is not a symmetric matrix.
    """

    # Check that M0 is a square matrix
    (m, n) = M0.shape
    if m != n or (M0.T - M0).any():
        raise ValueError("M0 must be a symmetric matrix")

    # Check if M0 needs no fixing:
    try:
        checkPD(M0)
    except:
        return False
    else:
        return True

def initialize(M0, M1, fbs, checkM0 = True, buildM1 = True):
    """
    The common part for all three methods: verification of the initial conditions
    of the input parameters and creation of `M1` via `fbs`, if so requested.
    Also, positive definiteness of `M0` is checked and `alpha` = 0 returned if it is.

    This function is not meant to be called directly.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    M1 : ndarray or matrix
        A positive definite target matrix.
    fb : int or a list of ints
        Dimension(s) of fixed block(s) in `M0`, used to create M1 as
        `M1` = diag(A11, A22,..., Akk, I),
        where Aii are the diagonal blocks of `M0`, ``k = len(fbs)``, and I is the identity matrix.
    checkM0 : bool
        If True, check that M0 is not positive definite. If it is, returns `None`,
        which later translates to alpha = 0.

    Returns
    -------
    M1 : matrix or NoneType
        If `None`, then `M0` is positive definite and the method should return `alpha` = 1.
        Otherwise, the created and dimension-verified `M1`.

    Raises
    ------
    ValueError
        If the input values are wrong, i.e.,
        - if `M0` is not a symmetric matrix, or
        - neither `M1` nor `fbs` arguments are provided, or
        - both `M1` and `fbs` arguments are provided, or
        - `M1` is provided, but its shape doesn't match that of `M0`.
    """

    global lastIterations
    lastIterations = 0

    # Check M0 for symmetricity and positive definiteness
    if checkM0 and initCheckM0(M0): return None

    # Check that M1 or fbs are provided (but not both)
    if (M1 is None) == (fbs is None):
        raise ValueError("You must provide either M1 or the number of fbs (but not both)")
    if M1 is None:
        # Return block diagonal target created from M0 and fbs
        return blocks2target(M0, fbs) if buildM1 else True
    else:
        # Check that M1 is a square matrix
        if len(M0.shape) == 2 and len(M1.shape) == 2 and (M0.shape == M1.shape):
            return M1
        raise ValueError("M1 must be of the same order as M0")



### Bisections ###

def bisection(M0, *, M1 = None, fbs = None, tol = 10**(-6), checkM0 = True):
    """
    Implementation of the bisection algorithm.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    M1 : ndarray or matrix
        A positive definite target matrix.
    fbs : int or a list of ints
        Dimension(s) of fixed block(s) in `M0`, used to create `M1` as
        ``M1 = diag(A11, A22,..., Akk, I)``,
        where Aii are the diagonal blocks of `M0`, ``k=len(fbs)``, and `I` is the identity matrix.
    tol : float
        Tolerance.
    checkM0 : bool
        If True, check that M0 is not positive definite (if it turns out
        to be so, the function just returns alpha = 0.0).

    Returns
    -------
    alpha : float
        The optimal shrinking parameter for (M0, M1).
    """

    global lastIterations

    # Common initialization
    M1 = initialize(M0, M1, fbs, checkM0, buildM1 = False)
    if M1 is None: return 0.0

    # Bisection
    left = 0.0
    right = 1.0
    dM0M1 = None if fbs else M0 - M1
    while right - left >= tol:
        lastIterations += 1
        if lastIterations > 53: # tol=10**(-16) is done in 53 steps; anything more is unreliable
            raise ValueError("Not converging, probably due to too small tolerance")
        alpha = (left + right) / 2
        try:
            if fbs:
                checkPD(SFB(M0, fbs, alpha))
            else:
                checkPD(Sdif(M0, dM0M1, alpha))
        except:
            left = alpha
        else:
            right = alpha
    return right

def bisectionFB(M0, fbSize = None, tol = 10**(-6), which = 2, checkM0 = True):
    """
    A special case implementation of the bisection algorithm
    for correlation matrices (i.e., M0 must have a unit diagonal),
    when :math:`M_1 = \\operatorname{diag}(A_1, B_1)`, where
    :math:`A_1 = A` or :math:`A_1 = I`, :math:`B_1 = B` or
    :math:`B_1 = I`, and :math:`A` and :math:`B` are the two
    diagonal blocks of :math:`M_0`.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    fbSize : int
        Dimension of the leading principal block A in M0.
    tol : float
        Tolerance.
    which : int
        Which blocks are to be fixed. Two-bit integer, with first bit
        denoting whether the first block is fixed (0 = no, 1 = yes),
        and the second bit denoting whether the second block is fixed.
        The possible values are:
        0: M1 = I,
        1: M1 = diag(I, B),
        2: M1 = diag(A, I),
        3: M1 = diag(A, B),
        where `A` and `B` are (positive definite) diagonal blocks of M0.
    checkM0 : bool
        If True, check that M0 is not positive definite (if it turns out
        to be so, the function just returns alpha = 0.0).

    Returns
    -------
    alpha : float
        The optimal shrinking parameter for (`M0`, `M1`), where `M1`
        depends on `M0` and the argument `which`.
    """

    global lastIterations

    # Check that M0 has a unit diagonal
    if any(x != 1 for x in np.diag(M0)):
        raise ValueError("M0 must have unit diagonal")

    # Check that which has an allowed value
    if which not in {0, 1, 2, 3}:
        raise ValueError("The argument 'which' must be in {0, 1, 2, 3}")

    # Check M0 for symmetricity and positive definiteness
    if checkM0 and initCheckM0(M0): return 0.0

    # Initializations
    lastIterations = 0
    left = 0.0
    right = 1.0
    if which > 0:
        m = fbSize
        n = M0.shape[0] - m

    # If which == 3 and m < n, we can swap A and B to improve the results,
    # because the fixed-block bisection benefits from larger first block.
    # The swap variable will be used later on to properly define Y.
    if (which == 3) and (m < n):
        swap = True
        B = np.matrix(M0[:m, :m])
        A = np.matrix(M0[m:, m:])
    elif which > 0:
        swap = False
        A = np.matrix(M0[:m, :m])
        B = np.matrix(M0[m:, m:])

    # Compute L11 and/or L22 and raise an exception if not a positive definite matrix
    try:
        if which in {2, 3}:
            L11 = scipy.linalg.cholesky(A, lower = True, check_finite = False)
        if which in {1, 3}:
            L22 = scipy.linalg.cholesky(B, lower = True, check_finite = False)
    except:
        raise ValueError("The fixed blocks in M0 must be positive definite")

    # The rest of the initializations:
    # get Y, compute X from L11 * X = Y, form Z = X.T * X,
    # prepare Im (identity of order m
    if which > 0:
        Y = np.matrix(M0[m:,:m] if swap else M0[:m,m:])
        if which in {2, 3}:
            X = np.matrix(scipy.linalg.solve_triangular(L11, Y, lower = True, check_finite = False))
        elif which == 1:
            X = np.matrix(scipy.linalg.solve_triangular(L22, Y.T, lower = True, check_finite = False))
        Z = X.T * X

    # Prepare A or B (a NOT fixed block) or M0 without a diagonal (if neither of the diagonal blocks is fixed)
    if which == 0:
        zeroedM0 = M0.copy()
        np.fill_diagonal(zeroedM0, 0)
    elif which in {1, 2}:
        if which == 1: B = A # At this point, we no longer need to distinguish the blocks
        np.fill_diagonal(B, 0)

    # The main loop
    while right - left > tol:
        lastIterations += 1
        if lastIterations > 53: # tol=10**(-16) is done in 53 steps; anything more is unreliable
            raise ValueError("Not converging, probably due to too small tolerance")
        alpha = (left + right) / 2

        # Computing np.fill_diagonal(alphaI, alpha) with a preexisting
        # square diagonal matrix alphaI is much faster than computing
        # np.diag([alpha] * n) or recreating alphaI from zeros each time,
        # which are both much faster than computing alpha * I, where
        # I = np.identity(n) is precached
        # np.fill_diagonal(alphaI, alpha)
        # T = alphaI + (1 - alpha) * B - (1 - alpha)**2 * Z

        # This seems to be the fastest way (but only slightly faster than
        # the one above
        if which == 0:
            T = (1 - alpha) * zeroedM0
            np.fill_diagonal(T, 1)
        elif which in {1, 2}:
            T = (1 - alpha) * B
            np.fill_diagonal(T, 1)
            T = T - (1 - alpha)**2 * Z
        elif which == 3:
            T = B - (1 - alpha)**2 * Z

        try:
            checkPD(T)
        except:
            left = alpha
        else:
            right = alpha
    return right



### Newton ###

def x(S):
    """
    A helper function for Newton's method that computes the unit eigenvector for the
    smallest eigenvalue of :math:`S(\\alpha)`.

    Parameters
    ----------
    S : matrix or ndarray
        A symmetric matrix.

    Return value
    ------------
    x : array
        A unit eigenvector for the smallest eigenvalue of S.
    """

    v = scipy.linalg.eigh(S, eigvals = (0, 0), check_finite = False)[1]
    return np.matrix(v)
    #w, v = scipy.linalg.eigh(S, check_finite = False)
    #return np.matrix(v[:, 0:1])

def newton(M0, *, M1 = None, fbs = None, tol = 10**(-6), maxIterations = None, checkM0 = True):
    """
    Implementation of the Newton's algorithm.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    M1 : ndarray or matrix
        A positive definite target matrix.
    fb : int or a list of ints
        Dimension(s) of fixed block(s) in M0, used to create M1 as
        M1 = diag(A11, A22,..., Akk, I),
        where Aii are the diagonal blocks of M0, k=len(fbs), and I is the identity matrix.
    tol : float
        Tolerance.
    checkM0 : bool
        If True, check that M0 is not positive definite (if it turns out
        to be so, the function just returns alpha = 0.0).

    Returns
    -------
    alpha : float
        The optimal shrinking parameter for (M0, M1).
    """

    global lastIterations

    # Common initialization
    M1 = initialize(M0, M1, fbs, checkM0)
    if M1 is None: return 0.0

    # Newton's method
    alpha = 0
    dM0M1 = M0 - M1
    while True:
        lastIterations += 1
        if maxIterations and lastIterations > maxIterations:
            raise NotConvergingError("Not converging")
        if fbs:
            vecx = x(SFB(M0, fbs, alpha))
        else:
            vecx = x(Sdif(M0, dM0M1, alpha))
        newAlpha = (vecx.T * M0 * vecx)[0, 0] / (vecx.T * dM0M1 * vecx)[0, 0]
        if abs(newAlpha - alpha) < tol * newAlpha:
            return newAlpha
        alpha = newAlpha



### Generalized eigenvalues ###

def GEP(M0, *, M1 = None, fbs = None, posdefM1 = True, checkM0 = True):
    """
    Implementation of the generalized eigenvalue algorithm.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    M1 : ndarray or matrix
        A positive definite target matrix.
    fbs : int or a list of ints
        Dimension(s) of fixed block(s) in M0, used to create M1 as
        M1 = diag(A11, A22,..., Akk, I),
        where Aii are the diagonal blocks of M0, k=len(fbs), and I is the identity matrix.
    posdefM1 : bool
        If True, M1 is assumed to be positive definite (as opposed
        to positive semidefinite) and the problem is converted to
        a generalized positive definite eigenvalue problem, which
        is then solved by using a routine optimized for this case.
    checkM0 : bool
        If True, check that M0 is not positive definite (if it turns out
        to be so, the function just returns alpha = 0.0).

    Returns
    -------
    alpha : float
        The optimal shrinking parameter for (M0, M1).
    """

    global lastIterations

    # Common initialization
    M1 = initialize(M0, M1, fbs, checkM0)
    if M1 is None: return 0.0
    lastIterations = 1

    if posdefM1:
        ev = (scipy.linalg.eigvalsh(M0, M1, check_finite = False, eigvals = (0, 0)))[0]
        return 1 + 1 / (ev-1)
    else:
        evals = scipy.linalg.eigvals(M0, M0 - M1, check_finite = True)
        return max(ev.real for ev in evals if 0 < ev.real <= 1)

def GEPFB(M0, fbSize = None, which = 2, checkM0 = True):
    """
    A special case implementation of the generalized eigenvalue algorithm
    for correlation matrices (i.e., M0 must have a unit diagonal),
    when :math:`M_1 = \\operatorname{diag}(A_1, B_1)`, where
    :math:`A_1 = A` or :math:`A_1 = I`, :math:`B_1 = B` or
    :math:`B_1 = I`, and :math:`A` and :math:`B` are the two
    diagonal blocks of :math:`M_0`.

    Parameters
    ----------
    M0 : ndarray or matrix
        A symmetric indefinite matrix to be shrunk.
    fbSize : int
        Dimension of the leading principal block A in M0.
    which : int
        Which blocks are to be fixed. Two-bit integer, with first bit
        denoting whether the first block is fixed (0 = no, 1 = yes),
        and the second bit denoting whether the second block is fixed.
        The possible values are:
        0: M1 = I,
        1: M1 = diag(I, B),
        2: M1 = diag(A, I),
        3: M1 = diag(A, B),
        where `A` and `B` are (positive definite) diagonal blocks of M0.
    checkM0 : bool
        If True, check that M0 is not positive definite (if it turns out
        to be so, the function just returns alpha = 0.0).

    Returns
    -------
    alpha : float
        The optimal shrinking parameter for (`M0`, `M1`), where `M1`
        depends on `M0` and the argument `which`.
    """

    global lastIterations

    # Check that M0 has a unit diagonal
    if any(x != 1 for x in np.diag(M0)):
        raise ValueError("M0 must have unit diagonal")

    # Common initialization
    lastIterations = 1

    if which == 0:
        F = M0.copy()
        np.fill_diagonal(F, 0)
        evals = scipy.linalg.eigvalsh(F, check_finite = False, eigvals = (0, 0))
        ev = evals[0]
        return (0 if ev > -1 else 1 + 1 / ev)

    # Common initializations when M1 != I
    m = fbSize
    A = np.matrix(M0[:m, :m])
    B = np.matrix(M0[m:, m:])
    Y = np.matrix(M0[:m,m:])

    if which == 1:
        n = M0.shape[0] - m
        L22 = scipy.linalg.cholesky(B, lower = True, check_finite = False)
        X = np.matrix(scipy.linalg.solve_triangular(L22, Y.T, lower = True, check_finite = False))
        np.fill_diagonal(A, 0)
        C = np.bmat([[ np.zeros((n, n)), X], [X.T, A]])
        evals = scipy.linalg.eigvalsh(C, check_finite = False, eigvals = (0, 0))
        ev = evals[0]
        return (0 if ev > -1 else 1 + 1 / ev)

    if which == 2:
        L11 = scipy.linalg.cholesky(A, lower = True, check_finite = False)
        X = np.matrix(scipy.linalg.solve_triangular(L11, Y, lower = True, check_finite = False))
        np.fill_diagonal(B, 0)
        C = np.bmat([[ np.zeros((m, m)), X], [X.T, B]])
        evals = scipy.linalg.eigvalsh(C, check_finite = False, eigvals = (0, 0))
        ev = evals[0]
        return (0 if ev > -1 else 1 + 1 / ev)

    if which == 3:
        L11 = scipy.linalg.cholesky(A, lower = True, check_finite = False)
        X = np.matrix(scipy.linalg.solve_triangular(L11, Y, lower = True, check_finite = False))
        L22 = scipy.linalg.cholesky(B, lower = True, check_finite = False)
        X = np.matrix(scipy.linalg.solve_triangular(L22, X.T, lower = True, check_finite = False))
        singvals = scipy.linalg.svdvals(X, check_finite = False)
        sv = singvals[0]
        return (0 if sv < 1 else 1 - 1 / sv)

    raise ValueError("The argument 'which' must be in {0, 1, 2, 3}")



### Main ###

if __name__ == "__main__":
    # Some test matrices, M1 being of form diag(A, I), so that
    # bisectionFB can be tested as well
    M0 = np.matrix([[1, 6/5], [6/5, 1]])
    M1 = np.matrix([[1, 0], [0, 1]])

    # Auxiliary timing function
    def dtime():
        global start_time
        end_time = time.time()
        dt = end_time - start_time
        start_time = end_time
        return dt

    start_time = time.time()
    print("Digits:            {0}0{0}".format("".join(str(x) for x in range(1,10))))
    print("Bisection alpha: %.17f (%g seconds)" % (bisection(M0, M1 = M1), dtime()))
    print("Bisection F.B.:  %.17f (%g seconds)" % (bisectionFB(M0, 1), dtime()))
    print("Newton alpha:    %.17f (%g seconds)" % (newton(M0, M1 = M1), dtime()))
    print("GEP alpha:       %.17f (%g seconds)" % (GEP(M0, M1 = M1), dtime()))
    print("GEP F.B.:        %.17f (%g seconds)" % (GEPFB(M0, 1), dtime()))


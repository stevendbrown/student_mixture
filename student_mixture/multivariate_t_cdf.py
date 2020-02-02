import numpy as np
from scipy.linalg import pinv, sqrtm, cholesky
from scipy.integrate import quad
from scipy.stats import t
from scipy.special import betainc
from .multivariate_t_mc import _multivariate_t_cdf_single_sample

###############################################################################
# Functions that estimate the cumulative distribution function of the standard multivariate t-distribution

def _standard_t_cdf_bivariate(x, corr_mat, dof, abseps=1e-8, releps=1e-8):
    """
    Bivariate Student's t cumulative density function.
    Adapted from MATLAB's mvtcdf.bvtcdf_generalNu function.
    Algorithm based on:
    Genz, A. (2004) "Numerical Computation of Rectangular Bivariate
    and Trivariate Normal and t Probabilities", Statistics and
    Computing, 14(3):251-260.

    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, 2)
    corr_mat : array_like, shape (2, 2)
        Correlation matrix of the distribution,must be symmetric and positive
        definite, with all elements of the diagonal being 1
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    abseps: float
        Absolute error tolerance for integration
    releps: float
        Relative error tolerance for integration
    

    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`
    """

    rho = corr_mat[0, 1]
    n =  x.shape[0]
    if rho >= 0:
        tau_s = t.cdf(x.min(axis=1), dof)
    else:
        tau_s = np.maximum(t.cdf(x[:, 0], dof) - t.cdf(x[:, 1], dof), np.zeros(n))

    lower_bound = np.sign(rho) + np.pi/2 if rho == 0 else np.sign(rho)
    upper_bound = np.arcsin(rho)

    ig = np.zeros_like(tau_s)

    for i in range(n):
        x1 = x[i, 0]
        x2 = x[i, 1]
        if np.isfinite(x1) and np.isfinite(x2):
            ig[i] = quad(bivariate_intergrand, lower_bound, upper_bound, (x1, x2, dof),
                         epsabs=abseps, epsrel=releps)[0]
    return tau_s + ig / (2 * np.pi)


def _standard_t_cdf_trivariate(x, corr_mat, dof, abseps=1e-8, releps=1e-8):
    """
    Trivariate Student's t cumulative density function.
    Adapted from MATLAB's mvtcdf.tvtcdf function.
    Algorithm based on:
    Genz, A. (2004) "Numerical Computation of Rectangular Bivariate
    and Trivariate Normal and t Probabilities", Statistics and
    Computing, 14(3):251-260.

    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, 3)
    corr_mat : array_like, shape (3, 3)
        Correlation matrix of the distribution,must be symmetric and positive
        definite, with all elements of the diagonal being 1
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    abseps: float
        Absolute error tolerance for integration
    releps: float
        Relative error tolerance for integration
    

    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`
    """
    n = x.shape[0]

    permutation = {0: [2, 1, 0], 1: [0, 2, 1], 2: [0, 1, 2]}
    variable_switch = {0: [2, 1, 0], 1: [1, 0, 2], 2: [0, 1, 2]}
    rho = corr_mat.ravel()[[1, 2, 5]]
    i_max = rho.argmax()
    rho = rho[permutation[i_max]]
    (rho_12, rho_13, rho_23) = rho
    x = x[:, variable_switch[i_max]]

    if rho_23 >= 0:
        tau_r_tag = _standard_t_cdf_bivariate(np.array([x[:, 0], x[:, 1:].min(axis=1)]).T,
                                              np.eye(2), dof, abseps=abseps, releps=releps)
    else:
        tau_r_tag1 = _standard_t_cdf_bivariate(x[:, 0:2], np.eye(2), dof, abseps=abseps, releps=releps)
        tau_r_tag2 = _standard_t_cdf_bivariate(np.array([x[:, 0], -x[:, 2].min(axis=1)]).T,
                                               np.eye(2), dof, abseps=abseps, releps=releps)
        tau_r_tag = np.maximum(tau_r_tag1 - tau_r_tag2, np.zeros(n))

    tau_0 = np.zeros_like(tau_r_tag)
    tau_1 = np.zeros_like(tau_r_tag)
    tau_2 = np.zeros_like(tau_r_tag)

    lower_bound = np.sign(rho_23) + np.pi/2 if rho == 0 else np.sign(rho_23)
    upper_bound = np.arcsin(rho_23)
    for i in range(n):
        x1 = x[i, 0]
        x2 = x[i, 1]
        x3 = x[i, 2]
        if np.isfinite(x1) and np.isfinite(x2) and np.isfinite(x3):
            tau_0[i] = quad(trivariate_intergrand1,lower_bound, upper_bound, (x1, x2, x3, dof),
                            epsabs=abseps, epsrel=releps)[0]

    if abs(rho_12) > 0:
        lower_bound = 0
        upper_bound = np.arcsin(rho_12)
        for i in range(n):
            x1 = x[i, 0]
            x2 = x[i, 1]
            x3 = x[i, 2]
            if np.isfinite(x1) and np.isfinite(x2) and np.isfinite(x3):
                tau_1[i] = quad(trivariate_intergrand2, lower_bound, upper_bound,
                                (x1, x2, x3, dof, rho_12, rho_13, rho_23),
                                epsabs=abseps, epsrel=releps)[0]

    if abs(rho_12) > 0:
        lower_bound = 0
        upper_bound = np.arcsin(rho_13)
        for i in range(n):
            x1 = x[i, 0]
            x3 = x[i, 1]
            x2 = x[i, 2]
            if np.isfinite(x1) and np.isfinite(x2) and np.isfinite(x3):
                tau_1[i] = quad(trivariate_intergrand2, lower_bound, upper_bound,
                                (x1, x2, x3, dof, rho_13, rho_12, rho_23),
                                epsabs=abseps, epsrel=releps)[0]

    return tau_r_tag + (tau_0 + tau_1 + tau_2) / (2 * np.pi)


def _standard_t_cdf_multivariate(x, corr_mat, dof, tol=1e-4, max_evaluations=1e+7):
    """
    Wrapper function for .
    See multivariate_t_mc for more on algorithm

    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, n_features)
    corr_mat : array_like, shape (n_features, n_features)
        Correlation matrix of the distribution,must be symmetric and positive
        definite, with all elements of the diagonal being 1
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    tol: float
        Tolerance for quasi-Monte Carlo algorithm
    max_evaluations: float
        Maximum points to evaluate with quasi-Monte Carlo algorithm
    

    Returns
    -------
    cdf : ndarray or scalar
        Cumulative density function evaluated at `x`
    """
    n = x.shape[0]
    cdf = np.zeros(n)
    err = np.zeros(n)
    chol = cholesky(corr_mat)
    for i in range(n):
        cdf[i], err[i] = _multivariate_t_cdf_single_sample(x[i, :], chol, dof, tol=tol, max_evaluations=max_evaluations)
    return cdf

###############################################################################
# Integral functions

def bivariate_intergrand(theta, b1, b2, dof):
    sin_theta = np.sin(theta)
    cos_theta_squared = np.cos(theta) ** 2
    return (1 / (1 +
                 (((b1 * sin_theta - b2) ** 2) / cos_theta_squared +
                  (b1 ** 2)) / dof)) ** (dof / 2)


def compute_lower_tail(x, dof):
    p = betainc(dof / 2, 0.5, dof / (dof + (x ** 2))) / 2
    reflect = x > 0
    p[reflect] = 1 - p[reflect]
    return p

def u_k(sin_theta, cos_theta_squared, b1, bj, bk, rho_1k, rho_1j, rho_23):
    sin_phi = sin_theta * rho_1k / rho_1j
    numerator = bk * cos_theta_squared - b1 * (sin_phi - rho_23 * sin_theta) - bj * (rho_23 - sin_theta * sin_phi)
    denominator = np.sqrt(cos_theta_squared * (cos_theta_squared - sin_phi * sin_phi
                                               - rho_23 * (rho_23 - 2 * sin_theta * sin_theta)))
    return numerator / denominator


def trivariate_intergrand1(theta, b1, b2, b3, dof):
    sin_theta = np.sin(theta)
    cos_theta_squared = np.cos(theta) ** 2
    w = np.sqrt(1 / (1 + (((b2 * sin_theta - b3) ** 2) / cos_theta_squared + (b2 ** 2)) / dof))
    return (w ** dof) * compute_lower_tail(b1 * w, dof)


def trivariate_intergrand2(theta, b1, bj, bk, dof, rho_1j, rho_1k, rho_23):
    sin_theta = np.sin(theta)
    cos_theta_squared = np.cos(theta) ** 2
    w = np.sqrt(1 / (1 + (((b1 * sin_theta - bj) ** 2) / cos_theta_squared + (b1 ** 2)) / dof))
    return (w ** dof) * compute_lower_tail(u_k(sin_theta, cos_theta_squared, b1, bj, bk, rho_1j, rho_1k, rho_23), dof)

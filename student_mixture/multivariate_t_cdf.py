import numpy as np
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.special import betainc
from .multivariate_t_mc import _multivariate_t_cdf_single_sample

###############################################################################
# Functions that estimate the cumulative distribution function of the standard multivariate t-distribution


def _standard_t_cdf_univariate(x, dof, abseps=1e-8, releps=1e-8):
    """
    Standard univariate Student's t cumulative density function.
    Reference:
    Johnson, N. L., Kotz, S., Balakrishnan, N. ( 1995 ). Continuous
    Univariate Distributions. Vol. 2, 2nd ed. New York : Wiley.

    Parameters
    ----------
    x : array_like
        Standardized samples, shape (n_samples, )
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
    return 1.0 - 0.5 * betainc(dof / 2.0, 0.5, dof / (dof + x ** 2))


def _standard_t_cdf_bivariate(x, corr_mat, dof, abseps=1e-8, releps=1e-8):
    """
    Standard bivariate Student's t cumulative density function.
    Algorithm based on:
    Genz, A. (2004). Numerical computation of rectangular bivariate
    and trivariate normal and t probabilities. Statistics and
    Computing, 14(3), 251-260.

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
    n = x.shape[0]
    if rho >= 0:
        tau_s = _standard_t_cdf_univariate(x.min(axis=1), dof)
    else:
        tau_s = np.maximum(_standard_t_cdf_univariate(x[:, 0], dof) - _standard_t_cdf_univariate(-x[:, 1], dof), 0)

    integral = np.zeros_like(tau_s)
    lower_bound = np.sign(rho) * np.pi / 2.0 if rho != 0 else np.pi / 2.0
    upper_bound = np.arcsin(rho)

    for i in range(n):
        x1 = x[i, 0]
        x2 = x[i, 1]
        if np.isfinite(x1) and np.isfinite(x2):
            integral[i] = quad(bivariate_integrand, lower_bound, upper_bound, (x1, x2, dof),
                               epsabs=abseps, epsrel=releps)[0]
    return tau_s + integral / (2 * np.pi)


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


def bivariate_integrand(theta, b1, b2, nu):
    sin_theta = np.sin(theta)
    cos_theta_squared = np.cos(theta) ** 2
    return (1 + ((b1 ** 2) + (b2 ** 2) - 2 * sin_theta * b1 * b2)
            / (nu * cos_theta_squared)) ** (-nu / 2.0)

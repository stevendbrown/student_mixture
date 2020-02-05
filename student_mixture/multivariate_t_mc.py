import numpy as np
from scipy.special import erfc, erfcinv, gammaincinv


def normp(z):
    return 0.5 * erfc(-z / (2 ** 0.5))


def normq(z):
    return -(2 ** 0.5) * erfcinv(2 * z)


def chiq(z, nu):
    return (2 * gammaincinv(nu / 2, z)) ** (0.5)


def _multivariate_t_cdf_single_sample(x, chol, dof, tol=1e-4, max_evaluations=1e+7):
    """
    Wrapper function for estimating the cdf of single sample 'x'.

    Parameters
    ----------
    x : array_like
        Standardized sample, shape (n_features,)
    chol : array_like, shape (n_features, n_features)
        Cholesky decomposition of the correlation matrix of the distribution
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    tol: float
        Tolerance for quasi-Monte Carlo algorithm
    max_evaluations: float
        Maximum points to evaluate with quasi-Monte Carlo algorithm
    

    Returns
    -------
    cdf : scalar
        Cumulative density function evaluated at `x`
    """
    if np.all(x == np.inf):
        return 1, 0

    elif np.all(x == -np.inf):
        return 0, 0

    else:
        return _multivariate_t_cdf_quasi_monte_carlo(x, chol, dof, tol=tol, max_evaluations=max_evaluations)


def _multivariate_t_cdf_quasi_monte_carlo(x, chol, dof, tol=1e-4, max_evaluations=1e+7):
    """
    Function for estimating the cdf of single sample 'x'.
    Adapted from MATLAB's mvtcdfqmc function.
    Algorithm based on:
    Genz, A., & Bretz, F. (1999). Numerical computation of multivariate
    t-probabilities with application to power calculation of multiple
    contrasts. Journal of Statistical Computation and Simulation, 63(4),
    103-117.
    Genz, A., & Bretz, F. (2002). Comparison of methods for the computation
    of multivariate t probabilities. Journal of Computational and Graphical
    Statistics, 11(4), 950-971.

    Parameters
    ----------
    x : array_like
        Standardized sample, shape (n_features,)
    chol : array_like, shape (n_features, n_features)
        Cholesky decomposition of the correlation matrix of the distribution
    dof : float
        Degrees-of-freedom of the distribution, must be a positive real number
    tol: float
        Tolerance for quasi-Monte Carlo algorithm
    max_evaluations: float
        Maximum points to evaluate with quasi-Monte Carlo algorithm
    

    Returns
    -------
    cdf : scalar
        Cumulative density function evaluated at `x`
    """
    primes = [173, 263, 397, 593, 907, 1361, 2053, 3079, 4621, 6947, 10427, 15641, 23473,
              35221, 52837, 79259, 118891, 178349, 267523, 401287, 601942, 902933, 1354471, 2031713]
    m = chol.shape[0]
    c = np.diag(chol)
    x = x / c
    chol = chol / np.tile(c, (m, 1))

    mc_repetitions = 25
    p = 0
    sigma_squared = np.inf

    fun_evals = 0
    err = np.nan

    for i in primes:
        if (fun_evals + 2 * mc_repetitions * i) > max_evaluations:
            break

        q = 2 ** (np.arange(1, m + 1) / (m + 1))

        tau_hat, sigma_squared_tau_hat = _mvt_qmc(x, chol, dof, mc_repetitions, i, q)

        fun_evals += 2 * mc_repetitions * i

        p = p + (tau_hat - p) / (1 + sigma_squared_tau_hat / sigma_squared)
        sigma_squared = sigma_squared_tau_hat / (1 + sigma_squared_tau_hat / sigma_squared)
        err = 3.5 * np.sqrt(sigma_squared)

        if err < tol:
            return p, err
    return p, err


def _mvt_qmc(x, chol, dof, mc_repetitions, prime, q):
    qq = np.outer(np.arange(1, prime + 1), q)
    tau_hat = np.zeros(mc_repetitions)

    for rep in range(mc_repetitions):
        r = np.tile(np.random.uniform(low=0.0, high=1.0, size=len(q)), (prime, 1))
        w = abs(2 * ((qq + r) % 1) - 1)
        tau_hat[rep] = (_f_qrsvn(x, chol, dof, w) + _f_qrsvn(x, chol, dof, 1 - w)) / 2

    return np.mean(tau_hat), np.var(tau_hat) / mc_repetitions


def _f_qrsvn(x, chol, dof, w):
    n_points = w.shape[0]
    dim = x.shape[0]
    eps = np.finfo(float).eps

    s_nu = chiq(w[:, -1], dof) / (dof ** 0.5)
    emd = normp(s_nu * x[0])
    tau = np.copy(emd)
    y = np.zeros((n_points, dim))

    for i in range(1, dim):
        z = np.minimum(np.maximum(emd * w[:, -1 - i], eps / 2), 1 - eps / 2)
        y[:, i - 1] = normq(z)
        y_sum = np.dot(y, chol[:, i])
        emd = normp(s_nu * x[i] - y_sum)
        tau = tau * emd

    return tau.sum() / tau.shape[0]

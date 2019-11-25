"""Student's t-distribution Mixture Model."""

# Author: Omri Tomer <omritomer1@mail.tau.ac.il>
# License: BSD 3 clause

import numpy as np

from scipy import linalg
from scipy.special import gamma, gammaln, digamma, polygamma
from scipy.optimize import newton

from sklearn.mixture.base import BaseMixture, _check_shape, _check_X
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import row_norms

import warnings

from sklearn import cluster
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.utils.fixes import logsumexp


###############################################################################
# Student mixture shape checkers used by the StudentMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means


def _check_dofs(dofs, n_components):
    """Check the user provided 'dofs'.

    Parameters
    ----------
    dofs : array-like, shape (n_components,)
        The degrees-of-freedom of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    dofs = check_array(dofs, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(dofs, (n_components,), 'dofs')

    # check range
    if any(np.less(dofs, 2)):
        raise ValueError("The parameter 'dofs' should be in the range "
                         "[2, inf), but got min value %.5f"
                         % (np.min(dofs)))
    return dofs


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=covariance_type == 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[covariance_type],
                 '%s precision' % covariance_type)

    _check_precisions = {'full': _check_precisions_full,
                         'tied': _check_precision_matrix,
                         'diag': _check_precision_positivity,
                         'spherical': _check_precision_positivity}
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


###############################################################################
# Student mixture parameters estimators (used by the M-Step)

def _estimate_student_covariances_full(resp, gamma_priors, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * gamma_priors[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances


def _estimate_student_covariances_tied(resp, gamma_priors, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[::len(covariance) + 1] += reg_covar
    return covariance


def _estimate_student_covariances_diag(resp, gamma_priors, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot((resp * gamma_priors).T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot((resp * gamma_priors).T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_student_covariances_spherical(resp, gamma_priors, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_student_covariances_diag(resp, gamma_priors, X, nk,
                                               means, reg_covar).mean(1)


def _estimate_student_parameters(X, resp, gamma_priors, reg_covar, covariance_type):
    """Estimate the Student's t-distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot((resp * gamma_priors).T, X) / (resp * gamma_priors).T.sum(1)[:, np.newaxis]
    covariances = {"full": _estimate_student_covariances_full,
                   "tied": _estimate_student_covariances_tied,
                   "diag": _estimate_student_covariances_diag,
                   "spherical": _estimate_student_covariances_spherical
                   }[covariance_type](resp, gamma_priors, X, nk, means, reg_covar)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type in 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            print(covariances)
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol

def _compute_gamma_priors(X, means, precisions_chol, covariance_type, dofs):
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    if covariance_type == 'full':
        prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        prob = (np.sum((means ** 2 * precisions), 1) -
                2. * np.dot(X, (means * precisions).T) +
                np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        prob = (np.sum(means ** 2, 1) * precisions -
                2 * np.dot(X, means.T * precisions) +
                np.outer(row_norms(X, squared=True), precisions))

    return (dofs + n_features) / (dofs + prob)

###############################################################################
# Student mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol

def _estimate_log_student_prob(X, means, precisions_chol, covariance_type, dofs):
    """Estimate the log Student probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    dofs : array-like, shape (n_components,)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    log_prob = np.log(1 + (prob / dofs))
    return gammaln(0.5*(dofs + n_features)) - gammaln(0.5 * dofs) -.5 * (n_features * np.log(dofs * np.pi) + (dofs + n_features) * log_prob) + log_det


def _initialize_dofs(X, covarainces, n_components, resp, gamma_priors, current_dofs, n_features, max_iter=1000, tol=1.48e-08):
    if current_dofs is None:
        arg_resp = resp.argmax(1)
        current_dofs = np.zeros((n_components,))
        for k in range(n_components):
            norm_cov = np.cov(X[arg_resp == k, :].T);
            cov = covarainces[k]
            current_dofs[k] = np.minimum(np.maximum(2, np.mean(2 * (norm_cov / cov) / (norm_cov / cov - 1))), 30)
    return _estimate_dofs(resp, gamma_priors, current_dofs, n_features, max_iter=max_iter, tol=tol)

def _estimate_dofs(resp, gamma_priors, current_dofs, n_features, max_iter=1000, tol=1.48e-08):

    n_components = current_dofs.shape[0]
    dofs = np.zeros((n_components,))
    for k in range(n_components):
        constant = (
                1.0
                + (1.0 / resp[:, k].sum(axis=0))
                * (resp[:, k] * (np.log(gamma_priors[:, k]) - gamma_priors[:, k])).sum(axis=0)
                + digamma(1.0 * (current_dofs[k] + n_features) / 2.0)
                - np.log(1.0 * (current_dofs[k] + n_features) / 2.0)
        )

        def func(df):
            return -digamma(1.0 * df / 2.0) + np.log(1.0 * df / 2.0) + constant

        def first_derivative(df):
            return -(1.0 / 2.0) * polygamma(1, 1.0 * df / 2.0) + 1.0 / (2.0 * df)

        def second_derivative(df):
            return -(1.0 / 4.0) * polygamma(2, 1.0 * df / 2.0) - 1 / (2.0 * df * df)

        dofs[k] = np.asscalar(
            newton(
                func,
                current_dofs[k],
                first_derivative,
                args=(),
                maxiter=max_iter,
                tol=tol,
                fprime2=second_derivative,
                full_output=False,
                disp=False,
            )
        )
    return dofs


class StudentMixture(BaseMixture):
    """Student Mixture.

    Representation of a Student's t- mixture model probability distribution.
    This class allows to estimate the parameters of a Student's mixture
    distribution.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
        String describing the type of covariance parameters to use.
        Must be one of:

        'full'
            each component has its own general covariance matrix
        'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'spherical'
            each component has its own single variance

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    dof_tol : float, defaults to 1e-3.
        The degrees-of-freedom estimation convergence threshold.

    dof_max_iter : int, defaults to 100.
        The number of iterations to perform for degrees-of-freedom estimation.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    dofs_ : array-like, shape (n_components,)
        The degrees-of-freedom of each mixture components.

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Student's t-distributions
        can be equivalently parameterized by the precision matrices. Storing
        the precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Student's t-distributions can be equivalently parameterized by the
        precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.


    """

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 dof_tol=1e-3, dof_max_iter=100,
                 weights_init=None, means_init=None, precisions_init=None, dofs_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.dof_tol = dof_tol
        self.dof_max_iter = dof_max_iter
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.dofs_init = dofs_init

    def _check_parameters(self, X):
        """Check the Student mixture parameters are well defined."""
        _, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)
        if self.dofs_init is not None:
            self.dofs_init = _check_dofs(self.dofs_init,
                                               self.n_components)

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised. After fitting, it
        predicts the most probable label for the input data points.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = (-np.infty if do_init else self.lower_bound_)

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp, gamma_priors = self._e_step(X)
                self._m_step(X, log_resp, gamma_priors)
                lower_bound = self._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp, _ = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)
        gamma_priors = np.ones(resp.shape)

        self._initialize(X, resp, gamma_priors)


    def _initialize(self, X, resp, gamma_priors):
        """Initialization of the Student mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)

        gamma_priors : array-like, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape

        weights, means, covariances = _estimate_student_parameters(
            X, resp, gamma_priors, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init
        self.dofs_ = _initialize_dofs(X, covariances, self.n_components, resp, gamma_priors,  None, n_features) if self.dofs_init is None else self.dofs_init
        '''self.dofs_ = _estimate_dofs(resp, gamma_priors,  np.tile(15, self.means_.shape[0]), n_features)if self.means_init is None else self.means_init'''

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp, gamma_priors = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp, gamma_priors

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        gamma_priors = self._estimate_gamma_priors(X)
        return log_prob_norm, log_resp, gamma_priors

    def _estimate_gamma_priors(self, X):
        """Estimate the gamma priors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        gamma_priors : array, shape (n_samples, n_component)
        """
        return _compute_gamma_priors(X, self.means_, self.precisions_cholesky_, self.covariance_type, self.dofs_)

    def _m_step(self, X, log_resp, gamma_priors):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, n_features = X.shape
        resp = np.exp(log_resp)
        self.weights_, self.means_, self.covariances_ = (
            _estimate_student_parameters(X, resp, gamma_priors, self.reg_covar,
                                          self.covariance_type))
        self.dofs_ = _estimate_dofs(resp, gamma_priors, self.dofs_, n_features)
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def _estimate_log_prob(self, X):
        return _estimate_log_student_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type, self.dofs_)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_, self.dofs_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_, self.dofs_) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + 2 * self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

# Author: Wei Xue <xuewei4d@gmail.com>
#         Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import sys
import copy
import warnings
import pytest

import numpy as np
from scipy import stats, linalg

from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from io import StringIO
from sklearn.metrics.cluster import adjusted_rand_score
from student_mixture import StudentMixture
from student_mixture_functions import (
    _estimate_student_scales_full,
    _estimate_student_scales_tied,
    _estimate_student_scales_diag,
    _estimate_student_scales_spherical,
    _compute_precision_cholesky,
    _compute_log_det_cholesky,
)
from multivariate_t_functions import _multivariate_t_random

from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.extmath import fast_logdet
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.tests.test_validation import assert_raise_message, assert_warns_message, ignore_warnings


SCALES_TYPE = ['full', 'tied', 'diag', 'spherical']


def generate_data(n_samples, n_features, weights, locations, scales, dofs,
                  covariance_type):
    rng = np.random.RandomState(0)

    X = []
    if covariance_type == 'spherical':
        for _, (w, l, s, d) in enumerate(zip(weights, locations,
                                          scales['spherical'], dofs)):
            X.append(_multivariate_t_random(l, s * np.eye(n_features), d,
                                             int(np.round(w * n_samples)), rng))
    if covariance_type == 'diag':
        for _, (w, l, s, d) in enumerate(zip(weights, locations,
                                          scales['diag'], dofs)):
            X.append(_multivariate_t_random(l, np.diag(s), d,
                                             int(np.round(w * n_samples)), rng))
    if covariance_type == 'tied':
        for _, (w, l, d) in enumerate(zip(weights, locations, dofs)):
            X.append(_multivariate_t_random(l, scales['tied'], d,
                                             int(np.round(w * n_samples)), rng))
    if covariance_type == 'full':
        for _, (w, l, s, d) in enumerate(zip(weights, locations,
                                          scales['full'], dofs)):
            X.append(_multivariate_t_random(l, s, d,
                                             int(np.round(w * n_samples)), rng))

    X = np.vstack(X)
    return X


class RandomData:
    def __init__(self, rng, n_samples=200, n_components=2, n_features=2,
                 scaling=50):
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features

        self.weights = rng.rand(n_components)
        self.weights = self.weights / self.weights.sum()
        self.means = rng.rand(n_components, n_features) * scaling
        self.scales = {
            'spherical': .5 + rng.rand(n_components),
            'diag': (.5 + rng.rand(n_components, n_features)) ** 2,
            'tied': make_spd_matrix(n_features, random_state=rng),
            'full': np.array([
                make_spd_matrix(n_features, random_state=rng) * .5
                for _ in range(n_components)])}
        self.precisions = {
            'spherical': 1. / self.scales['spherical'],
            'diag': 1. / self.scales['diag'],
            'tied': linalg.inv(self.scales['tied']),
            'full': np.array([linalg.inv(scale)
                             for scale in self.scales['full']])}
        self.dofs = rng.randint(2, 50, (n_components,)).astype(float) + rng.rand(n_components)

        self.X = dict(zip(SCALES_TYPE, [generate_data(
            n_samples, n_features, self.weights, self.means, self.scales,
            self.dofs, scale_type) for scale_type in SCALES_TYPE]))
        self.Y = np.hstack([np.full(int(np.round(w * n_samples)), k,
                                    dtype=np.int)
                            for k, w in enumerate(self.weights)])


def test_student_mixture_attributes():
    # test bad parameters
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)

    n_components_bad = 0
    smm = StudentMixture(n_components=n_components_bad)
    assert_raise_message(ValueError,
                         "Invalid value for 'n_components': %d "
                         "Estimation requires at least one component"
                         % n_components_bad, smm.fit, X)

    # covariance_type should be in [spherical, diag, tied, full]
    scale_type_bad = 'bad_scale_type'
    smm = StudentMixture(scale_type=scale_type_bad)
    assert_raise_message(ValueError,
                         "Invalid value for 'scale_type': %s "
                         "'scale_type' should be in "
                         "['spherical', 'tied', 'diag', 'full']"
                         % scale_type_bad,
                         smm.fit, X)

    tol_bad = -1
    smm = StudentMixture(tol=tol_bad)
    assert_raise_message(ValueError,
                         "Invalid value for 'tol': %.5f "
                         "Tolerance used by the EM must be non-negative"
                         % tol_bad, smm.fit, X)

    reg_scale_bad = -1
    smm = StudentMixture(reg_scale=reg_scale_bad)
    assert_raise_message(ValueError,
                         "Invalid value for 'reg_scale': %.5f "
                         "regularization on scale must be "
                         "non-negative" % reg_scale_bad, smm.fit, X)

    max_iter_bad = 0
    smm = StudentMixture(max_iter=max_iter_bad)
    assert_raise_message(ValueError,
                         "Invalid value for 'max_iter': %d "
                         "Estimation requires at least one iteration"
                         % max_iter_bad, smm.fit, X)

    n_init_bad = 0
    smm = StudentMixture(n_init=n_init_bad)
    assert_raise_message(ValueError,
                         "Invalid value for 'n_init': %d "
                         "Estimation requires at least one run"
                         % n_init_bad, smm.fit, X)

    init_params_bad = 'bad_method'
    smm = StudentMixture(init_params=init_params_bad)
    assert_raise_message(ValueError,
                         "Unimplemented initialization method '%s'"
                         % init_params_bad,
                         smm.fit, X)

    algorithm_bad = 'bad_algorithm'
    smm = StudentMixture(algorithm=algorithm_bad)
    assert_raise_message(ValueError,
                         "Unimplemented algorithm '%s'"
                         % algorithm_bad,
                         smm.fit, X)

    # test good parameters
    n_components, tol, n_init, max_iter, reg_scale = 2, 1e-4, 3, 30, 1e-1
    scale_type, init_params = 'full', 'random'
    smm = StudentMixture(n_components=n_components, tol=tol, n_init=n_init,
                          max_iter=max_iter, reg_scale=reg_scale,
                          scale_type=scale_type,
                          init_params=init_params).fit(X)

    assert smm.n_components == n_components
    assert smm.covariance_type == scale_type
    assert smm.tol == tol
    assert smm.reg_covar == reg_scale
    assert smm.max_iter == max_iter
    assert smm.n_init == n_init
    assert smm.init_params == init_params


def test_check_X():
    from student_mixture_functions import _check_X
    rng = np.random.RandomState(0)

    n_samples, n_components, n_features = 10, 2, 2

    X_bad_dim = rng.rand(n_components - 1, n_features)
    assert_raise_message(ValueError,
                         'Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X_bad_dim.shape[0]),
                         _check_X, X_bad_dim, n_components)

    X_bad_dim = rng.rand(n_components, n_features + 1)
    assert_raise_message(ValueError,
                         'Expected the input data X have %d features, '
                         'but got %d features'
                         % (n_features, X_bad_dim.shape[1]),
                         _check_X, X_bad_dim, n_components, n_features)

    X = rng.rand(n_samples, n_features)
    assert_array_equal(X, _check_X(X, n_components, n_features))


def test_check_weights():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    n_components = rand_data.n_components
    X = rand_data.X['full']

    s = StudentMixture(n_components=n_components)

    # Check bad shape
    weights_bad_shape = rng.rand(n_components, 1)
    s.weights_init = weights_bad_shape
    assert_raise_message(ValueError,
                         "The parameter 'weights' should have the shape of "
                         "(%d,), but got %s" %
                         (n_components, str(weights_bad_shape.shape)),
                         s.fit, X)

    # Check bad range
    weights_bad_range = rng.rand(n_components) + 1
    s.weights_init = weights_bad_range
    assert_raise_message(ValueError,
                         "The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights_bad_range),
                            np.max(weights_bad_range)),
                         s.fit, X)

    # Check bad normalization
    weights_bad_norm = rng.rand(n_components)
    weights_bad_norm = weights_bad_norm / (weights_bad_norm.sum() + 1)
    s.weights_init = weights_bad_norm
    assert_raise_message(ValueError,
                         "The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f"
                         % np.sum(weights_bad_norm),
                         s.fit, X)

    # Check good weights matrix
    weights = rand_data.weights
    s = StudentMixture(weights_init=weights, n_components=n_components)
    s.fit(X)
    assert_array_equal(weights, s.weights_init)


def test_check_locations():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    n_components, n_features = rand_data.n_components, rand_data.n_features
    X = rand_data.X['full']

    s = StudentMixture(n_components=n_components)

    # Check means bad shape
    locations_bad_shape = rng.rand(n_components + 1, n_features)
    s.locations_init = locations_bad_shape
    assert_raise_message(ValueError,
                         "The parameter 'locations' should have the shape of ",
                         s.fit, X)

    # Check good locations matrix
    locations = rand_data.locations
    s.locations_init = locations
    s.fit(X)
    assert_array_equal(locations, s.locations_init)


def test_check_precisions():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    n_components, n_features = rand_data.n_components, rand_data.n_features

    # Define the bad precisions for each covariance_type
    precisions_bad_shape = {
        'full': np.ones((n_components + 1, n_features, n_features)),
        'tied': np.ones((n_features + 1, n_features + 1)),
        'diag': np.ones((n_components + 1, n_features)),
        'spherical': np.ones((n_components + 1))}

    # Define not positive-definite precisions
    precisions_not_pos = np.ones((n_components, n_features, n_features))
    precisions_not_pos[0] = np.eye(n_features)
    precisions_not_pos[0, 0, 0] = -1.

    precisions_not_positive = {
        'full': precisions_not_pos,
        'tied': precisions_not_pos[0],
        'diag': np.full((n_components, n_features), -1.),
        'spherical': np.full(n_components, -1.)}

    not_positive_errors = {
        'full': 'symmetric, positive-definite',
        'tied': 'symmetric, positive-definite',
        'diag': 'positive',
        'spherical': 'positive'}

    for scale_type in SCALES_TYPE:
        X = RandomData(rng).X[scale_type]
        s = StudentMixture(n_components=n_components,
                            scale_type=scale_type,
                            random_state=rng)

        # Check precisions with bad shapes
        s.precisions_init = precisions_bad_shape[scale_type]
        assert_raise_message(ValueError,
                             "The parameter '%s precision' should have "
                             "the shape of" % scale_type,
                             s.fit, X)

        # Check not positive precisions
        s.precisions_init = precisions_not_positive[scale_type]
        assert_raise_message(ValueError,
                             "'%s precision' should be %s"
                             % (scale_type, not_positive_errors[scale_type]),
                             s.fit, X)

        # Check the correct init of precisions_init
        s.precisions_init = rand_data.precisions[scale_type]
        s.fit(X)
        assert_array_equal(rand_data.precisions[scale_type], s.precisions_init)


def test_suffstat_sk_full():
    # compare the precision matrix compute from the
    # EmpiricalCovariance.covariance fitted on X*sqrt(resp)
    # with _sufficient_sk_full, n_components=1
    rng = np.random.RandomState(0)
    n_samples, n_features = 500, 2
    dof = 100

    # special case 1, assuming data is "centered"
    X = rng.rand(n_samples, n_features)
    resp = rng.rand(n_samples, 1)
    X_resp = np.sqrt(resp) * X
    nk = np.array([n_samples])
    xk = np.zeros((1, n_features))
    delta = (resp - resp.mean(0)) / resp.std(0)
    gamma_priors = (dof + n_features) / (dof + delta)
    scales_pred = _estimate_student_scales_full(resp, gamma_priors, X, nk, xk, 0)
    ecov = EmpiricalCovariance(assume_centered=True)
    ecov.fit(X_resp)
    assert_almost_equal(ecov.error_norm(scales_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(scales_pred[0], norm='spectral'), 0)

    # check the precision computation
    precs_chol_pred = _compute_precision_cholesky(scales_pred, 'full')
    precs_pred = np.array([np.dot(prec, prec.T) for prec in precs_chol_pred])
    precs_est = np.array([linalg.inv(scale) for scale in scales_pred])
    assert_array_almost_equal(precs_est, precs_pred)

    # special case 2, assuming resp are all ones
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean(axis=0).reshape((1, -1))
    gamma_priors = np.ones_like(resp)
    scales_pred = _estimate_student_scales_full(resp, gamma_priors, X, nk, xk, 0)
    ecov = EmpiricalCovariance(assume_centered=False)
    ecov.fit(X)
    assert_almost_equal(ecov.error_norm(scales_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(scales_pred[0], norm='spectral'), 0)

    # check the precision computation
    precs_chol_pred = _compute_precision_cholesky(covars_pred, 'full')
    precs_pred = np.array([np.dot(prec, prec.T) for prec in precs_chol_pred])
    precs_est = np.array([linalg.inv(cov) for cov in covars_pred])
    assert_array_almost_equal(precs_est, precs_pred)


def test_suffstat_sk_tied():
    # use equation Nk * Sk / N = S_tied
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 500, 2, 2

    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]

    covars_pred_full = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)
    covars_pred_full = np.sum(nk[:, np.newaxis, np.newaxis] * covars_pred_full,
                              0) / n_samples

    covars_pred_tied = _estimate_gaussian_covariances_tied(resp, X, nk, xk, 0)

    ecov = EmpiricalCovariance()
    ecov.covariance_ = covars_pred_full
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='spectral'), 0)

    # check the precision computation
    precs_chol_pred = _compute_precision_cholesky(covars_pred_tied, 'tied')
    precs_pred = np.dot(precs_chol_pred, precs_chol_pred.T)
    precs_est = linalg.inv(covars_pred_tied)
    assert_array_almost_equal(precs_est, precs_pred)


def test_suffstat_sk_diag():
    # test against 'full' case
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 500, 2, 2

    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    covars_pred_full = _estimate_gaussian_covariances_full(resp, X, nk, xk, 0)
    covars_pred_diag = _estimate_gaussian_covariances_diag(resp, X, nk, xk, 0)

    ecov = EmpiricalCovariance()
    for (cov_full, cov_diag) in zip(covars_pred_full, covars_pred_diag):
        ecov.covariance_ = np.diag(np.diag(cov_full))
        cov_diag = np.diag(cov_diag)
        assert_almost_equal(ecov.error_norm(cov_diag, norm='frobenius'), 0)
        assert_almost_equal(ecov.error_norm(cov_diag, norm='spectral'), 0)

    # check the precision computation
    precs_chol_pred = _compute_precision_cholesky(covars_pred_diag, 'diag')
    assert_almost_equal(covars_pred_diag, 1. / precs_chol_pred ** 2)


def test_gaussian_suffstat_sk_spherical():
    # computing spherical covariance equals to the variance of one-dimension
    # data after flattening, n_components=1
    rng = np.random.RandomState(0)
    n_samples, n_features = 500, 2

    X = rng.rand(n_samples, n_features)
    X = X - X.mean()
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean()
    covars_pred_spherical = _estimate_gaussian_covariances_spherical(resp, X,
                                                                     nk, xk, 0)
    covars_pred_spherical2 = (np.dot(X.flatten().T, X.flatten()) /
                              (n_features * n_samples))
    assert_almost_equal(covars_pred_spherical, covars_pred_spherical2)

    # check the precision computation
    precs_chol_pred = _compute_precision_cholesky(covars_pred_spherical,
                                                  'spherical')
    assert_almost_equal(covars_pred_spherical, 1. / precs_chol_pred ** 2)


def test_compute_log_det_cholesky():
    n_features = 2
    rand_data = RandomData(np.random.RandomState(0))

    for covar_type in COVARIANCE_TYPE:
        covariance = rand_data.covariances[covar_type]

        if covar_type == 'full':
            predected_det = np.array([linalg.det(cov) for cov in covariance])
        elif covar_type == 'tied':
            predected_det = linalg.det(covariance)
        elif covar_type == 'diag':
            predected_det = np.array([np.prod(cov) for cov in covariance])
        elif covar_type == 'spherical':
            predected_det = covariance ** n_features

        # We compute the cholesky decomposition of the covariance matrix
        expected_det = _compute_log_det_cholesky(_compute_precision_cholesky(
            covariance, covar_type), covar_type, n_features=n_features)
        assert_array_almost_equal(expected_det, - .5 * np.log(predected_det))


def _naive_lmvnpdf_diag(X, means, covars):
    resp = np.empty((len(X), len(means)))
    stds = np.sqrt(covars)
    for i, (mean, std) in enumerate(zip(means, stds)):
        resp[:, i] = stats.norm.logpdf(X, mean, std).sum(axis=1)
    return resp


def test_gaussian_mixture_log_probabilities():
    from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob

    # test against with _naive_lmvnpdf_diag
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_samples = 500
    n_features = rand_data.n_features
    n_components = rand_data.n_components

    means = rand_data.means
    covars_diag = rng.rand(n_components, n_features)
    X = rng.rand(n_samples, n_features)
    log_prob_naive = _naive_lmvnpdf_diag(X, means, covars_diag)

    # full covariances
    precs_full = np.array([np.diag(1. / np.sqrt(x)) for x in covars_diag])

    log_prob = _estimate_log_gaussian_prob(X, means, precs_full, 'full')
    assert_array_almost_equal(log_prob, log_prob_naive)

    # diag covariances
    precs_chol_diag = 1. / np.sqrt(covars_diag)
    log_prob = _estimate_log_gaussian_prob(X, means, precs_chol_diag, 'diag')
    assert_array_almost_equal(log_prob, log_prob_naive)

    # tied
    covars_tied = np.array([x for x in covars_diag]).mean(axis=0)
    precs_tied = np.diag(np.sqrt(1. / covars_tied))

    log_prob_naive = _naive_lmvnpdf_diag(X, means,
                                         [covars_tied] * n_components)
    log_prob = _estimate_log_gaussian_prob(X, means, precs_tied, 'tied')

    assert_array_almost_equal(log_prob, log_prob_naive)

    # spherical
    covars_spherical = covars_diag.mean(axis=1)
    precs_spherical = 1. / np.sqrt(covars_diag.mean(axis=1))
    log_prob_naive = _naive_lmvnpdf_diag(X, means,
                                         [[k] * n_features for k in
                                          covars_spherical])
    log_prob = _estimate_log_gaussian_prob(X, means,
                                           precs_spherical, 'spherical')
    assert_array_almost_equal(log_prob, log_prob_naive)

# skip tests on weighted_log_probabilities, log_weights


def test_gaussian_mixture_estimate_log_prob_resp():
    # test whether responsibilities are normalized
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_samples = rand_data.n_samples
    n_features = rand_data.n_features
    n_components = rand_data.n_components

    X = rng.rand(n_samples, n_features)
    for covar_type in COVARIANCE_TYPE:
        weights = rand_data.weights
        means = rand_data.means
        precisions = rand_data.precisions[covar_type]
        g = StudentMixture(n_components=n_components, random_state=rng,
                            weights_init=weights, means_init=means,
                            precisions_init=precisions,
                            covariance_type=covar_type)
        g.fit(X)
        resp = g.predict_proba(X)
        assert_array_almost_equal(resp.sum(axis=1), np.ones(n_samples))
        assert_array_equal(g.weights_init, weights)
        assert_array_equal(g.means_init, means)
        assert_array_equal(g.precisions_init, precisions)


def test_gaussian_mixture_predict_predict_proba():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        Y = rand_data.Y
        g = StudentMixture(n_components=rand_data.n_components,
                            random_state=rng, weights_init=rand_data.weights,
                            means_init=rand_data.means,
                            precisions_init=rand_data.precisions[covar_type],
                            covariance_type=covar_type)

        # Check a warning message arrive if we don't do fit
        assert_raise_message(NotFittedError,
                             "This StudentMixture instance is not fitted "
                             "yet. Call 'fit' with appropriate arguments "
                             "before using this estimator.", g.predict, X)

        g.fit(X)
        Y_pred = g.predict(X)
        Y_pred_proba = g.predict_proba(X).argmax(axis=1)
        assert_array_equal(Y_pred, Y_pred_proba)
        assert adjusted_rand_score(Y, Y_pred) > .95


@pytest.mark.filterwarnings("ignore:.*did not converge.*")
@pytest.mark.parametrize('seed, max_iter, tol', [
    (0, 2, 1e-7),    # strict non-convergence
    (1, 2, 1e-1),    # loose non-convergence
    (3, 300, 1e-7),  # strict convergence
    (4, 300, 1e-1),  # loose convergence
])
def test_gaussian_mixture_fit_predict(seed, max_iter, tol):
    rng = np.random.RandomState(seed)
    rand_data = RandomData(rng)
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        Y = rand_data.Y
        g = StudentMixture(n_components=rand_data.n_components,
                            random_state=rng, weights_init=rand_data.weights,
                            means_init=rand_data.means,
                            precisions_init=rand_data.precisions[covar_type],
                            covariance_type=covar_type,
                            max_iter=max_iter, tol=tol)

        # check if fit_predict(X) is equivalent to fit(X).predict(X)
        f = copy.deepcopy(g)
        Y_pred1 = f.fit(X).predict(X)
        Y_pred2 = g.fit_predict(X)
        assert_array_equal(Y_pred1, Y_pred2)
        assert adjusted_rand_score(Y, Y_pred2) > .95


def test_gaussian_mixture_fit_predict_n_init():
    # Check that fit_predict is equivalent to fit.predict, when n_init > 1
    X = np.random.RandomState(0).randn(1000, 5)
    gm = StudentMixture(n_components=5, n_init=5, random_state=0)
    y_pred1 = gm.fit_predict(X)
    y_pred2 = gm.predict(X)
    assert_array_equal(y_pred1, y_pred2)


def test_gaussian_mixture_fit():
    # recover the ground truth
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_features = rand_data.n_features
    n_components = rand_data.n_components

    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = StudentMixture(n_components=n_components, n_init=20,
                            reg_covar=0, random_state=rng,
                            covariance_type=covar_type)
        g.fit(X)

        # needs more data to pass the test with rtol=1e-7
        assert_allclose(np.sort(g.weights_), np.sort(rand_data.weights),
                        rtol=0.1, atol=1e-2)

        arg_idx1 = g.means_[:, 0].argsort()
        arg_idx2 = rand_data.means[:, 0].argsort()
        assert_allclose(g.means_[arg_idx1], rand_data.means[arg_idx2],
                        rtol=0.1, atol=1e-2)

        if covar_type == 'full':
            prec_pred = g.precisions_
            prec_test = rand_data.precisions['full']
        elif covar_type == 'tied':
            prec_pred = np.array([g.precisions_] * n_components)
            prec_test = np.array([rand_data.precisions['tied']] * n_components)
        elif covar_type == 'spherical':
            prec_pred = np.array([np.eye(n_features) * c
                                 for c in g.precisions_])
            prec_test = np.array([np.eye(n_features) * c for c in
                                 rand_data.precisions['spherical']])
        elif covar_type == 'diag':
            prec_pred = np.array([np.diag(d) for d in g.precisions_])
            prec_test = np.array([np.diag(d) for d in
                                 rand_data.precisions['diag']])

        arg_idx1 = np.trace(prec_pred, axis1=1, axis2=2).argsort()
        arg_idx2 = np.trace(prec_test, axis1=1, axis2=2).argsort()
        for k, h in zip(arg_idx1, arg_idx2):
            ecov = EmpiricalCovariance()
            ecov.covariance_ = prec_test[h]
            # the accuracy depends on the number of data and randomness, rng
            assert_allclose(ecov.error_norm(prec_pred[k]), 0, atol=0.15)


def test_gaussian_mixture_fit_best_params():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    n_init = 10
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = StudentMixture(n_components=n_components, n_init=1, reg_covar=0,
                            random_state=rng, covariance_type=covar_type)
        ll = []
        for _ in range(n_init):
            g.fit(X)
            ll.append(g.score(X))
        ll = np.array(ll)
        g_best = StudentMixture(n_components=n_components,
                                 n_init=n_init, reg_covar=0, random_state=rng,
                                 covariance_type=covar_type)
        g_best.fit(X)
        assert_almost_equal(ll.min(), g_best.score(X))


def test_gaussian_mixture_fit_convergence_warning():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=1)
    n_components = rand_data.n_components
    max_iter = 1
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = StudentMixture(n_components=n_components, n_init=1,
                            max_iter=max_iter, reg_covar=0, random_state=rng,
                            covariance_type=covar_type)
        assert_warns_message(ConvergenceWarning,
                             'Initialization %d did not converge. '
                             'Try different init parameters, '
                             'or increase max_iter, tol '
                             'or check for degenerate data.'
                             % max_iter, g.fit, X)


def test_multiple_init():
    # Test that multiple inits does not much worse than a single one
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 5, 2
    X = rng.randn(n_samples, n_features)
    for cv_type in COVARIANCE_TYPE:
        train1 = StudentMixture(n_components=n_components,
                                 covariance_type=cv_type,
                                 random_state=0).fit(X).score(X)
        train2 = StudentMixture(n_components=n_components,
                                 covariance_type=cv_type,
                                 random_state=0, n_init=5).fit(X).score(X)
        assert train2 >= train1


def test_gaussian_mixture_n_parameters():
    # Test that the right number of parameters is estimated
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 5, 2
    X = rng.randn(n_samples, n_features)
    n_params = {'spherical': 13, 'diag': 21, 'tied': 26, 'full': 41}
    for cv_type in COVARIANCE_TYPE:
        g = StudentMixture(
            n_components=n_components, covariance_type=cv_type,
            random_state=rng).fit(X)
        assert g._n_parameters() == n_params[cv_type]


def test_bic_1d_1component():
    # Test all of the covariance_types return the same BIC score for
    # 1-dimensional, 1 component fits.
    rng = np.random.RandomState(0)
    n_samples, n_dim, n_components = 100, 1, 1
    X = rng.randn(n_samples, n_dim)
    bic_full = StudentMixture(n_components=n_components,
                               covariance_type='full',
                               random_state=rng).fit(X).bic(X)
    for covariance_type in ['tied', 'diag', 'spherical']:
        bic = StudentMixture(n_components=n_components,
                              covariance_type=covariance_type,
                              random_state=rng).fit(X).bic(X)
        assert_almost_equal(bic_full, bic)


def test_gaussian_mixture_aic_bic():
    # Test the aic and bic criteria
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 3, 2
    X = rng.randn(n_samples, n_features)
    # standard gaussian entropy
    sgh = 0.5 * (fast_logdet(np.cov(X.T, bias=1)) +
                 n_features * (1 + np.log(2 * np.pi)))
    for cv_type in COVARIANCE_TYPE:
        g = StudentMixture(
            n_components=n_components, covariance_type=cv_type,
            random_state=rng, max_iter=200)
        g.fit(X)
        aic = 2 * n_samples * sgh + 2 * g._n_parameters()
        bic = (2 * n_samples * sgh +
               np.log(n_samples) * g._n_parameters())
        bound = n_features / np.sqrt(n_samples)
        assert (g.aic(X) - aic) / n_samples < bound
        assert (g.bic(X) - bic) / n_samples < bound


def test_gaussian_mixture_verbose():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        g = StudentMixture(n_components=n_components, n_init=1, reg_covar=0,
                            random_state=rng, covariance_type=covar_type,
                            verbose=1)
        h = StudentMixture(n_components=n_components, n_init=1, reg_covar=0,
                            random_state=rng, covariance_type=covar_type,
                            verbose=2)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            g.fit(X)
            h.fit(X)
        finally:
            sys.stdout = old_stdout


@pytest.mark.filterwarnings('ignore:.*did not converge.*')
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_warm_start(seed):
    random_state = seed
    rng = np.random.RandomState(random_state)
    n_samples, n_features, n_components = 500, 2, 2
    X = rng.rand(n_samples, n_features)

    # Assert the warm_start give the same result for the same number of iter
    g = StudentMixture(n_components=n_components, n_init=1, max_iter=2,
                        reg_covar=0, random_state=random_state,
                        warm_start=False)
    h = StudentMixture(n_components=n_components, n_init=1, max_iter=1,
                        reg_covar=0, random_state=random_state,
                        warm_start=True)

    g.fit(X)
    score1 = h.fit(X).score(X)
    score2 = h.fit(X).score(X)

    assert_almost_equal(g.weights_, h.weights_)
    assert_almost_equal(g.means_, h.means_)
    assert_almost_equal(g.precisions_, h.precisions_)
    assert score2 > score1

    # Assert that by using warm_start we can converge to a good solution
    g = StudentMixture(n_components=n_components, n_init=1,
                        max_iter=5, reg_covar=0, random_state=random_state,
                        warm_start=False, tol=1e-6)
    h = StudentMixture(n_components=n_components, n_init=1,
                        max_iter=5, reg_covar=0, random_state=random_state,
                        warm_start=True, tol=1e-6)

    g.fit(X)
    assert not g.converged_

    h.fit(X)
    # depending on the data there is large variability in the number of
    # refit necessary to converge due to the complete randomness of the
    # data
    for _ in range(1000):
        h.fit(X)
        if h.converged_:
            break
    assert h.converged_


@ignore_warnings(category=ConvergenceWarning)
def test_convergence_detected_with_warm_start():
    # We check that convergence is detected when warm_start=True
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    X = rand_data.X['full']

    for max_iter in (1, 2, 50):
        smm = StudentMixture(n_components=n_components, warm_start=True,
                              max_iter=max_iter, random_state=rng)
        for _ in range(100):
            smm.fit(X)
            if smm.converged_:
                break
        assert smm.converged_
        assert max_iter >= smm.n_iter_


def test_score():
    covar_type = 'full'
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    X = rand_data.X[covar_type]

    # Check the error message if we don't call fit
    smm1 = StudentMixture(n_components=n_components, n_init=1,
                           max_iter=1, reg_covar=0, random_state=rng,
                           covariance_type=covar_type)
    assert_raise_message(NotFittedError,
                         "This StudentMixture instance is not fitted "
                         "yet. Call 'fit' with appropriate arguments "
                         "before using this estimator.", smm1.score, X)

    # Check score value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        smm1.fit(X)
    smm_score = smm1.score(X)
    smm_score_proba = smm1.score_samples(X).mean()
    assert_almost_equal(smm_score, smm_score_proba)

    # Check if the score increase
    smm2 = StudentMixture(n_components=n_components, n_init=1, reg_covar=0,
                           random_state=rng,
                           covariance_type=covar_type).fit(X)
    assert smm2.score(X) > smm1.score(X)


def test_score_samples():
    covar_type = 'full'
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    X = rand_data.X[covar_type]

    # Check the error message if we don't call fit
    smm = StudentMixture(n_components=n_components, n_init=1, reg_covar=0,
                          random_state=rng, covariance_type=covar_type)
    assert_raise_message(NotFittedError,
                         "This StudentMixture instance is not fitted "
                         "yet. Call 'fit' with appropriate arguments "
                         "before using this estimator.", smm.score_samples, X)

    smm_score_samples = smm.fit(X).score_samples(X)
    assert smm_score_samples.shape[0] == rand_data.n_samples


def test_monotonic_likelihood():
    # We check that each step of the EM without regularization improve
    # monotonically the training set likelihood
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components

    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        smm = StudentMixture(n_components=n_components,
                              covariance_type=covar_type, reg_covar=0,
                              warm_start=True, max_iter=1, random_state=rng,
                              tol=1e-7)
        current_log_likelihood = -np.infty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            # Do one training iteration at a time so we can make sure that the
            # training log likelihood increases after each iteration.
            for _ in range(600):
                prev_log_likelihood = current_log_likelihood
                try:
                    current_log_likelihood = smm.fit(X).score(X)
                except ConvergenceWarning:
                    pass
                assert (current_log_likelihood >=
                                     prev_log_likelihood)

                if smm.converged_:
                    break

            assert smm.converged_


def test_regularisation():
    # We train the StudentMixture on degenerate data by defining two clusters
    # of a 0 covariance.
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = np.vstack((np.ones((n_samples // 2, n_features)),
                   np.zeros((n_samples // 2, n_features))))

    for covar_type in COVARIANCE_TYPE:
        smm = StudentMixture(n_components=n_samples, reg_covar=0,
                              covariance_type=covar_type, random_state=rng)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert_raise_message(ValueError,
                                 "Fitting the mixture model failed because "
                                 "some components have ill-defined empirical "
                                 "covariance (for instance caused by "
                                 "singleton or collapsed samples). Try to "
                                 "decrease the number of components, or "
                                 "increase reg_covar.", smm.fit, X)

            smm.set_params(reg_covar=1e-6).fit(X)


def test_property():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components

    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]
        smm = StudentMixture(n_components=n_components,
                              covariance_type=covar_type, random_state=rng,
                              n_init=5)
        smm.fit(X)
        if covar_type == 'full':
            for prec, covar in zip(smm.precisions_, smm.covariances_):

                assert_array_almost_equal(linalg.inv(prec), covar)
        elif covar_type == 'tied':
            assert_array_almost_equal(linalg.inv(smm.precisions_),
                                      smm.covariances_)
        else:
            assert_array_almost_equal(smm.precisions_, 1. / smm.covariances_)


def test_sample():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7, n_components=3)
    n_features, n_components = rand_data.n_features, rand_data.n_components

    for covar_type in COVARIANCE_TYPE:
        X = rand_data.X[covar_type]

        smm = StudentMixture(n_components=n_components,
                              covariance_type=covar_type, random_state=rng)
        # To sample we need that StudentMixture is fitted
        assert_raise_message(NotFittedError, "This StudentMixture instance "
                             "is not fitted", smm.sample, 0)
        smm.fit(X)

        assert_raise_message(ValueError, "Invalid value for 'n_samples",
                             smm.sample, 0)

        # Just to make sure the class samples correctly
        n_samples = 20000
        X_s, y_s = smm.sample(n_samples)

        for k in range(n_components):
            if covar_type == 'full':
                assert_array_almost_equal(smm.covariances_[k],
                                          np.cov(X_s[y_s == k].T), decimal=1)
            elif covar_type == 'tied':
                assert_array_almost_equal(smm.covariances_,
                                          np.cov(X_s[y_s == k].T), decimal=1)
            elif covar_type == 'diag':
                assert_array_almost_equal(smm.covariances_[k],
                                          np.diag(np.cov(X_s[y_s == k].T)),
                                          decimal=1)
            else:
                assert_array_almost_equal(
                    smm.covariances_[k], np.var(X_s[y_s == k] - smm.means_[k]),
                    decimal=1)

        means_s = np.array([np.mean(X_s[y_s == k], 0)
                           for k in range(n_components)])
        assert_array_almost_equal(smm.means_, means_s, decimal=1)

        # Check shapes of sampled data, see
        # https://github.com/scikit-learn/scikit-learn/issues/7701
        assert X_s.shape == (n_samples, n_features)

        for sample_size in range(1, 100):
            X_s, _ = smm.sample(sample_size)
            assert X_s.shape == (sample_size, n_features)


@ignore_warnings(category=ConvergenceWarning)
def test_init():
    # We check that by increasing the n_init number we have a better solution
    for random_state in range(15):
        rand_data = RandomData(np.random.RandomState(random_state),
                               n_samples=50, scale=1)
        n_components = rand_data.n_components
        X = rand_data.X['full']

        smm1 = StudentMixture(n_components=n_components, n_init=1,
                               max_iter=1, random_state=random_state).fit(X)
        smm2 = StudentMixture(n_components=n_components, n_init=10,
                               max_iter=1, random_state=random_state).fit(X)

        assert smm2.lower_bound_ >= smm1.lower_bound_

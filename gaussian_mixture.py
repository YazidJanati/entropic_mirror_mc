from jax import vmap, jit, debug
from jax.lax import while_loop
from jax.numpy.linalg import norm
import jax.numpy as jnp
from jax.nn import logsumexp
from jax.random import normal, split, uniform, randint
from jax.scipy.linalg import cholesky, solve_triangular, eigh
from jax.tree_util import Partial as partial
from numpyro.distributions import MultivariateNormal, Categorical, \
    MixtureSameFamily
from jax.config import config
from typing import NamedTuple

config.update("jax_enable_x64", True)

"""
Jax implementation of the Expectation-Maximization algorithm.
"""

class Gaussian_Mixture(NamedTuple):
    n_components: int
    params: jnp.ndarray

def _precision_chol(covs):
    n_comp, dim = covs.shape[0], covs.shape[-1]
    covs_chol = cholesky(covs, lower=True)
    chol = solve_triangular(covs_chol, jnp.array([jnp.eye(dim)] * n_comp), lower=True)
    log_det = jnp.trace(jnp.log(chol), axis1=1, axis2=2)
    return chol, log_det


def _mvn_logprob(x, means, prec_chol, log_det_chol):
    dim = x.shape[-1]
    diff = (x - means)[:, :, jnp.newaxis]
    y = ((prec_chol @ diff) ** 2).sum((-1, -2))
    return - 0.5 * (y + jnp.log(2 * jnp.pi) * dim) + log_det_chol


def _joint_logprob(x, means, prec_chol, log_det_chol, log_weights):
    likelihood = _mvn_logprob(x, means, prec_chol, log_det_chol)
    return likelihood + log_weights


def _update_means(x, posterior):
    return posterior[:, jnp.newaxis] @ x[jnp.newaxis, :]


def _update_covs(x, posterior, means):
    diff = x - means
    return (diff[:, :, jnp.newaxis] @ diff[:, jnp.newaxis, :]) * posterior[:, jnp.newaxis, jnp.newaxis]


def _full_update(x, log_posterior):
    cov_reg = jnp.eye(x.shape[-1]) * 1e-6
    n_samples = x.shape[0]

    log_sum_posterior = logsumexp(log_posterior, 0)
    sum_posterior = jnp.exp(log_sum_posterior) + 10 * jnp.finfo(log_sum_posterior.dtype).eps
    new_means = vmap(_update_means)(x, jnp.exp(log_posterior))
    new_means = new_means.sum(0) / sum_posterior[:, jnp.newaxis]

    cov_update = partial(_update_covs, means=new_means)
    new_covs = vmap(cov_update)(x, jnp.exp(log_posterior))
    new_covs = (new_covs.sum(0) / sum_posterior[:, jnp.newaxis, jnp.newaxis]) + cov_reg

    return new_means, new_covs, log_sum_posterior - jnp.log(n_samples)


def em_step(x, val):
    iter, means, covs, log_weights, prev_llk, _ = val
    prec_chol, log_det = _precision_chol(covs)
    partial_jlp = partial(_joint_logprob, means=means, prec_chol=prec_chol, log_det_chol=log_det,
                          log_weights=log_weights)
    joint_logprob = vmap(partial_jlp)(x)
    log_likelihood = logsumexp(joint_logprob, -1)
    log_posterior = joint_logprob - log_likelihood[:, jnp.newaxis]

    new_means, new_covs, new_logweights = _full_update(x, log_posterior)
    return iter + 1, new_means, new_covs, new_logweights, log_likelihood.mean(), prev_llk


def initialize(key, x, n_components):
    key1, _ = split(key)
    dim = x.shape[-1]
    means = kmeans(key1, x, n_components, eps=1e-4)[0]
    covs = jnp.array([jnp.eye(dim)] * n_components)
    log_weights = jnp.log(jnp.array([1 / n_components] * n_components))
    return (means, covs, log_weights)


def train(samples, init_params, n_components, key=None, eps=1e-3, max_iter=100):
    if init_params == "kmeans":
        init_params = initialize(key, samples, n_components)

    body_func = lambda val: partial(em_step, x=samples)(val=val)
    init_val = (0, *init_params, jnp.inf, 0)
    _, means, covs, log_weights, *_ = while_loop(lambda val: (abs(val[-2] - val[-1]) > eps) * (val[0] <= max_iter),
                                                 body_func,
                                                 init_val)
    return means, covs, log_weights


def tomixture(means, covs, weights):
    cat = Categorical(weights)
    norm = MultivariateNormal(means, covs)
    return MixtureSameFamily(cat, norm)


def assign(points, centroids):
    substrac = vmap(lambda x, y: x - y, in_axes=(None, 0))
    distances = norm(substrac(points, centroids), axis=-1)
    assignments = jnp.argmin(distances, axis=0)
    return assignments, distances


def kmeans_step(points, n_centroids, centroids, distances, prev_distances):
    assignments, new_dist = assign(points, centroids)
    labels = jnp.arange(n_centroids)
    assigned = ((assignments[:, jnp.newaxis] == labels[jnp.newaxis, :]) * 1)
    n_assigned = assigned.sum(0, keepdims=True).T
    n_assigned = jnp.clip(n_assigned, a_min=1)
    return jnp.where(assigned.T[:, :, jnp.newaxis], points, jnp.zeros(points.shape)).sum(
        axis=1) / n_assigned, new_dist.sum(), distances


def kmeans(key, points, n_centroids, eps):
    init_centroids_idxs = randint(key, (n_centroids,), 0, points.shape[0])
    init_centroids = points[init_centroids_idxs, :]
    init_val = (init_centroids, 0, jnp.inf)
    body_func = lambda val: kmeans_step(points, n_centroids, *val)
    return while_loop(lambda val: abs(val[1] - val[2]) > eps, body_func, init_val)

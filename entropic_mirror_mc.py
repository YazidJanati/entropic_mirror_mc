import jax.numpy as jnp
from jax.lax import fori_loop
from jax.random import split, categorical, uniform, normal
from numpyro.distributions import MixtureGeneral, Categorical
from gaussian_mixture import Gaussian_Mixture, params_to_gm
from gaussian_mixture import train as train_gm
from jax.lax import cond
from jax.nn import logsumexp
from typing import NamedTuple, Callable
from jax.tree_util import Partial as partial


class MCMC_kernel(NamedTuple):
    sampler: Callable
    n_chains: int


def emc(key, logpdf, n_train, n_samples, model, global_kernel, local_kernel, n_chains, heavy_distr,
        mixed_proposal_weights=jnp.array([.8, .2])):
    key_train, key_init_proposal = split(key, 2)
    if isinstance(model, Gaussian_Mixture):
        key_means, key_covs, key_log_weights = split(key_init_proposal, 3)
        init_proposal = params_to_gm(means=normal(key_means, (model.n_components, model.dim)),
                                     covs=normal(key_covs, (model.n_components, model.dim, model.dim)),
                                     log_weights=jnp.log(jnp.array([1 / model.n_components] * model.n_components)))

        def train_func(key, samples):
            params = train_gm(samples=samples, init_params="kmeans", n_components=model.n_components, key=key)
            return params_to_gm(*params)

    keys = split(key_train, n_train)
    partial_emc_step = partial(emc_step, key=keys, n_samples=n_samples, logpdf=logpdf, global_kernel=global_kernel,
                               local_kernel=local_kernel,
                               train_func=train_func, heavy_distr=heavy_distr,
                               mixed_proposal_weights=mixed_proposal_weights)
    return fori_loop(0, n_train, partial_emc_step, init_proposal)


def emc_step(i, key, n_samples, logpdf, proposal, global_kernel, local_kernel, train_func, heavy_distr,
             mixed_proposal_weights):
    key_empirical, key_resample, key_lker = split(key[i], 3)
    if heavy_distr is None:
        mixed_proposal = proposal
    else:
        mixed_proposal = MixtureGeneral(Categorical(mixed_proposal_weights), [proposal, heavy_distr])

    empirical_iterate = empirical_update(key_empirical, logpdf, pow, n_samples, global_kernel, mixed_proposal)
    samples = resample_empirical_update(key_resample, empirical_iterate, n_samples)
    samples = local_kernel.sampler(keys=split(key_lker, local_kernel.n_chains),
                                   state=samples,
                                   steps=n_samples // global_kernel.n_chains)
    return train_func(proposal, samples)


def empirical_update(key, logpdf, pow, n_samples, global_kernel, prev_proposal):
    proposal_samples = prev_proposal.sample(key, (n_samples,))
    proposal_log_weights = pow * (logpdf(proposal_samples) - prev_proposal.log_prob(proposal_samples))
    global_kernel_samples = global_kernel.sampler(keys=split(key, global_kernel.n_chains),
                                                  state=proposal_samples,
                                                  steps=n_samples // global_kernel.n_chains)
    global_kernel_log_weights = pow * (logpdf(global_kernel_samples) - prev_proposal.log_prob(global_kernel_samples))
    global_kernel_lse, proposal_lse = logsumexp(global_kernel_log_weights), logsumexp(proposal_log_weights)
    alpha = cond((global_kernel_lse - jnp.log(n_samples) > 0),
                 lambda _: (global_kernel_lse - jnp.log(n_samples)) / (global_kernel_lse - proposal_lse),
                 lambda _: 0.5,
                 operand=None)
    return proposal_samples, proposal_log_weights, global_kernel_samples, global_kernel_log_weights, alpha


def resample_empirical_update(key, empirical_update, n_samples):
    key_resample_gker, key_resample_proposal, key_cat = split(key, 3)
    proposal_samples, proposal_log_weights, global_kernel_samples, global_kernel_log_weights, alpha = empirical_update
    n_proposal_resamples = categorical(key_cat, alpha, shape=(n_samples,)).sum()
    n_global_kernel_resamples = n_samples - n_proposal_resamples
    global_kernel_resamples = resample(key_resample_gker, global_kernel_samples, global_kernel_log_weights,
                                       n_global_kernel_resamples)
    proposal_resamples = resample(key_resample_proposal, proposal_samples, proposal_log_weights, n_proposal_resamples)
    return jnp.vstack([proposal_resamples, global_kernel_resamples])


def resample(key, samples, log_weights, n_samples):
    idxs = categorical(key, log_weights, shape=(n_samples,))
    return samples[idxs, :]

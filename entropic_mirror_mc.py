import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop
from jax.random import split, categorical, uniform, normal, bernoulli
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


def emc(key, logpdf, pow_eps, n_train, n_samples, model, global_kernel, local_kernel, n_chains, heavy_distr,
        mixed_proposal_weights=jnp.array([.8, .2])):
    key_train, key_init_proposal = split(key, 2)
    if isinstance(model, Gaussian_Mixture):
        key_means, key_covs, key_log_weights = split(key_init_proposal, 3)
        init_proposal = params_to_gm(means=normal(key_means, (model.n_components, model.dim)),
                                     covs=jnp.eye(model.dim)[jnp.newaxis,:].repeat(model.n_components, 0),
                                     log_weights=jnp.log(jnp.array([1 / model.n_components] * model.n_components)))

        def train_func(key, proposal, samples):
            params = train_gm(samples=samples, init_params="kmeans", n_components=model.n_components, key=key)
            return params_to_gm(*params)

    keys = split(key_train, n_train)
    partial_emc_step = partial(emc_step, key=keys, pow_eps=pow_eps, n_samples=n_samples, logpdf=logpdf, global_kernel=global_kernel,
                               local_kernel=local_kernel,
                               train_func=train_func, heavy_distr=heavy_distr,
                               mixed_proposal_weights=mixed_proposal_weights)
    return fori_loop(0, n_train, partial_emc_step, init_proposal)


def emc_step(i, proposal, pow_eps, key, n_samples, logpdf, global_kernel, local_kernel, train_func, heavy_distr,
             mixed_proposal_weights):
    key_empirical, key_resample, key_lker, key_train = split(key[i], 4)
    if heavy_distr is None:
        mixed_proposal = proposal
    else:
        mixed_proposal = MixtureGeneral(Categorical(mixed_proposal_weights), [proposal, heavy_distr])

    empirical_iterate = empirical_update(key_empirical, logpdf, pow_eps, n_samples, global_kernel, mixed_proposal)
    samples = resample_empirical_update(key_resample, empirical_iterate, n_samples)
    samples = local_kernel(keys=split(key_lker, n_samples), state=samples)
    return train_func(key=key_train, proposal=proposal, samples=samples)


def empirical_update(key, logpdf, pow_eps, n_samples, global_kernel, prev_proposal):
    proposal_samples = prev_proposal.sample(key, (n_samples,))
    proposal_log_weights = pow_eps * (logpdf(proposal_samples) - prev_proposal.log_prob(proposal_samples))
    global_kernel_samples = global_kernel(keys=split(key, n_samples),
                                          state=proposal_samples)
    global_kernel_log_weights = pow_eps * (
                logpdf(global_kernel_samples) - prev_proposal.log_prob(global_kernel_samples))
    global_kernel_lse, proposal_lse = logsumexp(global_kernel_log_weights), logsumexp(proposal_log_weights)
    alpha = cond((global_kernel_lse - jnp.log(n_samples) > 0),
                 lambda _: (global_kernel_lse - jnp.log(n_samples)) / (global_kernel_lse - proposal_lse),
                 lambda _: 0.5,
                 operand=None)
    return proposal_samples, proposal_log_weights, global_kernel_samples, global_kernel_log_weights, alpha


def resample_empirical_update(key, empirical_update, n_samples):
    # TODO: make it more efficient by resampling then propagating
    key_resample_gker, key_resample_proposal, key_cat = split(key, 3)
    proposal_samples, proposal_log_weights, global_kernel_samples, global_kernel_log_weights, alpha = empirical_update
    samples = jnp.vstack([proposal_samples, global_kernel_samples])
    log_weights = jnp.concatenate([alpha * proposal_log_weights, (1 - alpha) * global_kernel_log_weights])
    resample_idxs = categorical(key_cat, log_weights, shape=(n_samples,))
    return samples[resample_idxs, :]
    # n_proposal_resamples = bernoulli(key_cat, p=alpha, shape=(n_samples,)).sum()
    # n_global_kernel_resamples = n_samples - n_proposal_resamples
    # global_kernel_resamples = resample(key_resample_gker, global_kernel_samples, global_kernel_log_weights,
    #                                    n_global_kernel_resamples)
    # proposal_resamples = resample(key_resample_proposal, proposal_samples, proposal_log_weights, n_proposal_resamples)
    # return jnp.vstack([proposal_resamples, global_kernel_resamples])

def resample(key, samples, log_weights, n_samples):
    idxs = categorical(key, log_weights, shape=(n_samples,))
    return samples[idxs, :]

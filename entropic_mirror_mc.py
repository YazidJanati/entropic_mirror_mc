import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop
from jax.random import split, categorical, uniform, normal, bernoulli
from numpyro.distributions import MixtureGeneral, Categorical
from gaussian_mixture import Gaussian_Mixture, params_to_gm
from gaussian_mixture import train as train_gm
from jax.lax import cond
from jax.nn import logsumexp, log_softmax
from typing import NamedTuple, Callable
from jax.tree_util import Partial as partial
from realnvp import RealNVP
from utils import estimate_kl
from jax import debug
from utils import display_samples


class MCMC_kernel(NamedTuple):
    sampler: Callable
    n_chains: int


def em2c_kl(key, logpdf, pow_eps, n_train, n_samples, model, global_kernel, local_kernel, n_chains, heavy_distr,
            mixed_proposal_weights=jnp.array([.8, .2]), target_samples=None):

    key_train, key_init_proposal = split(key, 2)
    if isinstance(model, Gaussian_Mixture):
        key_means, key_covs, key_log_weights = split(key_init_proposal, 3)
        init_proposal = params_to_gm(means=normal(key_means, (model.n_components, model.dim)),
                                     covs=jnp.eye(model.dim)[jnp.newaxis, :].repeat(model.n_components, 0),
                                     log_weights=jnp.log(jnp.array([1 / model.n_components] * model.n_components)))

        def train_func(key, proposal, samples):
            params = train_gm(samples=samples, init_params="kmeans", n_components=model.n_components, key=key)
            return params_to_gm(*params)

    if isinstance(model, RealNVP):
        pass


    keys = split(key_train, n_train)
    kl_vals = jnp.array([estimate_kl(logpdf, init_proposal, target_samples)] + [0] * (n_train - 1))
    partial_emc_step = partial(em2c_kl_step, key=keys, pow_eps=pow_eps, n_samples=n_samples, logpdf=logpdf,
                               global_kernel=global_kernel,
                               local_kernel=local_kernel,
                               train_func=train_func,
                               heavy_distr=heavy_distr,
                               mixed_proposal_weights=mixed_proposal_weights,
                               target_samples=target_samples)
    return fori_loop(1, n_train, partial_emc_step, (init_proposal, kl_vals))


def emd_kl(key, logpdf, pow_eps, n_train, n_samples, model, global_kernel, local_kernel, n_chains, heavy_distr,
           mixed_proposal_weights=jnp.array([.8, .2]), target_samples=None):
    key_train, key_init_proposal = split(key, 2)
    if isinstance(model, Gaussian_Mixture):
        key_means, key_covs, key_log_weights = split(key_init_proposal, 3)
        init_proposal = params_to_gm(means=normal(key_means, (model.n_components, model.dim)),
                                     covs=jnp.eye(model.dim)[jnp.newaxis, :].repeat(model.n_components, 0),
                                     log_weights=jnp.log(jnp.array([1 / model.n_components] * model.n_components)))

        def train_func(key, proposal, samples):
            params = train_gm(samples=samples, init_params="kmeans", n_components=model.n_components, key=key)
            return params_to_gm(*params)

    keys = split(key_train, n_train)
    kl_vals = jnp.array([estimate_kl(logpdf, init_proposal, target_samples)] + [0] * (n_train - 1))
    partial_emd_step = partial(emd_kl_step,
                               key=keys,
                               pow_eps=pow_eps,
                               n_samples=n_samples,
                               logpdf=logpdf,
                               global_kernel=global_kernel,
                               local_kernel=local_kernel,
                               train_func=train_func,
                               heavy_distr=heavy_distr,
                               mixed_proposal_weights=mixed_proposal_weights,
                               target_samples=target_samples)
    return fori_loop(1, n_train, partial_emd_step, (init_proposal, kl_vals))


def em2c_kl_step(i, proposal_state, pow_eps, key, n_samples, logpdf, global_kernel, local_kernel, train_func,
                 heavy_distr,
                 mixed_proposal_weights, target_samples):
    key_empirical, key_resample, key_lker, key_train = split(key[i], 4)
    proposal, kl_vals = proposal_state

    if heavy_distr is None:
        mixed_proposal = proposal
    else:
        mixed_proposal = MixtureGeneral(Categorical(mixed_proposal_weights),
                                        [proposal, heavy_distr])

    empirical_iterate = em2c_empirical_update(key_empirical, logpdf, pow_eps, n_samples, global_kernel, mixed_proposal)
    samples = em2c_resample_empirical_update(key_resample, empirical_iterate, n_samples)
    samples = local_kernel(keys=split(key_lker, n_samples), state=samples)
    proposal = train_func(key=key_train, proposal=proposal, samples=samples)
    kl_vals = kl_vals.at[i].set(estimate_kl(logpdf, proposal, target_samples))
    return (proposal, kl_vals)


def emd_kl_step(i, proposal_state, pow_eps, key, n_samples, logpdf, global_kernel, local_kernel, train_func,
                heavy_distr,
                mixed_proposal_weights, target_samples):
    key_empirical, key_resample, key_lker, key_train = split(key[i], 4)

    proposal, kl_vals = proposal_state
    if heavy_distr is None:
        mixed_proposal = proposal
    else:
        mixed_proposal = MixtureGeneral(Categorical(mixed_proposal_weights),
                                        [proposal, heavy_distr])

    empirical_iterate = emd_empirical_update(key_empirical, logpdf, pow_eps, n_samples, global_kernel, mixed_proposal)
    samples = emd_resample_empirical_update(key_resample, empirical_iterate, n_samples)
    samples = local_kernel(keys=split(key_lker, n_samples), state=samples)
    proposal = train_func(key=key_train, proposal=proposal, samples=samples)
    kl_vals = kl_vals.at[i].set(estimate_kl(logpdf, proposal, target_samples))
    return (proposal, kl_vals)


def em2c_empirical_update(key, logpdf, pow_eps, n_samples, global_kernel, prev_proposal):
    proposal_samples = prev_proposal.sample(key, (n_samples,))
    log_weights = logpdf(proposal_samples) - prev_proposal.log_prob(proposal_samples)
    proposal_log_weights = pow_eps * log_weights
    global_kernel_samples = global_kernel(keys=split(key, n_samples),
                                          state=proposal_samples)
    global_kernel_log_weights = pow_eps * (
            logpdf(global_kernel_samples) - prev_proposal.log_prob(global_kernel_samples))
    global_kernel_lse, proposal_lse = logsumexp(global_kernel_log_weights), logsumexp(proposal_log_weights)
    log_normconst = logsumexp(log_weights)
    # alpha = cond(global_kernel_lse - log_normconst > 0,
    #              lambda _: (global_kernel_lse - log_normconst) / (global_kernel_lse - proposal_lse),
    #              lambda _: 0.5,
    #              operand=None)
    alpha = pow_eps
    return (proposal_samples,
            log_softmax(proposal_log_weights),
            global_kernel_samples,
            log_softmax(global_kernel_log_weights),
            alpha)


def emd_empirical_update(key, logpdf, pow_eps, n_samples, global_kernel, prev_proposal):
    proposal_samples = prev_proposal.sample(key, (n_samples,))
    return (proposal_samples, pow_eps * (logpdf(proposal_samples) - prev_proposal.log_prob(proposal_samples)))


def emd_resample_empirical_update(key, empirical_update, n_samples):
    key_cat, _ = split(key, 2)
    samples, log_weights = empirical_update
    resample_idxs = categorical(key_cat, log_weights, shape=(n_samples,))
    return samples[resample_idxs, :]


def em2c_resample_empirical_update(key, empirical_update, n_samples):
    # TODO: make it more efficient by resampling then propagating
    key_resample_gker, key_resample_proposal, key_cat = split(key, 3)
    proposal_samples, proposal_log_weights, global_kernel_samples, global_kernel_log_weights, alpha = empirical_update
    samples = jnp.vstack([proposal_samples, global_kernel_samples])
    log_weights = jnp.concatenate(
        [jnp.log(alpha) + proposal_log_weights, jnp.log(1 - alpha) + global_kernel_log_weights])
    resample_idxs = categorical(key_cat, log_weights, shape=(n_samples,))
    return samples[resample_idxs, :]




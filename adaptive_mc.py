import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import fori_loop, cond
from mcmc import independent_mh, IMH
from jax.random import split, categorical, uniform, normal, bernoulli
from numpyro.distributions import MixtureGeneral, Categorical
from gaussian_mixture import Gaussian_Mixture, params_to_gm
from gaussian_mixture import train as train_gm
from jax.lax import cond
from jax.nn import logsumexp
from typing import NamedTuple, Callable
from jax.tree_util import Partial as partial


def amc(key, logpdf, n_train, n_samples, model, global_kernel, k_global, local_steps):
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
    partial_adaptivemc_step = partial(amc_step, key=keys, n_samples=n_samples, logpdf=logpdf,
                                      global_kernel=global_kernel,
                                      k_global=k_global, local_steps=local_steps, train_func=train_func)
    return fori_loop(0, n_train, partial_adaptivemc_step, init_proposal)


def amc_step(i, proposal, key, n_samples, logpdf, global_kernel, k_global, local_steps, train_func):
    proposal_samples = proposal.sample(key[i], (n_samples,))
    key_train, key_sampler = split(key[i], 2)
    def imh_samples(key_sampler, proposal):
        keys_imh = split(key_sampler, n_samples)
        imh_params = IMH(proposal=proposal)
        partial_imh = partial(independent_mh, params=imh_params, logpdf=logpdf, burn_in_steps=local_steps, steps=1)
        imh = vmap(partial_imh, in_axes=(0, 0))
        return imh(keys_imh, proposal_samples).reshape(n_samples, -1)

    samples = cond(i % k_global == 0,
                   lambda _: imh_samples(key_sampler, proposal),
                   lambda _: global_kernel(keys=split(key_sampler, n_samples),
                                           state=proposal_samples),
                   operand=None)
    # if i % k_global == 0:
    #     keys_imh = split(key_sampler, n_samples)
    #     imh_params = IMH(proposal=proposal)
    #     partial_imh = partial(independent_mh, params=imh_params, logpdf=logpdf, burn_in_steps=local_steps, steps=1)
    #     imh = vmap(partial_imh, in_axes=(0, 0))
    #     samples = imh(keys_imh, proposal_samples)
    # else:
    #     samples = global_kernel(keys=split(key_sampler, n_samples),
    #                             state=proposal_samples)
    return train_func(key=key_train, proposal=proposal, samples=samples)

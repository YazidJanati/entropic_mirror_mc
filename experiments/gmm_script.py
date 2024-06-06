import jax.numpy as jnp
import jax
from pathlib import Path
from jax.lax import fori_loop
from adaptive_mc import amc
from gaussian_mixture import Gaussian_Mixture, gm_to_params
from entropic_mirror_mc import em2c_kl, emd_kl, MCMC_kernel
from jax.tree_util import Partial as partial
from mcmc import random_walk_mh, unadjusted_langevin, RWM, adjusted_langevin, MALA
from targets import mog4_blockdiag_cov, mog25, mog2_blockdiag_cov
from jax import grad, vmap, default_device, devices, jit
from jax.nn import logsumexp
from jax.random import split, PRNGKey, normal, categorical
from numpyro.distributions import MultivariateNormal, MixtureSameFamily, Categorical, MultivariateStudentT
import matplotlib.pyplot as plt
import blackjax
import os


def run_gmm(key, target, dim, n_components, n_samples):

    logpdf = lambda x: jnp.log(10.) + target.log_prob(x)
    model = Gaussian_Mixture(dim=dim, n_components=n_components)
    heavy_distr = MultivariateStudentT(df=2., loc=jnp.zeros(dim), scale_tril=jnp.eye(dim))

    mala_params = MALA(step_size=1e-1, grad_logpdf=lambda x: grad(logpdf)(x))
    partial_mala = partial(adjusted_langevin, logpdf=logpdf, steps=1, params=mala_params,
                           burn_in_steps=1000)
    # partial_rwm = partial(random_walk_mh, logpdf=target.log_prob, params=rwm_params, burn_in_steps=100, steps=1)
    partial_ula = partial(unadjusted_langevin, grad_logpdf=grad(lambda x: logpdf(x)), burn_in_steps=300,
                          steps=1,
                          step_size=1e-1)

    global_kernel = jit(
        lambda keys, state: vmap(partial_ula, in_axes=(0, 0))(keys, state).reshape(-1, state.shape[-1]))
    local_kernel = jit(
        lambda keys, state: vmap(partial_mala, in_axes=(0, 0))(keys, state).reshape(-1, state.shape[-1]))

    key_target, key_algo = split(key, 2)
    target_samples = target.sample(key_target, (1000,))

    em2c_proposal, kl_vals = em2c_kl(key_algo,
                                     pow_eps=.9,
                                     logpdf=logpdf,
                                     n_train=10,
                                     n_samples=n_samples,
                                     model=model,
                                     global_kernel=global_kernel,
                                     local_kernel=local_kernel,
                                     n_chains=20,
                                     heavy_distr=heavy_distr,
                                     mixed_proposal_weights=jnp.array([.9, .1]),
                                     target_samples=target_samples
                                     )
    emd_proposal, kl_vals = em2c_kl(key_algo,
                                    pow_eps=.9,
                                    logpdf=logpdf,
                                    n_train=10,
                                    n_samples=n_samples,
                                    model=model,
                                    global_kernel=global_kernel,
                                    local_kernel=local_kernel,
                                    n_chains=20,
                                    heavy_distr=heavy_distr,
                                    mixed_proposal_weights=jnp.array([.9, .1]),
                                    target_samples=target_samples
                                    )

    def estimate_norm_const(key, target, proposal):
        n_samples = 5000
        samples = target.sample(key, (n_samples,))
        log_weights = target.log_prob(samples) - proposal.log_prob(samples)
        return logsumexp(log_weights).exp() / n_samples

    key_normconst, _ = split(key_algo, 2)
    em2c_norm_const = estimate_norm_const(key_normconst, target, em2c_proposal)
    emd_norm_const = estimate_norm_const(key_normconst, target, emd_proposal)

    return {'em2c': {'params': gm_to_params(em2c_proposal), 'kl': kl_vals, 'normconst': em2c_norm_const},
            'emd': {'params': gm_to_params(emd_proposal), 'kl': kl_vals, 'normconst': emd_norm_const}}

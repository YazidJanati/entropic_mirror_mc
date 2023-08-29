import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import grad, default_device, devices, vmap
import os
from numpyro.distributions import MultivariateNormal
from mcmc import random_walk_mh, RWM, MALA, adjusted_langevin, independent_mh, IMH
from jax.random import split, PRNGKey, normal, multivariate_normal
import matplotlib.pyplot as plt
from jax.tree_util import Partial as partial
from targets import mog2_blockdiag_cov
import blackjax

with default_device(devices('cpu')[0]):
    with jax.disable_jit(False):
        dim = 10
        key = PRNGKey(126)
        n_chains = 300
        burn_in_steps = 1000

        cov = jnp.array([[10., 1.], [-5., 1.]])
        target = mog2_blockdiag_cov(dim=dim, means=[0., 10.], mini_cov=cov,
                                    weights=jnp.array([.5, .5]))
        proposal = MultivariateNormal(jnp.zeros(dim), 100*jnp.eye(dim))

        keys_imh = split(key, n_chains)
        imh_params = IMH(proposal=proposal)
        partial_imh = partial(independent_mh, params=imh_params, logpdf=target.log_prob, burn_in_steps=25000,
                              steps=1)
        imh = vmap(partial_imh, in_axes=(0, 0))
        init_samples = proposal.sample(key, (n_chains,))
        target.log_prob(init_samples)
        # rwm_samples = rwmh(key_rwm, init_samples).reshape(-1, dim)
        imh_samples = imh(keys_imh, init_samples).reshape(-1, dim)
        target_samples = target.sample(key, (1000,))

        plt.figure(figsize=(10, 5))
        plt.scatter(imh_samples[:, 0], imh_samples[:, 1], label='mala samples')
        plt.scatter(target_samples[:, 0], target_samples[:, 1], label='target samples', alpha=.3)
        # plt.scatter(rwm_samples[:, 0], rwm_samples[:, 1], label='rwm samples')
        # plt.scatter(bkjx_rwmh_samples[:,0], bkjx_rwmh_samples[:,1], label='bkjx rwmh samples')
        plt.legend()
        plt.show()

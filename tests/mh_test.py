import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import grad, default_device, devices, vmap
import os
from numpyro.distributions import MultivariateNormal
from mcmc import random_walk_mh, RWM, MALA, adjusted_langevin
from jax.random import split, PRNGKey, normal, multivariate_normal
import matplotlib.pyplot as plt
from jax.tree_util import Partial as partial
from targets import mog2_blockdiag_cov
import blackjax


def bkjx_loop(key, init_state, kernel, steps):
    keys = split(key, steps)

    def one_step(i, state):
        state, _ = kernel.step(keys[i], state)
        return state

    return fori_loop(0, steps, one_step, kernel.init(init_state))


os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['JAX_LOG_COMPILES'] = '1'

with default_device(devices('cpu')[0]):
    with jax.disable_jit(False):
        dim = 2
        key = PRNGKey(124)
        n_chains = 300
        burn_in_steps = 1000

        cov = jnp.array([[10., 1.], [-5., 1.]])
        target = mog2_blockdiag_cov(dim=dim, means=[0., 10.], mini_cov=cov,
                                    weights=jnp.array([.5, .5]))
        init_distr = MultivariateNormal(5 + jnp.zeros(dim), jnp.eye(dim))

        rwm_params = RWM(cov=.5 * jnp.eye(dim))
        pow = 1.
        logpdf = lambda x: (1 + pow) * target.log_prob(x) - pow * init_distr.log_prob(x)
        init_state = jnp.zeros((n_chains, dim))
        rng_key = PRNGKey(0)
        target.log_prob(init_state)
        # samples = random_walk_mh(rng_key, init_state, logpdf, burn_in_steps, 100, rwm_params)
        # mala_params = MALA(step_size=1e-1, grad_logpdf=lambda x: grad(target.log_prob)(x))

        key_rwm, key_init, key_target = split(PRNGKey(2), 3)
        init_state = normal(key_init, (n_chains, dim))
        # partial_rwm = partial(random_walk_mh, logpdf=target.log_prob, params=rwm_params, burn_in_steps=burn_in_steps,
        #                       steps=1)
        # partial_mala = partial(adjusted_langevin, logpdf=target.log_prob, steps=1, params=mala_params,
        #                        burn_in_steps=burn_in_steps)
        # rwmh = vmap(partial_rwm, in_axes=(0, 0))
        # mala = vmap(partial_mala, in_axes=(0, 0))

        key_init, key_bkjx_mala, key_bkjx_rwmh, key_rwm, key_mala = split(key, 5)
        key_rwm, key_mala = split(key_rwm, n_chains), split(key_mala, n_chains)

        bkjx_mala = blackjax.mala(logpdf, step_size=1e-1)
        bkjx_rwmh = blackjax.rmh(target.log_prob, sigma=.5 * jnp.eye(dim))
        # init_sample = init_distr.sample(key_init)
        # mala_loop(key_init, init_sample, kernel=bkjx_mala, steps=1000)
        bkjx_mala = vmap(partial(bkjx_loop, kernel=bkjx_mala, steps=burn_in_steps), in_axes=(0, 0))
        bkjx_rwmh = vmap(partial(bkjx_loop, kernel=bkjx_rwmh, steps=burn_in_steps), in_axes=(0, 0))

        # rwm_samples = rwmh(key_rwm, init_samples).reshape(-1, dim)
        # mala_samples = bkjx_mala(key_mala, init_samples).reshape(-1, dim)
        bkjx_mala_samples, *_ = bkjx_mala(split(key_bkjx_mala, n_chains), init_state)
        # bkjx_rwmh_samples, *_ = bkjx_rwmh(split(key_bkjx_rwmh, n_chains), init_samples)
        # target_samples = target.sample(key_target, (1000,))

        plt.figure(figsize=(10, 5))
        plt.scatter(bkjx_mala_samples[:, 0], bkjx_mala_samples[:, 1], label='bkjx samples')
        # plt.scatter(mala_samples[:, 0], mala_samples[:, 1], label='mala samples')
        # plt.scatter(target_samples[:, 0], target_samples[:, 1], label='target samples', alpha=.3)
        # plt.scatter(rwm_samples[:, 0], rwm_samples[:, 1], label='rwm samples')
        # plt.scatter(bkjx_rwmh_samples[:,0], bkjx_rwmh_samples[:,1], label='bkjx rwmh samples')
        plt.legend()
        plt.show()

        # steps = 100
        #
        # import jax
        #
        # init_states = jnp.zeros((n_chains, dim))
        # rng_keys = split(PRNGKey(0), n_chains)
        # partial_rwm = partial(random_walk_mh, logpdf=logpdf, params=rwm_params,
        #                       burn_in_steps=burn_in_steps, steps=steps)
        # rwmh = vmap(partial_rwm, in_axes=(0, 0))
        # samples = rwmh(rng_keys, init_state)
        #
        # n_chains = 50000
        # init_states = jax.numpy.zeros((n_chains,))
        # rng_keys = jax.random.split(jax.random.PRNGKey(0), n_chains)
        # partial_rwm = partial(random_walk_mh, logpdf=logpdf, params=rwm_params,
        #                       burn_in_steps=burn_in_steps, steps=steps)
        # samples = jax.jit(jax.vmap(partial_rwm))(rng_keys, init_states)

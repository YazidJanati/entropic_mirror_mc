import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import grad, default_device, devices, vmap
import os
from numpyro.distributions import MultivariateNormal
from mcmc import random_walk_mh, RWM, MALA, adjusted_langevin
from jax.random import split, PRNGKey, normal
import matplotlib.pyplot as plt
from jax.tree_util import Partial as partial
from targets import mog2_blockdiag_cov
import blackjax

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print(jax.local_devices())
# print(devices('gpu')[0])
def mala_loop(key, init_state, kernel, steps):
    keys = split(key, steps)

    def one_step(i, state):
        state, _ = kernel.step(keys[i], state)
        return state

    return fori_loop(0, steps, one_step, kernel.init(init_state))


with default_device(devices('cpu')[0]):
    with jax.disable_jit():
        dim = 2
        cov = jnp.array([[10., 1.], [-5., 1.]])
        target = mog2_blockdiag_cov(dim=dim, means=[0., 10.], mini_cov=cov,
                                    weights=jnp.array([.5, .5]))
        init_distr = MultivariateNormal(-5. + jnp.zeros(dim), jnp.eye(dim))

        rwm_params = RWM(cov=jnp.eye(dim))
        mala_params = MALA(step_size=1e-1, grad_logpdf=lambda x: grad(target.log_prob)(x))

        key_rwm, key_init, key_target = split(PRNGKey(2), 3)
        init_state = normal(key_init, (dim,))
        partial_rwm = partial(random_walk_mh, logpdf=target.log_prob, params=rwm_params, burn_in_steps=1000, steps=1)
        partial_mala = partial(adjusted_langevin, logpdf=target.log_prob, steps=1, params=mala_params, burn_in_steps=1000)
        rwmh = vmap(partial_rwm, in_axes=(0, 0))
        mala = vmap(partial_mala, in_axes=(0, 0))

        n_chains = 100
        key = PRNGKey(1234)
        key_init, key_bkjx, key_rwm, key_mala = split(key, 4)
        key_rwm, key_mala = split(key_rwm, n_chains), split(key_mala, n_chains)
        init_samples = target.sample(key_init, (n_chains,))
        target.log_prob(init_samples)

        bkjx_mala = blackjax.mala(target.log_prob, step_size=1e-1)
        # init_sample = init_distr.sample(key_init)
        # mala_loop(key_init, init_sample, kernel=bkjx_mala, steps=1000)
        bkjx_mala = vmap(partial(mala_loop, kernel=bkjx_mala, steps=1000), in_axes=(0, 0))

        mala_samples = mala(key_mala, init_samples).reshape(-1, dim)
        bkjx_mala_samples, *_ = bkjx_mala(split(key_bkjx, n_chains), init_samples)
        rwm_samples = rwmh(key_rwm, init_samples).reshape(-1, dim)
        target_samples = target.sample(key_target, (1000,))

        plt.figure(figsize=(10, 5))
        # plt.scatter(rwm_samples[:, 0], rwm_samples[:, 1], label='rwm samples')
        plt.scatter(bkjx_mala_samples[:,0], bkjx_mala_samples[:, 1], label = 'bkjx samples')
        plt.scatter(mala_samples[:, 0], mala_samples[:, 1], label='mala samples')
        plt.scatter(target_samples[:, 0], target_samples[:, 1], label='target samples', alpha=.3)
        plt.legend()
        plt.show()

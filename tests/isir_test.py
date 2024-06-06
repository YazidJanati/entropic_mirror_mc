import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import grad, default_device, devices, vmap
import os
from numpyro.distributions import MultivariateNormal, MultivariateStudentT
from mcmc import isir
from jax.random import split, PRNGKey, normal, multivariate_normal
import matplotlib.pyplot as plt
from jax.tree_util import Partial as partial
from targets import mog2_blockdiag_cov
import blackjax

os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['JAX_LOG_COMPILES'] = '1'
with default_device(devices('gpu')[0]):
    with jax.disable_jit(False):
        dim = 10
        key = PRNGKey(126)
        n_samples = 1000
        n_chains = 300
        burn_in_steps = 1000

        cov = jnp.array([[10., 1.], [-5., 1.]])
        target = mog2_blockdiag_cov(dim=dim, means=[0., 5.], mini_cov=cov,
                                    weights=jnp.array([.5, .5]))

        proposal = MultivariateStudentT(df=2., loc=jnp.zeros(dim), scale_tril=10 * jnp.eye(dim))

        key_isir, key_init, key_target = split(key, 3)
        partial_isir = partial(isir, logpdf=target.log_prob,
                               burn_in_steps=50000,
                               proposal=proposal,
                               n_proposals=n_samples,
                               steps=1)
        isir = vmap(partial_isir, in_axes=(0, 0))
        init_samples = proposal.sample(key_init, (n_chains,))
        target.log_prob(init_samples)
        # rwm_samples = rwmh(key_rwm, init_samples).reshape(-1, dim)
        # isir_samples = isir(key_isir, init_state=init_samples, logpdf=target.log_prob, proposal=proposal,
        #                     burn_in_steps=500, steps=1, n_proposals=300)
        isir_samples = isir(split(key_isir, n_chains), init_samples).reshape(-1, dim)
        print(isir_samples.shape)
        target_samples = target.sample(key_target, (1000,))

        plt.figure(figsize=(10, 5))
        plt.scatter(isir_samples[:, 0], isir_samples[:, 1], label='isir samples')
        plt.scatter(target_samples[:, 0], target_samples[:, 1], label='target samples', alpha=.3)
        # plt.scatter(rwm_samples[:, 0], rwm_samples[:, 1], label='rwm samples')
        # plt.scatter(bkjx_rwmh_samples[:,0], bkjx_rwmh_samples[:,1], label='bkjx rwmh samples')
        plt.legend()
        plt.show()

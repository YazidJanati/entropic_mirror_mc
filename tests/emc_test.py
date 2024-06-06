import jax.numpy as jnp
import jax
from jax.lax import fori_loop
from adaptive_mc import amc
from gaussian_mixture import Gaussian_Mixture
from entropic_mirror_mc import em2c_kl, emd_kl, MCMC_kernel
from jax.tree_util import Partial as partial
from mcmc import random_walk_mh, unadjusted_langevin, RWM, adjusted_langevin, MALA
from targets import mog4_blockdiag_cov, mog25, mog2_blockdiag_cov
from jax import grad, vmap, default_device, devices, jit
from jax.random import split, PRNGKey, normal, categorical
from numpyro.distributions import MultivariateNormal, MixtureSameFamily, Categorical, MultivariateStudentT
import matplotlib.pyplot as plt
import blackjax
import os


def mala_loop(key, init_state, kernel, steps):
    keys = split(key, steps)

    def one_step(i, state):
        state, _ = kernel.step(keys[i], state)
        return state

    return fori_loop(0, steps, one_step, kernel.init(init_state))


os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['JAX_LOG_COMPILES'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

with default_device(devices('gpu')[1]):
    with jax.disable_jit(False):
        dim = 2
        key = PRNGKey(11)
        n_samples = 2000
        sigma_y = 0.3

        heavy_distr = MultivariateStudentT(df=2., loc=jnp.zeros(dim), scale_tril=jnp.eye(dim))
        cov = jnp.array([[10., 1.], [-5., 1.]])
        target = mog2_blockdiag_cov(dim=dim, means=[0., 10.], mini_cov=cov,
                                    weights=jnp.array([.8, .2]))
        logpdf = lambda x: jnp.log(10.) + target.log_prob(x)

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

        key_emc, key_amc = split(key, 2)
        key_target, _ = split(key_emc, 2)
        target_samples = target.sample(key_target, (1000,))
        # target.log_prob(target_samples)
        model = Gaussian_Mixture(dim=dim, n_components=2)

        emc_proposal, kl_vals  = emd_kl(key_emc,
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

    print('\nEMC')
    # print(f'mean: {emc_proposal.component_distribution.mean}')
    # print(f'covs: {emc_proposal.component_distribution.covariance_matrix}')
    print(f'weights: {emc_proposal.mixing_distribution.probs}')
    print(kl_vals)
    print('\nAMC')
    # print(f'mean: {amc_proposal.component_distribution.mean}')
    # print(f'covs: {amc_proposal.component_distribution.covariance_matrix}')
    # print(f'weights: {amc_proposal.mixing_distribution.probs}')
    key_emc_proposal, key_amc_proposal, key_target = split(key_emc, 3)
    emc_proposal_samples = emc_proposal.sample(key_emc_proposal, (1000,))
    # amc_proposal_samples = amc_proposal.sample(key_amc_proposal, (1000,))

    plt.figure(figsize=(10, 5))
    plt.scatter(emc_proposal_samples[:, 0], emc_proposal_samples[:, 1])
    # plt.scatter(amc_proposal_samples[:, 0], amc_proposal_samples[:, 1])
    plt.scatter(target_samples[:, 0], target_samples[:, 1])
    plt.show()

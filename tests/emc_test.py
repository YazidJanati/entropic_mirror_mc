import jax.numpy as jnp
import jax
from jax.lax import fori_loop
from gaussian_mixture import Gaussian_Mixture
from entropic_mirror_mc import emc, MCMC_kernel
from jax.tree_util import Partial as partial
from mcmc import random_walk_mh, unadjusted_langevin, RWM
from targets import mog4_blockdiag_cov, mog25, mog2_blockdiag_cov
from jax import grad, vmap
from jax.random import split, PRNGKey, normal, categorical
from numpyro.distributions import MultivariateNormal, MixtureSameFamily, Categorical, MultivariateStudentT
import matplotlib.pyplot as plt
import blackjax
from jax import default_device, devices

def mala_loop(key, init_state, kernel, steps):
    keys = split(key, steps)

    def one_step(i, state):
        state, _ = kernel.step(keys[i], state)
        return state

    return fori_loop(0, steps, one_step, kernel.init(init_state))

with default_device(devices("cpu")[0]):
    dim = 2
    heavy_distr = MultivariateStudentT(df=2., loc=jnp.zeros(dim), scale_tril=jnp.eye(dim))
    cov = jnp.array([[10., 1.], [-5., 1.]])
    target = mog2_blockdiag_cov(dim=dim, means=[0., 10.], mini_cov=cov,
                                weights=jnp.array([.05, .95]))
    # means = jnp.array([[0., 0.], [5., 5.]])
    # covs = .2 * jnp.eye(dim)[jnp.newaxis, :].repeat(2, 0)
    # target = MixtureSameFamily(Categorical(jnp.array([.9, .1])), MultivariateNormal(means, covs))
    rwm_params = RWM(cov=jnp.eye(dim))
    bkjx_mala = blackjax.mala(target.log_prob, step_size=1e-1)
    partial_mala = partial(mala_loop, kernel=bkjx_mala, steps=1000)
    partial_rwm = partial(random_walk_mh, logpdf=target.log_prob, params=rwm_params, burn_in_steps=100, steps=1)
    partial_ula = partial(unadjusted_langevin, grad_logpdf=grad(lambda x: target.log_prob(x)), burn_in_steps=100,
                          steps=1,
                          step_size=5e-2)

    global_kernel = lambda keys, state: vmap(partial_ula, in_axes=(0, 0))(keys, state).reshape(-1, state.shape[-1])
    # local_kernel = lambda keys, state: vmap(partial_rwm, in_axes=(0, 0))(keys, state).reshape(-1, state.shape[-1])
    local_kernel = lambda keys, state: vmap(partial_mala, in_axes=(0, 0))(keys, state)[0].reshape(-1, state.shape[-1])

    key_emc = PRNGKey(1)
    model = Gaussian_Mixture(dim=dim, n_components=2)

    with jax.disable_jit(False):
        proposal = emc(key_emc, pow_eps=.5, logpdf=target.log_prob, n_train=10, n_samples=1000, model=model,
                       global_kernel=global_kernel,
                       local_kernel=local_kernel,
                       n_chains=20,
                       heavy_distr=None,
                       mixed_proposal_weights=jnp.array([.9, .1]))

    print(
        f'mean: {proposal.component_distribution.mean}\n covs: {proposal.component_distribution.covariance_matrix}\n weights: {proposal.mixing_distribution.probs}')
    key_proposal, key_target = split(key_emc, 2)
    proposal_samples = proposal.sample(key_proposal, (1000,))
    target_samples = target.sample(key_target, (1000,))

    plt.figure(figsize=(10, 5))
    plt.scatter(proposal_samples[:, 0], proposal_samples[:, 1])
    plt.scatter(target_samples[:, 0], target_samples[:, 1])
    plt.show()

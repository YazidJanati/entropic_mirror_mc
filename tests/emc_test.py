import jax.numpy as jnp
import jax
from gaussian_mixture import Gaussian_Mixture
from entropic_mirror_mc import emc, MCMC_kernel
from jax.tree_util import Partial as partial
from mcmc import random_walk_mh, unadjusted_langevin, RWM
from targets import mog4_blockdiag_cov, mog25, mog2_blockdiag_cov
from jax import grad, vmap
from jax.random import split, PRNGKey, normal
from numpyro.distributions import MultivariateNormal, MixtureSameFamily, Categorical
import matplotlib.pyplot as plt

dim = 2
# target = mog4_blockdiag_cov(dim=dim, mini_cov=jnp.eye(2))
means = jnp.array([[0., 0.], [2., 2.]])
covs = .2*jnp.eye(dim)[jnp.newaxis,:].repeat(2, 0)
target = MixtureSameFamily(Categorical(jnp.array([.5, .5])), MultivariateNormal(means, covs))
rwm_params = RWM(cov=jnp.eye(dim))

partial_rwm = partial(random_walk_mh, logpdf=target.log_prob, params=rwm_params, burn_in_steps=30)
partial_ula = partial(unadjusted_langevin, grad_logpdf=grad(lambda x: target.log_prob(x)), burn_in_steps=30,
                      step_size=5e-2)
ula = vmap(lambda steps, keys, state: partial_ula(steps=steps, key=keys, init_state=state), in_axes=(None, 0, 0))
rwm = vmap(lambda steps, keys, state: partial_rwm(steps=steps, key=keys, init_state=state), in_axes=(None, 0, 0))

local_kernel = MCMC_kernel(sampler=lambda steps, keys, state: ula(steps, keys, state), n_chains=20)
global_kernel = MCMC_kernel(sampler=lambda steps, keys, state: rwm(steps, keys, state), n_chains=20)

key_emc = PRNGKey(0)
model = Gaussian_Mixture(dim=dim, n_components=2)
proposal = emc(key_emc, logpdf=target.log_prob, n_train=10, n_samples=1000, model=model, global_kernel=global_kernel,
               local_kernel=local_kernel, n_chains=20, heavy_distr=None)

n_chains = 20
key_init = PRNGKey(0)
keys_ula = split(key_init, n_chains)
init_state = normal(key_init, (n_chains, dim))
target.log_prob(init_state)
rwm_samples = ula(100, keys_ula, init_state).reshape(-1,dim)

plt.figure(figsize=(10,5))
plt.scatter(rwm_samples[:,0], rwm_samples[:,1])
plt.show()



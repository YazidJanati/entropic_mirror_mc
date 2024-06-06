import jax
import jax.numpy as jnp
from targets import Funnel
from jax.lax import fori_loop
from jax.random import normal, PRNGKey, split
from numpyro.distributions import MixtureSameFamily, Categorical
import matplotlib.pyplot as plt
from jax import vmap
from jax.tree_util import Partial as partial
from utils import display_samples
import blackjax
from jax import default_device, devices
import os

def bkjx_loop(key, init_state, kernel, steps):
    keys = split(key, steps)

    def one_step(i, state):
        state, _ = kernel.step(keys[i], state)
        return state

    return fori_loop(0, steps, one_step, kernel.init(init_state))

os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['JAX_LOG_COMPILES'] = '1'
with default_device(devices("gpu")[0]):
    n_components = 5
    dim = 10
    key = PRNGKey(0)
    sigma_y = 0.1
    burn_in_steps = 1000
    n_chains = 100

    key_rot, key_loc, key_fun = split(key, 3)
    rotations = normal(key_rot, (n_components, dim, dim))
    # U, _, VT = jnp.linalg.svd(rotations)
    # rot = U @ VT
    rot = jnp.eye(dim)[jnp.newaxis, :].repeat(n_components, 0)
    loc = 10 * normal(key_loc, (n_components, dim))

    cat = Categorical(jnp.array([1 / n_components] * n_components))
    fun = Funnel(a=1., b=.5, loc=loc, rot=rot)
    fun_mixture = MixtureSameFamily(cat, fun)

    key_mixt, key_sample = split(key_fun)
    samples = fun_mixture.sample(key_mixt, (1000,))
    from utils import display_samples
    display_samples(samples)
    mixt_fun_sample = fun_mixture.sample(key_sample, sample_shape=(1,))[0]
    y = mixt_fun_sample[0]
    step_size = 1e-3
    inverse_mass_matrix = jnp.array([1.]*dim)
    def posterior_logprob(x):
        return - ((y - x['loc'][1]) ** 2).sum() / (2 * sigma_y ** 2) + fun_mixture.log_prob(x['loc'])
    # posterior_logprob = lambda x:  - ((y - x['loc'][0]) ** 2).sum() / (2 * sigma_y ** 2) + fun_mixture.log_prob(x['loc'])

    init_state = {"loc": mixt_fun_sample}
    nuts = blackjax.nuts(posterior_logprob, step_size, inverse_mass_matrix)
    # bkjx_nuts = lambda keys: partial(bkjx_loop, kernel=nuts, init_state=init_state,
    #                                  steps=burn_in_steps)(keys)[0]
    bkjx_nuts = lambda keys: vmap(partial(bkjx_loop, kernel=nuts, init_state=init_state,
                                          steps=burn_in_steps))(keys).position['loc']

    key_init, _ = split(key_mixt)
    keys_nuts = split(key_init, n_chains)
    # init_samples = fun_mixture.sample(key_init, (n_chains,))

    nuts_samples = bkjx_nuts(keys_nuts)

    plt.figure(figsize=(10, 5))
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(nuts_samples[:, 0], nuts_samples[:, 1])
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.show()

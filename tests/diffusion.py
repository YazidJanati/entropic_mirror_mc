import jax.numpy as jnp
from targets import Funnel
from numpyro.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from mcmc import unadjusted_langevin, MALA, adjusted_langevin
from jax.tree_util import Partial as partial
from jax import grad, vmap, jit
from jax.random import PRNGKey, split, normal
import matplotlib.pyplot as plt
from jax.lax import fori_loop


def diffusion_sampler(key, logpdf, dim, alphas, n_chains, ula_steps):
    def diffusion_sampler_step(i, prev_samples, keys, ula_steps):
        key = split(keys[i], n_chains)
        logpdf_t = lambda x: (alphas[i] ** 2) * logpdf(x / alphas[i])
        # logpdf_t = lambda x: logpdf(x)
        ula_t = partial(unadjusted_langevin, grad_logpdf=grad(logpdf_t),
                        burn_in_steps=ula_steps,
                        steps=1,
                        step_size=1e-1)
        samples = vmap(ula_t, in_axes=(0, 0))(key, prev_samples).reshape(-1, dim)
        return samples

    key_sampler, key_init = split(key)
    init_samples = normal(key_init, (n_chains, dim))
    keys = split(key_sampler, len(alphas))
    partial_diff_sampler = partial(diffusion_sampler_step, keys=keys, ula_steps=ula_steps)
    return fori_loop(0, len(alphas), partial_diff_sampler, init_samples)


dim = 500
means = jnp.array([[-10.] * dim, [-5.] * dim, [50.] * dim])
covs = jnp.eye(dim)[jnp.newaxis, :].repeat(means.shape[0], 0)
weights = jnp.array([1/means.shape[0]] * means.shape[0])
print(means.shape, covs.shape, weights.shape)
norm = MultivariateNormal(means, covs)
cat = Categorical(weights)

fun = Funnel(a=1., b=.5, loc=means, rot=covs)
target = MixtureSameFamily(cat, fun)

target = MixtureSameFamily(cat, norm)
# partial_ula = partial(unadjusted_langevin, grad_logpdf=grad(lambda x: logpdf(x)),
#                       burn_in_steps=300,
#                       steps=1,
#                       step_size=1e-1)

n_chains = 200
key_sampler, key_init = split(PRNGKey(0), 2)
alphas = jnp.linspace(1., 0.8, 50)
alphas = jnp.flip(jnp.cumprod(alphas))
print(alphas)
# keys_ula = split(key_ula, n_chains)
#
init_sample = normal(key_init, (n_chains, dim))
target.log_prob(init_sample)
# ula_samples = jit(vmap(partial_ula, in_axes=(0, 0)))
partial_diff_sampler = partial(diffusion_sampler, logpdf=target.log_prob, dim=dim, alphas=alphas, n_chains=n_chains,
                               ula_steps=300)
samples = jit(partial_diff_sampler)(key_sampler)
target_samples = target.sample(key_init, (200,))

plt.figure(figsize=(10, 5))
plt.scatter(*target_samples[:, :2].T, alpha=0.4, color='crimson', label="$\\pi$")
plt.scatter(*samples[:, :2].T, alpha=0.3, color='royalblue', label="new sampler")
plt.legend()
plt.show()

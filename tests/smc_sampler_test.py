from jax.random import categorical
from numpyro.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from jax import jit, vmap, random, grad
import jax.numpy as jnp
from jax.lax import fori_loop, cond
from jax import lax
import numpyro
from jax.tree_util import Partial as partial
from jax import disable_jit

key = random.PRNGKey(20)

class Funnel(numpyro.distributions.Distribution):
    arg_constraints = {
        "a": numpyro.distributions.constraints.positive,
        "b": numpyro.distributions.constraints.positive,
    }
    reparametrized_params = ["a", "b"]

    def __init__(self, a, b, dim, validate_args=True):
        super(Funnel).__init__()
        # not really functional, I know...
        self.a = a
        self.b = b
        batch_shape = a.shape
        event_shape = (dim,)
        super(Funnel, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape):
        key_free_coords, _ = random.split(key)
        free_coords = random.normal(key_free_coords, shape=(*sample_shape, *self._batch_shape, *self._event_shape))
        stds = jnp.ones_like(free_coords)
        stds = stds.at[..., 0].multiply(self.a)
        stds = stds.at[..., 1:].multiply(jnp.exp(free_coords[..., 0] * self.b)[..., None])
        samples_funnel = free_coords * stds
        return samples_funnel

    def log_prob(self, value):
        value_shape = value.shape
        stds = jnp.ones_like(value)
        stds = stds.at[..., 0].multiply(self.a)
        stds = stds.at[..., 1:].multiply(jnp.exp(value[..., 0] * self.b)[..., None])
        return -jnp.linalg.norm(value / stds, axis=-1) ** 2


class RigidMotionTransform(numpyro.distributions.transforms.Transform):

    def __init__(self, loc, rot):
        self.loc = loc
        self.rot = rot
        self.inv_rot = jnp.linalg.inv(rot)

    def __call__(self, x):
        sample_shape = x.shape
        rotated_samples = (self.rot.reshape(*((1,) * (len(sample_shape) - 2)), *self.rot.shape) @ x[..., None])[
            ..., 0]
        return rotated_samples + self.loc.reshape(*((1,) * (len(sample_shape) - 2)), *self.loc.shape)

    def _inverse(self, y):
        value_shape = y.shape
        unscaled = y - self.loc.reshape(*((1,) * (len(value_shape) - 2)), *self.loc.shape)
        unrotated = (self.inv_rot.reshape(*((1,) * (len(value_shape) - 2)), *self.inv_rot.shape) @ unscaled[..., None])[
            ..., 0]
        return unrotated

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.broadcast_to(jnp.zeros(1), jnp.shape(x))

    def forward_shape(self, shape):
        return lax.broadcast_shapes(
            shape, getattr(self.loc, "shape", ())
        )

    def inverse_shape(self, shape):
        return lax.broadcast_shapes(
            shape, getattr(self.loc, "shape", ())
        )

    def tree_flatten(self):
        return (self.loc, self.rot, self.inv_rot), (("loc", "rot", "inv_rot"), dict())

    def __eq__(self, other):
        if not isinstance(other, RigidMotionTransform):
            return False
        return (
                jnp.array_equal(self.loc, other.loc)
                & jnp.array_equal(self.rot, other.rot)
        )


def mala(key,
         start_sample,
         log_prob_target,
         max_iter,
         step_size):
    val = (start_sample, key)
    grad_logpdf = grad(lambda x: log_prob_target(x))

    def mh_step(i, val):
        sample, key = val
        key, subkey_u, subkey_sample = random.split(key, 3)
        noise = random.normal(subkey_sample, (sample.shape[-1],))
        next = sample + step_size * grad_logpdf(sample) \
               + jnp.sqrt(2 * step_size) * noise
        fwd_ker_logprob = - (noise ** 2).sum() / 2
        bwd_ker_logprob = - ((sample - next - step_size * grad_logpdf(next)) ** 2).sum() / (4 * step_size)
        log_u = jnp.log(random.uniform(subkey_u))
        log_ratio = log_prob_target(next) + bwd_ker_logprob - log_prob_target(sample) - fwd_ker_logprob
        accept = log_ratio > log_u
        x = cond(accept,
                 lambda _: next,
                 lambda _: sample,
                 None)
        return (x, key)

    sample, k = fori_loop(0,
                          max_iter,
                          body_fun=mh_step,
                          init_val=val)
    return sample


def generate_random_rotation(key, dim):
    rotation = random.normal(key=key, shape=(dim, dim))
    U, _, VT = jnp.linalg.svd(rotation)
    rotation = U @ VT
    return rotation


def generate_prior_distribution(key, n_comp, dim, box_size=20):
    key_rot, key_loc, key_components = random.split(key, 3)
    rotations = vmap(partial(generate_random_rotation, dim=dim))(random.split(key_rot, n_comp))
    loc = random.uniform(key_loc, (n_comp, dim)) * (box_size * 2) - box_size / 2
    probs = random.uniform(key=key_components, shape=(n_comp,))
    probs = probs / probs.sum()

    cat = numpyro.distributions.Categorical(probs)
    comps = rotated_funnel = numpyro.distributions.TransformedDistribution(
        base_distribution=Funnel(jnp.ones(n_comp), jnp.ones(n_comp) * .7, dim),
        transforms=RigidMotionTransform(loc=loc,
                                        rot=rotations)
    )
    return numpyro.distributions.MixtureSameFamily(cat, comps)


box_size = 10
dim_x = 2
key, _ = random.split(key, 2)
prior = generate_prior_distribution(key,
                                    box_size,
                                    dim_x,
                                    box_size=box_size)


def generate_measurement(key, dim_x, dim_y, prior):
    key_A, key_S, key_covar, key_covar, key_prior, key_noise = random.split(key, 6)
    A = random.normal(key=key_A, shape=(dim_y, dim_x))
    U, S, VT = jnp.linalg.svd(A, full_matrices=False)
    S = random.uniform(key_S, shape=S.shape) * 2 - 1
    A = (U * S) @ VT

    covar = (random.uniform(key_covar, shape=(1,)) ** 2) * jnp.eye(dim_y)
    y = A @ prior.sample(key_prior, sample_shape=(1,))[0] + covar @ random.normal(key_noise, (dim_y,))
    return y, A, covar


key, _ = random.split(key, 2)
y, A, covar = generate_measurement(key, dim_x, 1, prior)


def loglik_gaussian(x, y, A, prec):
    res = (y - A @ x).T @ prec @ (y - A @ x)
    return -res


loglik_fun = partial(loglik_gaussian, y=y, A=A, prec=jnp.linalg.inv(covar))


@partial(jit, static_argnames=['n_samples', 'sampler', 'prior', 'likelihood_logpdf'])
def smc_sampler(key, temperatures, n_samples, sampler, prior, likelihood_logpdf):
    key_init, key_steps = random.split(key)
    keys = random.split(key_steps, len(temperatures))
    vmap_likelihood_logpdf = lambda x: vmap(likelihood_logpdf)(x)

    def smc_sampler_step(i, samples):
        key_cat, key_mcmc = random.split(keys[i])
        log_weights = (temperatures[i] - temperatures[i - 1]) * vmap_likelihood_logpdf(samples)
        ancestors = categorical(key_cat, log_weights, shape=(n_samples,))
        resampled = samples[ancestors, :]
        key_mcmc = random.split(key_mcmc, n_samples)
        return sampler(key_mcmc, resampled, lambda x: temperatures[i] * likelihood_logpdf(x) + prior.log_prob(x))

    init_samples = prior.sample(key_init, (n_samples,))
    return fori_loop(1, len(temperatures), smc_sampler_step, init_samples)


# mvns = MultivariateNormal(5 * jnp.array([[1., 1.], [-1., 1.]]), 0.2 * jnp.eye(2)[jnp.newaxis, :].repeat(2, 0))
# cat = Categorical(jnp.array([1 / 2, 1 / 2]))
# target = MixtureSameFamily(cat, mvns)
# prior = MultivariateNormal(jnp.zeros(2), jnp.eye(2))
#
# loglik_fun = lambda x: target.log_prob(x) - prior.log_prob(x)

sampler = lambda key, init_sample, target_logpdf: \
    partial(mala, max_iter=1_00, step_size=0.05)(key=key,
                                                 start_sample=init_sample,
                                                 log_prob_target=target_logpdf)
sampler = vmap(sampler, in_axes=(0, 0, None))
temperatures = jnp.linspace(0., 1., 100)

key, _ = random.split(key, 2)
n_samples = 1000
log_prob_posterior = lambda x: loglik_fun(x) + prior.log_prob(x)
log_prob_posterior(jnp.zeros(2))

from jax import disable_jit

with disable_jit(False):
    posterior_samples_mala = jit(vmap(partial(mala,
                                              log_prob_target=log_prob_posterior,
                                              max_iter=1_000,
                                              step_size=0.05)))(random.split(key, n_samples),
                                                                prior.sample(key, sample_shape=(n_samples,)))

with disable_jit(False):
    smc_samples = smc_sampler(key,
                              temperatures,
                              n_samples=1000,
                              sampler=sampler,
                              prior=prior,
                              likelihood_logpdf=loglik_fun)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(*posterior_samples_mala.T)
plt.scatter(*smc_samples.T)
plt.show()

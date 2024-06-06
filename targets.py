from numpyro.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from jax.numpy import array, eye, zeros, repeat, hstack, ones
from jax.scipy.linalg import block_diag
from numpyro.distributions import Distribution, Normal
from jax.random import split, normal
import jax.numpy as jnp

def mog2_blockdiag_cov(dim, means, mini_cov, weights):
    cov = block_diag(*[mini_cov] * (dim // 2))
    mvn = MultivariateNormal(
        array([means[0] * ones(dim), means[1] * ones(dim)]),
        array([eye(dim), cov]))
    cat = Categorical(weights)
    return MixtureSameFamily(cat, mvn)


def mog25(dim, mode_std):
    means = array([[2 * i, 2 * j] for i in range(0, 5) for j in range(0, 5)])
    means = hstack([means for i in range(dim // 2)])
    covs = array([mode_std * eye(dim) for i in range(25)])
    mvn = MultivariateNormal(means, covs)
    cat = Categorical(array([1 / 25] * 25))
    return MixtureSameFamily(cat, mvn)


def mog4_blockdiag_cov(dim, mini_cov, weights=array([1 / 4] * 4)):
    means = array([[-10., -10.], [-10., 10.],
                   [10., 10.], [10., -10.]])
    means = hstack([means for i in range(dim // 2)])
    cov = block_diag(*[mini_cov] * (dim // 2))
    mvn = MultivariateNormal(means, cov)
    cat = Categorical(weights)
    return MixtureSameFamily(cat, mvn)

# class Funnel(Distribution):
#
#     def __init__(self, a, b, loc, rot):
#         super(Funnel).__init__()
#         self.a = a
#         self.b = b
#         self.free_noise_dist = Normal(0, 1)
#         self._batch_shape = loc.shape[:-1]
#         self._event_shape = loc.shape[-1:]
#         self.loc = loc
#         self.rotations = rot
#         self.inv_rot = jnp.linalg.inv(rot)
#
#     def sample(self, key, sample_shape):
#         key_free_coords, _ = split(key)
#         free_coords = self.free_noise_dist.sample(key_free_coords,
#                                                   sample_shape=(
#                                                   *sample_shape, *self._batch_shape, *self._event_shape))
#         stds = jnp.ones_like(free_coords)
#         stds = stds.at[..., 0].multiply(self.a)
#         stds = stds.at[..., 1:].multiply(jnp.exp(free_coords[..., 0] * self.b)[..., None])
#         samples_funnel = free_coords * stds
#         rotated_funnels = \
#         (self.rotations.reshape(*((1,) * len(sample_shape)), *self.rotations.shape) @ samples_funnel[..., None])[..., 0]
#         return rotated_funnels + self.loc.reshape(*((1,) * len(sample_shape)), *self.loc.shape)
#
#     def log_prob(self, value):
#         value_shape = value.shape[:-2]
#         unscaled = value - self.loc.reshape(*((1,)*len(value_shape)), *self.loc.shape)
#         unrotated = (self.inv_rot.reshape(*((1,)*len(value_shape)), *self.inv_rot.shape) @ unscaled[..., None])[..., 0]
#         stds = jnp.ones_like(unrotated)
#         stds = stds.at[..., 0].multiply(self.a)
#         stds = stds.at[..., 1:].multiply(jnp.exp(unrotated[..., 0]*self.b)[..., None])
#         return self.free_noise_dist.log_prob(unrotated / stds).sum(-1)

class Funnel(Distribution):

    def __init__(self, a, b, loc, rot):
        super(Funnel).__init__()
        self.a = a
        self.b = b
        self._batch_shape = loc.shape[:-1]
        self._event_shape = loc.shape[-1:]
        self.loc = loc
        self.rotations = rot
        self.inv_rot = jnp.linalg.inv(rot)

    def sample(self, key, sample_shape):
        key_free_coords, _ = split(key)
        free_coords = normal(key_free_coords, shape=(*sample_shape, *self._batch_shape, *self._event_shape))
        stds = jnp.ones_like(free_coords)
        stds = stds.at[..., 0].multiply(self.a)
        stds = stds.at[..., 1:].multiply(jnp.exp(free_coords[..., 0] * self.b)[..., None])
        samples_funnel = free_coords * stds
        rotated_funnels = \
            (self.rotations.reshape(*((1,) * len(sample_shape)), *self.rotations.shape) @ samples_funnel[..., None])[
                ..., 0]
        return rotated_funnels + self.loc.reshape(*((1,) * len(sample_shape)), *self.loc.shape)

    def log_prob(self, value):
        value_shape = value.shape[:-2]
        unscaled = value - self.loc.reshape(*((1,) * len(value_shape)), *self.loc.shape)
        unrotated = (self.inv_rot.reshape(*((1,) * len(value_shape)), *self.inv_rot.shape) @ unscaled[..., None])[
            ..., 0]
        stds = jnp.ones_like(unrotated)
        stds = stds.at[..., 0].multiply(self.a)
        stds = stds.at[..., 1:].multiply(jnp.exp(unrotated[..., 0] * self.b)[..., None])
        return -jnp.linalg.norm(unrotated / stds, axis=-1) ** 2
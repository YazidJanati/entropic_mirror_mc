from numpyro.distributions import Distribution, constraints
from typing import Sequence, Callable
import optax
from jax.lax import fori_loop
from jax.random import split
from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from jax import value_and_grad, debug

class MLP(nn.Module):
    """
    Multi-layer perceptron in Flax. We use a gaussian kernel with a standard deviation
    of `init_weight_scale=1e-4` by default.

    Args:
        features: (list of int) The number of features in each layer.
        activation: (callable) The activation function at each level
        use_bias: (bool) Whether to use bias in the layers.
        init_weight_scale: (float) The initial weight scale for the layers.
        kernel_init: (callable) The kernel initializer for the layers.
    """

    features: Sequence[int]
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-4
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.layers = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
            )
            for feat in self.features
        ]

    def __call__(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

class AffineCoupling(nn.Module):
    """
    Affine coupling layer.
    (Defined in the RealNVP paper https://arxiv.org/abs/1605.08803)
    We use tanh as the default activation function.

    Args:
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        mask: (ndarray) Alternating mask for the affine coupling layer.
        dt: (float) Scaling factor for the affine coupling layer.
    """

    n_features: int
    n_hidden: int
    mask: jnp.array
    dt: float = 1

    def setup(self):
        self.scale_MLP = MLP([self.n_features, self.n_hidden, self.n_features])
        self.translate_MLP = MLP([self.n_features, self.n_hidden, self.n_features])

    def __call__(self, x):
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s)
        t = self.mask * self.translate_MLP(x * (1 - self.mask))
        s = self.dt * s
        t = self.dt * t
        log_det = s.reshape(s.shape[0], -1).sum(axis=-1)
        outputs = (x + t) * jnp.exp(s)
        return outputs, log_det

    def inverse(self, x):
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s)
        t = self.mask * self.translate_MLP(x * (1 - self.mask))
        s = self.dt * s
        t = self.dt * t
        log_det = -s.reshape(s.shape[0], -1).sum(axis=-1)
        outputs = x * jnp.exp(-s) - t
        return outputs, log_det


class RealNVP(nn.Module):
    """
    RealNVP mode defined in the paper https://arxiv.org/abs/1605.08803.
    MLP is needed to make sure the scaling between layers are more or less the same.

    Args:
        n_layer: (int) The number of affine coupling layers.
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        dt: (float) Scaling factor for the affine coupling layer.

    Properties:
        base_mean: (ndarray) Mean of Gaussian base distribution
        base_cov: (ndarray) Covariance of Gaussian base distribution
    """

    n_layer: int
    n_features: int
    n_hidden: int
    dt: float = 1

    def setup(self):
        affine_coupling = []
        for i in range(self.n_layer):
            mask = np.ones(self.n_features)
            mask[int(self.n_features / 2):] = 0
            if i % 2 == 0:
                mask = 1 - mask
            mask = jnp.array(mask)
            affine_coupling.append(
                AffineCoupling(self.n_features, self.n_hidden, mask, dt=self.dt)
            )
        self.affine_coupling = affine_coupling

    def __call__(self, x):
        log_det = jnp.zeros(x.shape[0])
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[i](x)
            log_det += log_det_i
        return x, log_det

    def inverse(self, x):
        # x = (x - self.base_mean.value) / jnp.sqrt(jnp.diag(self.base_cov.value))
        log_det = jnp.zeros(x.shape[0])
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[self.n_layer - 1 - i].inverse(x)
            log_det += log_det_i
        return x, log_det

    def sample(self, rng_key, sample_shape):
        gaussian = jax.random.multivariate_normal(
            rng_key, jnp.zeros(self.n_features), jnp.eye(self.n_features), shape=sample_shape
        )
        samples = self.inverse(gaussian)[0]
        # samples = samples * jnp.sqrt(jnp.diag(self.base_cov.value)) + self.base_mean.value
        return samples  # Return only the samples

    def log_prob(self, x):
        # x = (x - self.base_mean.value) / jnp.sqrt(jnp.diag(self.base_cov.value))
        y, log_det = self.__call__(x)
        log_det = log_det + jax.scipy.stats.multivariate_normal.logpdf(
            y, jnp.zeros(self.n_features), jnp.eye(self.n_features)
        )
        return log_det
class RNVPDistr(Distribution):
    support = constraints.real_vector

    def __init__(self, model, param):
        self.model = model
        self.param = param
        self._batch_shape = ()
        self._event_shape = (model.n_features,)

    def log_prob(self, x):
        return self.model.apply({'params': self.param}, x, method=self.model.log_prob)

    def sample(self, key, sample_shape):
        return self.model.apply({'params': self.param}, key, sample_shape, method=self.model.sample)

def mle_training(key, samples, rnvp, loss_fn, batch_size, epochs, init_params=None, lr=1e-3,
                 target_logpdf=None, target_samples=None):
    n_samples = samples.shape[0]
    steps_per_epoch = n_samples // batch_size

    optim = optax.adam(lr)

    def eval_fn(params):
        rnvp_logprob = rnvp.apply({'params': params}, target_samples, method=rnvp.log_prob)
        return (target_logpdf(target_samples) - rnvp_logprob()).mean()

    def train_epoch(i, state, keys):

        def grad_step(i, state):
            params, opt_state = state
            loss, grad = value_and_grad(loss_fn, argnums=(1))(rnvp, params, samples[idxs[i], :])
            updates, opt_state = optim.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state)

        key = keys[i]
        n_samples = samples.shape[0]
        idxs = jax.random.permutation(key, n_samples)[:steps_per_epoch * batch_size]
        idxs = idxs.reshape(-1, batch_size)

        return fori_loop(0, steps_per_epoch, grad_step, state)
        # params, opt_state = state
        # for i in range(idxs.shape[0]):
        #     loss, grad = value_and_grad(loss_fn, argnums=(1))(rnvp, params, samples[idxs[i], :])
        #     updates, opt_state = optim.update(grad, opt_state)
        #     params = optax.apply_updates(params, updates)
        #
        # return (params, opt_state)

    key_init, key_train = split(key)
    keys = split(key, epochs)
    train = partial(train_epoch, keys=keys)

    if init_params is None:
        init_params = rnvp.init(key_init, jnp.ones((1, rnvp.n_features)))["params"]

    opt_state = optim.init(init_params)
    return fori_loop(0, epochs, train, (init_params, opt_state))
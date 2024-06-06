import jax
import jax.numpy as jnp
import optax
from jax.lax import fori_loop
from numpyro.distributions import MultivariateNormal
from realnvp import RealNVP, RNVPDistr, mle_training
from jax.random import PRNGKey, split, multivariate_normal, normal
from jax.tree_util import Partial as partial
from targets import mog2_blockdiag_cov, mog25, mog4_blockdiag_cov
from utils import estimate_kl
import yaml
from tqdm import trange
import os
from jax import default_device, devices, jit, vmap, debug, value_and_grad


def old_mle_training(key, samples, rnvp, batch_size, epochs, lr=1e-3):
    n_samples = samples.shape[0]
    steps_per_epoch = n_samples // batch_size

    optim = optax.adam(lr)

    def mean_likelihood(params, batch):
        log_prob = rnvp.apply({'params': params}, batch, method=rnvp.log_prob)
        return - log_prob.mean()

    def train_epoch(i, state, keys):
        debug.print('{i}', i=i)
        key = keys[i]
        n_samples = samples.shape[0]
        idxs = jax.random.permutation(key, n_samples)[:steps_per_epoch * batch_size]
        idxs = idxs.reshape(-1, batch_size)

        params, opt_state = state

        def grad_step(i, state):
            params, opt_state = state
            loss, grad = value_and_grad(mean_likelihood, argnums=(0))(params, samples[idxs[i], :])
            updates, opt_state = optim.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state)

        # for i in range(idxs.shape[0]):
        #     loss, grad = value_and_grad(mean_likelihood, argnums=(0))(params, samples[idxs[i], :])
        #     updates, opt_state = optim.update(grad, opt_state)
        #     params = optax.apply_updates(params, updates)
        #
        # return (params, opt_state)
        return fori_loop(0, idxs.shape[0], grad_step, (params, opt_state))

    key_init, key_train = split(key)
    keys = split(key, epochs)
    train = partial(train_epoch, keys=keys)

    init_params = rnvp.init(key_init, jnp.ones((1, rnvp.n_features)))["params"]
    opt_state = optim.init(init_params)
    return fori_loop(0, epochs, train, (init_params, opt_state))


def mean_likelihood(model, params, batch):
    log_prob = rnvp.apply({'params': params}, batch, method=rnvp.log_prob)
    return - log_prob.mean()


os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['JAX_LOG_COMPILES'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

with default_device(devices('gpu')[2]):
    with jax.disable_jit(False):
        n_layers = 10
        n_hidden = 128

        dim = 50
        key_rnvp, key_train, key_target, key_init = split(PRNGKey(0), 4)
        rnvp = RealNVP(n_features=dim, n_layer=n_layers, n_hidden=n_hidden)
        # target = mog4_blockdiag_cov(dim=dim, mini_cov=jnp.eye(dim))
        # cov = jnp.array([[10., 1.], [-5., 1.]])
        # target = mog2_blockdiag_cov(dim=dim, means=[5., 6.], mini_cov=cov,
        #                             weights=jnp.array([.5, .5]))
        # target = MultivariateNormal(10 * jnp.ones(dim), jnp.eye(dim))
        mini_cov2 = jnp.array([[10., 1.], [-5., 1.]])
        mini_cov4 = jnp.array([[3., 4.], [4., 10.]])
        # target = mog25(dim=dim, mode_std=.01)
        # target = mog4_blockdiag_cov(dim=dim, mini_cov=mini_cov4)
        target = mog2_blockdiag_cov(dim=dim, means=[0., 20.],
                                    mini_cov=mini_cov2,
                                    weights=jnp.array([0.2, 0.8]))
        target_samples = target.sample(key_target, (10_000,))
        init_params = rnvp.init(key_init, jnp.ones((1, rnvp.n_features)))["params"]
        # optim = optax.adam(learning_rate)
        # params, _ = vi_training(key_train, rnvp=rnvp, logpdf=target.log_prob, optim=optim, n_samples=100, n_train=200)

        jit_mle_training = jit(partial(mle_training,
                                       loss_fn=mean_likelihood,
                                       rnvp=rnvp,
                                       batch_size=1024,
                                       epochs=2000,
                                       lr=1e-3))

        # jit_mle_training = jit(partial(old_mle_training,
        #                            samples=target_samples,
        #                            rnvp=rnvp,
        #                            batch_size=1024,
        #                            epochs=2000,
        #                            lr=1e-3))

        # params, _ = mle_training(key_train, samples=target_samples, rnvp=rnvp, batch_size=512, epochs=30, lr=1e-3)
        print('JIT done, starting training...')
        params, _ = jit_mle_training(key=key_train, init_params=init_params, samples=target_samples)

        key_samples, key_target = split(key_rnvp)
        rnvp_distr = RNVPDistr(rnvp, params)
        target_samples = target.sample(key_target, (20_000,))
        print(f'opt_kl: {estimate_kl(target.log_prob, rnvp_distr, target_samples)}')
        # nf_samples = rnvp.apply({'params': params}, key_samples, 10000, method=rnvp.sample)
        nf_samples = rnvp_distr.sample(key_samples, (2000,))

        import matplotlib.pyplot as plt

        target_samples = target_samples[:2000, :]

        plt.figure(figsize=(10, 5))
        plt.scatter(nf_samples[:, 0], nf_samples[:, 1])
        plt.scatter(target_samples[:, 0], target_samples[:, 1])
        plt.show()

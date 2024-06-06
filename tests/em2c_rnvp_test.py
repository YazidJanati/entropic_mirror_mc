import os
import sys

sys.path.append('/home/yjanati/projects/entropic_mirror_mc/')

import jax
import jax.numpy as jnp
import optax
from jax.lax import fori_loop
from numpyro.distributions import MultivariateNormal, MultivariateStudentT
from realnvp import RealNVP, RNVPDistr, mle_training
from jax.random import PRNGKey, split, multivariate_normal, normal
from jax.tree_util import Partial as partial
from targets import mog2_blockdiag_cov, mog25, mog4_blockdiag_cov
from mcmc import random_walk_mh, unadjusted_langevin, RWM, adjusted_langevin, MALA
from jax import default_device, devices, jit, vmap, debug, value_and_grad, grad, config
from em2c import em2c, estimate_kl
import yaml

config.update('jax_enable_x64', False)
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['JAX_LOG_COMPILES'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def train_mle(key, model, samples, train_args):
    def mean_likelihood(model, params, batch):
        log_prob = model.apply({'params': params}, batch, method=model.log_prob)
        return - log_prob.mean()

    key_train, key_init = split(key)
    batch_size, epochs, lr = train_args['batch_size'], train_args['epochs'], train_args['lr']
    init_params = model.init(key_init, jnp.ones((1, model.n_features)))["params"]

    return mle_training(key, samples, model, mean_likelihood, batch_size, epochs, init_params, lr)


n_modes = int(sys.argv[1])
dim = int(sys.argv[2])

with default_device(devices('gpu')[3]):
    with jax.disable_jit(False):
        with open(f'../configs/mog{n_modes}.yaml', 'r') as conf:
            config = yaml.safe_load(conf)

        n_layers, n_hidden, n_samples, batch_size, epochs, lr \
            = config['n_layers'], config['n_hidden'], config['n_samples'], config['batch_size'], config['epochs'], \
            jnp.float32(config['lr'])

        # learning_rate = 0.001
        # momentum = 0.9
        # n_layers = 10
        # n_hidden = 128
        # dt = 1 / n_layers
        heavy_distr = MultivariateStudentT(df=2., loc=jnp.zeros(dim),
                                           scale_tril=jnp.eye(dim))
        mini_cov2 = jnp.array([[10., 1.], [-5., 1.]])
        mini_cov4 = jnp.array([[3., 4.], [4., 10.]])
        target = mog25(dim=dim, mode_std=.01)
        # target = mog4_blockdiag_cov(dim=dim, mini_cov=mini_cov4)
        target = mog2_blockdiag_cov(dim=dim,
                                    means=jnp.array([0., 20.]),
                                    mini_cov=mini_cov2,
                                    weights=jnp.array([0.2, 0.8]))
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

        train_args = {'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 'train_mode': 'renyi'}
        key_emc, key_train, key_target = split(PRNGKey(0), 3)
        target_samples = target.sample(key_target, (10_000,))
        rnvp = RealNVP(n_features=dim, n_layer=n_layers, n_hidden=n_hidden)
        opt_params, _ = train_mle(key_train, rnvp, target_samples, train_args)

        opt_rnvp = RNVPDistr(rnvp, opt_params)
        kl = estimate_kl(logpdf, opt_rnvp, target_samples)
        print(f'optimal kl: {kl}')

        partial_emc = partial(em2c,
                              pow_eps=.9,
                              logpdf=logpdf,
                              n_train=10,
                              n_samples=n_samples,
                              model=rnvp,
                              train_args=train_args,
                              global_kernel=global_kernel,
                              local_kernel=local_kernel,
                              heavy_distr=heavy_distr,
                              mixed_proposal_weights=jnp.array([.9, .1]),
                              target_samples=target_samples
                              )
        emc_proposal, kl_vals = partial_emc(key_emc)
        print(kl_vals)
        # nf_samples = rnvp.apply({'params': params}, key_samples, 10000, method=rnvp.sample)
        # nf_samples = rnvp_distr.sample(key_samples, (10000, ))
        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(10, 5))
        # plt.scatter(target_samples[:, 0], target_samples[:, 1])
        # plt.scatter(nf_samples[:, 0], nf_samples[:, 1])
        # plt.show()

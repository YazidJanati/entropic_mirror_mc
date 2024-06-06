import jax.numpy as jnp
import os
import sys
sys.path.append('/home/yjanati/projects/entropic_mirror_mc/')

import jax
from pathlib import Path
from jax.lax import fori_loop
from adaptive_mc import amc
from gaussian_mixture import Gaussian_Mixture, train, params_to_gm
from entropic_mirror_mc import em2c_kl, emd_kl, MCMC_kernel
from jax.tree_util import Partial as partial
from mcmc import random_walk_mh, unadjusted_langevin, RWM, adjusted_langevin, MALA
from targets import mog4_blockdiag_cov, mog25, mog2_blockdiag_cov
from jax import grad, vmap, default_device, devices, jit
from jax.random import split, PRNGKey, normal, categorical
from numpyro.distributions import MultivariateNormal, MixtureSameFamily, Categorical, MultivariateStudentT
import matplotlib.pyplot as plt

from gmm_script import run_gmm
import torch



os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['JAX_LOG_COMPILES'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

folder = '/mnt/data/yazid/em2c_experiments/'

n_modes = int(sys.argv[1])
device = int(sys.argv[2])
repeat = int(sys.argv[3])

save_folder = os.path.join(folder, f'gmm_{n_modes}_exp')
Path(save_folder).mkdir(exist_ok=True)

with default_device(devices('gpu')[device]):
    key = PRNGKey(10)
    dims_nsamples = [(2, 2000), (20, 5000), (100, 10000), (200, 10000)]

    for dim in dims_nsamples:
        dim, n_samples = dim
        if n_modes == 2:
            cov = jnp.array([[10., 1.], [-5., 1.]])
            target = mog2_blockdiag_cov(dim=dim, means=[0., 10.], mini_cov=cov,
                                        weights=jnp.array([.2, .8]))
            components = [2]

        elif n_modes == 4:
            cov = jnp.array([[3., 4.], [4., 10.]])
            target = mog4_blockdiag_cov(dim=dim, mini_cov=cov)
            components = [2, 4, 8]

        elif n_modes == 25:
            target = mog25(dim=dim, mode_std=0.25)
            components = [10, 30]

        dim_path = os.path.join(save_folder, f'gmm_{n_modes}_{dim}')
        Path(dim_path).mkdir(exist_ok=True)

        for n_components in components:

            key, key_em = split(key)
            target_samples = target.sample(key, (n_samples,))
            means, covs, log_weights = train(samples=target_samples, init_params='kmeans', n_components=n_components,
                                             key=key_em)
            opt_proposal = params_to_gm(means, covs, log_weights)
            log_weights = jnp.log(10) + target.log_prob(target_samples) - opt_proposal.log_prob(target_samples)
            opt_kl = log_weights.mean()

            partial_run_gmm = partial(run_gmm, target=target, dim=dim, n_components=n_components, n_samples=n_samples)
            jit_run_gmm = jit(partial_run_gmm)
            runs = {'exp_detail': {'nmodes': n_modes, 'ncomponents': n_components, 'dim': dim, 'n_samples': n_samples,
                                   'n_repeat': repeat},
                    'results': []}

            for r in range(repeat):
                print(f'{dim}: {r} / {repeat}')
                key, _ = split(key)
                res = jit_run_gmm(key)
                res['optKL'] = opt_kl
                res['exp_details'] = {''}
                runs['results'].append(res)

                print(res['em2c']['kl'])
                print(res['emd']['kl'])
                print(f'optimal KL: {opt_kl}')

                torch.save(runs, os.path.join(dim_path, f'gmm_{n_modes}_{dim}_{n_components}'))

import jax.numpy as jnp
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
import os
import sys
from gmm_script import run_gmm
import torch

n_modes = 2
path = f'/mnt/data/yazid/em2c_experiments/gmm_{n_modes}_exp/'

for dim_folder in os.listdir(path):
    for component_file in os.listdir(os.path.join(path, dim_folder)):
        results = torch.load(os.path.join(path, dim_folder, component_file))
        dim = results['exp_detail']['dim']
        for rep in range(len(results)):

            print(results['results'][0]['em2c']['kl'])

from gaussian_mixture import train, params_to_gm
from targets import mog4_blockdiag_cov, mog2_blockdiag_cov
from jax.nn import softmax
import jax.numpy as jnp
from jax.random import PRNGKey, split
import matplotlib.pyplot as plt
import os
from jax import default_device, devices

with default_device(devices("cpu")[0]):
    dim = 20
    key_target, key_em, key_gm = split(PRNGKey(0), 3)
    # target = mog4_blockdiag_cov(dim=dim, mini_cov=jnp.eye(2))
    cov = jnp.array([[10., 1.], [-5., 1.]])
    target = mog2_blockdiag_cov(dim=dim, means=[0., 20.], mini_cov=cov,
                                weights=jnp.array([.1, .9]))
    target_samples = target.sample(key_target, (1000,))

    means, covs, log_weights = train(samples=target_samples, init_params='kmeans', n_components=4, key=key_em)

    gm = params_to_gm(means, covs, softmax(log_weights, 0))
    gm_samples = gm.sample(key_gm, (1000,))

    plt.figure(figsize=(10,5))
    plt.scatter(target_samples[:,0], target_samples[:,1])
    plt.scatter(gm_samples[:,0], gm_samples[:,1])
    plt.show()
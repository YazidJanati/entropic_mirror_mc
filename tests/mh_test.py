import jax
import jax.numpy as jnp
from jax import grad, default_device, devices

import os
from numpyro.distributions import MultivariateNormal
from mcmc import random_walk_mh, RWM, MALA, adjusted_langevin
from jax.random import split, PRNGKey, normal
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(jax.local_devices())
print(devices('gpu')[0])
with default_device(devices('gpu')[0]):
    dim = 2
    a = jnp.ones((100, 100))
    print(a * a)
    mvn = MultivariateNormal(jnp.zeros(dim), jnp.array([[6., 1.], [3, 1.]]))

    rwm_params = RWM(cov=jnp.eye(dim))
    mala_params = MALA(step_size=1e-1, grad_logpdf=lambda x: grad(mvn.log_prob)(x).sum())

    key_rwm, key_init, key_target = split(PRNGKey(2), 3)
    init_state = normal(key_init, (dim,))
    rwm_samples = random_walk_mh(key_rwm, init_state, logpdf=mvn.log_prob, params=rwm_params, burn_in_steps=30, steps=1000)
    mala_samples = adjusted_langevin(key_rwm, init_state, logpdf=mvn.log_prob, params=mala_params, burn_in_steps=30,
                                     steps=1000)
    print(rwm_samples)
    target_samples = mvn.sample(key_target, (1000,))

    plt.figure(figsize=(10,5))
    plt.scatter(target_samples[:,0], target_samples[:,1])
    plt.scatter(rwm_samples[:,0], rwm_samples[:,1])
    plt.scatter(mala_samples[:,0], mala_samples[:,1])
    plt.show()
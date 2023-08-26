import jax.numpy as jnp
import optax
from flowMC.nfmodel.realNVP import RealNVP
from jax.random import PRNGKey, split

num_epochs = 3000
batch_size = 10000
learning_rate = 0.001
momentum = 0.9
n_layers = 10
n_hidden = 128
dt = 1 / n_layers

dim = 2
key_rnvp = PRNGKey(0)
rnvp = RealNVP(n_features=dim, n_layer=n_layers, n_hidden=n_hidden, key=key_rnvp)

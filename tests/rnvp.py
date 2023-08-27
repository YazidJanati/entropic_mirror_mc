import jax.numpy as jnp
import optax
from flowMC.nfmodel.utils import make_training_loop
from flowMC.nfmodel.realNVP import RealNVP
from jax.random import PRNGKey, split
from targets import mog4_blockdiag_cov

num_epochs = 3000
batch_size = 10000
learning_rate = 0.001
momentum = 0.9
n_layers = 10
n_hidden = 128
dt = 1 / n_layers

dim = 2
key_rnvp, key_train, key_target = split(PRNGKey(0), 3)
rnvp = RealNVP(n_features=dim, n_layer=n_layers, n_hidden=n_hidden, key=key_rnvp)
target = mog4_blockdiag_cov(dim=dim, mini_cov=jnp.eye(dim))
target_samples = target.sample(key_target, (1000,))

optim = optax.adam(learning_rate)
train_flow, _, _ = make_training_loop(optim)

key, model, loss = train_flow(key_train, rnvp, target_samples, num_epochs, batch_size, verbose=True)

key_samples, _ = split(key_rnvp)
nf_samples = model.sample(key_rnvp, 5000)

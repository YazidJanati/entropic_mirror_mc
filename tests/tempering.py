import jax.numpy as jnp
from numpyro.distributions import Normal, MixtureSameFamily, Categorical
from jax.random import PRNGKey, split
import matplotlib.pyplot as plt

means = jnp.array([-5., 7.])
norms = Normal(means, jnp.array([1., 1.]))
cat = Categorical(jnp.array([.5, .5]))
mixt = MixtureSameFamily(cat, norms)
norm = Normal(0., 1.)

key = PRNGKey(0)
lambdas = jnp.linspace(0., 1., 10)

x = jnp.linspace(-10, 10, 500)
plt.figure(figsize=(10, 5))
for t in range(len(lambdas)):
    temp_seq = lambda x: lambdas[t] * mixt.log_prob(x) + (1 - lambdas[t]) * norm.log_prob(x)
    plt.plot(x, temp_seq(x))
    plt.show()

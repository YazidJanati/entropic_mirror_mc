from numpyro.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from jax.numpy import array, eye, zeros, repeat, hstack, ones
from jax.scipy.linalg import block_diag

def mog2_blockdiag_cov(dim, means, mini_cov, weights):
    cov = block_diag(*[mini_cov] * (dim // 2))
    mvn = MultivariateNormal(
        array([means[0] * ones(dim), means[1] * ones(dim)]),
        array([eye(dim), cov]))
    cat = Categorical(weights)
    return MixtureSameFamily(cat, mvn)


def mog25(dim, mode_std):
    means = array([[2 * i, 2 * j] for i in range(0, 5) for j in range(0, 5)])
    means = hstack([means for i in range(dim // 2)])
    covs = array([mode_std * eye(dim) for i in range(25)])
    mvn = MultivariateNormal(means, covs)
    cat = Categorical(array([1 / 25] * 25))
    return MixtureSameFamily(cat, mvn)


def mog4_blockdiag_cov(dim, mini_cov, weights=array([1 / 4] * 4)):
    means = array([[-10., -10.], [-10., 10.],
                   [10., 10.], [10., -10.]])
    means = hstack([means for i in range(dim // 2)])
    cov = block_diag(*[mini_cov] * (dim // 2))
    mvn = MultivariateNormal(means, cov)
    cat = Categorical(weights)
    return MixtureSameFamily(cat, mvn)
import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop, stop_gradient
from jax.random import split, categorical, uniform, normal, bernoulli
from numpyro.distributions import MixtureGeneral, Categorical
from gaussian_mixture import Gaussian_Mixture, params_to_gm
from gaussian_mixture import train as train_gm
from jax.lax import cond
from jax.nn import logsumexp, log_softmax
from typing import NamedTuple, Callable
from jax.tree_util import Partial as partial
from jax.nn import softmax
from realnvp import RealNVP, mle_training, RNVPDistr
from jax import debug
from utils import display_samples


class MCMC_kernel(NamedTuple):
    sampler: Callable
    n_chains: int


def em2c(key,
         logpdf,
         pow_eps,
         n_train,
         n_samples,
         model,
         train_args,
         global_kernel,
         local_kernel,
         heavy_distr,
         mixed_proposal_weights=jnp.array([.8, .2]),
         target_samples=None):
    key_train, key_init = split(key, 2)

    if isinstance(model, RealNVP):
        batch_size, epochs, lr, train_mode \
            = train_args['batch_size'], train_args['epochs'], train_args['lr'], train_args['train_mode']
        init_params = model.init(key_init, jnp.ones((1, model.n_features)))["params"]

        train_fn = renyi_training if train_mode == 'renyi' else kl_training
        partial_train = partial(train_fn, logpdf=logpdf, model=model, batch_size=batch_size, epochs=epochs, lr=lr)
        train_fn = jit(partial_train)

        to_distr_fn = lambda model, params: RNVPDistr(model=model, param=params)
        init_proposal = to_distr_fn(model, init_params)

    keys = split(key_train, n_train)
    kl_vals = jnp.array([estimate_kl(logpdf, init_proposal, target_samples)] + [0] * (n_train - 1))
    partial_em2c_step = partial(em2c_step,
                                key=keys,
                                pow_eps=pow_eps,
                                n_samples=n_samples,
                                logpdf=logpdf,
                                global_kernel=global_kernel,
                                local_kernel=local_kernel,
                                model=model,
                                train_fn=train_fn,
                                to_distr_func=to_distr_fn,
                                heavy_distr=heavy_distr,
                                mixed_proposal_weights=mixed_proposal_weights,
                                target_samples=target_samples)

    return fori_loop(1, n_train, partial_em2c_step, (init_params, kl_vals))


def em2c_step(i,
              proposal_state,
              pow_eps,
              key,
              n_samples,
              logpdf,
              model,
              train_fn,
              to_distr_func,
              global_kernel,
              local_kernel,
              heavy_distr,
              mixed_proposal_weights,
              target_samples):

    key_empirical, key_resample, key_lker, key_train = split(key[i], 4)
    proposal_params, kl_vals = proposal_state
    proposal_distr = to_distr_func(model, proposal_params)

    if heavy_distr is None:
        mixed_proposal = proposal_distr
    else:
        mixed_proposal = MixtureGeneral(mixing_distribution=Categorical(mixed_proposal_weights),
                                        component_distributions=[proposal_distr, heavy_distr])

    empirical_iterate = em2c_empirical_update(key_empirical, logpdf, pow_eps, n_samples, global_kernel, mixed_proposal)
    samples = em2c_resample_empirical_update(key_resample, empirical_iterate, n_samples)
    samples = local_kernel(keys=split(key_lker, n_samples), state=samples)
    proposal_params, _ = train_fn(key=key_train, params=proposal_params, samples=samples)
    kl_vals = kl_vals.at[i].set(estimate_kl(logpdf, to_distr_func(model, proposal_params), target_samples))
    debug.print('{kl}', kl=kl_vals[i])
    return (proposal_params, kl_vals)


def em2c_empirical_update(key, logpdf, pow_eps, n_samples, global_kernel, prev_proposal):
    proposal_samples = prev_proposal.sample(key, (n_samples,))
    log_weights = logpdf(proposal_samples) - prev_proposal.log_prob(proposal_samples)
    proposal_log_weights = pow_eps * log_weights
    global_kernel_samples = global_kernel(keys=split(key, n_samples),
                                          state=proposal_samples)
    global_kernel_log_weights = pow_eps * (
            logpdf(global_kernel_samples) - prev_proposal.log_prob(global_kernel_samples))
    global_kernel_lse, proposal_lse = logsumexp(global_kernel_log_weights), logsumexp(proposal_log_weights)
    log_normconst = logsumexp(log_weights)
    # alpha = cond(global_kernel_lse - log_normconst > 0,
    #              lambda _: (global_kernel_lse - log_normconst) / (global_kernel_lse - proposal_lse),
    #              lambda _: 0.5,
    #              operand=None)
    alpha = pow_eps
    return (proposal_samples,
            log_softmax(proposal_log_weights),
            global_kernel_samples,
            log_softmax(global_kernel_log_weights),
            alpha)




def em2c_resample_empirical_update(key, empirical_update, n_samples):
    # TODO: make it more efficient by resampling then propagating
    key_resample_gker, key_resample_proposal, key_cat = split(key, 3)
    proposal_samples, proposal_log_weights, global_kernel_samples, global_kernel_log_weights, alpha = empirical_update
    samples = jnp.vstack([proposal_samples, global_kernel_samples])
    log_weights = jnp.concatenate(
        [jnp.log(alpha) + proposal_log_weights, jnp.log(1 - alpha) + global_kernel_log_weights])
    resample_idxs = categorical(key_cat, log_weights, shape=(n_samples,))
    return samples[resample_idxs, :]

def renyi_training(key, params, samples, logpdf, model, batch_size, epochs, lr):
    def sk_renyi(model, params, batch):
        log_prob = model.apply({'params': params}, batch, method=model.log_prob)
        log_weights = stop_gradient(logpdf(batch) - log_prob)
        return - (softmax(log_weights) * log_prob).sum()

    return mle_training(key, samples, model, sk_renyi, batch_size, epochs, init_params=params, lr=lr)


def kl_training(key, params, samples, logpdf, model, batch_size, epochs, lr):
    def mean_likelihood(model, params, batch):
        log_prob = model.apply({'params': params}, batch, method=model.log_prob)
        return - log_prob.mean()

    return mle_training(key, samples, model, mean_likelihood, batch_size, epochs, init_params=params, lr=lr)

def estimate_kl(logpdf, proposal, target_samples):
    return (logpdf(target_samples) - proposal.log_prob(target_samples)).mean()


# def resample(key, samples, log_weights, n_samples):
#     idxs = categorical(key, log_weights, shape=(n_samples,))
#     return samples[idxs, :]

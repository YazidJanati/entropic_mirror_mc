import jax.numpy as jnp
from jax.random import split, categorical
from jax.lax import cond
from jax.nn import logsumexp


def emc_step(key, n_samples, logpdf, proposal, train_func, global_kernel, local_kernel):
    pass


def empirical_update(key, logpdf, pow, n_samples, global_kernel, prev_proposal):
    proposal_samples = prev_proposal.sample(key, (n_samples,))
    proposal_log_weights = pow * (logpdf(proposal_samples) \
                                  - prev_proposal.log_prob(proposal_samples))

    key_gker = split(key, n_samples)
    gker_samples = global_kernel(key_gker, proposal_samples)
    gker_log_weights = pow * (logpdf(gker_samples) - prev_proposal.log_prob(gker_samples))
    gker_lse, proposal_lse = logsumexp(gker_log_weights), logsumexp(proposal_log_weights)
    alpha = cond((gker_lse - jnp.log(n_samples) > 0),
                 lambda _: (gker_lse - jnp.log(n_samples)) / (gker_lse - proposal_lse),
                 lambda _: 0.5,
                 operand=None)
    return (proposal_samples, proposal_log_weights), (gker_samples, gker_log_weights), alpha


def resample_empirical_update(key, empirical_iterate, n_samples):
    key1, key2, key3 = split(key, 3)
    emd, ker_emd, alpha = empirical_iterate
    resampled_ker_emd = resample(key1, ker_emd[0], ker_emd[1], n_samples)
    resampled_emd = resample(key2, emd[0], emd[1], n_samples)
    idxs = categorical(key3, alpha, shape=(n_samples,))
    return jnp.where(idxs[:, jnp.newaxis],
                     resampled_emd,
                     resampled_ker_emd)


def resample(key, samples, log_weights, n_samples):
    idxs = categorical(key, log_weights, shape=(n_samples,))
    return samples[idxs, :]

from jax.tree_util import Partial as partial
from jax.random import split, uniform, bernoulli, normal, multivariate_normal, categorical
import jax.numpy as jnp
from jax.lax import fori_loop, cond, scan
from typing import NamedTuple, Callable
from numpyro.distributions import Distribution
from utils import display_samples


class RWM(NamedTuple):
    cov: jnp.ndarray


class IMH(NamedTuple):
    proposal: Distribution


class MALA(NamedTuple):
    step_size: float
    grad_logpdf: Callable


def mh_accept(key, prev_state, new_state, log_mh_ratio):
    p_accept = jnp.clip(jnp.exp(log_mh_ratio), a_max=1)
    accepted = bernoulli(key, p_accept)
    return cond(accepted,
                lambda _: new_state,
                lambda _: prev_state,
                operand=None)


def mh_step(i, state, logpdf, transition, keys):
    key_state, key_acc = split(keys[i])
    new_state, rev_logprob, prev_state, fwd_logprob = transition(key_state, state)
    log_mh_ratio = logpdf(new_state) + rev_logprob - logpdf(prev_state) - fwd_logprob
    return mh_accept(key_acc, prev_state, new_state, log_mh_ratio)


def mh(init_state, key, logpdf, transition, burn_in_steps, steps, params):
    keys_mh = split(key, burn_in_steps + steps)
    partial_transition = partial(transition, params=params)
    partial_mh_burn_in = partial(mh_step, logpdf=logpdf, transition=partial_transition, keys=keys_mh[:burn_in_steps])
    partial_mh_chain = partial(mh_step, logpdf=logpdf, transition=partial_transition, keys=keys_mh[burn_in_steps:])

    def mh_chain(i, samples):
        return samples.at[i].set(partial_mh_chain(i, samples[i - 1]))

    first_state = fori_loop(0, burn_in_steps, partial_mh_burn_in, init_state)
    samples = jnp.empty((steps, *first_state.shape), dtype=first_state.dtype)
    samples = samples.at[0].set(first_state)
    return fori_loop(1, steps, mh_chain, samples)


def rwm_kernel(key, state, params):
    new_state = state + multivariate_normal(key, mean=jnp.zeros_like(state), cov=params.cov)
    return new_state, 0., state, 0.


def imh_kernel(key, state, params):
    new_state = params.proposal.sample(key)
    return new_state, params.proposal.log_prob(state), state, params.proposal.log_prob(new_state)


def mala_kernel(key, state, params):
    key_noise, _ = split(key)
    eps = normal(key_noise, state.shape)
    new_state = state + params.step_size * params.grad_logpdf(state) + jnp.sqrt(2 * params.step_size) * eps
    fwd_logprob = - (eps ** 2) / 2
    rev_logprob = - (state - new_state - params.step_size * params.grad_logpdf(new_state)) ** 2 / (4 * params.step_size)
    return new_state, rev_logprob.sum(-1), state, fwd_logprob.sum(-1)


def ula_kernel(i, state, keys, grad_logpdf, step_size):
    eps = normal(keys[i], state.shape)
    new_state = state + step_size * grad_logpdf(state) + jnp.sqrt(2 * step_size) * eps
    return new_state


def independent_mh(key, init_state, logpdf, burn_in_steps, steps, params):
    """
    Implements the independent Metropolis-Hastings algorithm.
    """
    return mh(init_state,
              key,
              logpdf=logpdf,
              transition=imh_kernel,
              burn_in_steps=burn_in_steps,
              steps=steps,
              params=params
              )


def random_walk_mh(key, init_state, logpdf, burn_in_steps, steps, params):
    """
    Implements the Random Walk Metropolis-Hastings.
    """
    return mh(init_state,
              key,
              logpdf=logpdf,
              transition=rwm_kernel,
              burn_in_steps=burn_in_steps,
              steps=steps,
              params=params
              )


def adjusted_langevin(key, init_state, logpdf, params, burn_in_steps, steps):
    """
    Implements the Metropolis Adjusted Langevin Algorithm (MALA).
    """
    return mh(init_state,
              key,
              logpdf=logpdf,
              transition=mala_kernel,
              burn_in_steps=burn_in_steps,
              steps=steps,
              params=params)


def unadjusted_langevin(key, init_state, grad_logpdf, step_size, burn_in_steps, steps):
    """
    Implements the unadjusted Langevin Algorithm (ULA).
    """
    keys = split(key, burn_in_steps + steps)
    partial_ula_burn_in = partial(ula_kernel, keys=keys[:burn_in_steps], grad_logpdf=grad_logpdf, step_size=step_size)
    partial_ula_chain = partial(ula_kernel, keys=keys[burn_in_steps:], grad_logpdf=grad_logpdf, step_size=step_size)

    def ula_chain(i, samples):
        return samples.at[i].set(partial_ula_chain(i, samples[i - 1]))

    first_state = fori_loop(0, burn_in_steps, partial_ula_burn_in, init_state)
    samples = jnp.empty((steps, *first_state.shape), dtype=first_state.dtype)
    samples = samples.at[0].set(first_state)
    return fori_loop(1, steps, ula_chain, samples)


def isir(key, init_state, logpdf, proposal, n_proposals, burn_in_steps, steps):
    keys = split(key, burn_in_steps + steps)

    def isir_step(i, state, keys, logpdf, proposal, n_proposals):
        key_samples, key_idx = split(keys[i])
        proposal_samples = proposal.sample(key_samples, (n_proposals,))
        log_weights = logpdf(proposal_samples) - proposal.log_prob(proposal_samples)
        return proposal_samples[categorical(key_idx, log_weights), :]

    partial_isir_burn_in = partial(isir_step, keys=keys[:burn_in_steps], logpdf=logpdf, proposal=proposal,
                                   n_proposals=n_proposals)
    partial_isir_chain = partial(isir_step, keys=keys[burn_in_steps:], logpdf=logpdf, proposal=proposal,
                                 n_proposals=n_proposals)

    first_state = fori_loop(0, burn_in_steps, partial_isir_burn_in, init_state)
    samples = jnp.empty((steps, *first_state.shape), dtype=first_state.dtype)
    samples = samples.at[0].set(first_state)

    def isir_chain(i, samples):
        return samples.at[i].set(partial_isir_chain(i, samples[i - 1]))

    return fori_loop(1, steps, isir_chain, samples)

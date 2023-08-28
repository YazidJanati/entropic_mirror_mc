from jax.tree_util import Partial as partial
from jax.random import split, uniform, bernoulli, normal, multivariate_normal
import jax.numpy as jnp
from jax.lax import fori_loop, cond
from typing import NamedTuple, Callable
from numpyro.distributions import Distribution


class RWM(NamedTuple):
    cov: jnp.ndarray


class IMH(NamedTuple):
    proposal: Distribution


class MALA(NamedTuple):
    step_size: float
    grad_logpdf: Callable


def mh_accept(key, prev_state, new_state, mh_ratio):
    p_accept = jnp.clip(jnp.exp(mh_ratio), a_max=1)
    accepted = bernoulli(key, p_accept)
    return cond(accepted,
                lambda _: new_state,
                lambda _: prev_state,
                operand=None)


def mh_step(i, state, logpdf, transition, keys):
    key_state, key_acc = split(keys[i])
    new_state, fwd_logprob, prev_state, bwd_logprob = transition(key_state, state)
    mh_ratio = logpdf(new_state) + fwd_logprob - logpdf(prev_state) - bwd_logprob
    return mh_accept(key_acc, prev_state, new_state, mh_ratio)


def mh(init_state, key, logpdf, transition, burn_in_steps, steps, params):
    keys = split(key, steps)
    partial_transition = partial(transition, params=params)
    partial_mh = partial(mh_step, logpdf=logpdf, transition=partial_transition, keys=keys)

    def mh_chain(i, samples):
        return samples.at[i].set(partial_mh(i, samples[i-1]))

    first_state = fori_loop(0, burn_in_steps, partial_mh, init_state)
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
    key, subkey = split(key)
    eps = normal(subkey, state.shape)
    new_state = state + params.step_size * params.grad_logpdf(state) + jnp.sqrt(2 * params.step_size) * eps
    log_prob = - (eps ** 2) / 2
    rev_logprob = - (state - new_state - params.step_size * params.grad_logpdf(new_state)) ** 2 / (4 * params.step_size)
    return new_state, log_prob.sum(-1), state, rev_logprob.sum(-1)


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
    keys = split(key, steps)
    partial_ula = partial(ula_kernel, keys=keys, grad_logpdf=grad_logpdf, step_size=step_size)

    def ula_chain(i, samples):
        return samples.at[i].set(partial_ula(i, samples[i - 1]))

    first_state = fori_loop(0, burn_in_steps, partial_ula, init_state)
    samples = jnp.empty((steps, *first_state.shape), dtype=first_state.dtype)
    samples = samples.at[0].set(first_state)
    return fori_loop(1, steps, ula_chain, samples)

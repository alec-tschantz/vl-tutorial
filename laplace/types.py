from typing import Callable, Optional, Self
from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.linalg import inv


@dataclass
class Model:
    prior_exp: jnp.ndarray
    prior_cov: jnp.ndarray
    prior_exp_hyp: jnp.ndarray
    prior_cov_hyp: jnp.ndarray
    design_matrix: jnp.ndarray
    likelihood_fn: Callable[[jnp.ndarray, Self, Optional[jnp.ndarray]], jnp.ndarray]
    inv_prior_cov = jnp.ndarray
    inv_prior_cov_hyp = jnp.ndarray
    num_params: int
    num_hyp: int

    def __init__(
        self,
        prior_exp,
        prior_cov,
        prior_exp_hyp,
        prior_cov_hyp,
        design_matrix,
        likelihood_fn,
        max_iterations=20,
    ):
        self.prior_exp = jnp.array(prior_exp).astype(jnp.float32)
        self.prior_cov = jnp.array(prior_cov).astype(jnp.float32)
        self.prior_exp_hyp = jnp.array(prior_exp_hyp).astype(jnp.float32)
        self.prior_cov_hyp = jnp.array(prior_cov_hyp).astype(jnp.float32)
        self.design_matrix = jnp.array(design_matrix).astype(jnp.float32)
        self.likelihood_fn = likelihood_fn
        self.max_iterations = int(max_iterations)

        self.inv_prior_cov = inv(self.prior_cov)
        self.inv_prior_cov_hyp = inv(self.prior_cov_hyp)

        self.num_params = self.prior_exp.shape[0]
        self.num_hyp = self.prior_exp_hyp.shape[0]


@dataclass
class Data:
    y: jnp.ndarray
    Q: jnp.ndarray

    def __init__(self, y, Q):
        self.y = jnp.array(y).astype(jnp.float32)
        self.Q = jnp.array(Q).astype(jnp.float32)

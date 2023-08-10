import jax.numpy as jnp

import numpy as np
from numpy.linalg import inv
from numpy.random import multivariate_normal as mvn

import laplace

num_obs = 100

params = np.array([[0.5], [0.1]])
hyperparams = np.array([[3]])
precision_components = np.array([np.eye(num_obs)])
precision = np.exp(hyperparams[0]) * precision_components[0]


def likelihood_fn(params: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(num_obs), np.linspace(1, 100, num_obs)))
    design[:, 1] = design[:, 1] - np.mean(design[:, 1])
    return np.dot(design, params)


noise = mvn(np.zeros(num_obs), inv(precision), 1)
data = likelihood_fn(params) + noise.T

beta_prior_mu = np.array([[0.0], [0.0]])
beta_prior_cov = np.eye(2) * 1
lambda_prior_mu = np.array([[4.0]])
lambda_prior_cov = np.array([[1.0]])

beta_post_mu, lambda_post_mu, beta_post_cov, vfe = laplace.invert(
    jnp.array(beta_prior_mu),
    jnp.array(beta_prior_cov),
    jnp.array(lambda_prior_mu),
    jnp.array(lambda_prior_cov),
    jnp.array(data),
    jnp.array(precision_components),
    likelihood_fn,
)

print(beta_post_mu)
print(lambda_post_mu)

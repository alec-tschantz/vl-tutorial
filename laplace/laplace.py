from typing import Callable, Tuple

import jax.numpy as jnp
from jax.numpy.linalg import slogdet
from jax.scipy.linalg import expm


def invert(
    beta_prior_mu: jnp.ndarray,
    beta_prior_cov: jnp.ndarray,
    lambda_prior_mu: jnp.ndarray,
    lambda_prior_cov: jnp.ndarray,
    data: jnp.ndarray,
    precision_components: jnp.ndarray,
    likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray],
    max_iter: int = 20,
    logv: float = -4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    beta_post_mu, lambda_post_mu = beta_prior_mu, lambda_prior_mu
    beta_prior_precision, lambda_prior_precision = jnp.linalg.inv(beta_prior_cov), jnp.linalg.inv(lambda_prior_cov)

    current = {
        "beta_post_mu": beta_post_mu,
        "beta_post_cov": beta_prior_cov,
        "lambda_post_mu": lambda_post_mu,
        "vfe": -jnp.inf,
    }

    for it in range(max_iter):
        jacobian = compute_jacobian(beta_post_mu, likelihood_fn)

        for _ in range(8):
            data_precision, precision_per_component = compute_data_precision(lambda_post_mu, precision_components)
            beta_post_cov = compute_posterior_covariance(jacobian, data_precision, beta_prior_precision)
            lambda_post_mu, lambda_post_cov, has_converged = update_lambda(
                lambda_post_mu,
                lambda_prior_mu,
                lambda_prior_precision,
                beta_post_mu,
                beta_post_cov,
                data,
                data_precision,
                precision_per_component,
                jacobian,
                likelihood_fn,
            )

            if has_converged:
                break

        vfe = compute_vfe(
            lambda_post_mu,
            lambda_prior_mu,
            lambda_post_cov,
            lambda_prior_cov,
            beta_post_mu,
            beta_prior_mu,
            beta_post_cov,
            beta_prior_cov,
            data,
            data_precision,
            likelihood_fn,
        )

        if vfe > current["vfe"] or it < 3:
            current = {
                "beta_post_mu": beta_post_mu,
                "beta_post_cov": beta_post_cov,
                "lambda_post_mu": lambda_post_mu,
                "vfe": vfe,
            }

            grad = compute_vfe_gradient(
                beta_post_mu, beta_prior_mu, beta_prior_precision, jacobian, data, data_precision, likelihood_fn
            )
            hessian = compute_hessian(jacobian, data_precision, beta_prior_precision)
            logv = min(logv + 1 / 2, 4)

        else:
            beta_post_mu, lambda_post_mu, beta_post_cov = (
                current["beta_post_mu"],
                current["lambda_post_mu"],
                current["beta_post_cov"],
            )
            logv = min(logv - 2, -4)

        beta_delta = update(hessian, grad, logv)
        beta_post_mu = beta_post_mu + beta_delta
    return beta_post_mu, lambda_post_mu, beta_post_cov, current["vfe"]


def compute_jacobian(beta_post_mu: jnp.ndarray, likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    bmu_0, num_params, jacobian = beta_post_mu, beta_post_mu.shape[0], []

    for p in range(num_params):
        beta_post_mu = bmu_0
        beta_post_mu = beta_post_mu.at[p].set(beta_post_mu[p] + jnp.exp(-8))
        jacobian_p = (likelihood_fn(beta_post_mu) - likelihood_fn(bmu_0)) / jnp.exp(-8)
        jacobian.append(jacobian_p)

    return jnp.array(jacobian).squeeze().T


def compute_hessian(jacobian: jnp.ndarray, data_precision: jnp.ndarray, beta_precision: jnp.ndarray) -> jnp.ndarray:
    return -jnp.matmul(jacobian.T, data_precision).dot(jacobian) - beta_precision


def update(hessian: jnp.ndarray, grad: jnp.ndarray, logv: float) -> jnp.ndarray:
    n = len(grad)
    _, hessian_logdet = slogdet(hessian)

    scale = jnp.exp(logv - hessian_logdet / n)
    delta = (expm(hessian * scale) - jnp.eye(n)) @ jnp.linalg.inv(hessian) @ grad
    return delta


def compute_data_precision(lambda_post_mu: jnp.ndarray, precision_components: jnp.ndarray) -> jnp.ndarray:
    num_data = precision_components[0].shape[0]

    precision, precision_per_component = jnp.zeros((num_data, num_data)), []
    for i in range(len(precision_components)):
        precision_per_component.append(precision_components[i] * (jnp.exp(-32) + jnp.exp(lambda_post_mu[i])))
        precision = precision + precision_per_component[i]

    return precision, precision_per_component


def compute_posterior_covariance(
    jacobian: jnp.ndarray, data_precision: jnp.ndarray, beta_precision: jnp.ndarray
) -> jnp.ndarray:
    hessian = -jnp.matmul(jacobian.T, data_precision).dot(jacobian) - beta_precision
    return jnp.linalg.inv(-hessian)


def compute_vfe(
    lambda_post_mu: jnp.ndarray,
    lambda_prior_mu: jnp.ndarray,
    lambda_post_cov: jnp.ndarray,
    lambda_prior_cov: jnp.ndarray,
    beta_post_mu: jnp.ndarray,
    beta_prior_mu: jnp.ndarray,
    beta_post_cov: jnp.ndarray,
    beta_prior_cov: jnp.ndarray,
    data: jnp.ndarray,
    data_precision: jnp.ndarray,
    likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray],
):
    data_err = data - likelihood_fn(beta_post_mu)
    beta_prior_err = beta_post_mu - beta_prior_mu
    lambda_prior_err = lambda_post_mu - lambda_prior_mu

    data_cov = jnp.linalg.inv(data_precision)

    _, data_logdet = slogdet(data_cov)
    _, lambda_post_logdet = slogdet(lambda_post_cov)
    _, lambda_prior_logdet = slogdet(lambda_prior_cov)
    _, beta_post_loget = slogdet(beta_post_cov)
    _, beta_prior_logdet = slogdet(beta_prior_cov)

    data_term = data_logdet + (data_err.T @ data_precision @ data_err)
    beta_prior_term = beta_prior_logdet + (beta_prior_err.T @ beta_prior_cov @ beta_prior_err)
    lambda_prior_term = lambda_prior_logdet + (lambda_prior_err.T @ lambda_prior_cov @ lambda_prior_err)
    entropy_term = beta_post_loget + lambda_post_logdet
    constant_term = data.shape[1] * jnp.log(2 * jnp.pi)

    return -0.5 * data_term - 0.5 * beta_prior_term - 0.5 * lambda_prior_term + 0.5 * entropy_term - 0.5 * constant_term


def compute_vfe_gradient(
    beta_post_mu: jnp.ndarray,
    beta_prior_mu: jnp.ndarray,
    beta_prior_precision: jnp.ndarray,
    jacobian: jnp.ndarray,
    data: jnp.ndarray,
    data_precision: jnp.ndarray,
    likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    data_err = data - likelihood_fn(beta_post_mu)
    prior_err = beta_post_mu - beta_prior_mu

    data_term = jnp.matmul(jnp.matmul(jacobian.T, data_precision), data_err)
    prior_term = jnp.dot(beta_prior_precision, prior_err)
    return data_term - prior_term


def update_lambda(
    lambda_post_mu: jnp.ndarray,
    lambda_prior_mu: jnp.ndarray,
    lambda_prior_precision: jnp.ndarray,
    beta_post_mu: jnp.ndarray,
    beta_post_cov: jnp.ndarray,
    data: jnp.ndarray,
    data_precision: jnp.ndarray,
    precision_per_component: jnp.ndarray,
    jacobian: jnp.ndarray,
    likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, bool]:
    data_err = data - likelihood_fn(beta_post_mu)
    lambda_err = lambda_post_mu - lambda_prior_mu
    data_cov = jnp.linalg.inv(data_precision)

    num_lambda = lambda_post_mu.shape[0]
    hessian = jnp.zeros((num_lambda, num_lambda))

    for l in range(num_lambda):
        hessian_entry = (
            -lambda_prior_precision[1, 1]
            + 0.5
            * jnp.trace(
                precision_per_component[l] * data_cov
                - precision_per_component[l] * data_cov * precision_per_component[l] * data_cov
            )
            - 0.5 * jnp.dot(jnp.dot(data_err.T, precision_per_component[l]), data_err)
            - 0.5
            * jnp.trace(jnp.dot(jnp.dot(jnp.dot(beta_post_cov, jacobian.T), precision_per_component[l]), jacobian))
        )
        hessian = hessian.at[l, l].set(hessian_entry[0, 0])

    lambda_post_precision = -hessian
    lambda_post_cov = jnp.linalg.inv(lambda_post_precision)

    lambda_grad = jnp.zeros((num_lambda, 1))
    for l in range(num_lambda):
        deh = jnp.zeros((num_lambda, 1))
        deh = deh.at[l].set(1)

        grad_entry = (
            0.5 * jnp.trace(precision_per_component[l] * data_cov)
            - 0.5 * (jnp.dot(jnp.dot(data_err.T, precision_per_component[l]), data_err))
            - 0.5
            * jnp.trace(jnp.dot(jnp.dot(jnp.dot(beta_post_cov, jacobian.T), precision_per_component[l]), jacobian))
            - jnp.dot(jnp.dot(deh.T, lambda_prior_precision), lambda_err)
        )

        lambda_grad = lambda_grad.at[l].set(grad_entry[0, 0])

    lambda_delta = update(hessian, lambda_grad, 4)
    lambda_delta = jnp.clip(lambda_delta, -1, 1)
    lambda_post_mu = lambda_post_mu + lambda_delta
    has_converged = jnp.dot(lambda_grad.T, lambda_delta) < 1e-2
    return lambda_post_mu, lambda_post_cov, has_converged

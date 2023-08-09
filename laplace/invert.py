import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jacrev, jacobian
from jax.numpy.linalg import slogdet
from jax.scipy.linalg import expm


def fit(model, data):
    y = data.y
    Q = data.Q
    num_data = y.shape[1]
    params = model.prior_exp
    hyp = model.prior_exp_hyp
    logv = -4

    curr_F = -jnp.inf
    curr_params = params
    curr_hyp = hyp
    curr_post_cov = None

    for it in range(model.max_iterations):
        J = compute_likelihood_grad(params, model.likelihood_fn)
        for it2 in range(8):
            iS, P = compute_precision(hyp, Q)
            _, post_cov = compute_hessian(J, iS, model.inv_prior_cov)

            hyp, hyper_cov, has_converged = update_hyperparameters(
                model, hyp, params, J, post_cov, P, iS, y, model.likelihood_fn
            )
            if has_converged:
                break

        F = free_energy(
            params,
            hyp,
            y,
            iS,
            model.likelihood_fn,
            post_cov,
            hyper_cov,
            num_data,
            model.prior_exp,
            model.prior_exp_hyp,
            model.prior_cov,
            model.inv_prior_cov,
            model.prior_cov_hyp,
            model.inv_prior_cov_hyp,
        )

        if F > curr_F or it < 3:
            curr_F = F
            curr_params = params
            curr_hyp = hyp
            curr_post_cov = post_cov

            dFdp = compute_vfe_gradient(params, y, J, iS, model.likelihood_fn, model.prior_exp, model.inv_prior_cov)
            dFdpp, _ = compute_hessian(J, iS, model.inv_prior_cov)

            logv = min(logv + 1 / 2, 4)

            print("VL: (+)")
        else:
            params = curr_params
            hyp = curr_hyp
            post_cov = curr_post_cov

            logv = min(logv - 2, -4)

        print(f"It {it}: log(v)={logv}, F={curr_F}")
        dp = update(dFdpp, dFdp, logv)
        params = params + dp

    return curr_params, curr_hyp, post_cov, curr_F


def update(hessian, grad_vfe, logv):
    n = len(grad_vfe)
    _, hessian_logdet = slogdet(hessian)

    scale = jnp.exp(logv - hessian_logdet / n)
    delta = (expm(hessian * scale) - jnp.eye(n)) @ jnp.linalg.inv(hessian) @ grad_vfe
    return delta


def compute_hessian(J, precision, inv_prior_cov):
    hessian = -jnp.matmul(J.T, precision).dot(J) - inv_prior_cov
    return hessian, invs(-hessian)


def compute_likelihood_grad(params, likelihood_fn):
    num_params, dx = params.shape[0], jnp.exp(-8)
    params_0, grad = params, []

    for p in range(num_params):
        params = params_0
        params = params.at[p].set(params[p] + dx)
        grad_p = (likelihood_fn(params) - likelihood_fn(params_0)) / dx
        grad.append(grad_p)

    grad = jnp.array(grad)
    return grad.T


def compute_vfe_gradient(params, data, J, precision, likelihood_fn, prior_exp, inv_prior_cov):
    data_err = (data - likelihood_fn(params)).T
    prior_err = params - prior_exp

    data_term = jnp.matmul(jnp.matmul(J.T, precision), data_err)
    prior_term = jnp.dot(inv_prior_cov, prior_err).reshape(-1, 1)
    grad = (data_term - prior_term).squeeze()
    return grad


def compute_precision(hyper_params, Q):
    num_data = Q[0].shape[0]

    precision, q_precisions = jnp.zeros((num_data, num_data)), []
    for i in range(len(Q)):
        q_precision = Q[i] * (jnp.exp(-32) + jnp.exp(hyper_params[i]))
        q_precisions.append(q_precision)
        precision = precision + q_precision

    return precision, q_precisions


def invs(mat):
    return jnp.linalg.inv(mat + jnp.eye(mat.shape[0]) * jnp.exp(-32))


def free_energy(
    params,
    hyper_params,
    data,
    precision,
    likelihood_fn,
    post_cov,
    hyper_cov,
    num_data,
    prior_exp,
    prior_exp_hyp,
    prior_cov,
    inv_prior_cov,
    prior_cov_hyp,
    inv_prior_cov_hyp,
):
    data_err = (data - likelihood_fn(params)).T
    prior_err = params - prior_exp
    hyper_err = hyper_params - prior_exp_hyp

    cov = invs(precision)

    _, cov_logdet = slogdet(cov)
    likelihood = cov_logdet + (data_err.T @ precision @ data_err)

    _, prior_cov_logdet = slogdet(prior_cov)
    prior = prior_cov_logdet + (prior_err.T @ inv_prior_cov @ prior_err)

    _, prior_cov_hyp_logdet = slogdet(prior_cov_hyp)
    hyper = prior_cov_hyp_logdet + (hyper_err.T @ inv_prior_cov_hyp @ hyper_err)

    _, post_cov_logdet = slogdet(post_cov)
    _, hyper_cov_logdet = slogdet(hyper_cov)
    entropy = post_cov_logdet + hyper_cov_logdet
    constants = num_data * jnp.log(2 * jnp.pi)

    vfe = -0.5 * likelihood - 0.5 * prior - 0.5 * hyper + 0.5 * entropy - 0.5 * constants
    return vfe

def update_hyperparameters(model, h, p, J, Cp, P, iS, y, likelihood_fn):
    ey = y - likelihood_fn(p)
    eh = h - model.prior_exp_hyp
    ey = ey.T

    ihC = model.inv_prior_cov_hyp
    S = invs(iS)

    num_hyp = model.num_hyp
    dFdhh = jnp.zeros((num_hyp, num_hyp))
    for i in range(num_hyp):
        at_i_1 = -ihC[1, 1]
        at_i_2 = 0.5 * jnp.trace(P[i] * S - P[i] * S * P[i] * S)
        x = jnp.dot(ey.T, P[i])
        x = jnp.dot(x, ey)
        at_i_3 = -0.5 * x
        at_i_4 = -0.5 * jnp.trace(jnp.dot(jnp.dot(jnp.dot(Cp, J.T), P[i]), J))
        at_i = at_i_1 + at_i_2 + at_i_3 + at_i_4
        # TODO
        dFdhh = dFdhh.at[i, i].set(at_i[0, 0])

    Ph = -dFdhh
    Ch = invs(Ph)

    dFdh = jnp.zeros((num_hyp, 1))
    for i in range(num_hyp):
        deh = jnp.zeros((num_hyp, 1))
        deh = deh.at[i].set(1)

        at_i = (
            0.5 * jnp.trace(P[i] * S)
            - 0.5 * (jnp.dot(jnp.dot(ey.T, P[i]), ey))
            - 0.5 * jnp.trace(jnp.dot(jnp.dot(jnp.dot(Cp, J.T), P[i]), J))
            - jnp.dot(jnp.dot(deh.T, ihC), eh)
        )

        dFdh = dFdh.at[i].set(at_i[0, 0])

    dh = update(dFdhh, dFdh, 4)
    dh = jnp.clip(dh, -1, 1)
    h = h + dh
    has_converged = jnp.dot(dFdh.T, dh) < 1e-2
    return h, Ch, has_converged
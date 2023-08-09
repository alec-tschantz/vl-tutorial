import numpy as np
from numpy.linalg import inv
from numpy.random import multivariate_normal as mvn

from laplace import Model, Data, fit

num_data = 100

def likelihood_fn(params):
    design_matrix = np.column_stack((np.ones(num_data), np.linspace(1, 100, num_data)))
    design_matrix[:, 1] = design_matrix[:, 1] - np.mean(design_matrix[:, 1])
    return np.matmul(design_matrix, params)


prior_exp = np.array([0, 0])
prior_cov = np.eye(2) * 1
prior_exp_hyp = np.array([4])
prior_cov_hyp = np.array([[1]])

num_data = 100
Q = np.array([np.eye(num_data)])

true_params = np.array([0.5, 0.1])
true_hyp = np.array([3])

design_matrix = np.column_stack((np.ones(num_data), np.linspace(1, 100, num_data)))
design_matrix[:, 1] = design_matrix[:, 1] - np.mean(design_matrix[:, 1])

model = Model(prior_exp, prior_cov, prior_exp_hyp, prior_cov_hyp, design_matrix, likelihood_fn)

P = np.zeros((num_data, num_data))
for i in range(len(Q)):
    P = P + np.exp(true_hyp[i]) * Q[i]

e = mvn(np.zeros(num_data), inv(P), 1)
y = likelihood_fn(true_params) + e

data = Data(y, Q)

params, hyper_params, cov, vfe  = fit(model, data)
print(params)
print(hyper_params)
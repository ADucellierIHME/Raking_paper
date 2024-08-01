import numpy as np
import pandas as pd
import pickle

from statistics import covariance

from raking_methods import get_margin_matrix_vector, raking_chi2_distance
from uncertainty_IFT import compute_variance

# Define variables and margins names
var1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
var2 = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
margins = ['var1_1', 'var1_2', 'var1_3', 'var2_1', 'var2_2', 'var2_3', 'var2_4', 'var2_5']

# Generate balanced table
rng = np.random.default_rng(0)

mu_ij = rng.uniform(low=2.0, high=3.0, size=(3, 5))
mu_i0 = np.sum(mu_ij, axis=1)
mu_0j = np.sum(mu_ij, axis=0)

# Then we respect the constraint:
# np.matmul(A, mu_ij.flatten()) = np.concatenate((mu_0j, mu_i0))

# Add noise to the data and generate samples from the MVN
mean = mu_ij.flatten(order='F') + rng.normal(0.0, 0.1, size=15)
cov = 0.01 * np.ones((15, 15))
np.fill_diagonal(cov, np.arange(0.01, 0.16, 0.01))

x = rng.multivariate_normal(mean, cov, 1000)

# Define constraints for the raking
(A, y) = get_margin_matrix_vector(np.repeat(1, 5), np.repeat(1, 3), mu_i0, mu_0j)

# Rake the mean
x_0 = np.mean(x, 0)
(mu_0, lambda_k) = raking_chi2_distance(x_0, np.ones(15), A, y)

df_raked = pd.DataFrame({'X1': var1, \
                         'X2': var2, \
                         'observations': x_0, \
                         'raked_values': mu_0})

# Compute the gradient
(dh_x, dh_y) = compute_variance(mu_0, lambda_k, x_0, np.ones(15), A, 'general', 1)

df_x = []
df_y = []
for i in range(0, 15):
    df_x.append(pd.DataFrame({'raked_1': np.repeat(var1[i], 15), \
                              'raked_2': np.repeat(var2[i], 15), \
                              'X1': var1, \
                              'X2': var2, \
                              'grad_x': dh_x[i, :]}))
    df_y.append(pd.DataFrame({'raked_1': np.repeat(var1[i], 8), \
                              'raked_2': np.repeat(var2[i], 8), \
                              'margins': margins, \
                              'grad_y': dh_y[i, :]}))
df_x = pd.concat(df_x)
df_y = pd.concat(df_y)

# Compute the covariance
dh_x = dh_x.flatten()
dh_y = dh_y.flatten()
covariance_mean = np.zeros((15, 15))
for i in range(0, 15):
    for j in range(0, 15):
        dh_x_i = dh_x[(15 * i):(15 * i + 15)]
        dh_y_i = dh_y[(8 * i):(8 * i + 8)]
        dh_i = np.concatenate((dh_x_i, dh_y_i), axis=0)
        dh_x_j = dh_x[(15 * j):(15 * j + 15)]
        dh_y_j = dh_y[(8 * j):(8 * j + 8)]
        dh_j = np.concatenate((dh_x_j, dh_y_j), axis=0)
        sigma = np.concatenate(( \
            np.concatenate((cov, np.zeros((15, 8))), axis=1), \
            np.concatenate((np.zeros((8, 15)), np.zeros((8, 8))), axis=1)), axis=0)
        covariance_mean[i, j] = np.matmul(np.transpose(dh_i), np.matmul(sigma, dh_i))

# Rake each draw and compute the mean and the covariance
mu = np.zeros((1000, 15))
for n in range(0, 1000):
    x_n = x[n, :]
    (mu_n, lambda_k) = raking_chi2_distance(x_n, np.ones(15), A, y)
    mu[n, :] = mu_n
mean_draws = np.mean(mu, 0)
covariance_draws = np.zeros((15, 15))
for i in range(0, 15):
    for j in range(0, 15):
        covariance_draws[i, j] = covariance(mu[:, i], mu[:, j])

with open('synthetics.pkl', 'wb') as output_file:
    pickle.dump([mu_ij, df_raked, df_x, df_y, covariance_mean, mean_draws, covariance_draws], output_file)


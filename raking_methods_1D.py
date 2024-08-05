"""
This module implements the different raking methods in 1D.
"""

import numpy as np

def raking_chi2_distance(x_i, q_i, v_i, mu, direct=True):
    """
    Raking using the chi2 distance (mu - x)^2 / 2x.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if (mu == 0.0) & (np.sum(np.abs(x_i)) == 0.0):
        return (np.zeros(len(x_i)), 0.0)

    if direct:
        lambda_k = (np.sum(v_i * x_i) - mu) / np.sum(q_i * np.square(v_i) * x_i)
        mu_i = x_i * (1.0 - q_i * v_i * lambda_k)
    else:
        Phi = np.concatenate((np.concatenate((np.diag(1.0 / (q_i * x_i)), \
            v_i.reshape((-1, 1))), axis=1), \
            np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
        y = np.array((1.0 / q_i).tolist() + [mu])
        result = np.linalg.solve(Phi, y)
        mu_i = result[0:-1]
        lambda_k = result[-1]
    return (mu_i, lambda_k)

def raking_l2_distance(x_i, q_i, v_i, mu, direct=True):
    """
    Raking using the l2 distance (mu - x)^2 / 2.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if (mu == 0.0) & (np.sum(np.abs(x_i)) == 0.0):
        return (np.zeros(len(x_i)), 0.0)

    if direct:
        lambda_k = (np.sum(v_i * x_i) - mu) / np.sum(q_i * np.square(v_i))
        mu_i = x_i - q_i * v_i * lambda_k
    else:
        Phi = np.concatenate((np.concatenate((np.diag(1.0 / q_i), \
            v_i.reshape((-1, 1))), axis=1), \
            np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
        y = np.array((x_i / q_i).tolist() + [mu])
        result = np.linalg.solve(Phi, y)
        mu_i = result[0:-1]
        lambda_k = result[-1]
    return (mu_i, lambda_k)

def raking_entropic_distance(x_i, q_i, v_i, mu, gamma0=1.0, max_iter=500, return_num_iter=False, direct=True):
    """
    Raking using the entropic distance mu log(mu/x) + x - mu.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      gamma0: scalar, initial value for line search
      max_iter: integer, maximum number of iterations for Newton's method
      return_num_iter: boolean, whether we return the number of iterations
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if (mu == 0.0) & (np.sum(np.abs(x_i)) == 0.0):
        if return_num_iter:
            return (np.zeros(len(x_i)), 0.0, 0)
        else:
            return (np.zeros(len(x_i)), 0.0)

    epsilon = 1.0
    if direct:
        lambda_k = 0.0
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            f1 = mu - np.sum(v_i * x_i * np.exp(- q_i * v_i * lambda_k))
            f2 = np.sum(q_i * np.square(v_i) * x_i * np.exp(- q_i * v_i * lambda_k))
            lambda_k = lambda_k - f1 / f2
            mu_i = x_i * np.exp(- q_i * v_i * lambda_k)
            epsilon = abs(mu - np.sum(v_i * mu_i))
            iter_eps = iter_eps + 1
    else:
        mu_k = np.copy(x_i)
        lambda_k = 0.0
        result = np.array(mu_k.tolist() + [lambda_k])
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            Phi = np.concatenate((np.concatenate((np.diag(1.0 / (q_i * mu_k)), \
                v_i.reshape((-1, 1))), axis=1), \
                np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
            y = np.array((np.log(mu_k / x_i) / q_i + v_i * lambda_k).tolist() + [np.sum(v_i * mu_k) - mu])
            gamma = gamma0
            iter_gam = 0
            result_tmp = result - gamma * np.linalg.solve(Phi, y)
            mu_k_tmp = result_tmp[0:-1]
            lambda_k_tmp = result_tmp[-1]
            while (np.any(mu_k_tmp < 0.0)) & (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                result_tmp = result - gamma * np.linalg.solve(Phi, y)
                mu_k_tmp = result_tmp[0:-1]
                lambda_k_tmp = result_tmp[-1]
            result = result - gamma * np.linalg.solve(Phi, y)
            mu_k = result[0:-1]
            lambda_k = result[-1]
            epsilon = abs(mu - np.sum(v_i * mu_k))
            iter_eps = iter_eps + 1
        mu_i = mu_k
    if return_num_iter:
        return (mu_i, lambda_k, iter_eps)
    else:
        return (mu_i, lambda_k)

def raking_inverse_entropic_distance(x_i, q_i, v_i, mu, gamma0=1.0, max_iter=500, return_num_iter=False, direct=True):
    """
    Raking using the inverse entropic distance x log(x/mu) + mu - x.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      gamma0: scalar, initial value for line search
      max_iter: integer, maximum number of iterations for Newton's method
      return_num_iter: boolean, whether we return the number of iterations
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if (mu == 0.0) & (np.sum(np.abs(x_i)) == 0.0):
        if return_num_iter:
            return (np.zeros(len(x_i)), 0.0, 0)
        else:
            return (np.zeros(len(x_i)), 0.0)

    epsilon = 1.0
    if direct:
        lambda_k = 0.0
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            f1 = mu - np.sum(v_i * x_i /(1.0 + q_i * v_i * lambda_k))
            f2 = np.sum(q_i * np.square(v_i) * x_i / np.square(1.0 + q_i * v_i * lambda_k))
            lambda_k = lambda_k - f1 / f2
            mu_i = x_i / (1.0 + q_i * v_i * lambda_k)
            epsilon = abs(mu - np.sum(v_i * mu_i))
            iter_eps = iter_eps + 1
    else:
        mu_k = np.copy(x_i)
        lambda_k = 0.0
        result = np.array(mu_k.tolist() + [lambda_k])
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            Phi = np.concatenate((np.concatenate((np.diag(x_i / (q_i * np.square(mu_k))), \
                v_i.reshape((-1, 1))), axis=1), \
                np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
            y = np.array(((1.0 - x_i / mu_k) / q_i + v_i * lambda_k).tolist() + [np.sum(v_i * mu_k) - mu])
            gamma = gamma0
            iter_gam = 0
            result_tmp = result - gamma * np.linalg.solve(Phi, y)
            mu_k_tmp = result_tmp[0:-1]
            lambda_k_tmp = result_tmp[-1]
            while (np.any(mu_k_tmp < 0.0)) & (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                result_tmp = result - gamma * np.linalg.solve(Phi, y)
                mu_k_tmp = result_tmp[0:-1]
                lambda_k_tmp = result_tmp[-1]
            result = result - gamma * np.linalg.solve(Phi, y)
            mu_k = result[0:-1]
            lambda_k = result[-1]
            epsilon = abs(mu - np.sum(v_i * mu_k))
            iter_eps = iter_eps + 1
        mu_i = mu_k
    if return_num_iter:
        return (mu_i, lambda_k, iter_eps)
    else:
        return (mu_i, lambda_k)

def raking_inverse_chi2_distance(x_i, q_i, v_i, mu, gamma0=1.0, max_iter=500, return_num_iter=False, direct=True):
    """
    Raking using the inverse chi2 distance (x - mu)^2 / 2mu.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      gamma0: scalar, initial value for line search
      max_iter: integer, maximum number of iterations for Newton's method
      return_num_iter: boolean, whether we return the number of iterations
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
     Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if (mu == 0.0) & (np.sum(np.abs(x_i)) == 0.0):
        if return_num_iter:
            return (np.zeros(len(x_i)), 0.0, 0)
        else:
            return (np.zeros(len(x_i)), 0.0)

    epsilon = 1.0
    if direct:
        lambda_k = 0.0
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            f1 = mu - np.sum(v_i * x_i / np.sqrt(1.0 + 2.0 * q_i * v_i * lambda_k))
            f2 = np.sum(q_i * np.square(v_i) * x_i / np.power(1 + 2.0 * q_i * v_i * lambda_k, 3.0 / 2.0))
            gamma = gamma0
            iter_gam = 0
            while (np.any(1 + 2.0 * q_i * v_i * (lambda_k - gamma * f1 / f2) <= 0.0)) & \
                  (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
            lambda_k = lambda_k - gamma * f1 / f2
            mu_i = x_i / np.sqrt(1.0 + 2.0 * q_i * v_i * lambda_k)
            epsilon = abs(mu - np.sum(v_i * mu_i))
            iter_eps = iter_eps + 1
    else:
        mu_k = np.copy(x_i)
        lambda_k = 0.0
        result = np.array(mu_k.tolist() + [lambda_k])
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            Phi = np.concatenate((np.concatenate((np.diag(np.square(x_i) / (q_i * np.power(mu_k, 3.0))), \
                v_i.reshape((-1, 1))), axis=1), \
                np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
            y = np.array(((1.0 - np.square(x_i) / np.square(mu_k)) / (2.0 * q_i) + v_i * lambda_k).tolist() + \
                [np.sum(v_i * mu_k) - mu])
            gamma = gamma0
            iter_gam = 0
            result_tmp = result - gamma * np.linalg.solve(Phi, y)
            mu_k_tmp = result_tmp[0:-1]
            lambda_k_tmp = result_tmp[-1]
            while ((np.any(mu_k_tmp < 0.0)) |
                   (np.any(1.0 + 2.0 * q_i * v_i * lambda_k_tmp <= 0.0))) & \
                  (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                result_tmp = result - gamma * np.linalg.solve(Phi, y)
                mu_k_tmp = result_tmp[0:-1]
                lambda_k_tmp = result_tmp[-1]
            result = result - gamma * np.linalg.solve(Phi, y)
            mu_k = result[0:-1]
            lambda_k = result[-1]
            epsilon = abs(mu - np.sum(v_i * mu_k))
            iter_eps = iter_eps + 1
        mu_i = mu_k
    if return_num_iter:
        return (mu_i, lambda_k, iter_eps)
    else:
        return (mu_i, lambda_k)

def raking_general_distance(x_i, q_i, v_i, alpha, mu, gamma0=1.0, max_iter=500, return_num_iter=False, direct=True):
    """
    Raking using general distance.
    Input:
      x_i: 1D Numpy array, observed values
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      alpha: scalar, paremeter of the general distance function
      mu: scalar, total of the linear constraint
      gamma0: scalar, initial value for line search
      max_iter: integer, maximum number of iterations for Newton's method
      return_num_iter: boolean, whether we return the number of iterations
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
     Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if (mu == 0.0) & (np.sum(np.abs(x_i)) == 0.0):
        if return_num_iter:
            return (np.zeros(len(x_i)), 0.0, 0)
        else:
            return (np.zeros(len(x_i)), 0.0)

    if alpha == 1:
        iter_eps = 1
        (mu_i, lambda_k) = raking_chi2_distance(x_i, q_i, v_i, mu, direct)
    else:
        epsilon = 1.0
        if direct:
            lambda_k = 0.0
            iter_eps = 0
            while (epsilon > 1.0e-6) & (iter_eps < max_iter):
                if alpha == 0:
                    f1 = mu - np.sum(v_i * x_i * np.exp(- q_i * v_i * lambda_k))
                    f2 = np.sum(q_i * np.square(v_i) * x_i * np.exp(- q_i * v_i * lambda_k))
                else:
                    f1 = mu - np.sum(v_i * x_i * np.power(1.0 - alpha * q_i * v_i * lambda_k, 1.0 / alpha))
                    f2 = np.sum(q_i * np.square(v_i) * x_i * np.power(1.0 - alpha * q_i * v_i * lambda_k, 1.0 / alpha - 1.0))
                if (alpha > 0.5) or (alpha < -1.0):
                    gamma = gamma0
                    iter_gam = 0
                    while (np.any(1 - alpha * q_i * v_i * (lambda_k - gamma * f1 / f2) <= 0.0)) & \
                          (iter_gam < max_iter):
                        gamma = gamma / 2.0
                        iter_gam = iter_gam + 1
                else:
                    gamma = gamma0
                lambda_k = lambda_k - gamma * f1 / f2
                if alpha == 0:
                    mu_i = x_i * np.exp(- q_i * v_i * lambda_k)
                else:
                    mu_i = x_i * np.power(1.0 - alpha * q_i * v_i * lambda_k, 1.0 / alpha)
                epsilon = abs(mu - np.sum(v_i * mu_i))
                iter_eps = iter_eps + 1
        else:
            mu_k = np.copy(x_i)
            lambda_k = 0.0
            result = np.array(mu_k.tolist() + [lambda_k])
            iter_eps = 0
            while (epsilon > 1.0e-6) & (iter_eps < max_iter):
                if alpha == 0:
                    Phi = np.concatenate((np.concatenate((np.diag(1.0 / (q_i * mu_k)), \
                       v_i.reshape((-1, 1))), axis=1), \
                       np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
                    y = np.array((np.log(mu_k / x_i) / q_i + v_i * lambda_k).tolist() + [np.sum(v_i * mu_k) - mu])
                else:
                    Phi = np.concatenate((np.concatenate((np.diag(np.power(mu_k / x_i, alpha - 1.0) / (q_i * x_i)), \
                        v_i.reshape((-1, 1))), axis=1), \
                        np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
                    y = np.array(((np.power(mu_k / x_i, alpha)  - 1.0) / (q_i * alpha) + v_i * lambda_k).tolist() + \
                        [np.sum(v_i * mu_k) - mu])
                gamma = gamma0
                iter_gam = 0
                result_tmp = result - gamma * np.linalg.solve(Phi, y)
                mu_k_tmp = result_tmp[0:-1]
                lambda_k_tmp = result_tmp[-1]
                if (alpha > 0.5) or (alpha < -1.0):
                    while ((np.any(mu_k_tmp < 0.0)) |
                           (np.any(1.0 - alpha * q_i * v_i * lambda_k_tmp <= 0.0))) & \
                          (iter_gam < max_iter):
                        gamma = gamma / 2.0
                        iter_gam = iter_gam + 1
                        result_tmp = result - gamma * np.linalg.solve(Phi, y)
                        mu_k_tmp = result_tmp[0:-1]
                        lambda_k_tmp = result_tmp[-1]
                else:
                    while (np.any(mu_k_tmp < 0.0)) & (iter_gam < max_iter):
                        gamma = gamma / 2.0
                        iter_gam = iter_gam + 1
                        result_tmp = result - gamma * np.linalg.solve(Phi, y)
                        mu_k_tmp = result_tmp[0:-1]
                        lambda_k_tmp = result_tmp[-1]
                result = result - gamma * np.linalg.solve(Phi, y)
                mu_k = result[0:-1]
                lambda_k = result[-1]
                epsilon = abs(mu - np.sum(v_i * mu_k))
                iter_eps = iter_eps + 1
            mu_i = mu_k
    if return_num_iter:
        return (mu_i, lambda_k, iter_eps)
    else:
        return (mu_i, lambda_k)

def raking_logit(x_i, l_i, h_i, q_i, v_i, mu, gamma0=1.0, max_iter=500, return_num_iter=False, direct=True):
    """
    Logit raking ensuring that l_i < mu_i < h_i.
    Input:
      x_i: 1D Numpy array, observed values
      l_i: 1D Numpy array, lower bound for the observations
      h_i: 1D Numpy array, upper bound for the observations
      q_i: 1D Numpy array, weights for the observations
      v_i: 1D Numpy array, weights for the linear constraint (usually 1)
      mu: scalar, total of the linear constraint
      gamma0: scalar, initial value for line search
      max_iter: integer, maximum number of iterations for Newton's method
    return_num_iter: boolean, whether we return the number of iterations
      direct: boolean, if True, we solve for lambda and compute mu;
          if False, we solve directly for lambda and mu
    Output:
      mu_i: 1D Numpy array, raked values
    """
    assert isinstance(x_i, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(l_i, np.ndarray), \
        'Lower bounds should be a Numpy array.'
    assert isinstance(h_i, np.ndarray), \
        'Upper bounds should be a Numpy array.'
    assert isinstance(q_i, np.ndarray), \
        'Weights should be a Numpy array.'
    assert isinstance(v_i, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert len(x_i) == len(l_i), \
        'Observations and lower bounds arrays should have the same size.'
    assert len(x_i) == len(h_i), \
        'Observations and upper bounds arrays should have the same size.'
    assert len(x_i) == len(q_i), \
        'Observations and weights arrays should have the same size.'
    assert len(x_i) == len(v_i), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if (mu == 0.0) & (np.sum(np.abs(x_i)) == 0.0):
        if return_num_iter:
            return (np.zeros(len(x_i)), 0.0, 0)
        else:
            return (np.zeros(len(x_i)), 0.0)

    epsilon = 1.0
    if direct:
        lambda_k = 0.0
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            f1 = mu - np.sum(v_i * (l_i * (h_i - x_i) + \
                h_i * (x_i - l_i) * np.exp(- q_i * v_i * lambda_k)) / \
                ((h_i - x_i) + (x_i - l_i) * np.exp(- q_i * v_i * lambda_k)))
            f2 = np.sum(q_i * np.square(v_i) * ((x_i - l_i) * (h_i - x_i) * (h_i - l_i) * \
                np.exp(- q_i * v_i * lambda_k)) / \
                np.square((h_i - x_i) + (x_i - l_i) * np.exp(- q_i * v_i * lambda_k)))  
            lambda_k = lambda_k - f1 / f2
            mu_i = v_i * (l_i * (h_i - x_i) + \
                h_i * (x_i - l_i) * np.exp(- q_i * v_i * lambda_k)) / \
                ((h_i - x_i) + (x_i - l_i) * np.exp(- q_i * v_i * lambda_k))
            epsilon = abs(mu - np.sum(v_i * mu_i))
            iter_eps = iter_eps + 1
    else:
        mu_k = np.copy(x_i)
        lambda_k = 0.0
        result = np.array(mu_k.tolist() + [lambda_k])
        iter_eps = 0
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
            Phi = np.concatenate((np.concatenate((np.diag((1.0 / (mu_k - l_i) + 1.0 / (h_i - mu_k)) / q_i), \
                v_i.reshape((-1, 1))), axis=1), \
                np.array(v_i.tolist() + [0.0]).reshape((-1, len(v_i) + 1))), axis=0)
            y = np.array(((np.log((mu_k - l_i) / (x_i - l_i)) - np.log((h_i - mu_k) / (h_i - x_i))) / q_i + v_i * lambda_k).tolist() + \
                [np.sum(v_i * mu_k) - mu])
            gamma = gamma0
            iter_gam = 0
            result_tmp = result - gamma * np.linalg.solve(Phi, y)
            mu_k_tmp = result_tmp[0:-1]
            lambda_k_tmp = result_tmp[-1]
            while (np.any(mu_k_tmp < 0.0)) & (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                result_tmp = result - gamma * np.linalg.solve(Phi, y)
                mu_k_tmp = result_tmp[0:-1]
                lambda_k_tmp = result_tmp[-1]
            result = result - gamma * np.linalg.solve(Phi, y)
            mu_k = result[0:-1]
            lambda_k = result[-1]
            epsilon = abs(mu - np.sum(v_i * mu_k))
            iter_eps = iter_eps + 1
        mu_i = mu_k
    if return_num_iter:
        return (mu_i, lambda_k, iter_eps)
    else:
        return (mu_i, lambda_k)

def raking_vectorized_entropic_distance(df, agg_var, constant_vars=[]):
    """
    Raking using the entropic distance mu log(mu/x) + x - mu.
    Input:
      df: Pandas dataframe, containing the columns:
          - value = Values to be raked
          - agg_var = Variable over which we want to rake.
          - constant_vars = Several other variables (not to be raked).
          - all_'agg_var'_value = Partial sums
      agg_var: String, variable over which we do the raking
      constant_vars: List of strings, several other variables (not to be raked).
    Output:
      df: Pandas dataframe with another additional column value_raked
    """
    assert 'value' in df.columns, \
        'The dataframe should contain a column with the values to be raked.'
    assert agg_var in df.columns, \
        'The dataframe should contain a column with the variable over which to do the raking.'
    assert 'all_' + agg_var + '_value' in df.columns, \
        'The dataframe should contain a column with the margins.' 
    if len(constant_vars) > 0:
        for var in constant_vars:
            assert var in df.columns, \
                'The dataframe should contain a column ' + var + '.'

    name1 = agg_var + '_total'
    name2 = 'all_' + agg_var + '_value'
    df[name1] = df.groupby(constant_vars).value.transform('sum')
    df['entropic_distance'] = df['value'] * df[name2] / df[name1]
    df.drop(columns=[name1], inplace=True)
    return df

def raking_vectorized_l2_distance(df, agg_var, constant_vars=[]):
    """
    Raking using the l2 distance (mu - x)^2 / 2.
    Input:
      df: Pandas dataframe, containing the columns:
          - value = Values to be raked
          - agg_var = Variable over which we want to rake.
          - constant_vars = Several other variables (not to be raked).
          - all_'agg_var'_value = Partial sums
      agg_var: String, variable over which we do the raking
      constant_vars: List of strings, several other variables (not to be raked).
    Output:
      df: Pandas dataframe with another additional column value_raked
    """
    assert 'value' in df.columns, \
        'The dataframe should contain a column with the values to be raked.'
    assert agg_var in df.columns, \
        'The dataframe should contain a column with the variable over which to do the raking.'
    assert 'all_' + agg_var + '_value' in df.columns, \
        'The dataframe should contain a column with the margins.' 
    if len(constant_vars) > 0:
        for var in constant_vars:
            assert var in df.columns, \
                'The dataframe should contain a column ' + var + '.'

    name1 = agg_var + '_total'
    name2 = 'all_' + agg_var + '_value'
    I = len(df[agg_var].unique().tolist())
    df[name1] = df.groupby(constant_vars).value.transform('sum')
    df['l2_distance'] = df['value'] - (df[name1] - df[name2]) / I
    df.drop(columns=[name1], inplace=True)
    return df


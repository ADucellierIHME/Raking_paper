"""
In this module, we compute the uncertainty on the raked values
using the implicit function theorem
"""

import numpy as np

def delta_method(dh_x, dh_y, sigma_xx, sigma_yy, sigma_xy):
    """
    """
    dh = np.concatenate((dh_x, dh_y), axis=0)
    sigma = np.concatenate(( \
        np.concatenate((sigma_xx, sigma_xy), axis=1), \
        np.concatenate((np.transpose(sigma_xy), sigma_yy), axis=1)), axis=0)
    variance = np.matmul(np.transpose(dh), np.matmul(sigma, dh))
    return variance

def compute_variance(mu_0, lambda_0, x, q, A, method, alpha=0, l=0, h=1):
    """
    """
    M = np.shape(A)[0]
    N = np.shape(A)[1]

    # Derivatives of distance function
    if method == 'general':
        H1_mu_diag = np.zeros(len(mu_0))
        H1_mu_diag[x!=0] = np.power(mu_0[x!=0] / x[x!=0], alpha - 1) / (q[x!=0] * x[x!=0])
        H1_mu_diag[x==0] = 0.0
        H1_mu = np.diag(H1_mu_diag)
        H1_x_diag = np.zeros(len(x))
        H1_x_diag[mu_0!=0] = - np.power(mu_0[mu_0!=0] / x[mu_0!=0], alpha + 1) / (q[mu_0!=0] * mu_0[mu_0!=0])
        H1_x_diag[mu_0==0] = 0.0
        H1_x = np.diag(H1_x_diag)
    elif method == 'l2':
        H1_mu = np.identity(len(mu_0))
        H1_x = - np.identity(len(x))
    elif method == 'logit':
        H1_mu_diag = np.zeros(len(mu_0))
        H1_mu_diag[(mu_0!=l)&(mu_0!=h)] = 1.0 / (mu_0[(mu_0!=l)&(mu_0!=h)] - l[(mu_0!=l)&(mu_0!=h)]) + \
                                          1.0 / (h[(mu_0!=l)&(mu_0!=h)] - mu_0[(mu_0!=l)&(mu_0!=h)])
        H1_mu_diag[(mu_0==l)|(mu_0==h)] = 0.0
        H1_mu = np.diag(H1_mu_diag)
        H1_x_diag = np.zeros(len(x))
        H1_x_diag[(x!=l)&(x!=h)] = - 1.0 / (x[(x!=l)&(x!=h)] - l[(x!=l)&(x!=h)]) - \
                                     1.0 / (h[(x!=l)&(x!=h)] - x[(x!=l)&(x!=h)])
        H1_x_diag[(x==l)|(x==h)] = 0.0
        H1_x = np.diag(H1_x_diag)

    # Gradient with respect to mu and lambda
    H1_lambda = np.transpose(np.copy(A))
    H2_mu = np.copy(A)
    H2_lambda = np.zeros((M, M))    
    DH_mu_lambda = np.concatenate(( \
        np.concatenate((H1_mu, H1_lambda), axis=1), \
        np.concatenate((H2_mu, H2_lambda), axis=1)), axis=0)

    # Gradient with respect to x and y
    H1_y = np.zeros((N, M))
    H2_x = np.zeros((M, N))
    H2_y = - np.identity(M)    
    DH_x_y = np.concatenate(( \
        np.concatenate((H1_x, H1_y), axis=1), \
        np.concatenate((H2_x, H2_y), axis=1)), axis=0)

    # Compute Moore-Penrose pseudo inverse of D_mu_lambda
    U, S, Vh = np.linalg.svd(DH_mu_lambda, full_matrices=True)
    V = np.transpose(Vh)
    Sdiag = np.diag(S)
    Sdiag[np.abs(Sdiag) <= 1.0e-12] = 1.0e-12
    Sinv = 1.0 / Sdiag
    Sinv[np.abs(Sdiag) <= 1.0e-12] = 0.0
    DH_mu_lambda_plus = np.matmul(V, np.matmul(Sinv, np.transpose(U)))
        
    # Gradient of mu and lambda with respect to x and y
    Dh = - np.matmul(DH_mu_lambda_plus, DH_x_y)
    dh_x = Dh[0:N, 0:N]
    dh_y = Dh[0:N, N:(N + M)]
    return (dh_x, dh_y)


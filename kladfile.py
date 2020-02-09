# import sys
# for p in sys.path:
#   print (p)

import pandas as pd
import arviz
import pymc3 as pm
import numpy as np
import matplotlib as mpl
plt = mpl.pyplot

from scipy.linalg import cholesky, cho_solve

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from timeit import default_timer as timer

import theano.tensor as tt


from scr import models_algorithms

from timeit import default_timer as timer

from scr import models_algorithms

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


amountTraining = 1000
modelListTraining = []
valuesFFTCallsTraining = pd.DataFrame(index=range(1), columns=range(amountTraining))
parametersModelsTraining = pd.DataFrame(index=range(amountTraining), columns=range(10))

for x in range(amountTraining):

    # generate pseudo-random numbers for the Training set

    stock_value = 1
    strike = np.random.uniform(0.4,1.6)*stock_value
    maturity = np.random.uniform(11/12, 1)
    interest = np.random.uniform(0.015,0.025)
    dividend_yield = np.random.uniform(0,0.05)

    # heston

    kappa = np.random.uniform(1.4, 2.6)
    rho = np.random.uniform(-0.85, -0.55)
    theta = np.random.uniform(0.45, 0.75)
    eta = np.random.uniform(0.01, 0.1)
    sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.9048)))

    modelListTraining.append(models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield))

for i, model in enumerate(modelListTraining):
    valuesFFTCallsTraining[i] = model.heston_carr_madan(0)
    for j, parameter in enumerate(model.get_parameters()):
        parametersModelsTraining.iat[i, j] = parameter



amountInducing = 10
modelListInducing = []
valuesFFTCallsInducing = pd.DataFrame(index=range(1), columns=range(amountInducing))
parametersModelsInducing= pd.DataFrame(index=range(amountInducing), columns=range(10))

for x in range(amountInducing):

    # generate pseudo-random numbers for the Training set

    stock_value = 1
    strike = np.random.uniform(0.4,1.6)*stock_value
    maturity = np.random.uniform(11/12, 1)
    interest = np.random.uniform(0.015,0.025)
    dividend_yield = np.random.uniform(0,0.05)

    # heston

    kappa = np.random.uniform(1.4, 2.6)
    rho = np.random.uniform(-0.85, -0.55)
    theta = np.random.uniform(0.45, 0.75)
    eta = np.random.uniform(0.01, 0.1)
    sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.9048)))

    modelListInducing.append(models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield))

for i, model in enumerate(modelListInducing):
    valuesFFTCallsInducing[i] = model.heston_carr_madan(0)
    for j, parameter in enumerate(model.get_parameters()):
        parametersModelsInducing.iat[i, j] = parameter

kmeans  = KMeans(n_clusters=amountInducing, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(parametersModelsTraining)
parametersModelsInducing = kmeans.cluster_centers_

kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))

# K_xx = kernel(parametersModelsTraining)
# K_ux = kernel(parametersModelsInducing,parametersModelsTraining)
# K_xu = kernel(parametersModelsTraining,parametersModelsInducing)
# K_uu = kernel(parametersModelsInducing)
#
# L_u = cholesky(K_uu, lower=True)
# Alpha = cho_solve((L_u, True), K_ux)    #d*n
# Q_xx = np.dot(K_xu,Alpha)
#
#
# Lambd = K_xx - Q_xx
# diag = np.einsum('ii->i', Lambd)
# save = diag.copy()
# Lambd[...] = 0
# diag[...] = save
# print(Lambd)
# Lambd_inv = np.linalg.inv(Lambd)
#
# test = np.linalg.inv(Q_xx +Lambd)
#
# A = np.linalg.solve(L_u,K_ux)
# A_l = np.dot(A,Lambd_inv)    #d*n
# L2_u = cholesky(np.eye(amountInducing) + np.dot(A_l,A.transpose()) , lower=True)
# c = np.linalg.solve(L2_u, A_l)
# inv = Lambd_inv - np.dot(c.transpose(),c)
#
# alpha = np.dot(inv, valuesFFTCallsTraining.transpose())
#
# # Compute log-likelihood (compare line 7)
# log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", valuesFFTCallsTraining.transpose(), alpha)  # eerste term
# log_likelihood_dims -= np.log(np.diag(L2_u)).sum() + 0.5 * np.log(np.diag(Lambd)).sum()  # determinant (2*1/2 = 1)
# log_likelihood_dims -= K_xx.shape[0] / 2 * np.log(2 * np.pi)  # cte term
# log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

K_xx = kernel(parametersModelsTraining)
K_ux = kernel(parametersModelsInducing,parametersModelsTraining)
K_xu = kernel(parametersModelsTraining,parametersModelsInducing)
K_uu = kernel(parametersModelsInducing)
K_xastu = kernel(parametersModelsTraining, parametersModelsInducing)
K_uastx = kernel(parametersModelsInducing, parametersModelsTraining)


L_u = cholesky(K_uu, lower=True)
Alpha = cho_solve((L_u, True), K_ux)    #d*n
Q_xx = np.dot(K_xu,Alpha)

Lambd = K_xx - Q_xx
diag = np.einsum('ii->i', Lambd)
save = diag.copy()
Lambd[...] = 0
diag[...] = save
print(Lambd)
Lambd_inv = np.linalg.inv(Lambd)

sigma = K_uu + np.dot(K_ux, np.dot(Lambd_inv, K_xu))
L_sigma = cholesky(sigma)
y_l = np.dot(Lambd_inv, valuesFFTCallsTraining.transpose())
a = np.dot(K_ux, y_l)
alpha = cho_solve((L_sigma, True), a)

pred = np.dot(K_xastu, alpha)

print(pred[:10])
#print(valuesFFTCallsTraining[:10])







#KAN GEWIST WORDEN? MAAR HOU IK ALS BACK UP

import pymc3 as pm

import theano
import theano.tensor as tt
import numpy as np

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from timeit import default_timer as timer

from scr import standard_gpr_newtry

from scr import models_algorithms

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# set the seed
np.random.seed(1)


amountTraining = 100
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

valuesFFTCallsTraining = np.squeeze(np.asarray(valuesFFTCallsTraining))
X = np.array(np.squeeze(np.asarray(parametersModelsTraining)),dtype=np.float64)

#
# amountinducing =100
# modelListInducing = []
# valuesFFTCallsInducing= pd.DataFrame(index=range(1), columns=range(amountinducing))
# parametersModelsInducing = pd.DataFrame(index=range(amountinducing), columns=range(10))
#
# for x in range(amountinducing):
#
#     # generate pseudo-random numbers for the Training set
#
#     stock_value = 1
#     strike = np.random.uniform(0.4,1.6)*stock_value
#     maturity = np.random.uniform(11/12, 1)
#     interest = np.random.uniform(0.015,0.025)
#     dividend_yield = np.random.uniform(0,0.05)
#
#     # heston
#
#     kappa = np.random.uniform(1.4, 2.6)
#     rho = np.random.uniform(-0.85, -0.55)
#     theta = np.random.uniform(0.45, 0.75)
#     eta = np.random.uniform(0.01, 0.1)
#     sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.9048)))
#
#     modelListTraining.append(models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield))
#
# for i, model in enumerate(modelListInducing):
#     valuesFFTCallsInducing[i] = model.heston_carr_madan(0)
#     for j, parameter in enumerate(model.get_parameters()):
#         parametersModelsInducing.iat[i, j] = parameter
#
# valuesFFTCallsInducing = np.squeeze(np.asarray(valuesFFTCallsTraining))
# X_unit = np.array(np.squeeze(np.asarray(parametersModelsInducing)),dtype=np.float64)



with pm.Model() as model:
    # Set priors on the hyperparameters of the covariance
    ls1 = pm.Gamma("ls1", alpha=2, beta=2)

    eta = pm.HalfNormal("eta", sigma=2)

    cov = eta ** 2 * pm.gp.cov.ExpQuad(10, ls1)  # cov_x1 must accept X1 without error


    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")

    # set flat prior for Xu  (flat ok? , normal better)
    #Xu = pm.Flat("Xu", shape=(amountinducing,10), testval=X_unit)
    Xu = pm.gp.util.kmeans_inducing_points(100, X)

    sigma = pm.HalfNormal("sigma", sigma=2)

    # Place a GP prior over the function f.
    y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, y=valuesFFTCallsTraining, noise=sigma)

    mp = pm.find_MAP()


amountTest = 100
modelListTest = []
valuesFFTCallsTest = pd.DataFrame(index=range(1), columns=range(amountTest))
parametersModelsTest = pd.DataFrame(index=range(amountTest),columns=range(10))

for x in range(amountTest):

    # generate pseudo-random numbers for the Training set

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

    modelListTest.append(models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield))

for i, Model in enumerate(modelListTest):
    for j, parameter in enumerate(Model.get_parameters()):
        parametersModelsTest.iat[i, j] = parameter

for i, Model in enumerate(modelListTest):
    valuesFFTCallsTest[i] = Model.heston_carr_madan(0)

valuesFFTCallsTest = np.squeeze(np.asarray(valuesFFTCallsTest))
X_new = np.array(np.squeeze(np.asarray(parametersModelsTest)),dtype=np.float64)

mu, var = gp.predict(X_new, point=mp, diag=True)


print(mu)
print(valuesFFTCallsTest)

MAE = np.sum(np.abs((valuesFFTCallsTest - mu)))/amountTest
print(MAE)
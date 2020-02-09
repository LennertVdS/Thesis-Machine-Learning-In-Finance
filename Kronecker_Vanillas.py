
import pandas as pd
import arviz
import pymc3 as pm
import numpy as np
import matplotlib as mpl
plt = mpl.pyplot

from timeit import default_timer as timer

from scr import models_algorithms

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


kappa_amount,eta_amount,theta_amount,rho_amount,sigma0_amount,strike_amount,maturity_amount,interest_amount = (3,3,3,5,3,7,1,1)

amountTraining = kappa_amount * eta_amount * theta_amount *rho_amount * sigma0_amount * strike_amount *maturity_amount *interest_amount
modelListTraining = []
valuesFFTCallsTraining = pd.DataFrame(index=range(1), columns=range(amountTraining))
parametersModelsTraining = pd.DataFrame(index=range(amountTraining), columns=range(10))


stock_value = 1

kappa = np.linspace(1.4, 2.6,kappa_amount)
eta = np.linspace(0.01, 0.1,eta_amount)
theta = np.linspace(0.45, 0.75,theta_amount)
rho = np.linspace(-0.85, -0.55,rho_amount)
sigma0 = np.sqrt(-np.log(np.linspace(0.99, 0.9048,sigma0_amount)))


strike = np.linspace(0.4, 1.6,strike_amount) * stock_value
maturity = np.linspace(11 / 12, 1,maturity_amount)
interest = np.linspace(0.010, 0.025,interest_amount)    #modified


X = pm.math.cartesian(kappa[:,None],eta[:,None],theta[:,None],rho[:,None],sigma0[:,None],strike[:,None],maturity[:,None],interest[:,None])

for element in X:
    modelListTraining.append(
        models_algorithms.vanilla_option_heston(element[0], element[1], element[2], element[3], element[4], element[5], element[6],
                                                1, element[7], 0))


for i, model in enumerate(modelListTraining):
    valuesFFTCallsTraining[i] = model.heston_carr_madan(0)
    for j, parameter in enumerate(model.get_parameters()):
        parametersModelsTraining.iat[i, j] = parameter

valuesFFTCallsTraining = np.squeeze(np.asarray(valuesFFTCallsTraining))

X = (np.squeeze(np.asarray(parametersModelsTraining.loc[:,[0,1,2,3,4,5,6,8]]))).astype(dtype = 'float32')


# poly = PolynomialFeatures(degree=3)
# X_poly = poly.fit_transform(X)
# pol_reg = LinearRegression()
# k = pol_reg.fit(X_poly, valuesFFTCallsTraining)
# valuesFFTCallsTraining = valuesFFTCallsTraining - pol_reg.predict(poly.fit_transform(X))


# this implementation takes a list of inputs for each dimension as input
Xs = [kappa[:,None],eta[:,None],theta[:,None],rho[:,None],sigma0[:,None],strike[:,None],maturity[:,None],interest[:,None]]
#Xs = [kappa[:,None],eta[:,None],theta[:,None],rho[:,None],sigma0[:,None],strike[:,None],maturity[:,None],interest[:,None],dividend_yield[:,None]]
eta_true = 1

with pm.Model() as model:
    # Set priors on the hyperparameters of the covariance
    ls1  = pm.Gamma("ls1", alpha=2, beta=2)
    #ls2 = pm.Gamma("ls2", alpha=2, beta=2)
    #ls3= pm.Gamma("ls3", alpha=2, beta=2)
    #ls4 = pm.Gamma("ls4", alpha=2, beta=2)
    #ls5 = pm.Gamma("ls5", alpha=2, beta=2)
    #ls6 = pm.Gamma("ls6", alpha=2, beta=2)

    eta = pm.HalfNormal("eta", sigma=2)

    # Specify the covariance functions for each Xi
    # Since the covariance is a product, only scale one of them by eta.
    # Scaling both overparameterizes the covariance function.
    cov_kappa = eta**2 * pm.gp.cov.ExpQuad(1, ls1)        # cov_x1 must accept X1 without error
    cov_eta = pm.gp.cov.ExpQuad(1, ls1)    # cov_x2 must accept X2 without error
    cov_theta = pm.gp.cov.ExpQuad(1, ls1)
    cov_rho = pm.gp.cov.ExpQuad(1, ls1)
    cov_sigma0 = pm.gp.cov.ExpQuad(1, ls1)
    cov_strike = pm.gp.cov.ExpQuad(1, ls1)
    cov_maturity = pm.gp.cov.ExpQuad(1, ls1)
    cov_interest = pm.gp.cov.ExpQuad(1, ls1)
    #cov_dividend_yield = pm.gp.cov.ExpQuad(1, ls1)

    # Specify the GP.  The default mean function is `Zero`.
#    gp = pm.gp.MarginalKron(cov_funcs=[cov_kappa, cov_eta,cov_theta,cov_rho,cov_sigma0,cov_strike,cov_maturity,cov_interest,cov_dividend_yield])
    gp = pm.gp.MarginalKron(cov_funcs=[cov_kappa, cov_eta,cov_theta,cov_rho,cov_sigma0,cov_strike,cov_maturity,cov_interest])

    # Set the prior on the variance for the Gaussian noise
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Place a GP prior over the function f.
    y_ = gp.marginal_likelihood("y", Xs=Xs, y=valuesFFTCallsTraining, sigma=sigma)

with model:
    mp = pm.find_MAP(method="BFGS")  #  find the maximum a-posteriori estimate
    trace = pm.sample(250, start=mp,tune=70, cores=1)


fig = pm.traceplot(trace, var_names=['ls1', 'eta'])
plt.show()
print(mp)
print(trace[0])
mp =trace[0]


amountTest = 100
modelListTest = []
valuesFFTCallsTest = pd.DataFrame(index=range(1), columns=range(amountTest))
parametersModelsTest = pd.DataFrame(index=range(amountTest),columns=range(10))

for x in range(amountTest):

    # generate pseudo-random numbers for the Training set

    strike = np.random.uniform(0.4,1.6)*stock_value
    maturity = np.random.uniform(11/12, 1)
    interest = np.random.uniform(0.010,0.025)
    dividend_yield = 0


    # heston

    kappa = np.random.uniform(1.4, 2.6)
    rho = np.random.uniform(-0.85, -0.55)
    theta = np.random.uniform(0.45, 0.75)
    eta = np.random.uniform(0.01, 0.1)
    sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.9048)))

    modelListTest.append(models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield))

for i, model in enumerate(modelListTest):
    for j, parameter in enumerate(model.get_parameters()):
        parametersModelsTest.iat[i, j] = parameter


startPredictingInSampleTimerFFT = timer()
for i, model in enumerate(modelListTest):
    valuesFFTCallsTest[i] = model.heston_carr_madan(0)
endPredictingInSampleTimerFFT = timer()

valuesFFTCallsTest = np.squeeze(np.asarray(valuesFFTCallsTest))

parametersModelsTest = (np.squeeze(np.asarray(parametersModelsTest.loc[:,[0,1,2,3,4,5,6,8]]))).astype(dtype = 'float32')

print(parametersModelsTest)

mu,cov = gp.predict(parametersModelsTest, point=mp)

# print(pol_reg.predict(poly.fit_transform(parametersModelsTest)))
#
# mu += pol_reg.predict(poly.fit_transform(parametersModelsTest))

mu = np.maximum(mu,0)

MAE = np.sum(np.abs((valuesFFTCallsTest - mu)))/amountTest
AAE = np.max(np.abs((valuesFFTCallsTest - mu)))

print(valuesFFTCallsTest[1:10])
print(mu[1:10])
print(MAE)
print(AAE)











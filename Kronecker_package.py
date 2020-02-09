import numpy as np
import pandas as pd
import itertools as it
import random as rn


import pymc3 as pm
import theano.tensor as tt
import numpy as np
import matplotlib as mpl
plt = mpl.pyplot

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from timeit import default_timer as timer


from random import randrange, uniform

from scr import standard_gpr_newtry

from scr import models_algorithms

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



np.random.seed(1234)

# One dimensional column vectors of inputs
n1, n2 = (50, 30)
x1 = np.linspace(0, 5, n1)
x2 = np.linspace(0, 3, n2)

# make cartesian grid out of each dimension x1 and x2
X = pm.math.cartesian(x1[:,None], x2[:,None])

l1_true = 0.8
l2_true = 1.0
eta_true = 1.0

# Although we could, we don't exploit kronecker structure to draw the sample
cov = eta_true**2 * pm.gp.cov.Matern52(2, l1_true, active_dims=[0]) *\
                    pm.gp.cov.Cosine(2, ls=l2_true, active_dims=[1])

K = cov(X).eval()
f_true = np.random.multivariate_normal(np.zeros(X.shape[0]), K, 1).flatten()

sigma_true = 0.25
y = f_true + sigma_true * np.random.randn(X.shape[0])

fig = plt.figure(figsize=(12,6))
ax = fig.gca(); cmap = 'terrain'
norm = mpl.colors.Normalize(vmin=-3, vmax=3)
plt.scatter(X[:,0], X[:,1], s=35, c=y, marker='o', norm=norm, cmap=cmap); plt.colorbar();
plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Simulated dataset");
plt.show()


# this implementation takes a list of inputs for each dimension as input
Xs = [x1[:,None], x2[:,None]]

with pm.Model() as model:
    # Set priors on the hyperparameters of the covariance
    ls1  = pm.Gamma("ls1", alpha=2, beta=2)
    ls2  = pm.Gamma("ls2", alpha=2, beta=2)
    eta = pm.HalfNormal("eta", sigma=2)

    # Specify the covariance functions for each Xi
    # Since the covariance is a product, only scale one of them by eta.
    # Scaling both overparameterizes the covariance function.
    cov_x1 = pm.gp.cov.Matern52(1, ls=ls1)        # cov_x1 must accept X1 without error
    cov_x2 = eta**2 * pm.gp.cov.Cosine(1, ls=ls2) # cov_x2 must accept X2 without error

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.MarginalKron(cov_funcs=[cov_x1, cov_x2])

    # Set the prior on the variance for the Gaussian noise
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Place a GP prior over the function f.
    y_ = gp.marginal_likelihood("y", Xs=Xs, y=y, sigma=sigma)

with model:
    mp = pm.find_MAP(method="powell")

    start = pm.find_MAP()
    # find starting value by optimization
    step = pm.NUTS(state=start)
    # instantiate MCMC sampling algorithm
    trace = pm.sample(100, step, start=start, progressbar =False)

x1new = np.linspace(5.1, 7.1, 20)
x2new = np.linspace(-0.5, 3.5, 40)
Xnew = pm.math.cartesian(x1new[:,None], x2new[:,None])

mu, var = gp.predict(Xnew, point=mp, diag=True)

fig = plt.figure(figsize=(12,6))
ax = fig.gca(); cmap = 'terrain'
norm = mpl.colors.Normalize(vmin=-3, vmax=3)
m=ax.scatter(X[:,0], X[:,1], s=30, c=y, marker='o', norm=norm, cmap=cmap); plt.colorbar(m);
ax.scatter(Xnew[:,0], Xnew[:,1], s=30, c=mu, marker='s', norm=norm, cmap=cmap);
ax.set_ylabel("x2"); ax.set_xlabel("x1")
ax.set_title("observed data 'y' (circles) with predicted mean (squares)");

plt.show()
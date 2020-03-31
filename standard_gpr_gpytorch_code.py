import torch
import gpytorch
import torch.cuda

import numpy as np
from timeit import default_timer as timer
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.linalg import cholesky, cho_solve

"""
This code executes a standard GPR using the BBMM framework 
"""

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def prediction(model, likelihood,test_x,train_x,alpha,kernel):

    # This implementation gives less accurate predictions

    # if torch.cuda.is_available():
    #     test_x = test_x.cuda()
    #
    # # Get into evaluation (predictive posterior) mode
    # model.eval()
    # likelihood.eval()
    #
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #
    #     #observed_pred = likelihood(model(test_x))
    #     #mean = observed_pred.mean
    #
    #     mean = model(test_x).mean

    K_ast = kernel(train_x, test_x)
    mean = (K_ast.transpose()).dot(alpha)

    return mean

def training(model,likelihood,train_x,train_y,training_iter):

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.3)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()                               #computes gradient
        print('Iter %d/%d - Loss: %.3f   signal variance: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.outputscale.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()

        ))
        optimizer.step()


    return model, likelihood


def training_extra(model, trainingParameters, trainingValues,):
    signal_variance = model.covar_module.outputscale.item()
    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    noise = model.likelihood.noise.item()
    kernel = C(signal_variance) * RBF(lengthscale)
    K = kernel(trainingParameters)
    K[np.diag_indices_from(K)] += noise ** 2
    L = cholesky_dec(K)
    alpha = cho_solve((L, True), trainingValues.transpose())

    return alpha,kernel

def cholesky_dec(matrix):
    try:
        L = cholesky(matrix, lower=True)  # Line 2
    except np.linalg.LinAlgError as exc:
        exc.args = ("The kernel, %s, is not returning a "
                    "positive definite matrix. Try gradually "
                    "increasing the 'alpha' parameter of your "
                    "GaussianProcessRegressor estimator.")
        raise
    return L




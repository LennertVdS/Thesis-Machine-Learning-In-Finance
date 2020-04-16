import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.linalg import cholesky, cho_solve

import numpy as np

"""
This code executes VFE using the BBMM framework 
"""

class GPRegressionModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, train_u,likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_u, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def prediction(model, likelihood,test_x,train_u,alpha,kernel):

    # if torch.cuda.is_available():
    #     test_x = test_x.cuda()
    #
    # # Get into evaluation (predictive posterior) mode
    # model.eval()
    # likelihood.eval()
    #
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     #observed_pred = likelihood(model(test_x))
    #     #mean = observed_pred.mean
    #
    #     mean = model(test_x).mean

    K_xastu = kernel(test_x, train_u)
    mean = np.dot(K_xastu, alpha)

    return mean


def training(model,likelihood,train_x,train_y,amount_iterations):

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(amount_iterations):

        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   signal variance: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, amount_iterations, loss.item(),
            model.base_covar_module.outputscale.item(),
            model.base_covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()

        ))
        optimizer.step()

    return model, likelihood


def training_extra(model,trainingParameters,parametersModelsInducing,trainingValues,inducingnoise):

    signal_variance = model.base_covar_module.outputscale.item()
    lengthscale = model.base_covar_module.base_kernel.lengthscale.item()
    noise = model.likelihood.noise.item()

    kernel = C(signal_variance) * RBF((lengthscale))
    K = kernel(trainingParameters)
    noise = noise ** 2
    K[np.diag_indices_from(K)] += noise
    K_uu = kernel(parametersModelsInducing, parametersModelsInducing)
    K_uu[np.diag_indices_from(K_uu)] += inducingnoise
    K_xu = kernel(trainingParameters, parametersModelsInducing)
    K_ux = kernel(parametersModelsInducing, trainingParameters)

    init = np.ones((np.shape(trainingParameters)[0]))
    inverse_noise = 1 / noise
    inv_lambd_vec = init * inverse_noise
    Lambd_inv = np.diag(inv_lambd_vec)

    sigma = K_uu + np.dot(K_ux, np.dot(Lambd_inv, K_xu))
    L_sigma = cholesky_dec(sigma)
    y_l = np.dot(Lambd_inv, trainingValues.transpose())
    a = np.dot(K_ux, y_l)
    alpha = cho_solve((L_sigma, True), a)

    return alpha, kernel

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
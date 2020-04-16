import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, ProductStructureKernel, GridInterpolationKernel



"""
Code for SKIP 
"""

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, amountinducing):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        dimension = train_x.size(-1)
        self.covar_module = ProductStructureKernel(

                GridInterpolationKernel(self.base_covar_module, grid_size=amountinducing, num_dims=1)
            , num_dims=dimension
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def prediction(model, likelihood,test_x):

    if torch.cuda.is_available():
        test_x = test_x.cuda()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
    return mean

    #predicting with skip

    # with gpytorch.settings.max_preconditioner_size(1000), torch.no_grad():
    #       with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(
    #               30), gpytorch.settings.fast_pred_var():
    #          mean = model(test_x).mean
    # return mean


def training(model,likelihood,train_x,train_y,training_iterations):

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.settings.use_toeplitz(True):

        for i in range(training_iterations):

            # Zero backprop gradients
            optimizer.zero_grad()
            with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30):   #amount of Lanczos iterations
                # Get output from model
                output = model(train_x)
                # Calc loss and backprop derivatives
                loss = -mll(output, train_y)
                loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()


    return model, likelihood

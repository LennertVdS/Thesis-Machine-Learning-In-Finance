import torch
import gpytorch

"""
This code executes a SVG where the initial input inducing points are chosen by kmeans
"""


class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

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
    with torch.no_grad():
        f_dist = model(test_x)
        mean = f_dist.mean
    return mean

def training(model,objective_function_cls,train_x,train_y,training_iter,zero_mean = False):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    objective_function = objective_function_cls(likelihood, model, num_data=train_y.numel())
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

    # Train
    model.train()
    likelihood.train()
    for i in range(training_iter):
        output = model(train_x)
        loss = -objective_function(output, train_y)
        loss.backward()
        optimizer.step()
        print('Iter %d/%d - Loss: %.3f ' % (
            i + 1, training_iter, loss.item()))
        optimizer.zero_grad()

    return model, likelihood

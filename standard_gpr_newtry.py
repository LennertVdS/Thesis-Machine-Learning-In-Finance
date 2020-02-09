import numpy as np
from operator import itemgetter
from scipy.linalg import cholesky, cho_solve
import scipy.optimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



# This code is based on the code of sklearn.gaussianprocess
# made more easy and modified

class gaussianprocessregression:

    def __init__(self,kernel, noise,X_train, y_train, zero_mean =False):
        self.kernel = kernel
        self.noise = noise
        self.X_train = X_train
        self.y_train = y_train
        self.zero_mean = zero_mean
        self.n_restarts_optimizer = 5


    def prediction(self,X_test):
        # K = self.kernel(self.X_train)
        # K[np.diag_indices_from(K)] += self.noise
        #
        # self.L = self.cholesky_dec(K)
        # self.alpha = cho_solve((self.L, True), self.y_train)

        K_ast = self.kernel(self.X_train,X_test)
        pred = (K_ast.transpose()).dot(self.alpha)

        if self.zero_mean is False:
            pred += self.pol_reg.predict(self.poly.fit_transform(X_test))

        return pred

    def fitting(self):

        if self.zero_mean is False:

            self.poly = PolynomialFeatures(degree=2)
            X_poly = self.poly.fit_transform(self.X_train)
            self.pol_reg = LinearRegression()
            k = self.pol_reg.fit(X_poly, self.y_train)
            self.y_train = self.y_train - self.pol_reg.predict(self.poly.fit_transform(self.X_train))

        def obj_func(theta):
            return -self.logmarginallikelihood(theta)
        optima = [(self.optimization(obj_func, self.kernel.theta, self.kernel.bounds))]

        bounds = self.kernel.bounds

        for iteration in range(self.n_restarts_optimizer):
            theta_initial =  [np.random.uniform(bounds[0, 0],bounds[1, 0]), np.random.uniform(bounds[1, 0],bounds[1, 1])]
            optima.append(
                self.optimization(obj_func, theta_initial, bounds))
        lml_values = list(map(itemgetter(1), optima))
        self.kernel.theta = optima[np.argmin(lml_values)][0]
        self.log_marginal_likelihood_value_ = -np.min(lml_values)

        K = self.kernel(self.X_train)
        K[np.diag_indices_from(K)] += self.noise

        L = self.cholesky_dec(K)
        self.alpha = cho_solve((L, True), self.y_train)

    def optimization(self, obj_func, initial_theta, bounds):
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B",bounds=bounds)
            theta_opt, func_min = opt_res.x, opt_res.fun
            return theta_opt, func_min

    def cholesky_dec(self, matrix):
        try:
            L = cholesky(matrix, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel,) + exc.args
            raise
        return L


    def logmarginallikelihood(self, theta):

        kernel = self.kernel
        kernel.theta = theta
        K = kernel(self.X_train)

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train
        # if y_train.ndim == 1:
        #     y_train = y_train[:, np.newaxis]        #put it in matrix form

        K[np.diag_indices_from(K)] += self.noise

        L = self.cholesky_dec(K)
        alpha = cho_solve((L, True), y_train)

        # Compute log-likelihood (compare line 7)
        log_likelihood = -0.5 * np.einsum("ik,ik->k", y_train, alpha)                      #eerste term
        log_likelihood -= np.log(np.diag(L)).sum()                                        #determinant (2*1/2 = 1)
        log_likelihood -= K.shape[0] / 2 * np.log(2 * np.pi)                               #cte term

        return log_likelihood


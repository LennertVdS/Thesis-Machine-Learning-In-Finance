import numpy as np
from operator import itemgetter
from scipy.linalg import cholesky, cho_solve
import scipy.optimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# This code is based on the code of sklearn.gaussianprocess, now for sparse GPR

class gaussianprocessregression:

    def __init__(self,kernel, noise,noise_u,X_train, U_induce, y_train, method, zero_mean =False):
        self.kernel = kernel
        self.noise = noise
        self.noise_u = noise_u
        self.X_train = X_train
        self.U_induce = U_induce
        self.y_train = y_train
        self.zero_mean = zero_mean
        self.method = method
        self.n_restarts_optimizer = 5


    def prediction(self,X_test):

        # #K_xx = self.kernel(self.X_train)
        # K_ux = self.kernel(self.U_induce, self.X_train)
        # K_xu = self.kernel(self.X_train, self.U_induce)
        # K_uu = self.kernel(self.U_induce)
        # K_uu[np.diag_indices_from(K_uu)] += self.noise
        #
        # # L_u = self.cholesky_dec(K_uu)
        # # Alpha = cho_solve((L_u, True), K_ux)
        # # Q_xx = K_xu.dot(Alpha)
        #
        # if (self.method == 'sparse_FITC'):
        #     # K_xx = np.diag(self.kernel.diag(self.X_train))
        #     # Lambd = K_xx - Q_xx + self.noise
        #     # diag = np.einsum('ii->i', Lambd)
        #     # save = diag.copy()
        #     # Lambd[...] = 0
        #     # diag[...] = save
        #     # Lambd_inv = np.linalg.inv(Lambd)
        #
        #     L_u = self.cholesky_dec(K_uu)
        #     Alpha = cho_solve((L_u, True), K_ux)
        #
        #     K_xx = np.diag(self.kernel.diag(self.X_train))
        #     Q_xx = np.diag(np.einsum('ij,ji->i', K_xu, Alpha))
        #
        #     Lambd = K_xx - Q_xx
        #     Lambd[np.diag_indices_from(Lambd)] += self.noise
        #     Lambd_inv = np.linalg.inv(Lambd)
        #
        # if (self.method == 'sparse_VFE'):
        #     Lambd = np.eye(np.shape(self.X_train)[0])
        #     Lambd[np.diag_indices_from(Lambd)] *= self.noise
        #     Lambd_inv = np.linalg.inv(Lambd)
        #
        # sigma = K_uu + np.dot(K_ux, np.dot(Lambd_inv,K_xu))
        # L_sigma = self.cholesky_dec(sigma)
        # y_l = np.dot(Lambd_inv,self.y_train)
        # a = np.dot(K_ux,y_l)
        # self.alpha = cho_solve((L_sigma, True), a)

        K_xastu = self.kernel(X_test, self.U_induce)

        pred = np.dot(K_xastu,self.alpha)

        if self.zero_mean is False:
            pred += self.pol_reg.predict(self.poly.fit_transform(X_test))

        return pred

    def fitting(self):

        self.n = np.shape(self.X_train)[0]

        if self.zero_mean is False:

            self.poly = PolynomialFeatures(degree=2)
            X_poly = self.poly.fit_transform(self.X_train)
            self.pol_reg = LinearRegression()
            k = self.pol_reg.fit(X_poly, self.y_train)
            self.y_train = self.y_train - self.pol_reg.predict(self.poly.fit_transform(self.X_train))

        def obj_func(theta):
            return -self.logmarginallikelihood(theta)

        #optima = [(self.optimization(obj_func, self.kernel.theta, self.kernel.bounds))]
        optima = []

        U_induce = np.matrix.flatten(np.squeeze(np.asarray(self.U_induce)))

        for iteration in range(self.n_restarts_optimizer):
            bounds = self.kernel.bounds
            theta_initial =  [np.random.uniform(bounds[0, 0],bounds[1, 0]), np.random.uniform(bounds[1, 0],bounds[1, 1])]
            for element in U_induce:
                theta_initial.append(element)
                bounds = np.vstack([bounds,[None,None]])
            optima.append(
                self.optimization(obj_func, theta_initial, bounds))
        lml_values = list(map(itemgetter(1), optima))
        self.kernel.theta = optima[np.argmin(lml_values)][0][:2]
        U_induce = optima[np.argmin(lml_values)][0][2:]
        U_induce.resize((np.shape(self.U_induce)[0],np.shape(self.X_train)[1]))
        self.U_induce = U_induce
        self.log_marginal_likelihood_value_ = -np.min(lml_values)

        #K_xx = self.kernel(self.X_train)
        K_ux = self.kernel(self.U_induce, self.X_train)
        K_xu = self.kernel(self.X_train, self.U_induce)
        K_uu = self.kernel(self.U_induce)
        K_uu[np.diag_indices_from(K_uu)] += self.noise_u

        # L_u = self.cholesky_dec(K_uu)
        # Alpha = cho_solve((L_u, True), K_ux)
        # Q_xx = K_xu.dot(Alpha)

        if (self.method == 'sparse_FITC'):
            L_u = self.cholesky_dec(K_uu)
            Alpha = cho_solve((L_u, True), K_ux)

            K_xx = self.kernel.diag(self.X_train)
            Q_xx = np.einsum('ij,ji->i', K_xu, Alpha)
            value = (K_xx - Q_xx) + self.noise
            # Lambd = np.diag(value)
            Lambd_inv = np.diag(1/value)

            # K_xx = np.diag(self.kernel.diag(self.X_train))
            # Q_xx = np.diag(np.einsum('ij,ji->i', K_xu, Alpha))
            #
            # Lambd = K_xx - Q_xx
            # Lambd[np.diag_indices_from(Lambd)] += self.noise
            # Lambd_inv = np.linalg.inv(Lambd)

        if (self.method == 'sparse_VFE'):
            init = np.ones((np.shape(self.X_train)[0]))
            #Lambd_vec = init * self.noise
            inverse_noise = 1/self.noise
            inv_lambd_vec = init * inverse_noise
            #Lambd = np.diag(Lambd_vec)
            Lambd_inv = np.diag(inv_lambd_vec)

            # Lambd = np.eye(np.shape(self.X_train)[0])
            # Lambd[np.diag_indices_from(Lambd)] *= self.noise
            # Lambd_inv = np.linalg.inv(Lambd)

        sigma = K_uu + np.dot(K_ux, np.dot(Lambd_inv,K_xu))
        L_sigma = self.cholesky_dec(sigma)
        y_l = np.dot(Lambd_inv,self.y_train)
        a = np.dot(K_ux,y_l)
        self.alpha = cho_solve((L_sigma, True), a)

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
        kernel.theta = theta[:2]

        U_induce = np.asarray(theta[2:])
        U_induce.resize((np.shape(self.U_induce)[0],np.shape(self.X_train)[1]))

        #K_xx = kernel(self.X_train)
        K_ux = kernel(self.U_induce, self.X_train)
        K_xu = kernel(self.X_train, self.U_induce)
        K_uu = kernel(self.U_induce)

        K_uu[np.diag_indices_from(K_uu)] += self.noise
        y_train = self.y_train

        L_u = self.cholesky_dec(K_uu)
        Alpha = cho_solve((L_u, True), K_ux)

        if (self.method == 'sparse_FITC'):
            K_xx = kernel.diag(self.X_train)
            Q_xx = np.einsum('ij,ji->i', K_xu, Alpha)
            value = (K_xx - Q_xx) + self.noise
            Lambd = np.diag(value)
            Lambd_inv = np.diag(1/value)
            trace = 0

            # K_xx = np.diag(kernel.diag(self.X_train))
            # Q_xx = np.diag(np.einsum('ij,ji->i', K_xu, Alpha))
            #
            # Lambd = K_xx - Q_xx
            # Lambd[np.diag_indices_from(Lambd)] += self.noise
            # Lambd_inv = np.linalg.inv(Lambd)
            #trace = 0

        if (self.method == 'sparse_VFE'):
            # K_xx = np.diag(kernel.diag(self.X_train))
            # Q_xx = K_xu.dot(Alpha)

            K_xx = kernel.diag(self.X_train)
            Q_xx = np.einsum('ij,ji->i', K_xu, Alpha)

            init = np.ones((np.shape(self.X_train)[0]))
            Lambd_vec = init * self.noise
            inverse_noise = 1 / self.noise
            inv_lambd_vec = init * inverse_noise
            Lambd = np.diag(Lambd_vec)
            Lambd_inv = np.diag(inv_lambd_vec)

            # Lambd = np.eye(np.shape(self.X_train)[0])
            # Lambd[np.diag_indices_from(Lambd)] *= self.noise
            # Lambd_inv = np.linalg.inv(Lambd)

            trace_mat = np.sum(K_xx) - np.sum(Q_xx)
            trace = (1 / (2 * self.noise)) * trace_mat

        A = np.linalg.solve(L_u, K_ux)
        A_l = np.dot(A, Lambd_inv)  # d*n
        L2_u = cholesky(np.eye(np.size(self.U_induce,0)) + np.dot(A_l, A.transpose()), lower=True)
        c = np.linalg.solve(L2_u, A_l)
        inv = Lambd_inv - np.dot(c.transpose(), c)
        alpha = np.dot(inv,y_train)

        # Compute log-likelihood (compare line 7)
        log_likelihood = -0.5 * np.einsum("ik,ik->k", y_train, alpha)                      #eerste term
        log_likelihood -= np.log(np.diag(L2_u)).sum()  + 0.5* np.log(np.diag(Lambd)).sum()         #determinant (2*1/2 = 1)
        log_likelihood -= self.n / 2 * np.log(2 * np.pi)                               #cte term

        return log_likelihood - trace


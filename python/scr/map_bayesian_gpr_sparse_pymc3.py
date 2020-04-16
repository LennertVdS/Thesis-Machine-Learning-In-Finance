import pymc3 as pm
import numpy as np
from timeit import default_timer as timer
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.linalg import cholesky, cho_solve

"""
This code executes the training and prediction of VFE using MAP and print its properties  
"""

def map_bayesian_gpr_sparse_pymc3_ex(amountTraining, amount_Inducing, amountTest, trainingValues, trainingParameters, testValues, testParameters):

    inducing_jitter = 0.0001

    valuesFFTCallsTraining = np.squeeze(np.asarray(trainingValues))
    X = np.array(np.asarray(trainingParameters),dtype=np.float64)

    valuesFFTCallsTest = np.squeeze(np.asarray(testValues))
    X_new = np.array(np.asarray(testParameters),dtype=np.float64)

    dimension = X.shape[1]

    startFittingTimer = timer()
    with pm.Model() as model2:

        # Set priors on the hyperparameters of the covariance
        ls1 = pm.Gamma("ls1", alpha=2, beta=2)

        eta = pm.HalfNormal("eta", sigma=2)

        cov = eta * pm.gp.cov.ExpQuad(dimension, ls1)  # cov_x1 must accept X1 without error

        # Specify the GP.  The default mean function is `Zero`.
        gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")

        Xu = pm.gp.util.kmeans_inducing_points(amount_Inducing, X)

        sigma = pm.HalfNormal("sigma", sigma=0.01)         #halfnormal always positive

        # Place a GP prior over the function f.
        gp.marginal_likelihood("y", X=X, Xu=Xu, y=valuesFFTCallsTraining, noise=sigma)

    with model2:
        mp = pm.find_MAP()

    # the package does the following calculations in the prediction, althought it is actually training.
    # this makes predicting in using very slow and not suitable for our data study
    # we write it ourself

    d = [*mp.values()]
    kernel = C(d[4]) * RBF((d[3]))
    K = kernel(X)
    noise = d[5] ** 2
    K[np.diag_indices_from(K)] += noise
    K_uu = kernel(Xu,Xu)
    K_uu[np.diag_indices_from(K_uu)] += inducing_jitter
    K_xu = kernel(X,Xu)
    K_ux = kernel(Xu,X)

    init = np.ones((np.shape(X)[0]))
    inverse_noise = 1 / noise
    inv_lambd_vec = init * inverse_noise
    Lambd_inv = np.diag(inv_lambd_vec)

    sigma = K_uu + np.dot(K_ux, np.dot(Lambd_inv, K_xu))
    L_sigma = cholesky_dec(sigma)
    y_l = np.dot(Lambd_inv, valuesFFTCallsTraining)
    a = np.dot(K_ux, y_l)
    alpha = cho_solve((L_sigma, True), a)


    endFittingTimer = timer()
    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # The prediction code in the package is very slow since it does not
    # fully optimise the training and it calculates the variance (or its trace)
    # We write the prediction ourself

    # startPredictingInSampleTimerGPR = timer()
    # mu, var = gp.predict(X, point=mp, diag=True)
    # endPredictingInSampleTimerGPR = timer()

    startPredictingInSampleTimerGPR = timer()
    for i in range(10):
        K_xastu = kernel(X, Xu)
        pred = np.dot(K_xastu, alpha)
    endPredictinginSampleTimerGPR = timer()

    print('Timer of predicting in sample GPR ' + str((
             endPredictinginSampleTimerGPR - startPredictingInSampleTimerGPR)/10))
    mu = np.squeeze(pred)

    mu = np.maximum(mu, 0)

    AEE = np.sum(np.abs((valuesFFTCallsTraining - mu)))/amountTraining
    MAE = np.max(np.abs((valuesFFTCallsTraining - mu)))


    print('In sample MAE ' + str(MAE))
    print('In sample AEE ' + str(AEE))

    # The prediction code in the package is very slow since it does not
    # fully optimise the training and it calculates the variance (or its trace)
    # We write the prediction ourself.

    # startPredictingOutSampleTimerGPR = timer()
    # mu, var = gp.predict(X_new, point=mp, diag=True)
    # endPredictingOutSampleTimerGPR = timer()
    # print('Timer of predicting out sample GPR ' + str(
    #     endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR))

    startPredictingOutSampleTimerGPR = timer()
    for i in range(10):
        K_xastu = kernel(X_new, Xu)
        pred = np.dot(K_xastu, alpha)
    endPredictingOutSampleTimerGPR = timer()

    print('Timer of predicting out sample GPR ' + str((
             endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR)/10))

    mu = np.squeeze(pred)

    mu = np.maximum(mu, 0)

    AEE = np.sum(np.abs((valuesFFTCallsTest - mu)))/amountTest
    MAE = np.max(np.abs((valuesFFTCallsTest - mu)))

    print('In sample MAE ' + str(MAE))
    print('In sample AEE ' + str(AEE))



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
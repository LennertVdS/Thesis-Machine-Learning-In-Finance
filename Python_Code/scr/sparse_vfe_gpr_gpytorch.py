import torch
import gpytorch
import numpy as np
from timeit import default_timer as timer
from scr import sparse_vfe_gpytorch_code
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




"""
This code calculates the fitting and predicting properties of VFE using the BBMM framework
"""

def sparse_vfe_gpr_pytorch_ex(amountTraining, amountInducing, amountTest, trainingValues,trainingParameters,testValues,testParameters):

    inducing_jitter =0.001

    # Inducing values

    startFittingTimer = timer()
    kmeans  = KMeans(n_clusters=amountInducing, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(trainingParameters)
    parametersModelsInducing = kmeans.cluster_centers_
    endFittingTimer = timer()
    print('Timer of kmeans ' + str(endFittingTimer - startFittingTimer))

    train_u = torch.Tensor(parametersModelsInducing)

    trainingParameters = np.array(trainingParameters.astype(np.float32))
    train_x = torch.Tensor(trainingParameters)

    trainingValues = np.array(trainingValues.astype(np.float32))

    train_y = torch.Tensor(trainingValues)

    testParameters = np.array(testParameters.astype(np.float32))

    testValues = np.array(testValues.astype(np.float32))

    # Instantiate a Gaussian Process model

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = sparse_vfe_gpytorch_code.GPRegressionModel(train_x, train_y, train_u, likelihood)


    startFittingTimer = timer()

    model, likelihood = sparse_vfe_gpytorch_code.training(model, likelihood, train_x, train_y, amount_iterations = 40)
    alpha, kernel = sparse_vfe_gpytorch_code.training_extra(model,trainingParameters,parametersModelsInducing,trainingValues,inducing_jitter)

    endFittingTimer = timer()

    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # In sample prediction

    startPredictingInSampleTimerGPR = timer()
    for i in range(10):
        y_pred = sparse_vfe_gpytorch_code.prediction(model, likelihood,trainingParameters,parametersModelsInducing,alpha,kernel)
    endPredictingInSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting in sample with GPR ' + str((endPredictingInSampleTimerGPR - startPredictingInSampleTimerGPR)/10))

    MAE = np.max(np.abs(((trainingValues).transpose() - y_pred)))
    AEE = np.sum(np.abs((trainingValues).transpose() - y_pred)) / amountTraining

    print('In sample MAE ' + str(MAE))
    print('In sample AEE ' + str(AEE))

    # Out of sample prediction

    startPredictingOutSampleTimerGPR = timer()
    for i in range(10):
        y_pred =  sparse_vfe_gpytorch_code.prediction(model, likelihood, testParameters,parametersModelsInducing,alpha,kernel)
    endPredictingOutSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred, 0)


    print('Timer of predicting out sample GPR ' + str((endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR)/10))

    MAE = np.max(np.abs(((testValues).transpose() - y_pred)))
    AEE = np.sum(np.abs(((testValues).transpose() - y_pred))) / amountTest

    print('Out of sample MAE ' + str(MAE))
    print('Out of sample AEE ' + str(AEE))

    startPredictingDerivative = timer()
    for i in range(10):
        derivatives = sparse_vfe_gpytorch_code.derivative(model, testParameters, 5, parametersModelsInducing, kernel, alpha)
    endPredictingDerivative = timer()

    print('Timer finding derivatives ' + str((endPredictingDerivative - startPredictingDerivative)/10))


    # fig = plt.figure(figsize=(12, 5))
    # ax = fig.gca()
    # ax.set_title('BBMM VFE fit')
    # ax.set_xlabel('Strike')
    # ax.set_ylabel('Price')
    # ax.plot(testParameters[:,5],y_pred,label = "BBMM VFE fit")
    # ax.plot(testParameters[:,5],testValues.transpose(),'bo', label = "Data Points")
    # legend = ax.legend(loc='upper right', shadow=True, prop={'size': 10},
    #            ncol=4)
    # plt.show()
    #
    #
    # fig = plt.figure(figsize=(12, 5))
    # ax = fig.gca()
    # ax.set_title('Derivative')
    # ax.set_xlabel('Strike')
    # ax.set_ylabel('Derivative Option Price Towards Strike')
    # plt.plot(testParameters[:,5],derivatives,label = "BBMM VFE Derivative")
    # plt.plot(testParameters[0:99,5],np.diff(np.squeeze(y_pred))*120,label = "Finite Differences")
    # legend = ax.legend(loc='upper left', shadow=True, prop={'size': 10},
    #            ncol=4)
    # plt.show()



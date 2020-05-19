import torch
import gpytorch
import numpy as np
from timeit import default_timer as timer
from scr import standard_gpr_gpytorch_code
import matplotlib.pyplot as plt


"""
This code calculates the fitting and predicting properties of a standard GPR using the BBMM framework
"""

def standard_gpr_pytorch_ex(amountTraining,amountTest,trainingValues,trainingParameters,testValues,testParameters):


    trainingParameters = np.array(trainingParameters.astype(np.float32))
    train_x = torch.Tensor(trainingParameters)

    trainingValues = np.array(trainingValues.astype(np.float32))
    train_y = torch.Tensor(trainingValues)


    testParameters = np.array(testParameters.astype(np.float32))

    testValues = np.array(testValues.astype(np.float32))

    # Instantiate a Gaussian Process model

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = standard_gpr_gpytorch_code.ExactGPModel(train_x, train_y, likelihood)

    # The proposed code approximates the predictions and is thus not suitable for our datastudy
    # it also calculates the variance in which we are not interested
    # we write predictions ourself

    startFittingTimer = timer()

    model, likelihood = standard_gpr_gpytorch_code.training(model, likelihood, train_x, train_y, training_iter = 30)
    alpha, kernel = standard_gpr_gpytorch_code.training_extra(model,trainingParameters,trainingValues)

    endFittingTimer = timer()

    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # In sample prediction

    startPredictingInSampleTimerGPR = timer()
    for i in range(10):
        y_pred = standard_gpr_gpytorch_code.prediction(model, likelihood, trainingParameters,trainingParameters,alpha,kernel)
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
        y_pred =  standard_gpr_gpytorch_code.prediction(model, likelihood, testParameters,trainingParameters,alpha,kernel)
    endPredictingOutSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting out sample GPR ' + str((endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR)/10))

    MAE = np.max(np.abs(((testValues).transpose() - y_pred)))
    AEE = np.sum(np.abs(((testValues).transpose() - y_pred))) / amountTest


    print('Out of sample MAE ' + str(MAE))
    print('Out of sample AEE ' + str(AEE))


    # fig = plt.figure(figsize=(12, 5))
    # ax = fig.gca()
    # ax.set_title('Error Histogram GPR BBMM')
    # ax.set_xlabel('Error Values')
    # ax.set_ylabel('Amount')
    # plt.hist((testValues).transpose() - y_pred, bins='auto')
    # plt.show()

    startPredictingDerivative = timer()
    for i in range(10):
        derivatives = standard_gpr_gpytorch_code.derivative(model, testParameters, 7, trainingParameters, kernel, alpha)
    endPredictingDerivative = timer()

    print('Timer finding derivatives ' + str((endPredictingDerivative - startPredictingDerivative)/10))


    fig = plt.figure(figsize=(12, 5))
    ax = fig.gca()
    ax.set_title('BBMM GPR fit')
    ax.set_xlabel('Time To Maturity')
    ax.set_ylabel('Price')
    ax.plot(testParameters[:,7],y_pred,label = "BBMM GPR fit")
    ax.plot(testParameters[:,7],testValues.transpose(),'bo', label = "Data Points")
    legend = ax.legend(loc='upper left', shadow=True, prop={'size': 10},
               ncol=4)
    plt.show()


    fig = plt.figure(figsize=(12, 5))
    ax = fig.gca()
    ax.set_title('Derivative')
    ax.set_xlabel('Time To Maturity')
    ax.set_ylabel('Derivative Option Price Towards Maturity')
    plt.plot(testParameters[:,7],derivatives, label = "BBMM GPR Derivative")
    legend = ax.legend(loc='upper right', shadow=True, prop={'size': 10},
               ncol=4)
    plt.show()





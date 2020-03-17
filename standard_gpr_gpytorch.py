import torch
import gpytorch
import numpy as np
from timeit import default_timer as timer
from scr import standard_gpr_gpytorch_code


from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.linalg import cholesky, cho_solve



"""
This code calculates the fitting and predicting properties of a standard GPR using the BBMM framework
"""

def standard_gpr_pytorch_ex(amountTraining,amountTest,trainingValues,trainingParameters,testValues,testParameters):


    trainingParameters = np.array(trainingParameters.astype(np.float32))
    train_x = torch.Tensor(trainingParameters)

    trainingValues = np.array(trainingValues.astype(np.float32))
    train_y = torch.Tensor(trainingValues)


    testParameters = np.array(testParameters.astype(np.float32))
    test_x = torch.Tensor(testParameters)

    testValues = np.array(testValues.astype(np.float32))

    # Instantiate a Gaussian Process model

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = standard_gpr_gpytorch_code.ExactGPModel(train_x, train_y, likelihood)

    # The proposed code approximates the predictions and is thus not suitable for our datastudy
    # it also calculates the variance in which we are not interested
    # we write predictions ourself

    startFittingTimer = timer()

    model, likelihood = standard_gpr_gpytorch_code.training(model, likelihood, train_x, train_y, 30)
    alpha, kernel = standard_gpr_gpytorch_code.training_extra(model,trainingParameters,trainingValues)

    endFittingTimer = timer()

    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # In sample prediction

    #startPredictingInSampleTimerGPR = timer()
    #y_pred = standard_gpr_gpytorch_code.prediction(model, likelihood, train_x,train_x,alpha,kernel)
    #endPredictingInSampleTimerGPR = timer()

    #y_pred = y_pred.numpy()

    startPredictingInSampleTimerGPR = timer()
    y_pred = standard_gpr_gpytorch_code.prediction(model, likelihood, trainingParameters,trainingParameters,alpha,kernel)
    endPredictingInSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting in sample with GPR ' + str(endPredictingInSampleTimerGPR - startPredictingInSampleTimerGPR))

    # MAE = np.max(np.abs(((np.squeeze(trainingValues)).transpose() - y_pred)))
    # AEE = np.sum(np.abs(((np.squeeze(trainingValues)).transpose() - y_pred))) / amountTraining

    MAE = np.max(np.abs(((trainingValues).transpose() - y_pred)))
    AEE = np.sum(np.abs((trainingValues).transpose() - y_pred)) / amountTraining

    print('In sample MAE ' + str(MAE))
    print('In sample AEE ' + str(AEE))

    # Out of sample prediction

    # startPredictingOutSampleTimerGPR = timer()
    # y_pred =  standard_gpr_gpytorch_code.prediction(model, likelihood, test_x,train_x,alpha,kernel)
    # endPredictingOutSampleTimerGPR = timer()

    #y_pred = y_pred.numpy()

    startPredictingOutSampleTimerGPR = timer()
    y_pred =  standard_gpr_gpytorch_code.prediction(model, likelihood, testParameters,trainingParameters,alpha,kernel)
    endPredictingOutSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting out sample GPR ' + str(endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR))

    # MAE = np.max(np.abs(((np.squeeze(testValues)).transpose() - y_pred)))
    # AEE = np.sum(np.abs(((np.squeeze(testValues)).transpose() - y_pred))) / amountTest

    MAE = np.max(np.abs(((testValues).transpose() - y_pred)))
    AEE = np.sum(np.abs(((testValues).transpose() - y_pred))) / amountTest


    print('Out of sample MAE ' + str(MAE))
    print('Out of sample AEE ' + str(AEE))



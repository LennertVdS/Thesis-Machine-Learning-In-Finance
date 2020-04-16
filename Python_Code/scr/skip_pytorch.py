import torch
import gpytorch
import numpy as np
from timeit import default_timer as timer
from scr import skip_pytorch_code

"""
This code calculates the fitting and predicting properties of SKIP  
"""

def skip_torch_ex(amountTraining, amountInducing, amountTest, trainingValues,trainingParameters,testValues,testParameters):


    trainingParameters = np.array(trainingParameters.astype(np.float32))
    train_x = torch.Tensor(trainingParameters)

    trainingValues = np.array(trainingValues.astype(np.float32))
    train_y = torch.Tensor(trainingValues)

    testParameters = np.array(testParameters.astype(np.float32))
    test_x = torch.Tensor(testParameters)

    testValues = np.array(testValues.astype(np.float32))

    # Instantiate a Gaussian Process model

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = skip_pytorch_code.GPRegressionModel(train_x, train_y, likelihood, amountInducing)

    startFittingTimer = timer()
    model, likelihood = skip_pytorch_code.training(model, likelihood, train_x, train_y, training_iterations=100)
    endFittingTimer = timer()

    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # In sample prediction

    startPredictingInSampleTimerGPR = timer()
    for i in range(10):
        y_pred = skip_pytorch_code.prediction(model, likelihood, train_x)
    endPredictingInSampleTimerGPR = timer()

    y_pred = y_pred.numpy()
    y_pred = np.maximum(y_pred, 0)

    print('Timer of predicting in sample with GPR ' + str(
        (endPredictingInSampleTimerGPR - startPredictingInSampleTimerGPR)/10))

    MAE = np.max(np.abs(((np.squeeze(trainingValues)).transpose() - y_pred)))
    AEE = np.sum(np.abs(((np.squeeze(trainingValues)).transpose() - y_pred))) / amountTraining

    print('In sample MAE ' + str(MAE))
    print('In sample AEE ' + str(AEE))

    # Out of sample prediction

    startPredictingOutSampleTimerGPR = timer()
    for i in range(10):
        y_pred = skip_pytorch_code.prediction(model, likelihood, test_x)
    endPredictingOutSampleTimerGPR = timer()

    y_pred = y_pred.numpy()
    y_pred = np.maximum(y_pred, 0)

    print(
        'Timer of predicting out sample GPR ' + str((endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR)/10))

    MAE = np.max(np.abs(((np.squeeze(testValues)).transpose() - y_pred)))
    AEE = np.sum(np.abs(((np.squeeze(testValues)).transpose() - y_pred))) / amountTest

    print('Out of sample MAE ' + str(MAE))
    print('Out of sample AEE ' + str(AEE))
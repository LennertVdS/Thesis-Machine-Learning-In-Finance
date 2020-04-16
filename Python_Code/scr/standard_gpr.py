import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from timeit import default_timer as timer
from scr import standard_gpr_code
import matplotlib.pyplot as plt


"""
This code calculates the fitting and predicting properties of a standard GPR
"""

def standard_gpr_ex(amountTraining,amountTest,trainingValues,trainingParameters,testValues,testParameters):

    noise = 0.000001

    # Instantiate a Gaussian Process model

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = standard_gpr_code.gaussianprocessregression(kernel, noise, trainingParameters, trainingValues.transpose(), zero_mean= True)
    startFittingTimer = timer()
    gp.fitting()
    endFittingTimer = timer()

    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # In sample prediction

    startPredictingInSampleTimerGPR = timer()
    for i in range(10):
        y_pred = gp.prediction(trainingParameters)
    endPredictingInSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting in sample with GPR ' + str((endPredictingInSampleTimerGPR - startPredictingInSampleTimerGPR)/10))

    MAE = (trainingValues.transpose() - y_pred).abs().max()
    AEE = (trainingValues.transpose() - y_pred).abs().sum() / amountTraining

    print('In sample MAE ' + str(MAE.to_numpy()))
    print('In sample AEE ' + str(AEE.to_numpy()))

    # Out of sample prediction

    startPredictingOutSampleTimerGPR = timer()
    for i in range(10):
        y_pred = gp.prediction(testParameters)
    endPredictingOutSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting out sample GPR ' + str((endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR)/10))

    MAE = (testValues.transpose() - y_pred).abs().max()
    AEE = (testValues.transpose() - y_pred).abs().sum() / amountTest


    print('Out of sample MAE ' + str(MAE.to_numpy()))
    print('Out of sample AEE ' + str(AEE.to_numpy()))

    # fig = plt.figure(figsize=(12, 5))
    # ax = fig.gca()
    # ax.set_title('Error Histogram GPR')
    # ax.set_xlabel('Error Values')
    # ax.set_ylabel('Amount')
    # plt.hist((testValues.transpose() - y_pred).transpose(), bins='auto')
    # plt.show()


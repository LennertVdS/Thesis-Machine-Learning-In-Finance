import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from timeit import default_timer as timer
from scr import standard_gpr_code


"""
This code calculates the fitting and predicting properties of a standard GPR
"""

def standard_gpr_ex(amountTraining,amountTest,trainingValues,trainingParameters,testValues,testParameters):


    # Instantiate a Gaussian Process model

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = standard_gpr_code.gaussianprocessregression(kernel, 0.000001, trainingParameters, trainingValues.transpose(), True)    #clean data
    #gp = standard_gpr_code.gaussianprocessregression(kernel, 0.0001, trainingParameters, trainingValues.transpose(), True)        #noisy data
    startFittingTimer = timer()
    gp.fitting()
    endFittingTimer = timer()

    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # In sample prediction

    startPredictingInSampleTimerGPR = timer()
    y_pred = gp.prediction(trainingParameters)
    endPredictingInSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting in sample with GPR ' + str(endPredictingInSampleTimerGPR - startPredictingInSampleTimerGPR))

    MAE = (trainingValues.transpose() - y_pred).abs().max()
    AEE = (trainingValues.transpose() - y_pred).abs().sum() / amountTraining

    print('In sample MAE ' + str(MAE.to_numpy()))
    print('In sample AEE ' + str(AEE.to_numpy()))

    # Out of sample prediction

    startPredictingOutSampleTimerGPR = timer()
    y_pred = gp.prediction(testParameters)
    endPredictingOutSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred,0)

    print('Timer of predicting out sample GPR ' + str(endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR))

    MAE = (testValues.transpose() - y_pred).abs().max()
    AEE = (testValues.transpose() - y_pred).abs().sum() / amountTest

    print('Out of sample MAE ' + str(MAE.to_numpy()))
    print('Out of sample AEE ' + str(AEE.to_numpy()))



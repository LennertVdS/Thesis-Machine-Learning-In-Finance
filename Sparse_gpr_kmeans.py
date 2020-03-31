import numpy as np
from sklearn.cluster import KMeans
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scr import sparse_gpr_kmeans_code
from timeit import default_timer as timer

"""
This code calculates the fitting and predicting properties of FITC and VFE using kmeans
"""

def sparse_gpr_kmeans_ex(amountTraining, amountInducing, amountTest, method,trainingValues,trainingParameters,testValues,testParameters):

    # inducing values

    startFittingTimer = timer()
    kmeans  = KMeans(n_clusters=amountInducing, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(trainingParameters)
    parametersModelsInducing = kmeans.cluster_centers_
    endFittingTimer = timer()
    print('Timer of kmeans ' + str(endFittingTimer - startFittingTimer))


    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    gp = sparse_gpr_kmeans_code.gaussianprocessregression(kernel, 0.000001, 0.00001, trainingParameters, parametersModelsInducing,
                                                          trainingValues.transpose(), method, True)

    startFittingTimer = timer()
    gp.fitting()
    endFittingTimer = timer()

    print('Timer of fitting in sample ' + str(endFittingTimer - startFittingTimer))

    # In sample prediction

    startPredictingInSampleTimerGPR = timer()
    y_pred = gp.prediction(trainingParameters)
    endPredictingInSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred, 0)

    print('Timer of predicting in sample with GPR ' + str(
        endPredictingInSampleTimerGPR - startPredictingInSampleTimerGPR))

    MAE = (trainingValues.transpose() - y_pred).abs().max()
    AEE = (trainingValues.transpose() - y_pred).abs().sum() / amountTraining

    print('In sample MAE ' + str(MAE.to_numpy()))
    print('In sample AEE ' + str(AEE.to_numpy()))


    # Out of sample prediction

    startPredictingOutSampleTimerGPR = timer()
    y_pred = gp.prediction(testParameters)
    endPredictingOutSampleTimerGPR = timer()

    y_pred = np.maximum(y_pred, 0)


    print('Timer of predicting out sample GPR ' + str(endPredictingOutSampleTimerGPR - startPredictingOutSampleTimerGPR))

    MAE = np.max(np.abs((testValues.transpose() - y_pred)))
    AAE = np.sum(np.abs((testValues.transpose() - y_pred))) / amountTest

    print('Out of sample MAE ' + str(MAE.to_numpy()))
    print('Out of sample AEE ' + str(AAE.to_numpy()))


def method_finder(amounttraining, amountinducing, amounttest, method, trainingValues,trainingParameters,testValues,testParameters):
    if method.find('sparse_kmeans_FITC') != -1:
        sparse_gpr_kmeans_ex(amounttraining, amountinducing, amounttest, 'sparse_kmeans_FITC', trainingValues, trainingParameters,
                             testValues, testParameters)
    if method.find('sparse_kmeans_VFE') != -1:
        sparse_gpr_kmeans_ex(amounttraining, amountinducing, amounttest, 'sparse_kmeans_VFE', trainingValues, trainingParameters,
                             testValues, testParameters)

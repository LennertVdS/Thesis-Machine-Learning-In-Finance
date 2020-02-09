import numpy as np
from sklearn.cluster import KMeans
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scr import sparse_gpr_newtry_kmeans
from scr import data_generator
from timeit import default_timer as timer


def sparse_gpr_kmeans_ex(amountTraining, amountInducing, amountTest, model, type, method):

    # Generate data

    if model == 'heston':
        if type == 'vanilla_call' or type == 'vanilla_put' :
            trainingValues , trainingParameters = \
                data_generator.data_generators_heston.training_data_heston_vanillas(amountTraining, type)
        if type == 'DOBP':
            trainingValues , trainingParameters = data_generator.data_generators_heston.training_data_heston_down_and_out(amountTraining)
    if type == 'american_call' or type == 'american_put' :
        trainingValues, trainingParameters = data_generator.data_generators_american.training_data_american(amountTraining,type)

    if model == 'heston':
        if type == 'vanilla_call' or type == 'vanilla_put' :
            testValues , testParameters = data_generator.data_generators_heston.test_data_heston_vanillas(amountTest, type)
        if type == 'DOBP':
            testValues, testParameters = data_generator.data_generators_heston.test_data_heston_down_and_out(amountTest)
    if type == 'american_call' or type == 'american_put' :
        testValues, testParameters = data_generator.data_generators_american.test_data_american(amountTest, type)

    print('Generating data done')

    # inducing values

    startFittingTimer = timer()
    kmeans  = KMeans(n_clusters=amountInducing, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(trainingParameters)
    parametersModelsInducing = kmeans.cluster_centers_
    endFittingTimer = timer()
    print('Timer of kmeans ' + str(endFittingTimer - startFittingTimer))


    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    gp = sparse_gpr_newtry_kmeans.gaussianprocessregression(kernel, 0.0000001, 0.000001, trainingParameters, parametersModelsInducing,
                                                            trainingValues.transpose(),method,True)

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

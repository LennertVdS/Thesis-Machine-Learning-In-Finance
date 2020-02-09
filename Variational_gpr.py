import numpy as np
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)

tf.disable_v2_behavior()
tfk = tfp.positive_semidefinite_kernels
tfd = tfp.distributions
from scr import data_generator
from scr import Variational_gpr_newtry
from timeit import default_timer as timer
import pymc3 as pm

def variational_gpr_ex(amountTraining, amount_Inducing, amountTest, model, type):

    # We'll use double precision throughout for better numerics.
    dtype = np.float64

    # Generate data
    if model == 'heston':
        if type == 'vanilla_call' or type == 'vanilla_put':
            trainingValues, trainingParameters = \
                data_generator.data_generators_heston.training_data_heston_vanillas(amountTraining, type)
        if type == 'DOBP':
            trainingValues, trainingParameters = data_generator.data_generators_heston.training_data_heston_down_and_out(
                amountTraining)
    if type == 'american_call' or type == 'american_put':
        trainingValues, trainingParameters = data_generator.data_generators_american.training_data_american(
            amountTraining, type)

    valuesTraining = np.squeeze(np.asarray(trainingValues))
    X = np.array(np.squeeze(np.asarray(trainingParameters)),dtype=np.float64)
    index_points = np.squeeze(X[..., np.newaxis])

    if model == 'heston':
        if type == 'vanilla_call' or type == 'vanilla_put' :
            testValues , testParameters = data_generator.data_generators_heston.test_data_heston_vanillas(amountTest, type)
        if type == 'DOBP':
            testValues, testParameters = data_generator.data_generators_heston.test_data_heston_down_and_out(amountTest)
    if type == 'american_call' or type == 'american_put' :
        testValues, testParameters = data_generator.data_generators_american.test_data_american(amountTest, type)

    valuesTest = np.squeeze(np.asarray(testValues))
    X_test = np.array(np.squeeze(np.asarray(testParameters)),dtype=np.float64)
    index_points_test = np.squeeze(X_test[..., np.newaxis])

    print('Generating data done')


    # Inducing values
    startFittingTimer = timer()
    inducing_index_points = tf.Variable(np.squeeze((pm.gp.util.kmeans_inducing_points(amount_Inducing, X)[..., np.newaxis])),
                                        dtype=dtype, name='inducing_index_points', use_resource=True)
    endFittingTimer = timer()
    print('Timer of kmeans ' + str(endFittingTimer - startFittingTimer))

    gp = Variational_gpr_newtry.Variational(amountTraining,0.00001,inducing_index_points,X,valuesTraining,index_points)

    startFittingTimer = timer()
    gp.fitting()
    endFittingTimer = timer()
    print('Timer of fitting in Sample ' + str(endFittingTimer - startFittingTimer))

    startFittingTimer = timer()
    mean_ = gp.prediction(X)
    endFittingTimer = timer()
    print('Timer of predicting in Sample ' + str(endFittingTimer - startFittingTimer))

    mean_ = np.maximum(mean_, 0)

    MAE = np.max(np.abs((valuesTraining - mean_)))
    AAE = np.sum(np.abs((valuesTraining - mean_)))/amountTraining

    print('In sample MAE ' + str(MAE))
    print('In sample AAE ' + str(AAE))



    # Out of sample prediction

    startFittingTimer = timer()
    mean_ = gp.prediction(X_test)
    endFittingTimer = timer()
    print('Timer of predicting out of Sample ' + str(endFittingTimer - startFittingTimer))
    mean_ = np.maximum(mean_, 0)

    MAE = np.max(np.abs((valuesTest - mean_)))
    AAE = np.sum(np.abs((valuesTest - mean_)))/amountTest

    print('Out of sample MAE ' + str(MAE))
    print('Out of sample AEE ' + str(AAE))
import numpy as np
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf

from scr import Variational_gpr_tensorflow_code
from timeit import default_timer as timer
import pymc3 as pm

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
tfk = tfp.positive_semidefinite_kernels
tfd = tfp.distributions

"""
This code calculates the fitting and predicting properties of SVG 
"""

def variational_gpr_ex(amountTraining, amount_Inducing, amountTest,trainingValues,trainingParameters,testValues,testParameters):

    # We'll use double precision throughout for better numerics.
    dtype = np.float64


    valuesTraining = np.squeeze(np.asarray(trainingValues))
    X = np.array(np.squeeze(np.asarray(trainingParameters)),dtype=np.float64)
    index_points = np.squeeze(X[..., np.newaxis])

    valuesTest = np.squeeze(np.asarray(testValues))
    X_test = np.array(np.squeeze(np.asarray(testParameters)),dtype=np.float64)
    index_points_test = np.squeeze(X_test[..., np.newaxis])


    # Inducing values
    startFittingTimer = timer()
    inducing_index_points = tf.Variable(np.squeeze((pm.gp.util.kmeans_inducing_points(amount_Inducing, X)[..., np.newaxis])),
                                        dtype=dtype, name='inducing_index_points', use_resource=True)
    endFittingTimer = timer()
    print('Timer of kmeans ' + str(endFittingTimer - startFittingTimer))

    gp = Variational_gpr_tensorflow_code.Variational(amountTraining, 0.0001, inducing_index_points, X, valuesTraining, index_points, True)

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
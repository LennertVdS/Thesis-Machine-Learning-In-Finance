
import numpy as np
#import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tfk = tfp.positive_semidefinite_kernels
tfd = tfp.distributions

import pandas as pd
import pymc3 as pm

from scr import models_algorithms

# We'll use double precision throughout for better numerics.
dtype = np.float64
# We'll use double precision throughout for better numerics.
dtype = np.float64
# set the seed

np.random.seed(1)

amountTraining = 1500
modelListTraining = []
valuesFFTCallsTraining = pd.DataFrame(index=range(1), columns=range(amountTraining))
parametersModelsTraining = pd.DataFrame(index=range(amountTraining), columns=range(10))

for x in range(amountTraining):

    # generate pseudo-random numbers for the Training set

    stock_value = 10
    strike = np.random.uniform(0.4,1.6)*stock_value
    maturity = np.random.uniform(11/12, 1)
    interest = np.random.uniform(0.015,0.025)
    dividend_yield = np.random.uniform(0,0.05)

    # heston

    kappa = np.random.uniform(1.4, 2.6)
    rho = np.random.uniform(-0.85, -0.55)
    theta = np.random.uniform(0.45, 0.75)
    eta = np.random.uniform(0.01, 0.1)
    sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.9048)))

    modelListTraining.append(models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield))

for i, model in enumerate(modelListTraining):
    valuesFFTCallsTraining[i] = model.heston_carr_madan(0)
    for j, parameter in enumerate(model.get_parameters()):
        parametersModelsTraining.iat[i, j] = parameter

valuesFFTCallsTraining = np.squeeze(np.asarray(valuesFFTCallsTraining))
X = np.array(np.squeeze(np.asarray(parametersModelsTraining)),dtype=np.float64)
X = X[:,(0,1,2,3,4,5,6,8,9)]


amplitude = (tf.math.softplus(                                              #Computes softplus: log(ex
  tf.Variable(.54, dtype=dtype, name='amplitude', use_resource=True)))     #variable with initial value
length_scale = (
  1e-5 +
  tf.math.softplus(
    tf.Variable(.54, dtype=dtype, name='length_scale', use_resource=True)))
kernel = tfk.ExponentiatedQuadratic(
    amplitude=amplitude,
    length_scale=length_scale)

observation_noise_variance = tf.math.softplus(
    tf.Variable(
      .54, dtype=dtype, name='observation_noise_variance', use_resource=True))


#Create trainable inducing point locations and variational parameters.
# num_inducing_points_ = 500
# inducing_index_points = tf.Variable(
#     np.squeeze(np.linspace((1.4,0.01,0.45,-0.85,0.1,0.4*stock_value,11/12,0.015,0),
#                 (2.6,0.1,0.75,-0.55,0.316228,1.6*stock_value,1,0.025,0.05), num_inducing_points_)[..., np.newaxis]),
#     dtype=dtype, name='inducing_index_points', use_resource=True)

num_inducing_points_ = 500
inducing_index_points = tf.Variable(np.squeeze((pm.gp.util.kmeans_inducing_points(num_inducing_points_, X)[..., np.newaxis])),
dtype=dtype, name='inducing_index_points', use_resource=True)


variational_loc, variational_scale = (
    tfd.VariationalGaussianProcess.optimal_variational_posterior(
        kernel=kernel,
        inducing_index_points=inducing_index_points,
        observation_index_points=X,
        observations=valuesFFTCallsTraining,
        observation_noise_variance=observation_noise_variance))


amountTest = 100
modelListTest = []
valuesFFTCallsTest = pd.DataFrame(index=range(1), columns=range(amountTest))
parametersModelsTest = pd.DataFrame(index=range(amountTest),columns=range(10))

for x in range(amountTest):

    # generate pseudo-random numbers for the Training set

    strike = np.random.uniform(0.4,1.6)*stock_value
    maturity = np.random.uniform(11/12, 1)
    interest = np.random.uniform(0.015,0.025)
    dividend_yield = np.random.uniform(0,0.05)

    # heston

    kappa = np.random.uniform(1.4, 2.6)
    rho = np.random.uniform(-0.85, -0.55)
    theta = np.random.uniform(0.45, 0.75)
    eta = np.random.uniform(0.01, 0.1)
    sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.9048)))

    modelListTest.append(models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield))

for i, Model in enumerate(modelListTest):
    for j, parameter in enumerate(Model.get_parameters()):
        parametersModelsTest.iat[i, j] = parameter

for i, Model in enumerate(modelListTest):
    valuesFFTCallsTest[i] = Model.heston_carr_madan(0)

valuesFFTCallsTest = np.squeeze(np.asarray(valuesFFTCallsTest))
X_test = np.array(np.squeeze(np.asarray(parametersModelsTest)),dtype=np.float64)
X_test = X_test[:,(0,1,2,3,4,5,6,8,9)]

index_points_ = np.squeeze(X_test[..., np.newaxis])


vgp = tfd.VariationalGaussianProcess(
    kernel,
    index_points=index_points_,
    inducing_index_points=inducing_index_points,
    variational_inducing_observations_loc=variational_loc,
    variational_inducing_observations_scale=variational_scale,
    observation_noise_variance=observation_noise_variance)

# For training, we use some simplistic numpy-based minibatching.
batch_size = 64
x_train_batch = tf.placeholder(dtype, [batch_size, 9], name='x_train_batch')
y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

# Create the loss function we want to optimize.
loss = vgp.variational_loss(
    observations=y_train_batch,
    observation_index_points=x_train_batch,
    kl_weight=float(batch_size) / float(amountTraining))

optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train_op = optimizer.minimize(loss)

num_iters = 150
num_logs = 10
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(num_iters):
    batch_idxs = np.random.randint(amountTraining, size=[batch_size])
    x_train_batch_ = X[batch_idxs, ...]
    y_train_batch_ = valuesFFTCallsTraining[batch_idxs]

    [_, loss_] = sess.run([train_op, loss],
                          feed_dict={x_train_batch: x_train_batch_,
                                     y_train_batch: y_train_batch_})
    if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
      print(i, loss_)


  num_samples = 1
  [
      samples_,
      mean_,
      inducing_index_points_,
      variational_loc_,
  ] = sess.run([
      vgp.sample(num_samples),
      vgp.mean(),
      inducing_index_points,
      variational_loc
  ])

print(inducing_index_points_)


MAE = np.sum(np.abs((valuesFFTCallsTest - mean_)))/amountTest
print(MAE)
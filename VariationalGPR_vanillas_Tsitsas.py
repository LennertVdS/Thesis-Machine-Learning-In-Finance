
import numpy as np
#import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tfk = tfp.positive_semidefinite_kernels
tfd = tfp.distributions
import pymc3 as pm

import matplotlib.pyplot as plt
# We'll use double precision throughout for better numerics.
dtype = np.float64
# We'll use double precision throughout for better numerics.
dtype = np.float64

# Generate noisy data from a known function.
f = lambda x: np.exp(-x[..., 0]**2 / 20.) * np.sin(1. * x[..., 0])
true_observation_noise_variance_ = dtype(1e-1) ** 2

num_training_points_ = 1000
x_train_ = np.random.uniform(-10., 10., [num_training_points_, 1])
y_train_ = (f(x_train_) +
            np.random.normal(
                0., np.sqrt(true_observation_noise_variance_),
                [num_training_points_]))

# Create kernel with trainable parameters, and trainable observation noise
# variance variable. Each of these is constrained to be positive.
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

# Create trainable inducing point locations and variational parameters.
num_inducing_points_ = 10

inducing_index_points = tf.Variable(
    np.linspace(-10., 10., num_inducing_points_)[..., np.newaxis],
    dtype=dtype, name='inducing_index_points', use_resource=True)

variational_loc, variational_scale = (
    tfd.VariationalGaussianProcess.optimal_variational_posterior(
        kernel=kernel,
        inducing_index_points=inducing_index_points,
        observation_index_points=x_train_,
        observations=y_train_,
        observation_noise_variance=observation_noise_variance))

# These are the index point locations over which we'll construct the
# (approximate) posterior predictive distribution.
num_predictive_index_points_ = 50
index_points_ = np.linspace(-13, 13,
                            num_predictive_index_points_,
                            dtype=dtype)[..., np.newaxis]

# Construct our variational GP Distribution instance.
vgp = tfd.VariationalGaussianProcess(
    kernel,
    index_points=index_points_,
    inducing_index_points=inducing_index_points,
    variational_inducing_observations_loc=variational_loc,
    variational_inducing_observations_scale=variational_scale,
    observation_noise_variance=observation_noise_variance)

# For training, we use some simplistic numpy-based minibatching.
batch_size = 64
x_train_batch = tf.placeholder(dtype, [batch_size, 1], name='x_train_batch')
y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

# Create the loss function we want to optimize.
loss = vgp.optimal_variational_posterior(
    kernel=kernel,
    inducing_index_points=inducing_index_points,
    observation_index_points=x_train_,
    observations=y_train_,
    observation_noise_variance=observation_noise_variance)

optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train_op = optimizer.minimize(loss)

num_iters = 300
num_logs = 10
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(num_iters):
    batch_idxs = np.random.randint(num_training_points_, size=[batch_size])
    x_train_batch_ = x_train_[batch_idxs, ...]
    y_train_batch_ = y_train_[batch_idxs]

    [_, loss_] = sess.run([train_op, loss],
                          feed_dict={x_train_batch: x_train_batch_,
                                     y_train_batch: y_train_batch_})
    if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
      print(i, loss_)

  # Generate a plot with
  #   - the posterior predictive mean
  #   - training data
  #   - inducing index points (plotted vertically at the mean of the
  #     variational posterior over inducing point function values)
  #   - 50 posterior predictive samples

  num_samples = 10
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
  plt.figure(figsize=(15, 5))
  plt.scatter(inducing_index_points_[..., 0], variational_loc_,
              marker='x', s=50, color='k', zorder=10)
  plt.scatter(x_train_[..., 0], y_train_, color='#00ff00', alpha=.1, zorder=9)
  plt.plot(np.tile(index_points_, num_samples),
           samples_.T, color='r', alpha=.1)
  plt.plot(index_points_, mean_, color='k')
  plt.plot(index_points_, f(index_points_), color='b')
  plt.show()
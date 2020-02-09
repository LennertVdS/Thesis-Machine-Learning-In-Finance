import numpy as np
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
tf.disable_v2_behavior()
tfk = tfp.positive_semidefinite_kernels
tfd = tfp.distributions


# this code can be found on the page of gaussian process regression of tensorflow

class Variational:

    def __init__(self,amount,noise,inducing_index_points, data, data_values, index_points, zero_mean = False):
        self.amount = amount
        self.observation_noise_variance = noise
        self.inducing_index_points = inducing_index_points
        self.data = data
        self.data_values = data_values
        self.index_points = index_points
        self.zero_mean = zero_mean


    def prediction(self,prediction_points):

        vgp1 = tfd.VariationalGaussianProcess(
            self.kernel,
            index_points=prediction_points,
            inducing_index_points=self.inducing_index_points_,
            variational_inducing_observations_loc=self.variational_loc_,
            variational_inducing_observations_scale=self.variational_scale_,
            observation_noise_variance=self.observation_noise_variance)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            [mean_] = sess.run([vgp1.mean()])

        if self.zero_mean is False:
            mean_ += self.pol_reg.predict(self.poly.fit_transform(prediction_points))

        return mean_

    def fitting(self):

        if self.zero_mean is False:
            self.poly = PolynomialFeatures(degree=2)
            X_poly = self.poly.fit_transform(self.data)
            self.pol_reg = LinearRegression()
            self.pol_reg.fit(X_poly, self.data_values)
            self.data_values = self.data_values - self.pol_reg.predict(self.poly.fit_transform(self.data))


        # We'll use double precision throughout for better numerics.
        dtype = np.float64

        amplitude = (tf.math.softplus(  # Computes softplus: log(ex
            tf.Variable(.54, dtype=dtype, name='amplitude', use_resource=True)))  # variable with initial value
        length_scale = (
                1e-5 +
                tf.math.softplus(
                    tf.Variable(.54, dtype=dtype, name='length_scale', use_resource=True)))
        self.kernel = tfk.ExponentiatedQuadratic(
            amplitude=amplitude,
            length_scale=length_scale)

        # self.observation_noise_variance = tf.math.softplus(
        #     tf.Variable(
        #         .54, dtype=dtype, name='observation_noise_variance', use_resource=True))

        variational_loc, variational_scale = (
            tfd.VariationalGaussianProcess.optimal_variational_posterior(
                kernel=self.kernel,
                inducing_index_points=self.inducing_index_points,
                observation_index_points=self.data,
                observations=self.data_values,
                observation_noise_variance=self.observation_noise_variance))

        vgp = tfd.VariationalGaussianProcess(
            self.kernel,
            index_points=self.index_points,
            inducing_index_points=self.inducing_index_points,
            variational_inducing_observations_loc=variational_loc,
            variational_inducing_observations_scale=variational_scale,
            observation_noise_variance=self.observation_noise_variance)

        # For training, we use some simplistic numpy-based minibatching.
        batch_size = 64
        x_train_batch = tf.placeholder(dtype, [batch_size, 10], name='x_train_batch')
        y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

        # Create the loss function we want to optimize.
        loss = vgp.variational_loss(
            observations=y_train_batch,
            observation_index_points=x_train_batch,
            kl_weight=float(batch_size) / float(self.amount))

        # optimizer = tf.train.AdamOptimizer(learning_rate=.01)
        # train_op = optimizer.minimize(loss)
        #
        # num_iters = 150
        # num_logs = 10
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     for i in range(num_iters):
        #         batch_idxs = np.random.randint(amount, size=[batch_size])
        #         x_train_batch_ = data[batch_idxs, ...]
        #         y_train_batch_ = data_values[batch_idxs]
        #
        #         [_, loss_] = sess.run([train_op, loss],
        #                               feed_dict={x_train_batch: x_train_batch_,
        #                                          y_train_batch: y_train_batch_})
        #         if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
        #             print(i, loss_)
        #
        #     num_samples = 1
        #     [
        #         samples_,
        #         mean_,
        #         inducing_index_points_,
        #         variational_loc_,
        #     ] = sess.run([
        #         vgp.sample(num_samples),
        #         vgp.mean(),
        #         inducing_index_points,
        #         variational_loc
        #     ])

        optimizer = tf.train.AdamOptimizer(learning_rate=.01)
        train_op = optimizer.minimize(loss)

        num_iters = 150
        num_logs = 10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_iters):
                batch_idxs = np.random.randint(self.amount, size=[batch_size])
                x_train_batch_ = self.data[batch_idxs, ...]
                y_train_batch_ = self.data_values[batch_idxs]

                [_, loss_] = sess.run([train_op, loss],
                                      feed_dict={x_train_batch: x_train_batch_,
                                                 y_train_batch: y_train_batch_})
                if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
                    print(i, loss_)

            num_samples = 1
            [
                samples_,
                mean_,
                self.inducing_index_points_,
                self.variational_loc_,
                self.variational_scale_
            ] = sess.run([
                vgp.sample(num_samples),
                vgp.mean(),
                self.inducing_index_points,
                variational_loc,
                variational_scale
            ])

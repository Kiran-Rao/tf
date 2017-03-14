import tensorflow as tf
import numpy as np

x = tf.placeholder("float")
y = tf.placeholder("float")

weight = tf.Variable([2.0, 1.0], name="weight")

y_model = tf.multiply(x, weight[1]) + weight[0]

error = tf.square(y - y_model, name="error")

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)


model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)

	for i in range(50):
		x_val = np.random.rand()
		y_val = x * 4 - 3
		session.run(train_op, feed_dict={})

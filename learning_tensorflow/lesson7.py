import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

weight = tf.Variable([2.0, 1.0], name="weight")

y_model = tf.multiply(x, weight[1]) + weight[0]

error = tf.square(y - y_model, name="error")

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(error)


model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)

	for i in range(500):
		x_val = np.random.rand()
		y_val = x_val * 4 - 3
		session.run(train_op, feed_dict={x: x_val, y: y_val})

	w_value = session.run(weight)
	print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[1], b=w_value[0]))

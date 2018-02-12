import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

# TF session
sess = tf.Session()
K.set_session(sess)

# Input placeholder
img = tf.placeholder(tf.float32, shape=(None, 784))

# Network
x_1 = Dense(128, activation='relu')(img)
x_2 = Dense(128, activation='relu')(x_1)
preds = Dense(10, activation='softmax')(x_2)

# Label placeholder
labels = tf.placeholder(tf.float32, shape=(None, 10))

# defining loss function
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# training data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# training accuracy
acc_value = accuracy(labels, preds)


# init all
init_op = tf.global_variables_initializer()
sess.run(init_op)


# train
with sess.as_default():
    res = acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels})
    print('Random Accuracy' , np.mean(res))

    for j in range(20):
        for i in range(50):
            batch = mnist_data.train.next_batch(50)
            train_step.run(feed_dict={img: batch[0], labels: batch[1]})

        res = acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels})
        print('Accuracy', (j + 1) * 50, np.mean(res))

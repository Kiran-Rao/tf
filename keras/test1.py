import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
from keras.layers import Dense, Dropout
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy

# TF session
sess = tf.Session()
K.set_session(sess)

# Input placeholder
img = tf.placeholder(tf.float32, shape=(None, 784))

# Network
x = Dense(256, activation='relu')(img)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

# Label placeholder
labels = tf.placeholder(tf.float32, shape=(None, 10))

# defining loss function
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# training data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# optimizer
train_step = tf.train.MomentumOptimizer(0.1, 0.5).minimize(loss)
train_step_late = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# training accuracy
acc_value = categorical_accuracy(labels, preds)


# init all
init_op = tf.global_variables_initializer()
sess.run(init_op)


# train
with sess.as_default():
    accuracy = acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels})
    print('Random Accuracy' , np.mean(accuracy))

    for j in range(5000//100):
        for i in range(100):
            batch = mnist_data.train.next_batch(50)
            train_step.run(feed_dict={img: batch[0], labels: batch[1]})

        accuracy = acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels})
        print('Accuracy', (j + 1) * 100, np.mean(accuracy))

    for j in range(5000//100):
        for i in range(100):
            batch = mnist_data.train.next_batch(50)
            train_step_late.run(feed_dict={img: batch[0], labels: batch[1]})

        accuracy = acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels})
        print('Accuracy, Late', (j + 1) * 100, np.mean(accuracy))

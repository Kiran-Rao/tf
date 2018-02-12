import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

print(image.shape)

plt.imshow(image)
plt.show()

x = tf.Variable(image, name='x')
model = tf.initialize_all_variables()

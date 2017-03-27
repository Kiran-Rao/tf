import unittest
import tensorflow as tf
from data_generator import create_samples, choose_random_centroids, plot_centroid_history


class TestDataGeneratorAssignToNearest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
         (samples, centroids) = create_samples(3, 20, 2, 50, 1337)
         cls.samples = samples
         cls.centroids = centroids

    def test_choose_random_centroids(self):
        random_centroids = choose_random_centroids(self.samples, 3)

        with tf.Session() as session:
            random_centroid_values = session.run(random_centroids)
            print('aoeu')

        self.assertTrue(True)

    def test_false(self):
        self.assertFalse(False)


if __name__ == '__main__':
    unittest.main()


(samples, centroids) = create_samples(3, 20, 2, 50, 1337)

print(samples)

model = tf.global_variables_initializer()

# with tf.Session() as session:
#     session.

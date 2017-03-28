import unittest
import tensorflow as tf
import numpy as np
from data_generator import create_samples, choose_random_centroids, plot_centroid_history


class TestDataGeneratorChooseRandomCentroids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (samples, centroids) = create_samples(3, 10, 3, 50, 1337)
        random_centroids = choose_random_centroids(samples, 3)

        with tf.Session() as session:
            output = session.run([samples, random_centroids])

        cls.sample_values = output[0]
        cls.random_centroid_values = output[1]


    def test_choose_random_centroids_centroids_are_unique(self):
        a = self.random_centroid_values[0]
        b = self.random_centroid_values[1]
        c = self.random_centroid_values[2]

        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(b, c))

    def test_choose_random_centroids_centroids_are_from_samples(self):
        a = self.random_centroid_values[0]
        b = self.random_centroid_values[1]
        c = self.random_centroid_values[2]

        self.assertTrue(a in self.sample_values)
        self.assertTrue(b in self.sample_values)
        self.assertTrue(c in self.sample_values)

    def test_choose_random_centroids_centroids_are_not_first_or_last(self):
        first_slice = self.sample_values[:3]
        last_slice = self.sample_values[-3:]

        self.assertFalse(np.all(first_slice == self.random_centroid_values))
        self.assertFalse(np.all(last_slice == self.random_centroid_values))


if __name__ == '__main__':
    unittest.main()

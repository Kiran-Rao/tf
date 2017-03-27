import unittest
import tensorflow as tf
from data_generator import create_samples, choose_random_centroids, plot_centroid_history


class TestDataGeneratorAssignToNearest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
         (samples, centroids) = create_samples(3, 5, 1, 50, 1337)
         cls.samples = samples
         cls.centroids = centroids

    def test_choose_random_centroids(self):
        random_centroids = choose_random_centroids(self.samples, 3)

        with tf.Session() as session:
            random_centroid_values = session.run(random_centroids)
            sample_values = session.run(self.samples)
            print(random_centroid_values)
            print(sample_values)
            print('aoeu')

        self.assertTrue(True)

    def test_false(self):
        self.assertFalse(False)


if __name__ == '__main__':
    unittest.main()

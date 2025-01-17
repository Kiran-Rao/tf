import tensorflow as tf
import numpy as np

from functions import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70


data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)

updated_centroids = initial_centroids

for i in range(10):
    nearest_indices = assign_to_nearest(samples, updated_centroids)
    updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

combined_centroids = tf.concat([initial_centroids, updated_centroids], 0)

model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = session.run(samples)
    combined_centroid_values = session.run(combined_centroids)
    updated_centroid_values = session.run(updated_centroids)
    print(updated_centroid_values)

plot_clusters(sample_values, combined_centroid_values, n_samples_per_cluster)

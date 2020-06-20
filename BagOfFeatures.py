import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift, estimate_bandwidth

from Classifier import Classifier


class BagOfFeatures:
    def __init__(self, descriptors, labels, method='kmeans50'):
        self.descriptors = descriptors
        self.cluster_centers = self.compute_cluster_centers(method)
        self.nearest_neighbors = NearestNeighbors(algorithm='auto')
        self.nearest_neighbors.fit(self.cluster_centers)
        self.codebook_histograms = self.calculate_histograms(self.descriptors)
        self.classifier = Classifier(self.codebook_histograms, labels)

    def compute_cluster_centers(self, method):
        if method == 'mean_shift':
            all_descriptors = []
            for i in self.descriptors:
                for j in i:
                    all_descriptors.append(j)
            all_descriptors = np.asarray(all_descriptors)

            bandwidth = estimate_bandwidth(all_descriptors, quantile=0.005, n_samples=10000)
            ms = MeanShift(bandwidth=bandwidth, max_iter=100)
            ms.fit(all_descriptors)
            return ms.cluster_centers_

        elif method == 'kmeans50':
            all_descriptors = []
            for i in self.descriptors:
                for j in i:
                    all_descriptors.append(j)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            km_ret, km_labels, km_centers = cv2.kmeans(np.asarray(all_descriptors), 50, None, criteria, 10,
                                                       cv2.KMEANS_RANDOM_CENTERS)

            return km_centers

        elif method == 'kmeans250':
            all_descriptors = []
            for i in self.descriptors:
                for j in i:
                    all_descriptors.append(j)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            km_ret, km_labels, km_centers = cv2.kmeans(np.asarray(all_descriptors), 250, None, criteria, 10,
                                                       cv2.KMEANS_RANDOM_CENTERS)

            return km_centers

        elif method == 'kmeans500':
            all_descriptors = []
            for i in self.descriptors:
                for j in i:
                    all_descriptors.append(j)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            km_ret, km_labels, km_centers = cv2.kmeans(np.asarray(all_descriptors), 500, None, criteria, 10,
                                                       cv2.KMEANS_RANDOM_CENTERS)

            return km_centers

        elif method == 'kmeans_ms_found_k':
            all_descriptors = []
            for i in self.descriptors:
                for j in i:
                    all_descriptors.append(j)
            all_descriptors = np.asarray(all_descriptors)

            bandwidth = estimate_bandwidth(all_descriptors, quantile=0.005, n_samples=10000)
            ms = MeanShift(bandwidth=bandwidth, max_iter=100)
            ms.fit(all_descriptors)

            mean_shift_found_k = len(ms.cluster_centers_)

            print("Mean shift found k: ", mean_shift_found_k)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            km_ret, km_labels, km_centers = cv2.kmeans(np.asarray(all_descriptors), mean_shift_found_k, None, criteria,
                                                       10, cv2.KMEANS_RANDOM_CENTERS)
            return km_centers

        else:
            print('Unknown codebook computation method')

    def calculate_histograms(self, descriptors):
        descriptor_histograms = []
        for i, descriptor in enumerate(descriptors):
            histogram_bins = [0 for i in range(len(self.cluster_centers))]
            cluster_indexes = self.nearest_neighbors.kneighbors(descriptor, 1, return_distance=False)
            for j, desc in enumerate(descriptor):
                histogram_bins[cluster_indexes[j][0]] += 1
            descriptor_histograms.append(histogram_bins)

        return descriptor_histograms

    def train(self):
        self.classifier.train()

    def predict(self, histograms):
        return self.classifier.predict(histograms)

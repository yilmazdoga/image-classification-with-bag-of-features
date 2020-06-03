import glob
import re
import random
import cv2
import natsort
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
from sklearn import svm


class DataLoader:
    def __init__(self, path='Dataset/'):
        self.path = path

    def get_training_set(self, shuffle=True):
        training_class_dirs = self.path + '*_train'
        TRAINING_CLASS_DIRS = natsort.natsorted(glob.glob(training_class_dirs))

        training_set_dirs = {}
        for class_dir in TRAINING_CLASS_DIRS:
            image_dirs = class_dir + '/*.jpg'
            IMAGE_DIRS = natsort.natsorted(glob.glob(image_dirs))
            training_set_dirs[re.split('/|_', class_dir)[1]] = IMAGE_DIRS

        images = []
        labels = []
        for element in training_set_dirs.items():
            for image_dir in element[1]:
                images.append(cv2.imread(image_dir))
                labels.append(element[0])

        if shuffle:
            data = list(zip(images, labels))
            random.shuffle(data)
            images, labels = zip(*data)

        return images, labels

    def get_validation_set(self, shuffle=True):
        validation_class_dirs = self.path + '*_validation'
        VALIDATION_CLASS_DIRS = natsort.natsorted(glob.glob(validation_class_dirs))

        validation_set_dirs = {}
        for class_dir in VALIDATION_CLASS_DIRS:
            image_dirs = class_dir + '/*.jpg'
            IMAGE_DIRS = natsort.natsorted(glob.glob(image_dirs))
            validation_set_dirs[re.split('/|_', class_dir)[1]] = IMAGE_DIRS

        images = []
        labels = []
        for element in validation_set_dirs.items():
            for image_dir in element[1]:
                images.append(cv2.imread(image_dir))
                labels.append(element[0])

        if shuffle:
            data = list(zip(images, labels))
            random.shuffle(data)
            images, labels = zip(*data)

        return images, labels


class FeatureExtractor:
    def __init__(self, images):
        self.images = images

    def get_keypoints(self, method='GFTT'):
        if method == 'GFTT':
            keypoints_GFTT = []
            for image in self.images:
                keypoint_diameter = 8

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray, 1000, 0.0002, 6)
                corners = np.int0(corners)

                keypoints = []
                for corner in corners:
                    x, y = corner[0]
                    keypoints.append(cv2.KeyPoint(x, y, keypoint_diameter))
                keypoints_GFTT.append(keypoints)
            return keypoints_GFTT

        elif method == 'grid0':
            keypoints_grid_0 = []
            for image in self.images:
                image_height, image_width, channels = image.shape
                patch_height_0 = image_height // 24
                patch_width_0 = image_width // 24
                keypoint_diameter = 8

                keypoints = []
                for y in range(patch_height_0 // 2, image_height - image_height % patch_height_0, patch_height_0):
                    for x in range(patch_width_0 // 2, image_width - image_width % patch_width_0, patch_width_0):
                        keypoints.append(cv2.KeyPoint(x, y, keypoint_diameter))
                keypoints_grid_0.append(keypoints)
            return keypoints_grid_0

        elif method == 'grid1':
            keypoints_grid_1 = []
            for image in self.images:
                image_height, image_width, channels = image.shape
                patch_height_1 = image_height // 32
                patch_width_1 = image_width // 32
                keypoint_diameter = 8

                keypoints = []
                for y in range(patch_height_1 // 2, image_height - image_height % patch_height_1, patch_height_1):
                    for x in range(patch_width_1 // 2, image_width - image_width % patch_width_1, patch_width_1):
                        keypoints.append(cv2.KeyPoint(x, y, keypoint_diameter))
                keypoints_grid_1.append(keypoints)
            return keypoints_grid_1

        else:
            print('Unknown feature extraction method')


class FeatureDescriptor:
    def __init__(self, images, keypoints):
        self.images = images
        self.keypoints = keypoints

    def get_SIFT_descriptors(self, return_keypoints=False):
        keypoints = []
        descriptors = []
        sift = cv2.xfeatures2d.SIFT_create()
        for i, image in enumerate(self.images):
            keypoint = self.keypoints[i]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, des = sift.compute(gray, keypoint)
            keypoints.append(kp)
            descriptors.append(des)
        if return_keypoints:
            return keypoints, descriptors
        else:
            return descriptors


class DictionaryComputationUnit:
    def __init__(self, descriptors):
        self.descriptors = descriptors

    def compute(self, method='mean_shift'):
        if method == 'mean_shift':
            all_descriptors = []
            for i in self.descriptors:
                for j in i:
                    all_descriptors.append(j)
            ms = MeanShift()
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
            print("not implemented")
        else:
            print('Unknown dictionary computation method')


class HistogramCalculator:
    def __init__(self, descriptors, cluster_centers):
        self.descriptors = descriptors
        self.cluster_centers = cluster_centers

    def calculate_histograms(self):
        descriptor_histograms = []
        for i, descriptor in enumerate(self.descriptors):
            histogram_bins = [0 for i in range(len(self.cluster_centers))]
            nearest_neighbors = NearestNeighbors(algorithm='auto')
            nearest_neighbors.fit(self.cluster_centers)
            cluster_indexes = nearest_neighbors.kneighbors(descriptor, 1, return_distance=False)
            for j, desc in enumerate(descriptor):
                histogram_bins[cluster_indexes[j][0]] += 1
            descriptor_histograms.append(histogram_bins)

        return descriptor_histograms


class Classifier:
    def __init__(self, histograms, labels):
        self.classifier = svm.SVC()
        self.histograms = histograms
        self.labels = labels
        self.labels_as_int = []
        self.conversation_table = {"cars": 1, "airplanes": 2, "motorbikes": 3, "faces": 4}

        self.convert_labels_to_int()
        self.classifier.fit(self.histograms, self.labels_as_int)

    def convert_labels_to_int(self):
        if isinstance(self.labels[0], str):
            for i in self.labels:
                self.labels_as_int.append(self.conversation_table[i])

    def convert_int_to_labels(self, results):
        results_as_str = []
        for result in results:
            results_as_str.append(
                list(self.conversation_table.keys())[list(self.conversation_table.values()).index(result)])
        return results_as_str

    def predict(self, histograms):
        return self.convert_int_to_labels(self.classifier.predict(histograms))


class ImageViewer:
    def __init__(self):
        self.images = []
        self.labels = []

    def add_to_plot(self, image, label):
        self.images.append(image)
        self.labels.append(label)

    def show(self):
        fig, axes = plt.subplots(nrows=len(self.images) // 2, ncols=2)

        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            ax.set_title(self.labels[i], fontsize=12)
            ax.imshow(cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
    dataloader = DataLoader(path='Dataset/')
    image_viewer = ImageViewer()

    training_images, training_labels = dataloader.get_training_set(shuffle=True)
    validation_images, validation_labels = dataloader.get_validation_set(shuffle=True)
    image_viewer.add_to_plot(training_images[0], training_labels[0])

    feature_extractor = FeatureExtractor(training_images)
    training_keypoints = feature_extractor.get_keypoints(method='GFTT')
    keypoint_image = training_images[0].copy()
    cv2.drawKeypoints(training_images[0], training_keypoints[0], keypoint_image)
    image_viewer.add_to_plot(keypoint_image, 'Keypoints GFTT ' + training_labels[0])

    feature_descriptor = FeatureDescriptor(training_images, training_keypoints)
    training_descriptors = feature_descriptor.get_SIFT_descriptors()

    dictionary_comp_unit = DictionaryComputationUnit(training_descriptors)
    descriptor_cluster_centers = dictionary_comp_unit.compute(method='kmeans500')

    histogram_calculator = HistogramCalculator(training_descriptors, descriptor_cluster_centers)
    training_descriptor_histograms = histogram_calculator.calculate_histograms()

    classifier = Classifier(training_descriptor_histograms, training_labels)

    # ---------------------------------------------------------------------------------------------

    feature_extractor_validation = FeatureExtractor(validation_images)
    validation_keypoints = feature_extractor_validation.get_keypoints(method='GFTT')

    validation_feature_descriptor = FeatureDescriptor(validation_images, validation_keypoints)
    validation_descriptors = validation_feature_descriptor.get_SIFT_descriptors()

    validation_dictionary_comp_unit = DictionaryComputationUnit(validation_descriptors)
    validation_descriptor_cluster_centers = validation_dictionary_comp_unit.compute(method='kmeans500')

    validation_histogram_calculator = HistogramCalculator(validation_descriptors, validation_descriptor_cluster_centers)
    validation_descriptor_histograms = validation_histogram_calculator.calculate_histograms()

    print(len(validation_descriptor_histograms))

    results = classifier.predict(validation_descriptor_histograms)

    correct_result_count = 0
    for i in range(len(validation_labels)):
        if validation_labels[i] == results[i]:
            correct_result_count += 1
    accuracy = (correct_result_count / len(results)) * 100

    print("Accuracy: ", accuracy, "%")

    image_viewer.show()

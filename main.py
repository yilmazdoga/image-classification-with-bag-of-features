__author__ = "Doga Yilmaz S011481 Department of Computer Science"

import time
import cv2
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from FeatureExtractor import FeatureExtractor
from FeatureDescriptor import FeatureDescriptor
from BagOfFeatures import BagOfFeatures


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

    # Dataset Path
    dataset_path = 'Dataset/'

    # Parameters
    feature_extractor_method = 'GFTT'
    clustering_method = 'kmeans50'
    show_images = True

    start = time.time()

    if show_images:
        image_viewer = ImageViewer()

    dataloader = DataLoader(path='Dataset/')
    training_images, training_labels = dataloader.get_training_set(shuffle=True)
    validation_images, validation_labels = dataloader.get_validation_set(shuffle=True)

    if show_images:
        image_viewer.add_to_plot(training_images[0], training_labels[0])

    feature_extractor = FeatureExtractor(training_images)
    training_keypoints = feature_extractor.get_keypoints(method=feature_extractor_method)

    if show_images:
        keypoint_image = training_images[0].copy()
        cv2.drawKeypoints(training_images[0], training_keypoints[0], keypoint_image)
        image_viewer.add_to_plot(keypoint_image, feature_extractor_method + " " + training_labels[0])

    feature_descriptor = FeatureDescriptor(training_images, training_keypoints)
    training_descriptors = feature_descriptor.get_SIFT_descriptors()

    BOF = BagOfFeatures(training_descriptors, training_labels, clustering_method)
    BOF.train()

    # ---------------------------------------------------------------------------------------------

    feature_extractor_validation = FeatureExtractor(validation_images)
    validation_keypoints = feature_extractor_validation.get_keypoints(method=feature_extractor_method)

    validation_feature_descriptor = FeatureDescriptor(validation_images, validation_keypoints)
    validation_descriptors = validation_feature_descriptor.get_SIFT_descriptors()

    validation_descriptor_histograms = BOF.calculate_histograms(validation_descriptors)

    results = BOF.predict(validation_descriptor_histograms)

    end = time.time()

    correct_result_count = 0
    for i in range(len(validation_labels)):
        if validation_labels[i] == results[i]:
            correct_result_count += 1
    accuracy = (correct_result_count / len(results)) * 100

    print("Accuracy:", str(accuracy) + "%", "Elapsed Time:", end - start)

    if show_images:
        image_viewer.show()

import glob
import re
import cv2
import natsort
import random


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

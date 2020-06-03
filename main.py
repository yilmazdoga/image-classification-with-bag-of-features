import glob
import re
import random
import cv2
import natsort
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

training_class_dirs = 'Dataset/*_train'
TRAINING_CLASS_DIRS = natsort.natsorted(glob.glob(training_class_dirs))

validation_class_dirs = 'Dataset/*_validation'
VALIDATION_CLASS_DIRS = natsort.natsorted(glob.glob(validation_class_dirs))

training_set_dirs = {}
for class_dir in TRAINING_CLASS_DIRS:
    image_dirs = class_dir + '/*.jpg'
    IMAGE_DIRS = natsort.natsorted(glob.glob(image_dirs))
    training_set_dirs[re.split('/|_', class_dir)[1]] = IMAGE_DIRS

validation_set_dirs = {}
for class_dir in VALIDATION_CLASS_DIRS:
    image_dirs = class_dir + '/*.jpg'
    IMAGE_DIRS = natsort.natsorted(glob.glob(image_dirs))
    validation_set_dirs[re.split('/|_', class_dir)[1]] = IMAGE_DIRS

training_set = []
for element in training_set_dirs.items():
    for image_dir in element[1]:
        training_set.append((element[0], cv2.imread(image_dir)))
random.shuffle(training_set)

validation_set = []
for element in validation_set_dirs.items():
    for image_dir in element[1]:
        validation_set.append((element[0], cv2.imread(image_dir)))
random.shuffle(validation_set)

keypoints_grid_0 = []
for data in training_set:
    label, image = data
    image_height, image_width, channels = image.shape
    patch_height_0 = image_height // 32
    patch_width_0 = image_width // 32
    keypoint_diameter = 8

    keypoints = []
    for y in range(patch_height_0 // 2, image_height - image_height % patch_height_0, patch_height_0):
        for x in range(patch_width_0 // 2, image_width - image_width % patch_width_0, patch_width_0):
            keypoints.append(cv2.KeyPoint(x, y, keypoint_diameter))
    keypoints_grid_0.append((label, image, keypoints))

keypoints_grid_1 = []
for data in training_set:
    label, image = data
    image_height, image_width, channels = image.shape
    patch_height_1 = image_height // 24
    patch_width_1 = image_width // 24
    keypoint_diameter = 8

    keypoints = []
    for y in range(patch_height_1 // 2, image_height - image_height % patch_height_1, patch_height_1):
        for x in range(patch_width_1 // 2, image_width - image_width % patch_width_1, patch_width_1):
            keypoints.append(cv2.KeyPoint(x, y, keypoint_diameter))
    keypoints_grid_1.append((label, image, keypoints))

keypoints_GFTT = []
for data in training_set:
    label, image = data
    image_height, image_width, channels = image.shape
    keypoint_diameter = 8

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 500, 0.00001, 10)
    corners = np.int0(corners)

    keypoints = []
    for corner in corners:
        x, y = corner[0]
        keypoints.append(cv2.KeyPoint(x, y, keypoint_diameter))
    if len(keypoints) >= 500:
        keypoints_GFTT.append((label, image, keypoints))

descriptors = []
sift = cv2.xfeatures2d.SIFT_create()
for data in keypoints_GFTT:
    label, image, keypoints = data
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = sift.compute(image, keypoints)
    descriptors.append((label, des))

centers_50 = []
for element in descriptors:
    label, des = element
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    km_ret, km_labels, km_centers = cv2.kmeans(np.asarray(des), 50, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers_50.append((label, km_centers))

centers_250 = []
for element in descriptors:
    label, des = element
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    km_ret, km_labels, km_centers = cv2.kmeans(np.asarray(des), 250, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers_250.append((label, km_centers))

centers_500 = []
for element in descriptors:
    label, des = element
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    km_ret, km_labels, km_centers = cv2.kmeans(np.asarray(des), 500, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers_500.append((label, km_centers))

descriptors = np.array([list(i) for i in zip(*descriptors)])

flattened_descriptors = []
for desc in descriptors[1]:
    flattened_descriptors.append(desc[0].flatten())

neigh = NearestNeighbors(n_neighbors=1, radius=0.4)
neigh.fit(flattened_descriptors)

sample = np.array(centers_50[0][1][0])

predictions = neigh.kneighbors([sample], 1, return_distance=False)

print(descriptors[0][predictions[0]])
print(centers_50[0][0])

# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.axis("off")
# ax1.set_title(training_set[10][0], fontsize=12)
# ax1.margins(0, 0)
# ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()


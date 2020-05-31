import glob
import re
import random
import cv2
import natsort
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

label, image = validation_set[50]

print(image.shape)

image_height, image_width, channels = image.shape

patch_height_0 = int(image_height/10)
patch_width_0 = int(image_width/10)

patches_0 = []

patch_height_1 = int(image_height/20)
patch_width_1 = int(image_width/20)

patches_1 = []

patches_keypoint = []

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis("off")
ax1.set_title(label, fontsize=12)
ax1.margins(0, 0)
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

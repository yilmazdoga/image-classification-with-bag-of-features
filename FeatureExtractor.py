import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self, images):
        self.images = images

    def get_keypoints(self, method='GFTT'):
        if method == 'GFTT':
            keypoints_GFTT = []
            for image in self.images:
                keypoint_diameter = 8

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray, 400, 0.002, 8)
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
                patch_height_0 = image_height // 8
                patch_width_0 = image_width // 8
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
                patch_height_1 = image_height // 12
                patch_width_1 = image_width // 12
                keypoint_diameter = 8

                keypoints = []
                for y in range(patch_height_1 // 2, image_height - image_height % patch_height_1, patch_height_1):
                    for x in range(patch_width_1 // 2, image_width - image_width % patch_width_1, patch_width_1):
                        keypoints.append(cv2.KeyPoint(x, y, keypoint_diameter))
                keypoints_grid_1.append(keypoints)
            return keypoints_grid_1

        else:
            print('Unknown feature extraction method')

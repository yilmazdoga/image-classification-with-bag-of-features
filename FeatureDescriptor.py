import cv2


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

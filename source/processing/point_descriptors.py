import cv2
import numpy as np


def load_bf_sift():
    # BFMatcher(Brute Force Matcher) with default setting
    return cv2.BFMatcher(cv2.NORM_L2)


def load_bf_orb():
    # BFMatcher(Brute Force Matcher) with default setting
    return cv2.BFMatcher(cv2.NORM_L2)


def load_flann_sift():
    # Parameters for FLANN with SIFT points
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    return cv2.FlannBasedMatcher(index_params, search_params)

def load_flann_orb():
    # Parameters for FLANN with ORB points
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    return cv2.FlannBasedMatcher(index_params, search_params)


def feature_matching(keypoints_image, keypoints_examples, matcher, threshold=0.75):
    nb_good_matches = list()
    for keypoints_example in keypoints_examples:
        # Sometimes there could be no more than 1 or 0 keypoints detected thus the ratio test cannot be done
        if not (keypoints_example is None):
            if len(keypoints_example)>1:
                matches = matcher.knnMatch(keypoints_image, keypoints_example, k=2)
                # Apply ratio test as in David Rowe's paper
                good_matches = []
                for match in matches:
                    if len(match)>1:
                        m,n = match
                        if m.distance < threshold * n.distance:
                            good_matches.append(m)
                nb_good_matches.append(len(good_matches))
            else:
                nb_good_matches.append(0)
        else:
            nb_good_matches.append(0)
    return nb_good_matches

import numpy as np
import pandas as pd


def calculateIntersection(a0: float, a1: float, b0: float, b1: float) -> float:
    """
    Computes the intersection between the segment [a0, a1] and the segment [b0, b1].
    Reference: https://stackoverflow.com/questions/48477130/
    find-area-of-overlapping-rectangles-in-python-cv2-with-a-raw-list-of-points
    :param a0: start of the first segment
    :param a1: end of the first segment
    :param b0: start of the second segment
    :param b1: end of the second segment
    :return: Value of the length of the intersection.
    """

    if a0 >= b0 and a1 <= b1:  # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1:  # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0:  # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1:  # Intersects left
        intersection = b1 - a0
    else:  # No intersection (either side)
        intersection = 0

    return intersection


def apply_detection(converted_img, detector_):
    """
    Applies the object detection to a properly converted image.
    :param converted_img: Converted image.
    :param detector_: Model of the object detector.
    :return: Objects detected and scores.
    """
    result = detector_(converted_img)
    result = {key:value.numpy() for key,value in result.items()}
    return result


def detect_motorcycles(obj_detected, architecture="mobilenet"):
    """
    Extracts the objects detected as motorcycles and their position on the image.
    :param obj_detected: Objects detected.
    :param architecture: Architecture used for the object detection task.
    :return: pd.DataFrame with the corners of the boxes and the scores of the motorcycles detected.
    """
    if architecture == "mobilenet":
        class_col = "detection_classes"
        moto_class = 4
    elif architecture == "inception":
        class_col = "detection_class_labels"
        moto_class = 300
    else:
        print("Choose one of the following architectures: mobilenet, inception.")
        return None
    return pd.DataFrame(np.concatenate((obj_detected["detection_boxes"][obj_detected[class_col] == moto_class],
                                        np.reshape(obj_detected["detection_scores"]\
                                                       [obj_detected[class_col] == moto_class].T,
                                                   (np.sum(obj_detected[class_col] == moto_class), 1))), axis=1),
                        columns = ["ymin", "xmin", "ymax", "xmax", "score"])


def remove_little_score(boxes_and_scores, threshold=0.5):
    """
    Removes objects detected with less than a threshold.
    :param boxes_and_scores: DataFrame with boxes where the object is detected and scores of those objects.
    :param threshold: Minimum score for an object to be considered.
    :return: pd.DataFrame of the boxes_and_scores filtered.
    """
    return boxes_and_scores[boxes_and_scores.score > threshold]


def remove_overlaying(boxes_and_scores, threshold=0.7):
    """
    Removes from DataFrame of boxes_and_scores those which overlay more than a certain threshold.
    It remains with the one which has the greatest score between those which overlay.
    :param boxes_and_scores: DataFrame with boxes where the object is detected and scores of those objects.
    :param threshold: Maximum ratio of overlay between the smallest and the greatest of the overlaying boxes.
    :return: pd.DataFrame of the boxes_and_scores filtered.
    """
    # Add area to DataFrame and sort by score.
    boxes_and_scores = boxes_and_scores.assign(area = (boxes_and_scores["ymax"]-boxes_and_scores["ymin"])
                                                      *(boxes_and_scores["xmax"]-boxes_and_scores["xmin"])
                                               ).sort_values("score", ascending = False).reset_index()
    # Iterate from the first to the last object detected, comparing with the previous ones (greater score)
    for idx in range(len(boxes_and_scores)):
        for i in range(idx):
            # Only consider if not already removed
            if i in boxes_and_scores.index:
                # The intersection is computed
                width = calculateIntersection(boxes_and_scores.loc[idx, "xmin"], boxes_and_scores.loc[idx, "xmax"],
                                              boxes_and_scores.loc[i, "xmin"], boxes_and_scores.loc[i, "xmax"])
                height = calculateIntersection(boxes_and_scores.loc[idx, "ymin"], boxes_and_scores.loc[idx, "ymax"],
                                              boxes_and_scores.loc[i, "ymin"], boxes_and_scores.loc[i, "ymax"])
                area_intersection = width * height
                # The ratio of intersection is computed and the row is removed if it is larger than the threshold
                if area_intersection/min(boxes_and_scores.loc[idx, "area"],
                                         boxes_and_scores.loc[i, "area"]) > threshold:
                    boxes_and_scores = boxes_and_scores.drop(idx)
                    # As the row has been removed, no more intersections need to be considered
                    break
    return boxes_and_scores.reset_index()


def extract_boxes(img, boxes_and_scores):
    """
    Denormalize the corners of the boxes where the objects have been detected.
    From [0,1] to [0, max_value_axis].
    :param img: Image where the objects have been detected.
    :param boxes_and_scores: DataFrame with boxes where the object is detected and scores of those objects.
    :return: list of images of the detected objects boxed.
    """
    im_height, im_width = img.shape[0:2]
    img_boxes = []
    for row in boxes_and_scores.index:
        ymin, xmin, ymax, xmax = boxes_and_scores.loc[row, ["ymin", "xmin", "ymax", "xmax"]]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        img_boxes.append(img[int(top):int(bottom), int(left):int(right), :])
    return img_boxes
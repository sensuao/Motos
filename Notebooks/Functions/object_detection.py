import numpy as np
import pandas as pd

def calculateIntersection(a0, a1, b0, b1):
    """
    Calcula la intersección entre el conjunto [a0, a1] y el conjunto [b0, b1].
    Reference: https://stackoverflow.com/questions/48477130/find-area-of-overlapping-rectangles-in-python-cv2-with-a-raw-list-of-points

    Parameters:
        a0(float32): inicio del primer conjunto
        a1(float32): final del primer conjunto
        b0(float32): inicio del segundo conjunto
        b1(float32): final del segundo conjunto

    Returns:
        (float32): Valor de la longitud de la intersección
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
    result = detector_(converted_img)
    result = {key:value.numpy() for key,value in result.items()}
    return result

def detect_motorcycles(object_detection_result, architecture = "mobilenet"):
    if architecture == "mobilenet":
        class_column = "detection_classes"
        motorcycle_class = 4
    elif architecture == "inception":
        class_column = "detection_class_labels"
        motorcycle_class = 300
    else:
        print("Elige una arquitectura de las siguientes: mobilenet, inception.")
        return None
    return pd.DataFrame(np.concatenate((object_detection_result["detection_boxes"][object_detection_result[class_column] == motorcycle_class],
                                       np.reshape(object_detection_result["detection_scores"][object_detection_result[class_column] == motorcycle_class].T, (np.sum(object_detection_result[class_column] == motorcycle_class),1))), axis=1), columns = ["ymin", "xmin", "ymax", "xmax", "score"])

def remove_little_score(boxes_and_scores, threshold = 0.5):
    return boxes_and_scores[boxes_and_scores.score > threshold]

def remove_overlaying(boxes_and_scores, threshold = 0.7):
    boxes_and_scores = boxes_and_scores.assign(area = (boxes_and_scores["ymax"]-boxes_and_scores["ymin"])*(boxes_and_scores["xmax"]-boxes_and_scores["xmin"])).sort_values("score", ascending = False).reset_index() # Y si se ordena por score??
    for idx in range(len(boxes_and_scores)):
        for i in range(idx):
            if i in boxes_and_scores.index:
                width = calculateIntersection(boxes_and_scores.loc[idx, "xmin"], boxes_and_scores.loc[idx, "xmax"],
                                              boxes_and_scores.loc[i, "xmin"], boxes_and_scores.loc[i, "xmax"])
                height = calculateIntersection(boxes_and_scores.loc[idx, "ymin"], boxes_and_scores.loc[idx, "ymax"],
                                              boxes_and_scores.loc[i, "ymin"], boxes_and_scores.loc[i, "ymax"])
                # SE CALCULA Y SE GUARDA EL AREA DE LA INTERSECCIÓN
                area_interseccion = width * height
                # SE CALCULA EL PORCENTAJE DE AREA QUE INTERSECTA Y SE ELIMINA LA FILA SI ES MAYOR QUE UN UMBRAL
                if area_interseccion/min(boxes_and_scores.loc[idx, "area"], boxes_and_scores.loc[i, "area"]) > threshold:
                    boxes_and_scores = boxes_and_scores.drop(idx)
                    break
    return boxes_and_scores.reset_index()

def extract_boxes(img, boxes_and_scores):
    im_height, im_width = img.shape[0:2]
    img_boxes = []
    for row in boxes_and_scores.index:
        ymin, xmin, ymax, xmax = boxes_and_scores.loc[row, ["ymin", "xmin", "ymax", "xmax"]]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        img_boxes.append(img[int(top):int(bottom), int(left):int(right), :])
    return img_boxes
import pandas as pd
import cv2

def saturate_well_illuminated_pixels(X):
    """
    Saturates the Hue depending on the Saturation and Value of the pixel as in S. Sural paper (2002).
    :param X: np.array - image.
    :return: pd.DataFrame - image with the well-defined colors saturated and the ill-defined colors converted to gray scale.
    """
    X_th = X.copy()
    X_th[:,0] = [X[j,0] if i else 0 for j,i in enumerate(list(X[:,1]>(255-0.8*X[:,2])))]
    X_th[:,1] = [255 if i else 0 for j,i in enumerate(list(X[:,1]>(255-0.8*X[:,2])))]
    X_th[:,2] = [255 if i else X[j,2] for j,i in enumerate(list(X[:,1]>(255-0.8*X[:,2])))]
    return pd.DataFrame(X_th)


def well_illuminated_pixels_mask(X):
    """
    Creates a list of the indexes corresponding to the pixels that are well-defined colorwise
    :param X: np.array - image
    :return: list - indexes of well-defined pixels colorwise
    """
    return list(X[:,1]>(255-0.8*X[:,2]))


def distance(a,b):
    return cv2.compareHist(a,b, cv2.HISTCMP_BHATTACHARYYA)



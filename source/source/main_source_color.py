import logging
import os
import sys
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.neighbors import KNeighborsClassifier
from processing.object_segmentation import *
from processing.point_descriptors import *
from processing.color_descriptors import *
from preprocessing.resizing import *
from data.data import *


s3_data_path = "Data"  # String: Path in s3 where the data is stored
data_zip_name = "Data_new"  # String: Name of the .zip of the data uploaded
machine_data_path = "Data"  # String: Path in machine for the data to be stored
images_folder = "Images"  # String: Folder inside the Data where the database of images is

# S3 parameters
s3 = boto3.resource('s3')
bucket = s3.Bucket("cbir-motos")

# # LOGGINGS
loggigns_file_name = 'loggings.log'
# # DOWNLOAD LOG FILE
bucket.download_file('Logs/' + loggigns_file_name, loggigns_file_name)
# create a logger
logger = logging.getLogger('logger')
# set logger level
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler(loggigns_file_name)
# # create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_data_from_s3(bucket, data_zip_name, machine_data_path):
    # DOWNLOAD ZIP FROM S3
    logger.info("Downloading zip file:" + data_zip_name)
    # Make folder if it does not exists
    if not os.path.exists(str(machine_data_path)):
        os.makedirs(str(machine_data_path))
    bucket.download_file(s3_data_path + '/' + data_zip_name + ".zip",
                         machine_data_path + "/" + data_zip_name + ".zip")
    logger.info("Zip file successfully downloaded")


def unpack_and_delete_zip_file(data_zip_name, machine_data_path):
    # UNPACK ZIP FILE
    logger.info("Unpacking zip file")
    shutil.unpack_archive(machine_data_path + '/' + data_zip_name + ".zip", machine_data_path, "zip")
    logger.info("Zip file successfully unpacked")
    os.remove(machine_data_path + '/' + data_zip_name + ".zip")


def load_train_paths(machine_data_path):
    # LOAD IMAGES PATHS
    img_paths = list()
    for image in os.listdir(os.path.join(machine_data_path, images_folder)):
        img_paths.append(os.path.join(machine_data_path, images_folder, image))
    df = pd.DataFrame(img_paths, columns=["image_path"])
    del img_paths
    return df


def load_query_paths(machine_data_path):
    # LOAD IMAGES PATHS
    img_paths_and_folder = list()
    folder = "Examples"
    global pos2nb_dict, folders_list
    for image in os.listdir(os.path.join(machine_data_path, folder)):
        img_paths_and_folder.append([os.path.join(machine_data_path, folder, image), image[:-4]])
    df = pd.DataFrame(img_paths_and_folder, columns=["image_path", "folder"])
    folders_list = [img_path_and_folder[1] for img_path_and_folder in img_paths_and_folder]
    pos2nb_dict = dict(zip(np.arange(len(folders_list)), folders_list))
    del img_paths_and_folder
    return df


def apply_is_and_hsv(df, interpreter, input_size, input_details, logger_info="Train"):
    logger.info("Applying Image Segmentation to " + logger_info)
    image_arrayed_hsv = list()
    for image_path in df.image_path:
        image = cv2.imread(image_path)
        # SEGMENTATION & HSV CONVERSION
        image_arrayed_hsv.append(get_person_and_moto_pixels_hsv(image, interpreter,
                                                                input_size, input_details))
    df["image_arrayed_hsv"] = image_arrayed_hsv
    del image_arrayed_hsv
    logger.info("Image Segmentation applied to " + logger_info)


def apply_saturation(df, logger_info="Train"):
    logger.info("Computing histograms of well illuminated pixels to " + logger_info)
    hists_hsv = []
    for image_arrayed_hsv in df.image_arrayed_hsv:
        X = image_arrayed_hsv
        X_th = saturate_well_illuminated_pixels(X)
        X = np.array(X_th.loc[well_illuminated_pixels_mask(X)])[:,0]
        hist = cv2.calcHist((X,), [0], None, [256], [0, 256])
        hists_hsv.append(np.array(hist).reshape(-1))
    df["hists_hsv"] = hists_hsv
    logger.info("Histograms of well illuminated pixels computed for " + logger_info)
    del hists_hsv


def train_knn_model(df_query):
    KNN = KNeighborsClassifier(metric=distance, n_neighbors=1)
    X_query = list(df_query.hists_hsv)
    y_query = list(df_query.folder)
    return KNN.fit(X_query, y_query)


def predict_folder(KNN, df_train):
    logger.info("Predicting folder with KNN model")
    X_test = np.array([hist_hsv for hist_hsv in df_train.hists_hsv])
    dists, y_test_pred = KNN.kneighbors(X_test, n_neighbors=3)
    logger.info("Folder predicted with KNN model")
    script_content = """\
import os

# Crea vectores    
folders = {!r}
paths = {!r}
images_folder = str({!r})

# Itera sobre cada fila del dataframe
for [folder, path] in zip(folders, paths):

    # Crea el directorio de destino si no existe
    if not os.path.exists(str(folder)):
        os.makedirs(str(folder))

    # Mueve el archivo a su destino
    filename = os.path.basename(path)
    origin = os.path.join(images_folder, filename)
    destination = os.path.join(folder, filename)
    os.rename(origin, destination)
    """.format([pos2nb_dict[pred] for pred in list(y_test_pred[:, 0])], list(df_train['image_path']), images_folder)

    # Genera el archivo .py
    with open('move_images.py', 'w') as file:
        file.write(script_content)


def main():

    get_data_from_s3(bucket, data_zip_name, machine_data_path)

    unpack_and_delete_zip_file(data_zip_name, machine_data_path)

    df_train = load_train_paths(machine_data_path)

    df_query = load_query_paths(machine_data_path)

    interpreter, input_size, input_details = download_and_load_model()

    apply_is_and_hsv(df_train, interpreter, input_size, input_details)

    apply_is_and_hsv(df_query, interpreter, input_size, input_details, "Examples")

    apply_saturation(df_train)

    apply_saturation(df_query, logger_info="Examples")

    # Predict
    predict_folder(train_knn_model(df_query), df_train)

    bucket.upload_file("move_images.py", 'Results/' + "move_images.py")

    # LOG FINALE
    logger.info("All finished")

    # UPLOAD LOG FILE
    bucket.upload_file(loggigns_file_name, 'Logs/' + loggigns_file_name)


if __name__ == "__main__":
    main()

import logging
import os
import sys
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

sys.path.append("preprocessing")

from sklearn.neighbors import KNeighborsClassifier
from processing.object_detection import *
from processing.object_segmentation import *
from processing.point_descriptors import *
from processing.color_descriptors import *
from preprocessing.resizing import *
from data.data import *


s3_data_path = "Data"
data_zip_name = "Data_resized_10"
machine_data_path = "Data_original"
machine_rs_data_path = machine_data_path + "/" + data_zip_name

# # S3 parameters
# s3 = boto3.resource('s3')
# bucket = s3.Bucket("cbir-motos")
#
# # LOGGINGS
loggigns_file_name = 'loggings_color.log'
# # DOWNLOAD LOG FILE
# bucket.download_file('Logs/' + loggigns_file_name, loggigns_file_name)
# create a logger
logger = logging.getLogger('logger')
# set logger level
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler(loggigns_file_name)
# # create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#
# # OBJECT DETECTION MODULE
# module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
# detector_mobilenet = hub.load(module_handle).signatures['serving_default']
#
# # SIFT FEATURE DETECTOR/DESCRIPTOR
# sift = cv2.SIFT_create(nfeatures=200)
# # ORB FEATURE DETECTOR/DESCRIPTOR
# orb = cv2.ORB().create()
# cv2.ORB.setMaxFeatures(orb, maxFeatures=200)


def get_data_from_s3(bucket, data_zip_name, machine_data_path):
    # DOWNLOAD ZIP FROM S3
    logger.info("Downloading zip file:" + data_zip_name)
    bucket.download_file(s3_data_path + '/' + data_zip_name + ".zip",
                         machine_data_path + "/" + data_zip_name + ".zip")
    logger.info("Zip file successfully downloaded")


def unpack_and_delete_zip_file(data_zip_name, machine_data_path):
    # UNPACK ZIP FILE
    logger.info("Unpacking zip file")
    shutil.unpack_archive(machine_data_path + '/' + data_zip_name + ".zip", machine_data_path, "zip")
    logger.info("Zip file successfully unpacked")
    os.remove(machine_data_path + '/' + data_zip_name + ".zip")


def resize_downloaded_images(data_zip_name, machine_data_path, machine_rs_data_path, scale_percent):
    # RESIZE ALL IMAGES
    logger.info("Resizing images:" + str(scale_percent) + "%")
    resize_all_images(data_path=machine_data_path + "/" + data_zip_name,
                      resized_data_path=machine_rs_data_path,
                      scale_percent=scale_percent)
    logger.info("Images resized")


def object_detection_motorcycle(image, architecture="mobilenet"):
    # OBJECT DETECTION
    if architecture == "mobilenet":
        converted_img = tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]
    elif architecture == "inception":
        converted_img = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
    else:
        print("Choose one of the following architectures: mobilenet, inception.")
        return image
    result = apply_detection(converted_img, detector_mobilenet)
    boxes_and_scores = detect_motorcycles(result, architecture=architecture)
    boxes_and_scores = remove_little_score(boxes_and_scores)
    boxes_and_scores = remove_overlaying(boxes_and_scores)
    img_boxed = extract_boxes(image, boxes_and_scores)
    return img_boxed


def apply_od_to_examples(machine_rs_data_path):
    logger.info("Applying Object Detection to Examples")
    for example_name in os.listdir(machine_rs_data_path + '/Examples'):
        example_image = cv2.imread(os.path.join(machine_rs_data_path + '/Examples', example_name))
        img_boxed = object_detection_motorcycle(example_image)
        if not len(img_boxed):
            logger.warning("No motorbike detected in example " + str(example_name[:-4]))
        else:
            cv2.imwrite(os.path.join(machine_rs_data_path + '/Examples', example_name), img_boxed[0])
    logger.info("Object Detection applied to Examples")


def mirroring(machine_rs_data_path):
    logger.info("Mirroring Example images")
    for example_name in os.listdir(machine_rs_data_path + '/Examples'):
        example_image_mirrored = cv2.flip(cv2.imread(os.path.join(machine_rs_data_path + '/Examples', example_name)), 1)
        cv2.imwrite(os.path.join(machine_rs_data_path + '/Examples', "mirrored"+example_name), example_image_mirrored)
    logger.info("Example images mirrored")

def compute_examples_keypoints(machine_rs_data_path, keypoints="SIFT", df_examples=pd.DataFrame()):
    # SIFT ALGORITHM TO EXAMPLES
    logger.info("Computing Examples " + keypoints + " points")
    if keypoints == "SIFT":
        keypoints_ = sift
    elif keypoints == "ORB":
        keypoints_ = orb
    else:
        logger.warning("Using SIFT keypoints as default. Specify the keypoints among SIFT & ORB.")
        keypoints_ = sift
    keypoints_list = list()
    folder_names_list = list()
    for example_name in os.listdir(machine_rs_data_path + '/Examples'):
        if example_name.startswith("mirrored"):
            folder_names_list.append(example_name[8:-4])
        else:
            folder_names_list.append(example_name[:-4])
        # SIFT POINTS
        kp, des = keypoints_.detectAndCompute(cv2.imread(machine_rs_data_path + '/Examples/' + example_name), None)
        keypoints_list.append(des)
    if len(df_examples):
        df_examples[keypoints] = keypoints_list
    else:
        df_examples = pd.DataFrame({"folder": folder_names_list, keypoints: keypoints_list})
    del folder_names_list, keypoints_list
    logger.info("Examples " + keypoints + " points computed")
    return df_examples


# TODO: SE USA
def load_train_paths(machine_rs_data_path):
    # LOAD IMAGES PATHS
    img_paths_and_folder = list()
    for folder in os.listdir(machine_rs_data_path):
        if folder != "Examples":
            for image in os.listdir(os.path.join(machine_rs_data_path, folder)):
                img_paths_and_folder.append([os.path.join(machine_rs_data_path, folder, image), folder])
    df = pd.DataFrame(img_paths_and_folder, columns=["image_path", "folder"])
    del img_paths_and_folder
    return df

def apply_od_to_train(df_train):
    logger.info("Applying Object Detection to Train")
    for train_path in df_train.image_path:
        train_image = cv2.imread(train_path)
        img_boxed = object_detection_motorcycle(train_image)
        if not len(img_boxed):
            cv2.imwrite(train_path, train_image)
        else:
            cv2.imwrite(train_path, img_boxed[0])
    logger.info("Object Detection applied to Train")


def compute_train_keypoints(df, keypoints="SIFT"):
    # KEYPOINTS TO TRAIN IMAGES
    logger.info("Computing Train " + keypoints + " points")
    if keypoints == "SIFT":
        keypoints_ = sift
    elif keypoints == "ORB":
        keypoints_ = orb
    else:
        logger.warning("Using SIFT keypoints as default. Specify the keypoints among SIFT & ORB.")
        keypoints_ = sift
    keypoints_list = list()
    for image_path in df.image_path:
        image = cv2.imread(image_path)
        # SIFT POINTS
        kp, des = keypoints_.detectAndCompute(image, None)
        keypoints_list.append(des)
    df[keypoints] = keypoints_list
    logger.info("Train " + keypoints + " points computed")
    del keypoints_list
    return df


def compare_keypoints_descriptors(df_train, df_examples, matcher="BF", keypoints="SIFT"):
    logger.info("Comparing " + keypoints + " keypoints with " + matcher + " matcher.")
    if matcher == "BF":
        if keypoints == "SIFT":
            matcher = load_bf_sift()
        elif keypoints == "ORB":
            matcher = load_bf_orb()
        else:
            logger.warning("Using SIFT keypoints as default. Specify the keypoints among SIFT & ORB.")
            matcher = load_bf_sift()
    elif matcher == "FLANN":
        if keypoints == "SIFT":
            matcher = load_flann_sift()
        elif keypoints == "ORB":
            matcher = load_flann_orb()
        else:
            logger.warning("Using SIFT keypoints as default. Specify the keypoints among SIFT & ORB.")
            matcher = load_flann_sift()
    else:
        if keypoints == "SIFT":
            matcher = load_bf_sift()
        elif keypoints == "ORB":
            matcher = load_bf_orb()
        else:
            logger.warning("Using SIFT keypoints as default. Specify the keypoints among SIFT & ORB.")
            matcher = load_bf_sift()
    nb_good_matches_list = list()
    keypoints_examples = df_examples[keypoints]
    for keypoints_train in df_train[keypoints]:
        # Sometimes there could be no more than 1 or 0 keypoints detected thus the ratio test cannot be done
        if not (keypoints_train is None):
            if len(keypoints_train)>1:
                nb_good_matches_list.append(feature_matching(keypoints_train, keypoints_examples, matcher))
            else:
                nb_good_matches_list.append(np.repeat(0, len(keypoints_examples)))
        else:
            nb_good_matches_list.append(np.repeat(0, len(keypoints_examples)))
    df_train["nb_good_matches_" + keypoints] = nb_good_matches_list
    return df_train


def predict_folder(df_train, df_examples, keypoints="SIFT"):
    if keypoints == "SIFT":
        keypoints_ = sift
    elif keypoints == "ORB":
        keypoints_ = orb
    else:
        logger.warning("Using SIFT keypoints as default. Specify the keypoints among SIFT & ORB.")
        keypoints_ = sift
    argmax_keypoints = list()
    for nb_good_matches in df_train["nb_good_matches_" + keypoints]:
        argmax_keypoints.append(np.argmax(nb_good_matches))
    return list(df_examples.loc[argmax_keypoints, "folder"])


# STARTING OF THE NEW WORK TODO: DELETE THE REST

# todo: esta funcion en un .py
def get_person_and_moto_pixels_hsv(img, interpreter, input_size, input_details):
    mask = get_person_and_moto_mask(img, interpreter, input_size, input_details)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[mask,:]


def load_train_paths(machine_rs_data_path):
    # LOAD IMAGES PATHS
    img_paths_and_folder = list()
    for folder in os.listdir(machine_rs_data_path):
        if folder != "Examples":
            for image in os.listdir(os.path.join(machine_rs_data_path, folder)):
                img_paths_and_folder.append([os.path.join(machine_rs_data_path, folder, image), folder])
    df = pd.DataFrame(img_paths_and_folder, columns=["image_path", "folder"])
    del img_paths_and_folder
    return df


def load_query_paths(machine_rs_data_path):
    # LOAD IMAGES PATHS
    img_paths_and_folder = list()
    folder = "Examples"
    global pos2nb_dict
    for image in os.listdir(os.path.join(machine_rs_data_path, folder)):
        img_paths_and_folder.append([os.path.join(machine_rs_data_path, folder, image), image[:-4]])
    df = pd.DataFrame(img_paths_and_folder, columns=["image_path", "folder"])
    folders_list = [img_path_and_folder[1] for img_path_and_folder in img_paths_and_folder]
    pos2nb_dict = dict(zip(np.arange(len(folders_list)), folders_list))
    del img_paths_and_folder
    return df


def apply_res_is_and_hsv(df, interpreter, input_size, input_details, logger_info="Train"):
    logger.info("Applying Image Segmentation to " + logger_info)
    image_arrayed_hsv = list()
    for image_path in df.image_path:
        image = cv2.imread(image_path)
        # RESIZING
        # NEW DIMENSIONS OF THE IMAGE ARE COMPUTED
        width = int(image.shape[1] * 10 / 100)
        height = int(image.shape[0] * 10 / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # SEGMENTATION
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
        hist = cv2.calcHist((X,), [0], None, [180], [0, 180])
        hist = hist / len(image_arrayed_hsv)
        hists_hsv.append(np.array(hist).reshape(-1))
    df["hists_hsv"] = hists_hsv
    logger.info("Histograms of well illuminated pixels computed for " + logger_info)
    del hists_hsv


def train_knn_model(df_query, distance):
    KNN = KNeighborsClassifier(metric=distance, n_neighbors=1)
    X_query = np.array([hist_hsv for hist_hsv in df_query.hists_hsv], dtype=np.float32)
    y_query = list(df_query.folder)
    return KNN.fit(X_query, y_query)


def predict_folder(KNN, df_train):
    logger.info("Predicting folder with KNN model")
    X_test = np.array([hist_hsv for hist_hsv in df_train.hists_hsv], dtype=np.float32)
    y_test = list(df_train.folder)
    dists, y_test_pred = KNN.kneighbors(X_test, n_neighbors=3)
    logger.info("Folder predicted with KNN model")
    return [y_test, [dists, y_test_pred]]


def main():

    # os.mkdir(machine_data_path)
    #
    # data_scale = "10"
    #
    # data_zip_name = "Data_resized_" + data_scale
    #
    # machine_rs_data_path = machine_data_path + "/" + data_zip_name
    #
    # get_data_from_s3(bucket, data_zip_name, machine_data_path)
    #
    # unpack_and_delete_zip_file(data_zip_name, machine_data_path)

    # resize_downloaded_images(data_zip_name, machine_data_path, machine_rs_data_path, 99)
    # Necessary to redefine machine_rs_data_path

    # machine_rs_data_path = "Data_resized"
    #

    #
    # #
    # machine_data_path = "../../Circuito_Almeria_171021"
    #
    # data_zip_name = "Medios"
    #
    # # resize_downloaded_images(data_zip_name, machine_data_path, machine_rs_data_path, 10)
    #
    # machine_rs_data_path = machine_data_path + "/" + data_zip_name
    #
    # df_train = load_train_paths(machine_rs_data_path)
    #
    # df_query = load_query_paths(machine_rs_data_path)
    #
    # interpreter, input_size, input_details = download_and_load_model()
    #
    # apply_res_is_and_hsv(df_train, interpreter, input_size, input_details)
    #
    # apply_res_is_and_hsv(df_query, interpreter, input_size, input_details, "Examples")
    #
    # apply_saturation(df_train)
    #
    # apply_saturation(df_query, logger_info="Examples")
    #
    # df_query.to_json("../Precomputed_data/df_query_medios.json")
    # df_train.to_json("../Precomputed_data/df_train_medios.json")

    logger.info("***************************************************************************************************")

    path_query_iniciados = "../Precomputed_data/df_query_iniciados.json"
    path_query_medios = "../Precomputed_data/df_query_medios.json"
    path_train_iniciados = "../Precomputed_data/df_train_iniciados.json"
    path_train_medios = "../Precomputed_data/df_train_medios.json"

    path_queries = [path_query_iniciados, path_query_medios]

    path_trains = [path_train_iniciados, path_train_medios]

    datasets_name = ["Iniciados", "Medios"]

    distances_name = ["Bhattacharrya", "Interseccion"]

    distances = [distance_bhat, distance_inter]

    logger.info("Loading Iniciados data")
    dataset_name = datasets_name[0]
    df_query = pd.read_json(path_query_iniciados)
    df_train = pd.read_json(path_train_iniciados)
    logger.info("Iniciados data loaded")
    # logger.info("Loading Medios data")
    # dataset_name = datasets_name[1]
    # df_query = pd.read_json(path_query_medios)
    # df_train = pd.read_json(path_train_medios)
    # logger.info("Medios data loaded")

    for distance_name, distance in zip(distances_name, distances):

        logger.info("*******************")
        logger.info("STARTING EXPERIMENT: Tres motos bien diferenciadas")
        logger.info("*******************")
        logger.info(dataset_name + " & " + distance_name)

        folders = list(df_query.folder.unique())

        pos2nb_dict = dict(zip(np.arange(len(folders)), folders))

        for n in range(3, 4):
            using_folders = [folders[0], folders[5], folders[16]]
            print("Using_folders:",using_folders)
            pos2nb_dict = dict(zip([0,1,2], [1,14,24]))
            logger.info("Number of vehicles: " + str(len(using_folders)))
            df_query_ = df_query[df_query['folder'].isin(using_folders)]
            df_train_ = df_train[df_train['folder'].isin(using_folders)]

            [y_test, [dists, y_test_pred]] = predict_folder(train_knn_model(df_query_, distance), df_train_)
            print("y_test_pred:",y_test_pred)

            accuracy = sum([y_test_i == np.vectorize(pos2nb_dict.get)(y_test_pred)[i,0] for i, y_test_i in enumerate(y_test)])/len(y_test)
            logger.info("KNN Histograms HSV Well Illuminated pixels Accuracy: " + str(round(accuracy * 100, 2)) + "%")
            accuracy = sum([y_test_i in np.vectorize(pos2nb_dict.get)(y_test_pred)[i,0:2] for i, y_test_i in enumerate(y_test)])/len(y_test)
            logger.info("KNN Histograms HSV Well Illuminated pixels Top2 Accuracy: " + str(round(accuracy * 100, 2)) + "%")
            accuracy = sum([y_test_i in np.vectorize(pos2nb_dict.get)(y_test_pred)[i,0:3] for i, y_test_i in enumerate(y_test)])/len(y_test)
            logger.info("KNN Histograms HSV Well Illuminated pixels Top3 Accuracy: " + str(round(accuracy * 100, 2)) + "%")

        using_folders = folders
        logger.info("Number of vehicles: " + str(len(using_folders)))
        df_query_ = df_query[df_query['folder'].isin(using_folders)]
        df_train_ = df_train[df_train['folder'].isin(using_folders)]

        [y_test, [dists, y_test_pred]] = predict_folder(train_knn_model(df_query_, distance), df_train_)

        accuracy = sum([y_test_i == np.vectorize(pos2nb_dict.get)(y_test_pred)[i,0] for i, y_test_i in enumerate(y_test)])/len(y_test)
        logger.info("KNN Histograms HSV Well Illuminated pixels Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = sum([y_test_i in np.vectorize(pos2nb_dict.get)(y_test_pred)[i,0:2] for i, y_test_i in enumerate(y_test)])/len(y_test)
        logger.info("KNN Histograms HSV Well Illuminated pixels Top2 Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = sum([y_test_i in np.vectorize(pos2nb_dict.get)(y_test_pred)[i,0:3] for i, y_test_i in enumerate(y_test)])/len(y_test)
        logger.info("KNN Histograms HSV Well Illuminated pixels Top3 Accuracy: " + str(round(accuracy * 100, 2)) + "%")

    # # Remove data
    # shutil.rmtree(machine_rs_data_path)
    #
    # LOG FINALE
    logger.info("All finished")
    #
    # # UPLOAD LOG FILE
    # bucket.upload_file(loggigns_file_name, 'Logs/' + loggigns_file_name)


if __name__ == "__main__":
    main()

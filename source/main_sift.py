import logging
import os
import sys
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

sys.path.append("preprocessing")

from processing.object_detection import *
from processing.point_descriptors import *
from preprocessing.resizing import *
from data.data import *


s3_data_path = "Data"
data_zip_name = "Data_resized_10"
machine_data_path = "Data_original"
machine_rs_data_path = machine_data_path + "/" + data_zip_name

# S3 parameters
s3 = boto3.resource('s3')
bucket = s3.Bucket("cbir-motos")

# LOGGINGS
loggigns_file_name = 'loggings_points_nfeatures_2000.log'
# DOWNLOAD LOG FILE
# bucket.download_file('Logs/' + loggigns_file_name, loggigns_file_name)
# create a logger
logger = logging.getLogger('logger')
# set logger level
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler(loggigns_file_name)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# OBJECT DETECTION MODULE
module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector_mobilenet = hub.load(module_handle).signatures['serving_default']

# SIFT FEATURE DETECTOR/DESCRIPTOR
sift = cv2.SIFT_create(nfeatures=2000)
# ORB FEATURE DETECTOR/DESCRIPTOR
orb = cv2.ORB().create()
cv2.ORB.setMaxFeatures(orb, maxFeatures=2000)


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


def main():

    for data_scale in ["10", "25", "33", "50"]:

        machine_data_path = "../../Circuito_Almeria_171021/Iniciados"

        data_zip_name = "Data"

        machine_rs_data_path = "Data_resized"

        resize_downloaded_images(data_zip_name, machine_data_path, machine_rs_data_path, int(data_scale))

        apply_od_to_examples(machine_rs_data_path)

        df_train = load_train_paths(machine_rs_data_path)

        apply_od_to_train(df_train)

        df_examples = compute_examples_keypoints(machine_rs_data_path, keypoints="SIFT")
        df_train = compute_train_keypoints(df_train, keypoints="SIFT")
        df_train = compare_keypoints_descriptors(df_train, df_examples, keypoints="SIFT")
        df_train["pred_folder_SIFT_BF"] = predict_folder(df_train, df_examples, keypoints="SIFT")
        df_train = compare_keypoints_descriptors(df_train, df_examples, matcher="FLANN", keypoints="SIFT")
        df_train["pred_folder_SIFT_FLANN"] = predict_folder(df_train, df_examples, keypoints="SIFT")

        df_examples = compute_examples_keypoints(machine_rs_data_path, keypoints="ORB", df_examples=df_examples)
        df_train = compute_train_keypoints(df_train, keypoints="ORB")
        df_train = compare_keypoints_descriptors(df_train, df_examples, keypoints="ORB")
        df_train["pred_folder_ORB_BF"] = predict_folder(df_train, df_examples, keypoints="ORB")
        df_train = compare_keypoints_descriptors(df_train, df_examples, matcher="FLANN", keypoints="ORB")
        df_train["pred_folder_ORB_FLANN"] = predict_folder(df_train, df_examples, keypoints="ORB")

        # METRICS
        accuracy = (sum(df_train.folder == df_train.pred_folder_SIFT_BF)-34)/(len(df_train)-34)
        print("SIFT BF Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("SIFT BF Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = (sum(df_train.folder == df_train.pred_folder_SIFT_FLANN)-34)/(len(df_train)-34)
        print("SIFT FLANN Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("SIFT FLANN Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = (sum(df_train.folder == df_train.pred_folder_ORB_BF)-34)/(len(df_train)-34)
        print("ORB BF Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("ORB BF Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = (sum(df_train.folder == df_train.pred_folder_ORB_FLANN)-34)/(len(df_train)-34)
        print("ORB FLANN Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("ORB FLANN Accuracy: " + str(round(accuracy * 100, 2)) + "%")

        # WITH MIRRORING
        mirroring(machine_rs_data_path)

        df_examples = compute_examples_keypoints(machine_rs_data_path, keypoints="SIFT")
        df_train = compare_keypoints_descriptors(df_train, df_examples, matcher="BF", keypoints="SIFT")
        df_train["pred_folder_SIFT_BF_mirroring"] = predict_folder(df_train, df_examples, keypoints="SIFT")

        df_train = compare_keypoints_descriptors(df_train, df_examples, matcher="FLANN", keypoints="SIFT")
        df_train["pred_folder_SIFT_FLANN_mirroring"] = predict_folder(df_train, df_examples, keypoints="SIFT")

        df_examples = compute_examples_keypoints(machine_rs_data_path, keypoints="ORB", df_examples=df_examples)
        df_train = compare_keypoints_descriptors(df_train, df_examples, matcher="BF", keypoints="ORB")
        df_train["pred_folder_ORB_BF_mirroring"] = predict_folder(df_train, df_examples, keypoints="ORB")

        df_train = compare_keypoints_descriptors(df_train, df_examples, matcher="FLANN", keypoints="ORB")
        df_train["pred_folder_ORB_FLANN_mirroring"] = predict_folder(df_train, df_examples, keypoints="ORB")


        # METRICS
        accuracy = (sum(df_train.folder == df_train.pred_folder_SIFT_BF_mirroring)-34)/(len(df_train)-34)
        print("SIFT BF Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("SIFT BF Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = (sum(df_train.folder == df_train.pred_folder_SIFT_FLANN_mirroring)-34)/(len(df_train)-34)
        print("SIFT FLANN Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("SIFT FLANN Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = (sum(df_train.folder == df_train.pred_folder_ORB_BF_mirroring)-34)/(len(df_train)-34)
        print("ORB BF Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("ORB BF Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        accuracy = (sum(df_train.folder == df_train.pred_folder_ORB_FLANN_mirroring)-34)/(len(df_train)-34)
        print("ORB FLANN Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")
        logger.info("ORB FLANN Mirroring Accuracy: " + str(round(accuracy * 100, 2)) + "%")

        # Remove data
        shutil.rmtree(machine_rs_data_path)

        logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info("DONE WITH "+ data_scale + "%")
        logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # LOG FINALE
    logger.info("All finished")

    # UPLOAD LOG FILE
    bucket.upload_file(loggigns_file_name, 'Logs/' + loggigns_file_name)


if __name__ == "__main__":
    main()

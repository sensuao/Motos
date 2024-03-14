import cv2
import os
import logging


def resize_image(image_path, image_resized_path, dim):
    """
    Resize an image given its path and store the resized image in a different path.
    :param image_path: Path of the original image.
    :param image_resized_path: Path of the resized image.
    :param dim: New dimensions of the image.
    :return: None
    """

    img = cv2.imread(image_path)

    # IMAGE IS RESIZED
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # RESIZED IMAGE IS STORED
    cv2.imwrite(image_resized_path, resized)


def resize_all_images(data_path: str, resized_data_path: str, scale_percent: float):
    """
    Creates a folder with all the resized images given a folder with the original size images.
    :param data_path: Original data path.
    :param resized_data_path: Path where the folder with the resized images is created.
    :param scale_percent: Percentage of resizing desired.
    :return:
    """

    # An example of an image is read
    img = cv2.imread(os.path.join(data_path,
                                  os.listdir(data_path)[0],
                                  os.listdir(os.path.join(data_path, os.listdir(data_path)[0]))[0]))
    # NEW DIMENSIONS OF THE IMAGE ARE COMPUTED
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    os.mkdir(resized_data_path)
    for folder in os.listdir(data_path):
        os.mkdir(os.path.join(resized_data_path, folder))
        for filename in os.listdir(os.path.join(data_path, folder)):
            resize_image(image_path=os.path.join(data_path, folder, filename),
                         image_resized_path=os.path.join(resized_data_path, folder, filename),
                         dim=dim)


def tst_resize_images(data_path: str, resized_data_path: str, scale_percent: float):
    """
    Testing: Creates a folder with all the resized images given a folder with the original size images.
    :param data_path: Original data path.
    :param resized_data_path: Path where the folder with the resized images is created.
    :param scale_percent: Percentage of resizing desired.
    :return:
    """

    # An example of an image is read
    img = cv2.imread(os.path.join(data_path,
                                  os.listdir(data_path)[0]))
    # NEW DIMENSIONS OF THE IMAGE ARE COMPUTED
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    os.mkdir(resized_data_path)
    for filename in os.listdir(data_path):
        print(os.path.join(data_path, filename))
        print(filename)
        resize_image(image_path=os.path.join(data_path, filename),
                     image_resized_path=os.path.join(resized_data_path, filename),
                     dim=dim)
import cv2
import os
import numpy as np

def crop_bounding_box_image(image, bounding_box):
    pass


def resize_image(image, image_size):
    return cv2.resize(image, image_size)


def save_image(image, path):
    cv2.imwrite(path, image)


def load_image(filename):
    return cv2.imread(filename)

def show_image(image_name, image):
    cv2.imshow(image_name, image)
    cv2.waitKey(1000)

def to_one_hot(value, depth):
    arr = np.zeros(depth)
    arr[value] = 1
    return arr

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return [x for x in files if x.endswith('.jpg')]

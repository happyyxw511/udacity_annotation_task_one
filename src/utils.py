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


def resample_unbalanced_data(file_list, label_list):
    file_list = np.array(file_list)
    label_list = np.array(label_list)
    index_list = np.array(range(len(label_list)))
    unique, counts = np.unique(label_list, return_counts=True)
    minimum_count = np.min(counts)
    resampled_file_list = np.array([])
    resampled_label_list = np.array([])
    for ind, v in enumerate(unique):
        indices = index_list[label_list[index_list] == v]
        np.random.shuffle(indices)
        selected_indices = indices[:minimum_count]
        resampled_file_list = np.append(resampled_file_list, file_list[selected_indices])
        resampled_label_list = np.append(resampled_label_list, label_list[selected_indices])
    return resampled_file_list, resampled_label_list




def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return [x for x in files if x.endswith('.jpg')]

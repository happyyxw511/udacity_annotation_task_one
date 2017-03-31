import cv2

def crop_bounding_box_image(image, bounding_box):
    pass


def resize_image(image, image_size):
    pass


def save_image(image, path):
    pass


def load_image(filename):
    return cv2.imread(filename)

def show_image(image_name, image):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

import tensorflow as tf
import os
import residual_cnn_classifier
import utils

BATCH_SIZE = 4
IMAGE_SIZE = [BATCH_SIZE, 50, 50, 3]
NUM_CLASSES = 3
CLASS_MAP = {
    'Car': 0,
    'Pedestrian': 1,
    'Truck': 2
}

flags = tf.app.flags
flags.DEFINE_string('input_path', '../detected_image/', 'the path of input images')
FLAGS = flags.FLAGS

def _get_files_and_labels(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files], [tf.one_hot(CLASS_MAP[x.split('_')[2]], 3) for x in files]

if __name__ == '__main__':
    _get_files_and_labels(FLAGS.input_path)
    neural_net = residual_cnn_classifier.nn_construction(IMAGE_SIZE, [BATCH_SIZE, NUM_CLASSES])

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
flags.DEFINE_string('checkpoint_path', '../checkout_point/', 'the path of checkpoint')
flags.DEFINE_float('dropout_probability', 0.5, 'the drop out probability')
FLAGS = flags.FLAGS

def _get_files_and_labels(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files], [utils.to_one_hot(CLASS_MAP[x.split('_')[2][:-4]], 3) for x in files]

if __name__ == '__main__':
    _get_files_and_labels(FLAGS.input_path)
    neural_net, loss, accuracy = residual_cnn_classifier.nn_construction(IMAGE_SIZE, [BATCH_SIZE, NUM_CLASSES])
    run_config = tf.ConfigProto(allow_soft_placement=True)
    img_files, labels = _get_files_and_labels(FLAGS.input_path)
    try:
        os.mkdir(FLAGS.checkpoint_path)
    except Exception:
        pass
    checkpoint_fullpath = os.path.join(FLAGS.checkpoint_path, 'save.ckpt')
    with tf.Session(config=run_config) as sess:
        residual_cnn_classifier.train(sess,
                                      loss,
                                      accuracy,
                                      img_files,
                                      labels,
                                      FLAGS.dropout_probability,
                                      checkpoint_fullpath,
                                      IMAGE_SIZE)



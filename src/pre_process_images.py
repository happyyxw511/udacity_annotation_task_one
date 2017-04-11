# It has the following steps:
# 1. read csv
# 2. get the image and the object bounding box and label.
# 3. crop the image out
# 4. resize the image
# 5. save the image to folder.

import pandas as pd
import tensorflow as tf
import os
import utils
from  collections import defaultdict

flags = tf.app.flags
flags.DEFINE_string('data_path', default_value='../data', docstring='path for input data')
flags.DEFINE_string('save_path', default_value='../detected_image/', docstring='path for input data')
FLAGS=flags.FLAGS
data_path = FLAGS.data_path
save_path = FLAGS.save_path
label_path = os.path.join(data_path, 'labels.csv')
labels_df = pd.read_csv(label_path)

print os.path.isfile(label_path)

try:
    os.mkdir(save_path)
except Exception:
    pass

image_size = (50, 50)

index_record = defaultdict(int)

for index, row in labels_df.iterrows():
    image_name = row['Frame']
    full_path_image_name = os.path.join(data_path, image_name)
    if os.path.isfile(full_path_image_name):
        image_name = image_name.split('.')[0]
        image = utils.load_image(full_path_image_name)
        xmin = row['xmin'] #if row['xmin'] < row['xmax'] else row['xmax']
        xmax = row['xmax'] #if row['xmin'] < row['xmax'] else row['xmin']
        ymin = row['ymin'] #if row['ymin'] < row['ymax'] else row['ymax']
        ymax = row['ymax'] #if row['ymin'] < row['ymax'] else row['ymin']

        cropped_image = image[xmax: ymax, xmin:ymin, :]
        label = row['Label']
        key = '{}_{}'.format(image_name, label)
        obj_index = index_record[key]
        obj_index += 1
        save_image_full_path = '{path}{image_name}_{obj_index}_{label}.jpg'.format(
            path=save_path,
            image_name=image_name,
            obj_index=obj_index,
            label=label
        )
        index_record[key] = obj_index
        resized_image = utils.resize_image(cropped_image, image_size)
        utils.save_image(resized_image, save_image_full_path)



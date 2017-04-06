import tensorflow as tf
import numpy as np
import utils

WEIGHTS_INIT_STDEV = .1

def nn_construction(x_input_shape, y_input_shape):
    x_placefolder = tf.placeholder(tf.float32, shape=x_input_shape, name='X_Input')
    y_placeholder = tf.placeholder(tf.float32, shape=y_input_shape, name='Y_Input')
    dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

    conv_l1 = _conv_layer(x_placefolder, 3, 32, 1)
    pool_l1 = _max_pool(conv_l1, 2, 1)
    conv_l2 = _conv_layer(pool_l1, 3, 64, 1)
    pool_l2 = _max_pool(conv_l2, 2, 1)
    fc_l1 = _fc_layer(pool_l2, 1024, dropout_placeholder)
    fc_l2 = _fc_layer(fc_l1, 3)
    output = tf.nn.softmax(fc_l2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_placeholder))
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return output, loss, accuracy


def train(sess, loss, accuracy, input_x_files, input_y, dropout, checkpoint_path, batch_shape, epoch=10):
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    num_imgs = len(input_x_files)
    batch_size = batch_shape[0]
    for i in xrange(epoch):
        cur_index = 0
        accuracy_list = []
        np.random.shuffle(input_x_files)
        while cur_index + batch_size < num_imgs:
            next_index = cur_index + batch_size
            batch_images = np.zeros(batch_shape, dtype=np.float32)
            for ind, filename in enumerate(input_x_files[cur_index: next_index]):
                batch_images[ind] = utils.load_image(filename)
            batch_images_labels = np.array(input_y[cur_index: next_index])
            _, accuracy_value = sess.run([optimizer, accuracy], feed_dict={
                'X_Input:0': batch_images,
                'Y_Input:0': batch_images_labels,
                'dropout:0': dropout
            })
            accuracy_list.append(accuracy_value)
            cur_index = next_index
            if len(accuracy_list) > 2:
                saver = tf.train.Saver()
                saver.save(sess, checkpoint_path)
                print np.mean(accuracy_list)
                accuracy_list = []


def infer(sess, checkout_point, output, input_x_files, batch_shape, dropout, show_image=False, class_mapping=None):
    saver = tf.train.Saver()
    saver.restore(sess, checkout_point)
    num_imgs = len(input_x_files)
    batch_size = batch_shape[0]
    curr = 0
    while curr + batch_size < num_imgs:
        next_index = curr + batch_size
        selected_files = input_x_files[curr: next_index]
        batch_images = np.zeros(batch_shape, dtype=np.float32)
        for ind, filename in enumerate(selected_files):
            print filename
            batch_images[ind] = utils.load_image(filename)
        preds = sess.run(
            [output],
            feed_dict={
                'X_Input:0': batch_images,
                'dropout:0': dropout
            }
        )
        curr = next_index
        if show_image and class_mapping:
            for ind in xrange(batch_size):
                class_ind = np.argmax(preds[ind])
                utils.show_image(class_mapping[class_ind], batch_images[ind].astype(np.uint8))





def _conv_layer(input, filter_size, num_out_channels, strides):
    input = _batch_norm(input)
    num_in_channels = input.get_shape()[3].value
    filter = _conv_init_filter(num_in_channels, num_out_channels, filter_size)
    output = tf.nn.conv2d(input, filter=filter, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.relu(output)


def _conv_init_filter(num_in_channels, num_out_channels, filter_size):
    weights_shape = [filter_size, filter_size, num_in_channels, num_out_channels]
    weights_init = tf.Variable(tf.truncated_normal(weights_shape, dtype=tf.float32), dtype=tf.float32)
    return weights_init


def _fc_weights(input_size, output_size):
    return tf.Variable(tf.truncated_normal([input_size, output_size], dtype=tf.float32), dtype=tf.float32)


def _fc_bias(size):
    return tf.Variable(tf.truncated_normal([size], dtype=tf.float32), dtype=tf.float32)


def _fc_layer(input, output_size, dropout=1):
    input_shape = input.get_shape().as_list()
    reshaped_input = tf.reshape(input, [input_shape[0], -1])
    feature_size = reshaped_input.get_shape().as_list()[1]
    weights = _fc_weights(feature_size, output_size)
    bias = _fc_bias(output_size)
    fc = tf.add(tf.matmul(reshaped_input, weights), bias)
    return tf.nn.dropout(fc, dropout)


def _max_pool(input, kernel_size, stride):
    return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME')


def _batch_norm(net):
    channels = net.get_shape()[3].value
    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.zeros(var_shape))
    mu, sigma = tf.nn.moments(net, [0, 1, 2], keep_dims=True)
    eps = 1e-3
    return tf.nn.batch_normalization(net, mu, sigma, shift, scale, eps)

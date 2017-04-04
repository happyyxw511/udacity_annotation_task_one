import tensorflow as tf

WEIGHTS_INIT_STDEV = .1

def nn_construction(x_input_shape, y_input_shape):
    x_placefolder = tf.placeholder(tf.float32, shape=x_input_shape, name='X_Input')
    y_placeholder = tf.placeholder(tf.float32, shape=y_input_shape, name='Y_Input')

    pass

def train(loss, input_x, input_y, checkpoint):
    pass

def infer(checkout_point, input_x, input_y=None):
    pass


def _conv_layer(input, filter_size, num_out_channels, strides):
    num_in_channels = input.get_shape()[3].value
    filter = _conv_init_filter(num_in_channels, num_out_channels, filter_size)
    output = tf.nn.conv2d(input, filter=filter, strides=[1, strides, strides, 1], padding='SAME')
    output = tf.nn.batch_normalization()

def _conv_init_filter(num_in_channels, num_out_channels, filter_size):
    weights_shape = [filter_size, filter_size, num_in_channels, num_out_channels]
    weights_init = tf.Variable(tf.truncated_norm(weights_shape, dtype=tf.float32), dtype=tf.float32)
    return weights_init

def _batch_norm(net):
    channels = net.get_shape()[3].value
    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.zeros(var_shape))
    mu, sigma = tf.nn.moments(net, [0, 1, 2], keep_dims=True)
    eps = 1e-3
    return tf.nn.batch_normalization(net, mu, sigma, shift, scale, eps)

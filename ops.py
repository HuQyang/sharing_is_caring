''' code from https://github.com/taki0112/Densenet-Tensorflow'''

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.layers import ReLU
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()
import numpy as np


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.compat.v1.name_scope(layer_name):
        network = tf.compat.v1.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Drop_out(x, rate, training) :
    return tf.compat.v1.layers.dropout(inputs=x, rate=rate, training=training)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.compat.v1.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.compat.v1.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Linear(x,class_num) :
    return tf.compat.v1.layers.dense(inputs=x, units=class_num, name='linear')

def concat(layers) :
    return tf.concat(layers, axis=3)


class DenseNet():
    def __init__(self, x, nb_blocks, filters,class_num,dropout_rate,training=True):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.compat.v1.name_scope(scope):
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate,training=self.training)

            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.compat.v1.name_scope(scope):
            x = BatchNormalization()(x)
            x = ReLU()(x)
            s = x.get_shape().as_list()
            in_channel = s[-1]
            x = conv_layer(x, filter=in_channel*0.5, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.compat.v1.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = concat(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = concat(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)

        for i in range(self.nb_blocks) :
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = GlobalAveragePooling2D()(x)
        x = Linear(x,self.class_num)

        x = tf.reshape(x, [-1, self.class_num])
        return x

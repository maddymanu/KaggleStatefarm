'''
 Note: some of this code is borrowed from kaggle kernels or github
 i didnt want to reinvent the wheel so the helper utils are taken from existing kaggle notebooks
'''


import numpy as np
import pandas as pd
import cv2


import tensorflow as tf
import pickle

import os
import time
import glob
from subprocess import check_output
from subprocess import call

from keras.utils import np_utils

import math
import tensorflow.contrib.slim as slim

use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 1

def get_driver_data():
    driver = dict()
    path = os.path.join("data" , "driver_imgs_list.csv")
    f = open(path , "r")
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        driver[arr[2]] = arr[0]
    f.close()
    return driver


def get_im_cv2(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)

    if img != None:
        resized = cv2.resize(img, (64, 64))
        return resized

def load_train_data(img_x=64, img_y=64, type=1):
    X_train = []
    y_train = []
    driver_id = []
    start_time = time.time()
    driver_data = get_driver_data()

    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join("data" , "imgs" , "train" , "c"+str(j) , "*.jpg")
        files = glob.glob(path)
        print(files)
        for file in files:
            flbase = os.path.basename(file)
            img = get_im_cv2(file, img_x , img_y , type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    start_time = time.time()
    path = os.path.join("data" , "imgs"  , 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, "wb")
        pickle._dump(data, file)
        file.close()
    else:
        print("Directory doesnt exist")



def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data










def read_and_normalize_train_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = load_train_data(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0] , color_type , img_rows , img_cols)
    train_target = np_utils.to_categorical(train_target , 10)
    train_data = train_data.astype("float32")
    train_data/=255

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers






def read_and_normalize_test_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id




def weigths(shape):
    return tf.Variable(tf.truncated_normal(shape=shape , stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))



img_rows , img_cols = 64, 64

train_data , train_target , driver_id , unique_drivers  = read_and_normalize_train_data(img_rows , img_cols , 1)
test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

yfull_train = dict()
yfull_test = []
unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']




tf.reset_default_graph()


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128


img_size = 64

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size , filter_size , num_input_channels , num_filters]
    w = weigths(shape)

    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=w , strides=[1,1,1,1] , padding="SAME")

    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1] , strides=[1,2,2,1] , padding="SAME")

    layer = tf.nn.relu(layer)
    return layer, w


def flatten_layer(layer):
    shape = layer.get_shape()
    num_f = shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1,num_f])

    return layer_flat, num_f


def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True):
    w = weigths(shape=[num_inputs, num_outputs])
    b = new_biases(num_outputs)

    layer = tf.matmul(input, w)
    layer += b

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x  = tf.placeholder(tf.float32, shape=[None, 1, 64, 64])
x_image = tf.reshape(x , [-1 ,img_size, img_size, num_channels ])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes])
y_true_cls = tf.argmax(y_true, dimension=1)

l1, w1 = new_conv_layer(x_image, num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
l2, w2 = new_conv_layer(l1, num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

l_flat , num_f = flatten_layer(l2)
l_fc1 = new_fc_layer(l_flat, num_inputs=num_f , num_outputs=num_classes , use_relu=True)

y_pred = tf.nn.softmax(l_fc1)

y_pred_cls = tf.argmax(y_pred, dimension=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=l_fc1,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()
session.run(tf.global_variables_initializer())


feed_dict_train = {x:train_data[:10000] , y_true:train_target[:10000]}
session.run(optimizer, feed_dict=feed_dict_train)

acc = session.run(accuracy, feed_dict=feed_dict_train)
print(acc)





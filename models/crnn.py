from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
import logging
import pandas as pd
import PIL.Image
import numpy as np
import os
import random
import math
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
import cv2

ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '1','2','3','4','5','6','7','9','0',
    '.',',',':','-','(',')','/',"'",
    ' ',
    '_'
]


def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset


wide_charset = None


def get_str_labels(char_map, v, add_eos=True):
    v = v.strip()
    v = v.lower()
    result = []
    for t in v:
        if t == '#' or t == '_':
            continue
        i = char_map.get(t, -1)
        if i >= 0:
            result.append(i)
    if len(result) < 1:
        return [0]
    if add_eos:
        result.append(len(char_map) - 1)
    return result


def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def get_input_fn(dataset_type):
    if dataset_type == 'synth-crop':
        return input_fn
    if dataset_type == 'full-generated':
        return full_generated_input_fn
    if dataset_type == 'tf-record':
        return tf_input_fn
    else:
        return generated_input_fn


def full_generated_input_fn(params, is_training):
    logging.info('Use full generated')
    max_width = params['max_width']
    char_map = params['charset']
    batch_size = params['batch_size']
    inputs = []
    with open(params['data_set'] + '/labels.txt', 'r') as f:
        for x in f:
            x = x.rstrip()
            ls = x.split(' ')
            if len(ls) < 2:
                continue
            img_name = params['data_set'] + '/' + ls[0]
            text = ' '.join(ls[1:])
            label = get_str_labels(char_map, text)
            if len(label) < 2:
                continue
            inputs.append([img_name, text])

    inputs = sorted(inputs, key=lambda row: row[0])
    input_size = len(inputs)
    logging.info('Dataset size {}'.format(input_size))

    def _input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        shuffle_size = input_size
        if is_training:
            logging.info("Shuffle by %d", shuffle_size)
            if shuffle_size == 0:
                shuffle_size = 10
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_size))

        def _features_labels(images, labels):
            return images, labels

        def _decode(filename_label):
            filename = str(filename_label[0], encoding='UTF-8')
            label = str(filename_label[1], encoding='UTF-8')
            label = get_str_labels(char_map, label)
            image = PIL.Image.open(filename)
            width, height = image.size
            min_ration = 10.0 / float(min(width, height))
            max_ratio = max(min_ration, 1.0)
            ratio = random.random() * (max_ratio - min_ration) + min_ration
            width = int(math.ceil(ratio * width))
            height = int(math.ceil(ratio * height))
            image = image.resize((width, height))
            ration_w = max(width / max_width, 1.0)
            ration_h = max(height / 32.0, 1.0)
            ratio = max(ration_h, ration_w)
            if ratio > 1:
                width = int(width / ratio)
                height = int(height / ratio)
                image = image.resize((width, height))
            image = np.asarray(image)
            pw = max(0, max_width - image.shape[1])
            ph = max(0, 32 - image.shape[0])
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
            image = image.astype(np.float32) / 127.5 - 1
            return image, np.array(label, dtype=np.int32)

        dataset = dataset.map(
            lambda filename_label: tuple(tf.py_func(_decode, [filename_label], [tf.float32, tf.int32])),
            num_parallel_calls=1)

        dataset = dataset.padded_batch(batch_size, padded_shapes=([32, max_width, 3], [None]))
        dataset = dataset.map(_features_labels, num_parallel_calls=1)
        dataset = dataset.prefetch(2)
        return dataset

    return _input_fn


def generated_input_fn(params, is_training):
    max_width = params['max_width']
    char_map = params['charset']
    batch_size = params['batch_size']
    inputs = []
    for img_file in glob.iglob(params['data_set'] + '/*.jpg'):
        name = os.path.basename(img_file)
        names = name.split('_')
        if len(names) > 1:
            label = get_str_labels(char_map, names[0])
            if len(label) < 2:
                continue
            inputs.append([img_file, names[0]])

    inputs = sorted(inputs, key=lambda row: row[0])
    input_size = len(inputs)
    logging.info('Dataset size {}'.format(input_size))

    def _input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        shuffle_size = input_size
        if is_training:
            logging.info("Shuffle by %d", shuffle_size)
            if shuffle_size == 0:
                shuffle_size = 10
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_size))

        def _features_labels(images, labels):
            return images, labels

        def _decode(filename_label):
            filename = str(filename_label[0], encoding='UTF-8')
            label = str(filename_label[1], encoding='UTF-8')
            label = get_str_labels(char_map, label)
            image = PIL.Image.open(filename)
            width, height = image.size
            min_ration = 6.0 / float(min(width, height))
            max_ratio = max(min_ration, 1.0)
            ratio = random.random() * (max_ratio - min_ration) + min_ration
            width = int(tf.ceil(ratio * width))
            height = int(tf.ceil(ratio * height))
            image = image.resize((width, height))
            ration_w = max(width / max_width, 1.0)
            ration_h = max(height / 32.0, 1.0)
            ratio = max(ration_h, ration_w)
            if ratio > 1:
                width = int(width / ratio)
                height = int(height / ratio)
                image = image.resize((width, height))
            image = np.asarray(image)
            pw = max(0, max_width - image.shape[1])
            ph = max(0, 32 - image.shape[0])
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
            image = image.astype(np.float32) / 255.0
            return image, np.array(label, dtype=np.int32)

        dataset = dataset.map(
            lambda filename_label: tuple(tf.py_func(_decode, [filename_label], [tf.float32, tf.int32])),
            num_parallel_calls=1)

        dataset = dataset.padded_batch(batch_size, padded_shapes=([32, max_width, 3], [None]))
        dataset = dataset.map(_features_labels, num_parallel_calls=1)
        dataset = dataset.prefetch(2)
        return dataset

    return _input_fn


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def _crop_py_and_roate(img, ymin, xmin, ymax, xmax, texts, gxs, gys):
    i = random.randrange(len(ymin))
    y0 = max(ymin[i] - 2, 0)
    x0 = max(xmin[i] - 2, 0)
    y1 = min(ymax[i] + 2, img.shape[0])
    x1 = min(xmax[i] + 2, img.shape[1])
    img = img[y0:y1, x0:x1, :]
    v = np.transpose(np.concatenate([[gxs[i]], [gys[i]]], axis=0))
    v = cv2.minAreaRect(v)
    if v[1][0] > v[1][1]:
        angle = -1 * v[2]
    else:
        angle = -1 * (90 + v[2])
    if angle != 0 and angle != 90 and angle != -90:
        img = rotate_bound(img, angle)
    label = str(texts[i], encoding='UTF-8')
    ilabel  = np.array(get_str_labels(wide_charset, label),np.int32)
    logging.info('use text: {} - {}'.format(label,ilabel))
    return np.array([img.shape[0], img.shape[1]], np.int32), img,ilabel

def _crop_py(img, ymin, xmin, ymax, xmax, texts):
    i = random.randrange(len(ymin))
    y0 = max(ymin[i] - 2, 0)
    x0 = max(xmin[i] - 2, 0)
    y1 = min(ymax[i] + 2, img.shape[0])
    x1 = min(xmax[i] + 2, img.shape[1])
    img = img[y0:y1, x0:x1, :]
    label = str(texts[i], encoding='UTF-8')
    ilabel  = np.array(get_str_labels(wide_charset, label),np.int32)
    logging.info('use text: {} - {}'.format(label,ilabel))
    return np.array([img.shape[0], img.shape[1]], np.int32), img, ilabel


def tf_input_fn(params, is_training):
    max_width = params['max_width']
    global wide_charset
    wide_charset = params['charset']
    batch_size = params['batch_size']
    datasets_files = []
    for tf_file in glob.iglob(params['data_set'] + '/*.record'):
        datasets_files.append(tf_file)
    random.shuffle(datasets_files)
    augs = params['aug'].split(',')
    rotate = 'rotate' in augs
    logging.info('Do rotation: {}'.format(rotate))
    inception = params['cnn_type']=='inception'
    def _input_fn():
        ds = tf.data.TFRecordDataset(datasets_files, buffer_size=256 * 1024 * 1024)

        def _parser(example):
            if rotate:
                features = {
                    'image/encoded':
                        tf.FixedLenFeature((), tf.string, default_value=''),
                    'image/shape':
                        tf.FixedLenFeature(3, tf.int64),
                    'image/object/bbox/xmin':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/xmax':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/ymin':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/ymax':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/label_text':
                        tf.VarLenFeature(tf.string),
                    'image/object/bbox/x1':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/x2':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/x3':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/x4':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/y1':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/y2':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/y3':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/y4':
                        tf.VarLenFeature(tf.float32),
                }
            else:
                features = {
                    'image/encoded':
                        tf.FixedLenFeature((), tf.string, default_value=''),
                    'image/shape':
                        tf.FixedLenFeature(3, tf.int64),
                    'image/object/bbox/xmin':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/xmax':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/ymin':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/ymax':
                        tf.VarLenFeature(tf.float32),
                    'image/object/bbox/label_text':
                        tf.VarLenFeature(tf.string),
                }
            res = tf.parse_single_example(example, features)
            img = tf.image.decode_jpeg(res['image/encoded'], channels=3)
            original_w = tf.cast(res['image/shape'][1], tf.int32)
            original_h = tf.cast(res['image/shape'][0], tf.int32)
            img = tf.reshape(img, [original_h, original_w, 3])
            original_w = tf.cast(original_w, tf.float32)
            original_h = tf.cast(original_h, tf.float32)
            ymin = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymin']), tf.float32) * original_h,
                           tf.int32)
            xmin = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmin']), tf.float32) * original_w,
                           tf.int32)
            xmax = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmax']), tf.float32) * original_w,
                           tf.int32)
            ymax = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymax']), tf.float32) * original_h,
                           tf.int32)
            texts = tf.sparse_tensor_to_dense(res['image/object/bbox/label_text'], default_value='')
            if rotate:
                x1 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x1']), tf.float32)
                x2 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x2']), tf.float32)
                x3 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x3']), tf.float32)
                x4 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/x4']), tf.float32)
                y1 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y1']), tf.float32)
                y2 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y2']), tf.float32)
                y3 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y3']), tf.float32)
                y4 = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/y4']), tf.float32)
                gxs = tf.cast(tf.transpose(tf.stack([x1, x2, x3, x4])) * original_w, tf.int32)
                gys = tf.cast(tf.transpose(tf.stack([y1, y2, y3, y4])) * original_h, tf.int32)
                size, img, label = tf.py_func(
                    _crop_py_and_roate,
                    [img, ymin, xmin, ymax, xmax, texts, gxs, gys],
                    [tf.int32, tf.uint8, tf.int32]
                )
            else:
                size, img, label = tf.py_func(
                    _crop_py,
                    [img, ymin, xmin, ymax, xmax, texts],
                    [tf.int32, tf.uint8, tf.int32]
                )
            original_h = size[0]
            original_w = size[1]
            img = tf.reshape(img, [original_h, original_w, 3])
            w = tf.maximum(tf.cast(original_w, tf.float32), 1.0)
            h = tf.maximum(tf.cast(original_h, tf.float32), 1.0)
            ratio = tf.random_uniform((), minval=0.5, maxval=1, dtype=tf.float32)
            w = tf.ceil(w * ratio)
            h = tf.ceil(h * ratio)
            img = tf.image.resize_images(img, [tf.cast(h, tf.int32), tf.cast(w, tf.int32)])
            if inception:
                img = tf.image.resize_images(img, [32,max_width])
            else:
                ratio_w = tf.maximum(w / max_width, 1.0)
                ratio_h = tf.maximum(h / 32.0, 1.0)
                ratio = tf.maximum(ratio_w, ratio_h)
                nw = tf.cast(tf.maximum(tf.floor_div(w, ratio), 1.0), tf.int32)
                nh = tf.cast(tf.maximum(tf.floor_div(h, ratio), 1.0), tf.int32)
                img = tf.image.resize_images(img, [nh, nw])
                padw = tf.maximum(0, int(max_width) - nw)
                padh = tf.maximum(0, 32 - nh)
                img = tf.image.pad_to_bounding_box(img, 0, 0, nh + padh, nw + padw)
                img = tf.cast(img, tf.float32) / 255.0
                label = tf.cast(label, tf.int32)
            return img, label

        ds = ds.map(_parser)

        def _fileter(img, labels):
            return tf.not_equal(0, tf.reduce_sum(labels))

        ds = ds.filter(_fileter)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(1000))
        ds = ds.padded_batch(batch_size, padded_shapes=([32, max_width, 3], [None]))
        return ds

    return _input_fn


def input_fn(params, is_training):
    max_width = params['max_width']
    char_map = params['charset']
    labels = pd.read_csv(params['data_set'] + '/labels.csv', converters={'text': str}, na_values=[],
                         keep_default_na=False)
    limit = params['limit_train']
    if limit is None or limit < 1:
        alldata = labels.iloc[:].values
    else:
        alldata = labels.iloc[:limit].values
    batch_size = params['batch_size']
    count = len(alldata) // batch_size

    def _input_fn():
        def _gen():
            for _ in range(params['epoch']):
                data = np.random.permutation(alldata)
                maxlen = 0
                for j in range(count):
                    features = []
                    labels = []
                    for i in range(batch_size):
                        k = j * batch_size + i
                        image = PIL.Image.open('{}/{}.png'.format(params['data_set'], data[k, 0]))
                        width, height = image.size
                        # logging.info("Width: {} Height: {}".format(width,height))
                        ration_w = max(width / max_width, 1.0)
                        ration_h = max(height / 32.0, 1.0)
                        ratio = max(ration_h, ration_w)
                        if ratio > 1:
                            width = int(width / ratio)
                            height = int(height / ratio)
                            image = image.resize((width, height))
                            w1, h1 = image.size
                            # logging.info("Resize Width: {} Height: {}".format(w1,h1))
                        image = np.asarray(image)
                        pw = max(0, max_width - image.shape[1])
                        ph = max(0, 32 - image.shape[0])
                        image = np.pad(image, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
                        image = image.astype(np.float32) / 127.5 - 1
                        # logging.info("Text {}".format(data[k,1]))
                        label = get_str_labels(char_map, data[k, 1])
                        features.append(image)
                        if len(label) > maxlen:
                            maxlen = len(label)
                        labels.append(np.array(label, dtype=np.int32))
                    for i in range(len(labels)):
                        l = len(labels[i])
                        if l < maxlen:
                            labels[i] = np.pad(labels[i], (0, maxlen - l), 'constant', constant_values=0)

                    yield (np.stack(features), np.stack(labels))

        ds = tf.data.Dataset.from_generator(_gen, (tf.float32, tf.int32), (
            tf.TensorShape([params['batch_size'], 32, max_width, 3]),
            tf.TensorShape([params['batch_size'], None])))
        ds = ds.prefetch(4)
        return ds

    return _input_fn


def _basic_lstm(mode, params, rnn_inputs):
    with tf.variable_scope('LSTM'):
        layers_list = []
        for _ in range(params['num_layers']):
            cell = tf.nn.rnn_cell.BasicLSTMCell(params['hidden_size'], state_is_tuple=True)
            layers_list.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(layers_list, state_is_tuple=True)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # make rnn state for training
        with tf.variable_scope('Hidden_state'):
            state_variables = []
            for state_c, state_h in cell.zero_state(params['batch_size'], tf.float32):
                state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                    tf.Variable(state_h, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
            rnn_state = tuple(state_variables)
    else:
        # use default for evaluation
        rnn_state = cell.zero_state(params['batch_size'], tf.float32)
    with tf.name_scope('LSTM'):
        rnn_inputs = tf.unstack(rnn_inputs,axis=1)
        rnn_output, new_states = tf.nn.static_rnn(cell, rnn_inputs, initial_state=rnn_state)
        rnn_output = tf.stack(rnn_output,axis=1)
    return rnn_output, rnn_state, new_states


def _cudnn_lstm_compatible(params, rnn_inputs):
    if params['lstm_direction_type'] == 'bidirectional':
        with tf.variable_scope('cudnn_lstm'):
            single_cell = lambda: tf.contrib.rnn.BasicLSTMCell(params['hidden_size'], forget_bias=0,
                                                               name="cudnn_compatible_lstm_cell")
            cells_fw = [single_cell() for _ in range(params['num_layers'])]
            cells_bw = [single_cell() for _ in range(params['num_layers'])]
            rnn_state_fw = [cell.zero_state(params['batch_size'], tf.float32) for cell in cells_fw]
            rnn_state_bw = [cell.zero_state(params['batch_size'], tf.float32) for cell in cells_bw]
        with tf.variable_scope('cudnn_lstm'):
            rnn_output, new_state_fw, new_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                rnn_inputs,
                initial_states_fw=rnn_state_fw,
                initial_states_bw=rnn_state_bw,
                sequence_length=None,
                time_major=True)
        return rnn_output, rnn_state_fw + rnn_state_bw, new_state_fw + new_states_bw
    else:
        with tf.variable_scope('cudnn_lstm'):
            single_cell = lambda: tf.contrib.rnn.BasicLSTMCell(params['hidden_size'], forget_bias=0,
                                                               name="cudnn_compatible_lstm_cell")
            cells = [single_cell() for _ in range(params['num_layers'])]
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            rnn_state = cell.zero_state(params['batch_size'], tf.float32)
        with tf.variable_scope('cudnn_lstm'):
            rnn_output, new_states = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=None,
                                                       initial_state=rnn_state, time_major=True)
        return rnn_output, rnn_state, new_states


def _cudnn_lstm(mode, params, rnn_inputs):
    with tf.variable_scope('LSTM'):
        dir = 2 if params['lstm_direction_type'] == 'bidirectional' else 1
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(params['num_layers'], params['hidden_size'],
                                              direction=params['lstm_direction_type'],
                                              dropout=float(params['output_keep_prob']))
        shape = [params['num_layers'] * dir, params['batch_size'], params['hidden_size']]
        rnn_state = (
            tf.Variable(tf.zeros(shape, tf.float32), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
            tf.Variable(tf.zeros(shape, tf.float32), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]))
    with tf.name_scope('LSTM'):
        rnn_output, new_states = cell(rnn_inputs, initial_state=rnn_state,
                                      training=(mode == tf.estimator.ModeKeys.TRAIN))
    return rnn_output, rnn_state, new_states

def _inception_v3_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.1,
                            batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'is_training': is_training,
        'decay': 0.9997,
        'epsilon': 0.001,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    normalizer_fn = slim.batch_norm
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu6,
                normalizer_fn=normalizer_fn,
                normalizer_params=batch_norm_params) as sc:
            return sc


def _inception(images, params):
    with slim.arg_scope(_inception_v3_arg_scope(is_training=True)):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.batch_norm],
                trainable=True):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout], is_training=True):
                net, _ = inception_v3.inception_v3_base(
                    images,
                    scope='InceptionV3',
                    final_endpoint='Mixed_5d')
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(1, 1), padding="valid", activation=tf.nn.relu)
    logging.info("conv7 {}".format(net.shape))

    return tf.reshape(net, [params['batch_size'], -1, 512])
    return net

def resnet_block(inputs,filters,strides,is_training):
    shortcut = inputs
    inputs = tf.layers.batch_normalization(inputs,training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3,strides=strides, padding="same")
    inputs = tf.layers.batch_normalization(inputs,training=is_training)
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3,strides=1, padding="same")
    return tf.concat([inputs,shortcut], 3)

def resnet(images,params,is_training):
    conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3, 3), padding="same")
    conv1 = tf.layers.batch_normalization(conv1,training=is_training)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    logging.info("conv1 {}".format(conv1.shape))

    conv2 = resnet_block(conv1,64,1,is_training)
    logging.info("conv2 {}".format(conv2.shape))
    conv3 = resnet_block(conv2,128,1,is_training)
    conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[1,2])
    logging.info("conv3 {}".format(conv3.shape))
    conv4 = resnet_block(conv3,256,1,is_training)
    conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=1)
    logging.info("conv3 {}".format(conv4.shape))
    conv5 = resnet_block(conv4,256,1,is_training)
    conv5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=[1,2])
    logging.info("conv5 {}".format(conv5.shape))
    conv6 = resnet_block(conv5,256,1,is_training)
    conv6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=1)
    logging.info("conv5 {}".format(conv6.shape))
    conv7 = resnet_block(conv6,256,1,is_training)
    conv7 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
    logging.info("conv7 {}".format(conv7.shape))


    return tf.reshape(conv7, [params['batch_size'], -1, 1280])

def plain_cnn(images,params,is_training):
    # 64 / 3 x 3 / 1 / 1
    conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv1 {}".format(conv1.shape))

    # 2 x 2 / 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    logging.info("pool1 {}".format(pool1.shape))

    # 128 / 3 x 3 / 1 / 1
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv2 {}".format(conv2.shape))
    # 2 x 2 / 1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    logging.info("pool2 {}".format(pool2.shape))

    # 256 / 3 x 3 / 1 / 1
    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv3 {}".format(conv3.shape))

    # Batch normalization layer
    bnorm1 = tf.layers.batch_normalization(conv3,training=is_training)

    # 256 / 3 x 3 / 1 / 1
    conv4 = tf.layers.conv2d(inputs=bnorm1, filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv4 {}".format(conv4.shape))

    # 1 x 2 / 1
    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")
    logging.info("pool3 {}".format(pool3.shape))

    # 512 / 3 x 3 / 1 / 1
    conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv5 {}".format(conv5.shape))

    # Batch normalization layer
    bnorm2 = tf.layers.batch_normalization(conv5,training=is_training)

    # 512 / 3 x 3 / 1 / 1
    conv6 = tf.layers.conv2d(inputs=bnorm2, filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv6 {}".format(conv6.shape))

    # 1 x 2 / 2
    pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")
    logging.info("pool4 {}".format(pool4.shape))
    # 512 / 2 x 2 / 1 / 0
    conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(2, 2), padding="valid", activation=tf.nn.relu)
    logging.info("conv7 {}".format(conv7.shape))

    return tf.reshape(conv7, [params['batch_size'], -1, 512])


def _crnn_model_fn(features, labels, mode, params=None, config=None):
    if isinstance(features, dict):
        features = features['images']
    max_width = params['max_width']
    global_step = tf.train.get_or_create_global_step()
    logging.info("Features {}".format(features.shape))
    features = tf.reshape(features, [params['batch_size'], 32, max_width, 3])
    images = tf.transpose(features, [0, 2, 1, 3])
    logging.info("Images {}".format(images.shape))
    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
        labels = tf.reshape(labels, [params['batch_size'], -1])
        tf.summary.image('image', features)
        idx = tf.where(tf.not_equal(labels, 0))
        sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx),
                                        [params['batch_size'], params['max_target_seq_length']])
        sparse_labels, _ = tf.sparse_fill_empty_rows(sparse_labels, params['num_labels'] - 1)

    if params['cnn_type']=='inception':
        reshaped_cnn_output = _inception(images,params)
    elif params['cnn_type']=='resnet':
        reshaped_cnn_output = resnet(images,params,mode == tf.estimator.ModeKeys.TRAIN)
    else:
        reshaped_cnn_output = plain_cnn(images,params,mode == tf.estimator.ModeKeys.TRAIN)

    if params['rnn_type'] != 'BasicLSTM':
        rnn_inputs = tf.transpose(reshaped_cnn_output, perm=[1, 0, 2])
        max_char_count = rnn_inputs.get_shape().as_list()[0]
    else:
        rnn_inputs = reshaped_cnn_output
        max_char_count = rnn_inputs.get_shape().as_list()[1]


    logging.info("max_char_count {}".format(max_char_count))
    input_lengths = tf.zeros([params['batch_size']], dtype=tf.int32) + max_char_count
    logging.info("InpuLengh {}".format(input_lengths.shape))

    if params['rnn_type'] == 'CudnnLSTM':
        rnn_output, rnn_state, new_states = _cudnn_lstm(mode, params, rnn_inputs)
    elif params['rnn_type'] == 'CudnnCompatibleLSTM':
        rnn_output, rnn_state, new_states = _cudnn_lstm_compatible(params, rnn_inputs)
    else:
        rnn_output, rnn_state, new_states = _basic_lstm(mode, params, rnn_inputs)

    with tf.variable_scope('Output_layer'):
        logits = tf.layers.dense(rnn_output, params['num_labels'],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

    if params['rnn_type'] == 'BasicLSTM':
        logits = tf.transpose(logits, perm=[1, 0, 2])

    if params['beam_search_decoder'] and  params['rnn_type'] != 'BasicLSTM':
        decoded, _log_prob = tf.nn.ctc_beam_search_decoder(logits, input_lengths, merge_repeated=False)
    else:
        decoded, _log_prob = tf.nn.ctc_greedy_decoder(logits, input_lengths,merge_repeated=False)

    prediction = tf.to_int32(decoded[0])

    metrics = {}
    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
        levenshtein = tf.edit_distance(prediction, sparse_labels, normalize=True)
        errors_rate = tf.metrics.mean(levenshtein)
        mean_error_rate = tf.reduce_mean(levenshtein)
        metrics['Error_Rate'] = errors_rate
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('Error_Rate', mean_error_rate)
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(sparse_labels, logits, input_lengths, ignore_longer_outputs_than_inputs=True)
            mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(input_lengths)))
            loss = mean_loss
    else:
        loss = None

    training_hooks = []

    if mode == tf.estimator.ModeKeys.TRAIN:

        opt = tf.train.AdamOptimizer(params['learning_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if params['grad_clip'] is None:
                train_op = opt.minimize(loss, global_step=global_step)
            else:
                gradients, variables = zip(*opt.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, params['grad_clip'])
                train_op = opt.apply_gradients([(gradients[i], v) for i, v in enumerate(variables)],
                                               global_step=global_step)
    elif mode == tf.estimator.ModeKeys.EVAL:
        train_op = None
    else:
        train_op = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.sparse_to_dense(tf.to_int32(prediction.indices),
                                         tf.to_int32(prediction.dense_shape),
                                         tf.to_int32(prediction.values),
                                         default_value=-1,
                                         name="output")
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}
    else:
        predictions = None
        export_outputs = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=predictions,
        loss=loss,
        training_hooks=training_hooks,
        export_outputs=export_outputs,
        train_op=train_op)


class BaseOCR(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _crnn_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config)

        super(BaseOCR, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )

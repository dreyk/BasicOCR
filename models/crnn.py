from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
import logging
import PIL.Image
import numpy as np
import random
import math
import cv2
import models.charset as charset





def null_dataset():
    def _input_fn():
        return None

    return _input_fn

def _features(img,width,labels):
    return {'image':img,'width':width},labels

def get_input_fn(dataset_type):
    if dataset_type == 'full-generated':
        return full_generated_input_fn
    if dataset_type == 'tf-record':
        return tf_input_fn
    else:
        return tf_input_fn


def full_generated_input_fn(params, is_training):
    logging.info('Use full generated')
    max_width = params['max_width']
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
            label = charset.string_to_label(text)
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
            label = charset.string_to_label(label)
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


def _crop_py_and_rotate(img, ymin, xmin, ymax, xmax, texts, gxs, gys):
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
    ilabel = np.array(charset.string_to_label(label), np.int32)
    return np.array([img.shape[0], img.shape[1]], np.int32), img, ilabel


def _crop_py(img, ymin, xmin, ymax, xmax, texts):
    i = random.randrange(len(ymin))
    y0 = max(ymin[i] - 2, 0)
    x0 = max(xmin[i] - 2, 0)
    y1 = min(ymax[i] + 2, img.shape[0])
    x1 = min(xmax[i] + 2, img.shape[1])
    img = img[y0:y1, x0:x1, :]
    label = str(texts[i], encoding='UTF-8')
    ilabel = np.array(charset.string_to_label(label), np.int32)
    return np.array([img.shape[0], img.shape[1]], np.int32), img, ilabel

def rescale_image( image ):
    image = tf.image.convert_image_dtype( image, tf.float32 )
    image = tf.subtract( image, 0.5 )
    return image

def normalize_image( image ):
    image = tf.image.rgb_to_grayscale( image )
    image = rescale_image( image )
    image_height = tf.cast(tf.shape(image)[0], tf.float64)
    image_width = tf.shape(image)[1]

    scaled_image_width = tf.cast(
        tf.round(
            tf.multiply(tf.cast(image_width,tf.float64),
                        tf.divide(32.0,image_height)) ),
        tf.int32)

    image = tf.image.resize_images(image, [32, scaled_image_width],
                                   tf.image.ResizeMethod.BICUBIC )

    return image,scaled_image_width

def tf_input_fn(params, is_training):
    batch_size = params['batch_size']
    datasets_files = []
    for tf_file in glob.iglob(params['data_set'] + '/*.record'):
        datasets_files.append(tf_file)
    random.shuffle(datasets_files)
    augs = params['aug'].split(',')
    rotate = 'rotate' in augs
    logging.info('Do rotation: {}'.format(rotate))
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
                    _crop_py_and_rotate,
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
            img,inference_w = normalize_image(img)
            return img,inference_w,label

        ds = ds.map(_parser)

        def _fileter(_img,_inference_w,labels):
            return tf.not_equal(0, tf.reduce_sum(labels))

        ds = ds.filter(_fileter)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(1000))
        ds = ds.padded_batch(batch_size, padded_shapes=([32, None,1],tf.TensorShape([]),[None]))
        ds = ds.map(_features)
        return ds

    return _input_fn



# Layer params:   Filts K  Padding  Name     BatchNorm?
layer_params = [ [  64, 3, 'valid', 'conv1', False],
                 [  64, 3, 'same',  'conv2', True],  # pool
                 [ 128, 3, 'same',  'conv3', False],
                 [ 128, 3, 'same',  'conv4', True],  # hpool
                 [ 256, 3, 'same',  'conv5', False],
                 [ 256, 3, 'same',  'conv6', True],  # hpool
                 [ 512, 3, 'same',  'conv7', False],
                 [ 512, 3, 'same',  'conv8', True] ] # hpool 3

rnn_size = 2**9    # Dimensionality of all RNN elements' hidden layers
dropout_rate = 0.5 # For RNN layers (currently not used--uncomment below)

def conv_layer( bottom, params, training ):
    """Build a convolutional layer using entry from layer_params)"""

    batch_norm = params[4] # Boolean

    if batch_norm:
        activation = None
    else:
        activation = tf.nn.relu

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer( value=0.0 )

    top = tf.layers.conv2d( bottom,
                            filters=params[0],
                            kernel_size=params[1],
                            padding=params[2],
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            name=params[3] )
    if batch_norm:
        top = norm_layer( top, training, params[3]+'/batch_norm' )
        top = tf.nn.relu( top, name=params[3]+'/relu' )

    return top


def pool_layer( bottom, wpool, padding, name ):
    """Short function to build a pooling layer with less syntax"""
    top = tf.layers.max_pooling2d( bottom,
                                   2,
                                   [2, wpool],
                                   padding=padding,
                                   name=name )
    return top


def norm_layer( bottom, training, name):
    """Short function to build a batch normalization layer with less syntax"""
    top = tf.layers.batch_normalization( bottom,
                                         axis=3, # channels last
                                         training=training,
                                         name=name )
    return top


def convnet_layers( inputs, widths, mode ):
    """
    Build convolutional network layers attached to the given input tensor
    """

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # inputs should have shape [ ?, 32, ?, 1 ]
    with tf.variable_scope( "convnet" ): # h,w

        conv1 = conv_layer( inputs, layer_params[0], training ) # 30,30
        conv2 = conv_layer( conv1, layer_params[1], training )  # 30,30
        pool2 = pool_layer( conv2, 2, 'valid', 'pool2' )        # 15,15
        conv3 = conv_layer( pool2, layer_params[2], training )  # 15,15
        conv4 = conv_layer( conv3, layer_params[3], training )  # 15,15
        pool4 = pool_layer( conv4, 1, 'valid', 'pool4' )        # 7,14
        conv5 = conv_layer( pool4, layer_params[4], training )  # 7,14
        conv6 = conv_layer( conv5, layer_params[5], training )  # 7,14
        pool6 = pool_layer( conv6, 1, 'valid', 'pool6')         # 3,13
        conv7 = conv_layer( pool6, layer_params[6], training )  # 3,13
        conv8 = conv_layer( conv7, layer_params[7], training )  # 3,13
        pool8 = tf.layers.max_pooling2d( conv8, [3, 1], [3, 1],
                                         padding='valid',
                                         name='pool8' )         # 1,13
        # squeeze row dim
        features = tf.squeeze( pool8, axis=1, name='features' )

        sequence_length = get_sequence_lengths( widths )

        # Vectorize
        sequence_length = tf.reshape( sequence_length, [-1], name='seq_len' )

        return features, sequence_length


def get_sequence_lengths( widths ):
    """Tensor calculating output sequence length from original image widths"""
    kernel_sizes = [params[1] for params in layer_params]

    with tf.variable_scope("sequence_length"):
        conv1_trim = tf.constant( 2 * (kernel_sizes[0] // 2),
                                  dtype=tf.int32,
                                  name='conv1_trim' )
        one = tf.constant( 1, dtype=tf.int32, name='one' )
        two = tf.constant( 2, dtype=tf.int32, name='two' )
        after_conv1 = tf.subtract( widths, conv1_trim, name='after_conv1' )
        after_pool2 = tf.floor_div( after_conv1, two, name='after_pool2' )
        after_pool4 = tf.subtract( after_pool2, one, name='after_pool4' )
        after_pool6 = tf.subtract( after_pool4, one, name='after_pool6' )
        after_pool8 = tf.identity( after_pool6, name='after_pool8' )
    return after_pool8


def rnn_layer( bottom_sequence, sequence_length, rnn_size, scope ):
    """Build bidirectional (concatenated output) RNN layer"""


    # Default activation is tanh
    cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell( rnn_size )
    cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_size )

    # Pre-CUDNN (slower) alternatve. Default activation is tanh .
    #cell_fw = tf.contrib.rnn.LSTMCell( rnn_size,
    #                                   initializer=weight_initializer)
    #cell_bw = tf.contrib.rnn.LSTMCell( rnn_size,
    #                                   initializer=weight_initializer)

    # Include?
    #cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw,
    #                                         input_keep_prob=dropout_rate )
    #cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw,
    #                                         input_keep_prob=dropout_rate )

    rnn_output,_ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope )

    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    rnn_output_stack = tf.concat( rnn_output, 2, name='output_stack' )

    return rnn_output_stack


def rnn_layers( features, sequence_length, num_classes ):
    """Build a stack of RNN layers from input features"""

    # Input features is [batchSize paddedSeqLen numFeatures]
    logit_activation = tf.nn.relu
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer( value=0.0 )

    with tf.variable_scope( "rnn" ):
        # Transpose to time-major order for efficiency
        rnn_sequence = tf.transpose( features,
                                     perm=[1, 0, 2],
                                     name='time_major' )
        rnn1 = rnn_layer( rnn_sequence, sequence_length, rnn_size, 'bdrnn1' )
        rnn2 = rnn_layer( rnn1, sequence_length, rnn_size, 'bdrnn2' )
        rnn_logits = tf.layers.dense( rnn2,
                                      num_classes+1,
                                      activation=logit_activation,
                                      kernel_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      name='logits' )
        return rnn_logits

def _crnn_model_fn(features, labels, mode, params=None, config=None):
    image = features['image']
    width = features['width']
    global_step = tf.train.get_or_create_global_step()
    conv_features,sequence_length = convnet_layers( image,width,mode )
    logits = rnn_layers( conv_features, sequence_length,charset.num_classes() )
    predictions = None
    export_outputs = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        idx = tf.where(tf.not_equal(labels,charset))
        maxl = tf.cast(tf.reduce_max(idx,axis=0),tf.int64) + 1
        sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx),
                                    [params['batch_size'], maxl])
        labels, _ = tf.sparse_fill_empty_rows(sparse_labels,charset.num_classes()+1)
        with tf.name_scope( "train" ):
            tf.summary.image('image', image)
            losses = tf.nn.ctc_loss( labels,
                                     logits,
                                     sequence_length,
                                     time_major=True,
                                     ignore_longer_outputs_than_inputs=True )
            decoded, _log_prob = tf.nn.ctc_greedy_decoder(logits, sequence_length)
            loss = tf.reduce_mean( losses )
            prediction = tf.to_int32(decoded[0])
            levenshtein = tf.edit_distance(prediction, labels, normalize=True)
            mean_error_rate = tf.reduce_mean(levenshtein)
            tf.summary.scalar('Error_Rate', mean_error_rate)

            # Update batch norm stats [http://stackoverflow.com/questions/43234667]
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(extra_update_ops):

                # Calculate the learning rate given the parameters
                learning_rate_tensor = tf.train.exponential_decay(
                    params['learning_rate'],
                    tf.train.get_global_step(),
                    params['decay_steps'],
                    params['decay_rate'],
                    staircase=params['decay_staircase'],
                    name='learning_rate')

                optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate_tensor,
                    beta1=params['momentum'])

                train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=global_step,
                    learning_rate=learning_rate_tensor,
                    optimizer=optimizer)

                tf.summary.scalar('learning_rate', learning_rate_tensor)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        training_hooks=[],
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

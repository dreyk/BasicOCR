from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging
import cv2
import numpy as np
import random
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter
import glob
import string

ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', ':', '(', ')', '.', ',', '/'
    # Apostrophe only for specific cases (eg. : O'clock)
                                  "'",
    " ",
    # "end of sentence" character for CTC algorithm
    '_'
]


def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset


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
    if add_eos:
        result.append(len(char_map) + 1)
    return result


def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def get_input_fn(dataset_type):
    return numbers_input_fn


def do_it():
    return random.randint(0, 1)


def random_string(l=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(l))


def fake_number():
    if random.randint(1, 10) < 4:
        sv = random_string(random.randint(3, 10))
        return sv, sv
    v = random.randint(1, 100000000)
    n = '{:,}'.format(v)
    if do_it():
        n = n.replace(',', ' , ')
    if do_it():
        n = '$' + n
    if random.randint(1, 10) < 4:
        sv = random_string(random.randint(1, 5))
        return str(v) + ' ' + sv, n + ' ' + sv
    return str(v), n


def erode(img, k):
    if k == 0:
        return img
    kernel = np.ones((k, k), np.uint8)
    for y in range(k):
        for x in range(k):
            if (x + 1) % 2 == 0 and y % 2 == 0:
                kernel[y, x] = 0

    img_erosion = cv2.erode(np.array(img), kernel, iterations=2)
    return Image.fromarray(img_erosion)


def bluring(img, r):
    if r == 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(r))


def box_geerator(text, fonts):
    clr = random.randint(0, 100)
    baks_size = random.randint(1, 36)
    f = random.randint(0, len(fonts) - 1)

    if do_it():
        baks_font = ImageFont.truetype(font=fonts[f], size=baks_size)
        baks_width, baks_height = baks_font.getsize('$ ')
    else:
        baks_font = None
        baks_width = 0
        baks_height = 0
    number_size = random.randint(12, 36)
    text_font = ImageFont.truetype(font=fonts[f], size=number_size)
    text_width, text_height = text_font.getsize(text)
    height = max(text_height, baks_height)
    height = height + random.randint(0, height)
    width = baks_width + text_width + random.randint(0, int(text_width / 2))
    img = Image.new('L', (width, height), (255))
    txt_draw = ImageDraw.Draw(img)
    baksy = random.randint(0, height - baks_height)
    baksx = random.randint(0, width - text_width - baks_width)
    if baks_font is not None:
        txt_draw.text((baksx, baksy), '$', fill=clr, font=baks_font)
    texty = random.randint(0, height - text_height)
    textx = baksx + baks_width + random.randint(0, max(0, width - text_width - baksx - baks_width))

    txt_draw.text((textx, texty), text, fill=clr, font=text_font)

    if do_it():
        txt_draw.rectangle([random.randint(0, baksx), random.randint(0, baksy), width - random.randint(0, 10),
                            height - random.randint(0, 10)], outline=0)
    return img


def numbers_input_fn(params, is_training):
    max_width = params['max_width']
    char_map = params['charset']
    batch_size = params['batch_size']
    fonts = glob.glob(params['data_set'] + '/*')
    logging.info('Fonts: {}'.format(fonts))

    def _input_fn():
        def _gen():
            for _ in range(params['epoch']):
                for j in range(1000):
                    text, show_text = fake_number()
                    # text = show_text
                    if j == 0:
                        logging.info("{} / {}".format(text, show_text))
                    image = box_geerator(show_text, fonts)
                    image = image.resize((max_width, 32))
                    image = np.asarray(image)
                    image = np.stack([image, image, image], axis=-1)
                    image = image.astype(np.float32) / 127.5 - 1
                    label = get_str_labels(char_map, text)
                    yield image, np.array(label, dtype=np.int32)

        ds = tf.data.Dataset.from_generator(_gen, (tf.float32, tf.int32), (
            tf.TensorShape([32, max_width, 3]),
            tf.TensorShape([None])))
        ds = ds.padded_batch(batch_size, padded_shapes=([32, max_width, 3], [None]))
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
        rnn_output, new_states = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=rnn_state, time_major=True)
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
        tf.summary.image('image', features, params['batch_size'])
        idx = tf.where(tf.not_equal(labels, 0))
        sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx),
                                        [params['batch_size'], params['max_target_seq_length']])
        sparse_labels, _ = tf.sparse_fill_empty_rows(sparse_labels, params['num_labels'] - 1)

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
    bnorm1 = tf.layers.batch_normalization(conv3)

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
    bnorm2 = tf.layers.batch_normalization(conv5)

    # 512 / 3 x 3 / 1 / 1
    conv6 = tf.layers.conv2d(inputs=bnorm2, filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    logging.info("conv6 {}".format(conv6.shape))

    # 1 x 2 / 2
    pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")
    logging.info("pool4 {}".format(pool4.shape))
    # 512 / 2 x 2 / 1 / 0
    conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(2, 2), padding="valid", activation=tf.nn.relu)
    logging.info("conv7 {}".format(conv7.shape))

    reshaped_cnn_output = tf.reshape(conv7, [params['batch_size'], -1, 512])
    rnn_inputs = tf.transpose(reshaped_cnn_output, perm=[1, 0, 2])

    max_char_count = rnn_inputs.get_shape().as_list()[0]
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

    if params['beam_search_decoder']:
        decoded, _log_prob = tf.nn.ctc_beam_search_decoder(logits, input_lengths, merge_repeated=False)
    else:
        decoded, _log_prob = tf.nn.ctc_greedy_decoder(logits, input_lengths)

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

import tensorflow as tf
import random
import glob
import numpy as np
import logging

ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0','1','2','3','4','5','6','7','8','9',
    '-',':','(',')','.',',','/','$',
    "'",
    " ",
    '_'
]

def read_charset():
    charset = {}
    inv_charset = {}
    for i,v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset

wide_charset = None

def get_str_labels(char_map, v, add_eos=True):
    v = v.strip()
    v = v.lower()
    result = []
    for t in v:
        if t == '#' or t=='_':
            continue
        i = char_map.get(t, -1)
        if i >= 0:
            result.append(i)
    if add_eos:
        result.append(len(char_map)-1)
    return result

def _crop_py(img, ymin, xmin, ymax,xmax,texts):
    i = random.randrange(len(ymin))
    y0 = max(ymin[i],0)
    x0 = max(xmin[i],0)
    y1 = min(ymax[i],img.shape[0])
    x1 = min(xmax[i],img.shape[1])
    img = img[y0:y1,x0:x1,:]
    label = str(texts[i], encoding='UTF-8')
    return np.array([img.shape[0],img.shape[1]],np.int32),img,np.array(get_str_labels(wide_charset, label),np.int32)


def tf_input_fn(params, is_training):
    max_width = params['max_width']
    global wide_charset
    wide_charset = params['charset']
    batch_size = params['batch_size']
    datasets_files = []
    for tf_file in glob.iglob(params['data_set'] + '/*.record'):
        datasets_files.append(tf_file)
    random.shuffle(datasets_files)
    def _input_fn():
        ds = tf.data.TFRecordDataset(datasets_files, buffer_size=256 * 1024 * 1024)
        def _parser(example):
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
            ymin = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymin']), tf.float32)*original_h,tf.int32)
            xmin = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmin']), tf.float32)*original_w,tf.int32)
            xmax = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmax']), tf.float32)*original_w,tf.int32)
            ymax = tf.cast(tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymax']), tf.float32)*original_h,tf.int32)
            texts = tf.sparse_tensor_to_dense(res['image/object/bbox/label_text'],default_value='')
            size,img,label = tf.py_func(
                _crop_py,
                [img,ymin, xmin, ymax,xmax,texts],
                [tf.int32,tf.uint8,tf.int32]
            )
            original_h = size[0]
            original_w = size[1]
            img = tf.reshape(img, [original_h, original_w, 3])
            w = tf.maximum(tf.cast(original_w, tf.float32),1.0)
            h = tf.maximum(tf.cast(original_h, tf.float32),1.0)
            min_ration = 10.0/tf.minimum(w,h)
            max_ratio = tf.maximum(min_ration,1.0)
            ratio = tf.random_uniform((),minval=min_ration,maxval=max_ratio,dtype=tf.float32)
            w = tf.ceil(w*ratio)
            h = tf.ceil(h*ratio)
            img = tf.image.resize_images(img, [tf.cast(h,tf.int32), tf.cast(w,tf.int32)])
            ratio_w = tf.maximum(w / max_width, 1.0)
            ratio_h = tf.maximum(h / 32.0, 1.0)
            ratio = tf.maximum(ratio_w, ratio_h)
            nw = tf.cast(tf.maximum(tf.floor_div(w , ratio),1.0), tf.int32)
            nh = tf.cast(tf.maximum(tf.floor_div(h , ratio),1.0), tf.int32)
            img = tf.image.resize_images(img, [nh, nw])
            padw = tf.maximum(0, int(max_width) - nw)
            padh = tf.maximum(0, 32 - nh)
            img = tf.image.pad_to_bounding_box(img, 0, 0, nh + padh, nw + padw)
            img = tf.cast(img, tf.float32) / 255.0
            label = tf.cast(label, tf.int32)
            return img, label

        ds = ds.map(_parser)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(1000))
        ds = ds.padded_batch(batch_size, padded_shapes=([32, max_width, 3], [None]))
        return ds

    return _input_fn

def _im2letter_model_fn(features, labels, mode, params=None, config=None):
    if isinstance(features, dict):
        features = features['images']

    global_step = tf.train.get_or_create_global_step()

    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
        labels = tf.reshape(labels, [params['batch_size'], -1])
        tf.summary.image('image', features)
        idx = tf.where(tf.not_equal(labels, 0))
        sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx),
                                        [params['batch_size'], params['max_target_seq_length']])
        sparse_labels, _ = tf.sparse_fill_empty_rows(sparse_labels, params['num_labels'] - 1)

    outputs = tf.layers.conv2d(
        features, filters=128, kernel_size=[32, 32], strides=3, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('1: {}'.format(outputs))
    outputs = tf.layers.conv2d(
        outputs, filters=128, kernel_size=[7, 7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('2: {}'.format(outputs))
    outputs = tf.layers.conv2d(
        outputs, filters=128, kernel_size=[7, 7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('5: {}'.format(outputs))
    outputs = tf.layers.conv2d(
        outputs, filters=128, kernel_size=[7, 7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('6: {}'.format(outputs))
    outputs = tf.layers.conv2d(
        outputs, filters=128, kernel_size=[7, 7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('7: {}'.format(outputs))
    outputs = tf.layers.conv2d(
        outputs, filters=128, kernel_size=[7, 7], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('8: {}'.format(outputs))
    outputs = tf.layers.conv2d(
        outputs, filters=256, kernel_size=[32, 32], strides=2, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('9: {}'.format(outputs))
    outputs = tf.layers.conv2d(
        outputs, filters=256, kernel_size=[2, 2], strides=1, padding='SAME', activation=tf.nn.relu
    )
    outputs = tf.layers.batch_normalization(outputs)
    logging.info('10: {}'.format(outputs))
    y_pred = tf.layers.conv2d(
        outputs, filters=params['num_labels'], kernel_size=[2, 2], strides=1, padding='SAME'
    )
    y_pred = tf.layers.batch_normalization(y_pred)
    logging.info('y_pred1: {}'.format(y_pred))
    y_pred = tf.layers.max_pooling2d(y_pred,[y_pred.shape[1],1],1)
    logging.info('y_pred2: {}'.format(y_pred))
    y_pred = tf.reshape(y_pred, shape=(params['batch_size'],y_pred.shape[1]*y_pred.shape[2], y_pred.shape[3]))
    y_pred = tf.nn.log_softmax(y_pred)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    max_char_count = y_pred.get_shape().as_list()[0]
    input_lengths = tf.zeros([params['batch_size']], dtype=tf.int32) + max_char_count
    if params['beam_search_decoder']:
        decoded, _log_prob = tf.nn.ctc_beam_search_decoder(y_pred, input_lengths,merge_repeated=False)
    else:
        decoded, _log_prob = tf.nn.ctc_greedy_decoder(y_pred, input_lengths)
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
            ctc_loss = tf.nn.ctc_loss(sparse_labels, y_pred, input_lengths, ignore_longer_outputs_than_inputs=True)
            mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(input_lengths)))
            loss = mean_loss
    else:
        loss = None

    training_hooks = []

    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdamOptimizer(params['learning_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)
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


class Image2Letter(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _im2letter_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config
            )

        super(Image2Letter, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )

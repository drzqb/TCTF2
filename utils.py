import tensorflow as tf
import collections
import re
import tensorflow.keras.backend as K
import numpy as np


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def convert2Uni(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        print(type(text))
        print('####################wrong################')


def load_vocab(vocab_file):  # 获取BERT字表方法
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            tmp = reader.readline()
            if not tmp:
                break
            token = convert2Uni(tmp)
            token = token.rstrip("\n")
            vocab[token] = index
            index += 1
    return vocab


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        'label': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    label = sequence_parsed['label']
    return sen, label


def single_example_parser_eb(serialized_example):
    context_features = {
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    sequence_features = {
        "sen": tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    label = context_parsed['label']

    return sen, label


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, shuffle=True, buffer_size=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset


def AccumOptimizer(BaseOptimizer):
    class NewOptimizer(BaseOptimizer):
        """
        带有梯度累积的优化器
        """

        def __init__(self, grad_accum_steps=2, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self.grad_accum_steps = grad_accum_steps

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, 'ag')

        def _resource_apply(self, grad, var, indices=None):
            # 更新判据
            cond = K.equal(self.iterations % self.grad_accum_steps, 0)
            # 获取梯度
            ag = self.get_slot(var, 'ag')

            old_update = K.update

            def new_update(x, new_x):
                new_x = K.switch(cond, new_x, x)
                return old_update(x, new_x)

            K.update = new_update
            ag_t = ag / self.grad_accum_steps
            op = super(NewOptimizer, self)._resource_apply(ag_t, var)
            K.update = old_update

            # 累积梯度
            with tf.control_dependencies([op]):
                ag_t = K.switch(cond, K.zeros_like(ag), ag)
                with tf.control_dependencies([K.update(ag, ag_t)]):
                    if indices is None:
                        ag_t = K.update(ag, ag + grad)
                    else:
                        ag_t = self._resource_scatter_add(ag, indices, grad)

            return ag_t

        def get_config(self):
            config = {
                'grad_accum_steps': self.grad_accum_steps,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


class replace():
    def __init__(self):
        rep = {
            '“': '"',
            '”': '"',
            ' ': ''
        }
        self.rep = dict((re.escape(k), v) for k, v in rep.items())
        self.pattern = re.compile("|".join(self.rep.keys()))

    def replace(self, words):
        return self.pattern.sub(lambda m: self.rep[re.escape(m.group(0))], words)


def softmax(a, mask):
    """
    :param a: B*ML1*ML2
    :param mask: B*ML1*ML2
    """
    return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)


def focal_loss(y_true, y_pred, gamma=15.0):
    """
    Focal Loss 针对样本不均衡
    :param y_true: 样本标签
    :param y_pred: 预测值（sigmoid）
    :return: focal loss
    """

    alpha = 0.5
    loss = tf.where(tf.equal(y_true, 1),
                    -alpha * (1.0 - y_pred) ** gamma * tf.math.log(y_pred),
                    -(1.0 - alpha) * y_pred ** gamma * tf.math.log(1.0 - y_pred))

    return tf.reduce_mean(loss)


def bce_loss_weight(y_true, y_pred):
    """
    bce Loss 针对样本不均衡，给出样本权重
    :param y_true: 样本标签
    :param y_pred: 预测值（sigmoid）
    :return: bce loss
    """
    class_weight = np.array([200., 2253.]) / (2253. + 200.)
    loss = tf.where(tf.equal(y_true, 1),
                    -class_weight[1] * tf.math.log(y_pred),
                    -class_weight[0] * tf.math.log(1.0 - y_pred))

    return tf.reduce_mean(loss)

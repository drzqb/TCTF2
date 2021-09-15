'''
    基于FastFormer的文本分类
    基于词
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, LayerNormalization, Embedding
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers.optimization_tf import AdamWeightDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import BinaryCrossentropy
from utils import focal_loss, bce_loss_weight
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import argparse
import os, jieba
import matplotlib.pyplot as plt

from utils import replace, load_vocab, single_example_parser_eb, batched_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--maxword', default=512, type=int, help='The max length of input sequence')
parser.add_argument('--vocab_size', default=289689, type=int, help='type_vocab_size')
parser.add_argument('--drop_rate', default=0.2, type=float, help='rate for dropout')
parser.add_argument('--block', type=int, default=2, help='number of Encoder submodel')
parser.add_argument('--head', type=int, default=10, help='number of multi_head attention')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=100, help='Epochs during training')
parser.add_argument('--patience', type=int, default=7, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-3, help='Initial learing rate')
parser.add_argument('--hidden_size', type=int, default=100, help='Embedding size for QA words')
parser.add_argument('--intermediate_size', type=int, default=100, help='Embedding size for QA words')
parser.add_argument('--check', type=str, default='modelfiles/textclassify_eb_fastformer_flw',
                    help='The path where modelfiles shall be saved')
parser.add_argument('--mode', type=str, default='train0', help='The mode of train or predict as follows: '
                                                               'train0: begin to train or retrain'
                                                               'tran1:continue to train'
                                                               'predict: predict')

params = parser.parse_args()

tf.random.set_seed(100)
np.random.seed(100)


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


def gelu(x):
    return x * 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        mask = tf.greater(sen, 0)
        mask = tf.where(mask,
                        tf.zeros_like(sen, tf.float32),
                        (1.0 - tf.pow(2.0, 31.0)) * tf.ones_like(sen, tf.float32))

        return tf.tile(tf.expand_dims(mask, axis=1), [1, params.head, 1])


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        newembedding = np.load("data/OriginalFile/embedding_matrix.npy")
        self.word_embeddings = Embedding(289689, 100, name='word_embeddings',
                                         embeddings_initializer=tf.constant_initializer(newembedding),
                                         dtype=tf.float32,
                                         # trainable=False,
                                         )
        self.position_embeddings = self.add_weight(name='position_embeddings',
                                                   shape=[params.maxword, params.hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=create_initializer())
        self.layernorm = LayerNormalization(name='layernorm-pre', epsilon=1e-6)
        self.dropout = Dropout(rate=params.drop_rate)
        super(Embeddings, self).build(input_shape)

    def call(self, sen, **kwargs):
        sen_embed = self.word_embeddings(sen)
        seq_length = tf.shape(sen)[1]
        return self.dropout(self.layernorm(sen_embed + self.position_embeddings[:seq_length]))


# class Attention(Layer):
#     def __init__(self, **kwargs):
#         super(Attention, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.dense_q = Dense(params.hidden_size,
#                              name='query',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#         self.dense_k = Dense(params.hidden_size,
#                              name='key',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#         self.dense_v = Dense(params.hidden_size,
#                              name='value',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#         self.alpha = Dense(params.head,
#                            name='alpha',
#                            dtype=tf.float32,
#                            kernel_initializer=create_initializer())
#         self.beta = Dense(params.head,
#                           name='beta',
#                           dtype=tf.float32,
#                           kernel_initializer=create_initializer())
#
#         self.dense_u = Dense(params.hidden_size,
#                              name='upvalue',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#
#         self.dense_o = Dense(params.hidden_size,
#                              name='output',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#
#         self.dropout1 = Dropout(rate=params.drop_rate)
#         self.dropout2 = Dropout(rate=params.drop_rate)
#         self.dropout3 = Dropout(rate=params.drop_rate)
#         self.layernorm = LayerNormalization(name='layernormattn', epsilon=1e-6)
#
#         super(Attention, self).build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         # x: B*N*768 mask:B*12*N
#         x, mask = inputs
#
#         batch_size = tf.shape(x)[0]
#         seqlen = tf.shape(x)[1]
#
#         # B*N*768
#         q = self.dense_q(x)
#         k = self.dense_k(x)
#         v = self.dense_v(x)
#
#         # B*N*12
#         alphascore = self.alpha(q) / (params.hidden_size / params.head) ** 0.5
#
#         # B*12*N
#         alphascore = tf.transpose(alphascore, [0, 2, 1])
#
#         # B*12*N
#         alphascore += mask
#
#         # B*12*N
#         alphaweight = self.dropout1(tf.nn.softmax(alphascore, axis=-1))
#
#         # B*12*1*N
#         alphaweight = tf.expand_dims(alphaweight, axis=2)
#
#         # B*N*12*64
#         qsplit = tf.reshape(q, [batch_size, seqlen, params.head, params.hidden_size // params.head])
#
#         # B*12*N*64
#         qsplit = tf.transpose(qsplit, [0, 2, 1, 3])
#
#         # B*12*1*64-->B*1*12*64
#         q_av = tf.transpose(tf.matmul(alphaweight, qsplit), [0, 2, 1, 3])
#
#         # B*1*768
#         q_av = tf.reshape(q_av, [-1, 1, params.hidden_size])
#
#         # B*N*768
#         q_av = tf.tile(q_av, [1, seqlen, 1])
#
#         #########################################################################
#
#         # B*N*768
#         p = k * q_av
#
#         # B*N*12
#         betascore = self.beta(p) / (params.hidden_size / params.head) ** 0.5
#
#         # B*12*N
#         betascore = tf.transpose(betascore, [0, 2, 1])
#
#         # B*12*N
#         betascore += mask
#
#         # B*12*N
#         betaweight = self.dropout2(tf.nn.softmax(betascore, axis=-1))
#
#         # B*12*1*N
#         betaweight = tf.expand_dims(betaweight, axis=2)
#
#         # B*N*12*64
#         psplit = tf.reshape(p, [batch_size, seqlen, params.head, params.hidden_size // params.head])
#
#         # B*12*N*64
#         psplit = tf.transpose(psplit, [0, 2, 1, 3])
#
#         # B*12*1*64
#         p_av = tf.matmul(betaweight, psplit)
#
#         # B*N*12*64
#         vsplit = tf.reshape(v, [batch_size, seqlen, params.head, params.hidden_size // params.head])
#
#         # B*12*N*64
#         vsplit = tf.transpose(vsplit, [0, 2, 1, 3])
#
#         # B*12*N*64
#         u = p_av * vsplit
#
#         # B*N*12*64
#         u = tf.transpose(u, [0, 2, 1, 3])
#
#         # B*N*768
#         u = tf.reshape(u, [batch_size, seqlen, params.hidden_size])
#
#         # B*N*768
#         r = self.dense_u(u)
#
#         # B*N*768
#         attention_output = self.dense_o(r + q)
#
#         return self.layernorm(x + self.dropout3(attention_output))

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_q = Dense(params.hidden_size,
                             name='query',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_k = Dense(params.hidden_size,
                             name='key',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_v = Dense(params.hidden_size,
                             name='value',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())

        self.alpha = Dense(1,
                           name='alpha',
                           dtype=tf.float32,
                           kernel_initializer=create_initializer())
        self.beta = Dense(1,
                          name='beta',
                          dtype=tf.float32,
                          kernel_initializer=create_initializer())

        self.dense_u = Dense(params.hidden_size // params.head,
                             name='upvalue',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())

        self.dense_o = Dense(params.hidden_size,
                             name='output',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())

        self.dropout1 = Dropout(rate=params.drop_rate)
        self.dropout2 = Dropout(rate=params.drop_rate)
        self.dropout3 = Dropout(rate=params.drop_rate)
        self.layernorm = LayerNormalization(name='layernormattn', epsilon=1e-6)

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # x: B*N*768 mask:B*12*N
        x, mask = inputs

        batch_size = tf.shape(x)[0]
        seqlen = tf.shape(x)[1]

        head_dim = params.hidden_size // params.head

        # B*N*768
        q = self.dense_q(x)
        k = self.dense_k(x)
        v = self.dense_v(x)

        # B*N*768 --> B*N*12*64 --> B*12*N*64
        qsplit = tf.transpose(tf.reshape(q, [batch_size, seqlen, params.head, head_dim]), [0, 2, 1, 3])
        ksplit = tf.transpose(tf.reshape(k, [batch_size, seqlen, params.head, head_dim]), [0, 2, 1, 3])
        vsplit = tf.transpose(tf.reshape(v, [batch_size, seqlen, params.head, head_dim]), [0, 2, 1, 3])

        # B*12*N*1
        alphascore = self.alpha(qsplit) / head_dim ** 0.5
        # B*12*N
        alphascore = tf.squeeze(alphascore, axis=-1)

        # B*12*N
        alphascore += mask

        # B*12*N
        alphaweight = self.dropout1(tf.nn.softmax(alphascore, axis=-1))

        # B*12*N B*12*N*64 --> B*12*64
        q_av = tf.einsum("ijk,ijkl->ijl", alphaweight, qsplit)

        #########################################################################

        # B*12*N*64
        p = ksplit * q_av[:, :, None, :]

        # B*12*N*1
        betascore = self.beta(p) / head_dim ** 0.5
        # B*12*N
        betascore = tf.squeeze(betascore, axis=-1)

        # B*12*N
        betascore += mask

        # B*12*N
        betaweight = self.dropout2(tf.nn.softmax(betascore, axis=-1))

        # B*12*N B*12*N*64 --> B*12*64
        p_av = tf.einsum("ijk,ijkl->ijl", betaweight, p)

        # B*12*N*64
        u = vsplit * p_av[:, :, None, :]

        # B*12*N*64
        r = self.dense_u(u)

        # B*N*12*64
        newr = tf.transpose(r + qsplit, [0, 2, 1, 3])

        # B*N*768
        newr = tf.reshape(newr, [batch_size, seqlen, params.hidden_size])

        # B*N*768
        attention_output = self.dense_o(newr)

        return self.layernorm(x + self.dropout3(attention_output))

class FeedFord(Layer):
    def __init__(self, **kwargs):
        super(FeedFord, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_ffgelu = Dense(params.intermediate_size,
                                  kernel_initializer=create_initializer(),
                                  dtype=tf.float32,
                                  name='intermediate',
                                  activation=gelu)
        self.dense_ff = Dense(params.hidden_size,
                              kernel_initializer=create_initializer(),
                              dtype=tf.float32,
                              name='output')
        self.dropout = Dropout(rate=params.drop_rate)
        self.layernorm = LayerNormalization(name='layernormffd', epsilon=1e-6)

        super(FeedFord, self).build(input_shape)

    def call(self, x, **kwargs):
        return self.layernorm(x + self.dropout(self.dense_ff(self.dense_ffgelu(x))))


class SplitPooler(Layer):
    def __init__(self, **kwargs):
        super(SplitPooler, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 0]


class Pooler(Layer):
    def __init__(self, **kwargs):
        super(Pooler, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(params.hidden_size,
                           name='pooler',
                           kernel_initializer=create_initializer(),
                           dtype=tf.float32,
                           activation=tf.tanh)
        super(Pooler, self).build(input_shape)

    def call(self, x, **kwargs):
        return self.dense(x)


class Project(Layer):
    def __init__(self, **kwargs):
        super(Project, self).__init__(**kwargs)

    def build(self, input_shape):
        self.project = Dense(1, activation="sigmoid", kernel_initializer=create_initializer())

    def call(self, inputs, **kwargs):
        return self.project(inputs)


class TextClassify():
    def __init__(self):
        self.word_dict = load_vocab("data/OriginalFile/word_dict.txt")

    def build_model(self, summary=True):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)

        mask = Mask()(sen)

        now = Embeddings(name='embeddings')(sen)

        for layers in range(params.block):
            now = Attention(name='attention-' + str(layers))(inputs=(now, mask))
            now = FeedFord(name='feedford-' + str(layers))(now)

        cls = SplitPooler(name="splitpooler")(now)

        pool = Pooler(name="pooler")(cls)

        logits = Project(name="project")(pool)

        model = Model(inputs=[sen], outputs=[logits])

        tf.keras.utils.plot_model(model, to_file="FastFormer.jpg", show_shapes=True, dpi=200)

        if summary:
            model.summary(line_length=200)
            for tv in model.variables:
                print(tv.name, tv.shape)

        return model

    def train(self, train_file, val_file):
        model = self.build_model()

        if params.mode == "train1":
            model.load_weights(params.check + "/textclassify.h5")

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        # optimizer = Adam(learning_rate=params.lr)

        # sigmoid crossentropy loss
        # lossobj = BinaryCrossentropy()

        # Focal Loss
        lossobj = focal_loss

        train_batch = batched_data(train_file,
                                   single_example_parser_eb,
                                   params.batch_size,
                                   ([-1], []),
                                   shuffle=False)
        val_batch = batched_data(val_file,
                                 single_example_parser_eb,
                                 params.batch_size,
                                 ([-1], []),
                                 shuffle=False)

        model.compile(optimizer=optimizer,
                      loss=lossobj,
                      metrics=["acc",
                               Precision(name="precision"),
                               Recall(name="recall"),
                               F1Score(name="F1", num_classes=2, threshold=0.5, average="micro"),
                               ])

        history = model.fit(train_batch,
                            validation_data=val_batch,
                            epochs=params.epochs,
                            callbacks=[
                                EarlyStopping(monitor='val_F1', patience=params.patience, mode="max"),
                                ModelCheckpoint(filepath=params.check + "/textclassify.h5",
                                                monitor='val_F1',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode="max")
                            ],
                            class_weight={0: 172. / (2238. + 172.), 1: 2238. / (2238. + 172.)}
                            )

        with open(params.check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history.history))

    def predict(self, sentences):
        rep = replace()
        m_samples = len(sentences)

        sents2id = []
        leng = []
        for sent in sentences:
            sen = jieba.lcut(rep.replace(sent.lower().strip()))

            sen2id = [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict["[UNK]"] for word in
                      sen]
            sents2id.append(sen2id)
            leng.append(len(sen2id))

        max_len = max(np.max(leng), 5)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [0] * (max_len - leng[i])
                sents2id[i] += pad

        model = self.build_model(summary=False)
        model.load_weights(params.check + "/textclassify.h5")

        prediction = model.predict(sents2id)[:, 0]
        for i in range(m_samples):
            print(sentences[i] + " ----> ", prediction[i])

    def test(self, test_file):
        model = self.build_model(summary=False)
        model.load_weights(params.check + "/textclassify.h5")

        test_batch = batched_data(test_file,
                                  single_example_parser_eb,
                                  params.batch_size,
                                  ([-1], []),
                                  shuffle=False)

        # sigmoid crossentropy loss
        lossobj = BinaryCrossentropy()

        model.compile(loss=lossobj,
                      metrics=["acc",
                               Precision(name="precision"),
                               Recall(name="recall"),
                               F1Score(name="F1", num_classes=2, threshold=0.5, average="micro"),
                               ])

        loss, acc, p, r, F1 = model.evaluate(test_batch, verbose=0)

        print("loss: %f acc: %f precision: %f recall: %f F1: %f" % (loss, acc, p, r, F1))

    def plot(self):
        with open(params.check + "/history.txt", "r", encoding="utf-8") as fr:
            history = fr.read()
            history = eval(history)

            plt.subplot(2, 2, 1)
            plt.plot(history["val_loss"])
            plt.title("val_loss")
            plt.grid()

            plt.subplot(2, 2, 2)
            plt.plot(history["val_acc"])
            plt.title("val_acc")
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(history["val_precision"])
            plt.plot(history["val_recall"])
            plt.plot(history["val_F1"])
            plt.title("val_precision,recall,F1")
            plt.legend(['P', 'R', 'F1'], loc='best', prop={'size': 10})
            plt.grid()

            plt.tight_layout()
            plt.savefig(params.check + "/record.png", dpi=500, bbox_inches="tight")
            plt.show()


def main():
    if not os.path.exists(params.check):
        os.makedirs(params.check)

    train_file = [
        'data/TFRecordFile/train_eb_word_fl.tfrecord',
    ]
    val_file = [
        'data/TFRecordFile/val_eb_word_fl.tfrecord',
    ]
    tc = TextClassify()

    if params.mode.startswith('train'):
        tc.train(train_file, val_file)

    elif params.mode == 'predict':
        sentences = [
            "这台电视不错！",
            "给服务赞一个",
            "不怎么样",
            "太差劲了",
            "非常棒！！！"
        ]
        tc.predict(sentences)

    elif params.mode == "plot":
        tc.plot()

    elif params.mode == "test":
        tc.test(train_file)


if __name__ == "__main__":
    main()

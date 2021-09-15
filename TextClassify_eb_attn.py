'''
    基于TextCNN + attention的评价二分类
    基于词
'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, Conv2D, Embedding, LayerNormalization
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from utils import focal_loss, bce_loss_weight
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import replace

import numpy as np
import os
import jieba
import matplotlib.pyplot as plt
from utils import load_vocab, single_example_parser_eb, batched_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

params_check = "modelfiles/textclassify_eb_attn_flw"
params_hidden_size = 100
params_label_size = 2
params_batch_size = 10
params_num_filters = 64
params_kernel_sizes = [2, 3, 4, 5]
params_drop_rate = 0.2
params_epochs = 100
params_lr = 1.0e-3
params_patience = 7
params_mode = "train0"

tf.random.set_seed(100)
np.random.seed(100)


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

        newembedding = np.load("data/OriginalFile/embedding_matrix.npy")
        self.word_embeddings = Embedding(289689, 100, name='word_embeddings',
                                         embeddings_initializer=tf.constant_initializer(newembedding),
                                         dtype=tf.float32,
                                         # trainable=False,
                                         )
        self.dropout = Dropout(params_drop_rate)

    def call(self, sen, **kwargs):
        sen_embed = self.dropout(self.word_embeddings(sen))

        return tf.expand_dims(sen_embed, axis=-1)


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_q = Dense(params_num_filters,
                             name='query',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_k = Dense(params_num_filters,
                             name='key',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_v = Dense(params_num_filters,
                             name='value',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_o = Dense(params_num_filters,
                             name='output',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dropoutsoft = Dropout(params_drop_rate)
        self.dropoutres = Dropout(params_drop_rate)
        self.layernorm = LayerNormalization(name='layernormattn')

        self.dropoutattn = Dropout(params_drop_rate)

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, y = inputs
        q = self.dense_q(y)
        k = self.dense_k(x)
        v = self.dense_v(x)
        qk = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / tf.sqrt(params_num_filters * 1.0)
        attention_output = self.dense_o(tf.matmul(self.dropoutsoft(tf.nn.softmax(qk, axis=-1)), v))
        attn = tf.squeeze(self.dropoutattn(self.layernorm(y + self.dropoutres(attention_output))), axis=1)
        return attn


class TextCNN(Layer):
    def __init__(self, **kwargs):
        super(TextCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.convs = []
        for i in range(len(params_kernel_sizes)):
            self.convs.append(
                Conv2D(filters=params_num_filters, kernel_size=[params_kernel_sizes[i], params_hidden_size],
                       padding="VALID", activation="relu"))
        self.dropout = [Dropout(params_drop_rate) for _ in range(len(params_kernel_sizes))]

        self.attn = Attention(name="attn")

        super(TextCNN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        now = inputs

        convs = []
        for i in range(len(params_kernel_sizes)):
            conv = self.convs[i](now)
            dropconv = self.dropout[i](conv)
            pool = tf.reduce_max(dropconv, axis=1)
            convs.append(pool)

        q = tf.concat([convs[i] for i in range(len(params_kernel_sizes) - 1)], axis=1)
        k = convs[-1]

        return q, k


class Project(Layer):
    def __init__(self, **kwargs):
        super(Project, self).__init__(**kwargs)

    def build(self, input_shape):
        self.project = Dense(1, activation="sigmoid", kernel_initializer=TruncatedNormal(stddev=0.02))

        super(Project, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.project(inputs)


class TextClassify():
    def __init__(self):
        self.word_dict = load_vocab("data/OriginalFile/word_dict.txt")

    def build_model(self, summary=True):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)  # 句子

        now = Embeddings(name="embeddings")(sen)

        q, k = TextCNN(name="textcnn")(now)

        attn = Attention(name="attn")(inputs=(q, k))

        logits = Project()(attn)

        model = Model(inputs=[sen], outputs=[logits])

        if summary:
            model.summary()
            for tv in model.variables:
                print(tv.name, " : ", tv.shape)

        return model

    def train(self, train_file, val_file):

        model = self.build_model()

        if params_mode == "train1":
            model.load_weights(params_check + "/textclassify.h5")

        optimizer = Adam(learning_rate=params_lr)

        # sigmoid crossentropy loss
        # lossobj = BinaryCrossentropy()

        # Focal Loss
        lossobj = focal_loss

        train_batch = batched_data(train_file,
                                   single_example_parser_eb,
                                   params_batch_size,
                                   ([-1], []),
                                   shuffle=False)
        val_batch = batched_data(val_file,
                                 single_example_parser_eb,
                                 params_batch_size,
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
                            epochs=params_epochs,
                            callbacks=[
                                EarlyStopping(monitor='val_F1', patience=params_patience, mode="max"),
                                ModelCheckpoint(filepath=params_check + "/textclassify.h5",
                                                monitor='val_F1',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode="max")
                            ],
                            class_weight={0: 172. / (2238. + 172.), 1: 2238. / (2238. + 172.)}
                            )

        with open(params_check + "/history.txt", "w", encoding="utf-8") as fw:
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
        model.load_weights(params_check + "/textclassify.h5")

        prediction = model.predict(sents2id)[:, 0]
        for i in range(m_samples):
            print(sentences[i] + " ----> ", prediction[i])

    def test(self, test_file):
        model = self.build_model()
        model.load_weights(params_check + "/textclassify.h5")

        test_batch = batched_data(test_file,
                                  single_example_parser_eb,
                                  params_batch_size,
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
        with open(params_check + "/history.txt", "r", encoding="utf-8") as fr:
            history = fr.read()
            history = eval(history)

            plt.subplot(2, 2, 1)
            plt.plot(history["val_loss"])
            plt.title("val_loss")
            plt.subplot(2, 2, 2)
            plt.plot(history["val_acc"])
            plt.title("val_acc")
            plt.subplot(2, 1, 2)
            plt.plot(history["val_precision"])
            plt.plot(history["val_recall"])
            plt.plot(history["val_F1"])
            plt.title("val_precision,recall,F1")
            plt.legend(['P', 'R', 'F1'], loc='best', prop={'size': 4})
            plt.tight_layout()
            plt.savefig(params_check + "/record.png", dpi=500, bbox_inches="tight")
            plt.show()


def main():
    if not os.path.exists(params_check):
        os.makedirs(params_check)

    train_file = [
        'data/TFRecordFile/train_eb_word_fl.tfrecord',
    ]
    val_file = [
        'data/TFRecordFile/val_eb_word_fl.tfrecord',
    ]
    tc = TextClassify()

    if params_mode.startswith('train'):
        tc.train(train_file, val_file)

    elif params_mode == 'predict':
        sentences = [
            "这台电视不错！",
            "给服务赞一个",
            "不怎么样",
            "太差劲了",
            "非常棒！！！"
        ]
        tc.predict(sentences)

    elif params_mode == "plot":
        tc.plot()

    elif params_mode == "test":
        tc.test(val_file)


if __name__ == "__main__":
    main()

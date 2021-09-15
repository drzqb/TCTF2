'''
    基于BERT的评价二分类
    基于字
'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras.initializers import TruncatedNormal
from transformers import TFBertModel, BertConfig, BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from utils import focal_loss, bce_loss_weight
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import os, time
import matplotlib.pyplot as plt
from utils import single_example_parser_eb, batched_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

params_check = "modelfiles/textclassify_eb_bert_bce"
params_batch_size = 4
params_drop_rate = 0.2
params_epochs = 100
params_lr = 1.0e-5
params_patience = 7
params_mode = "test"

tf.random.set_seed(100)
np.random.seed(100)


class BERT(Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        Config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")
        Config.attention_probs_dropout_prob = params_drop_rate
        Config.hidden_dropout_prob = params_drop_rate

        self.bert = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=Config)

    def call(self, inputs, **kwargs):
        return self.bert(input_ids=inputs,
                         token_type_ids=tf.zeros_like(inputs),
                         attention_mask=tf.cast(tf.greater(inputs, 0), tf.int32))[1]


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
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model(self, summary=True):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)

        now = BERT(name="bert")(sen)

        logits = Project()(now)

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
        lossobj = BinaryCrossentropy()

        # Focal Loss
        # lossobj = focal_loss

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
                            # class_weight={0: 181. / (2265. + 181.), 1: 2265. / (2265. + 181.)}
                            )

        with open(params_check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history.history))

    def predict(self, sentences):
        m_samples = len(sentences)

        sents2id = self.tokenizer(sentences, padding=True, return_tensors="tf")["input_ids"]

        model = self.build_model(summary=False)
        model.load_weights(params_check + "/textclassify.h5")

        prediction = model.predict(sents2id)[:, 0]
        for i in range(m_samples):
            print(sentences[i] + " ----> ", prediction[i])

    def test(self, test_file):
        model = self.build_model(summary=False)
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
        print("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n\n")

        loss, acc, p, r, F1 = model.evaluate(test_batch, verbose=0)

        print("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n\n")

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
        'data/TFRecordFile/train_eb_char_fl.tfrecord',
    ]
    val_file = [
        'data/TFRecordFile/val_eb_char_fl.tfrecord',
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
        tc.test(train_file)


if __name__ == "__main__":
    main()

'''
    Ebusiness 数字化
    基于字
'''
import tensorflow as tf
import numpy as np
import sys
from transformers import BertTokenizer
import pandas as pd
from sklearn.utils import shuffle

sys.path.append("../")


class Lang():
    def toid(self):
        m_samples_train = 0
        m_samples_train_pos = 0
        m_samples_train_neg = 0
        m_samples_val = 0
        m_samples_val_pos = 0
        m_samples_val_neg = 0

        train_writer = tf.io.TFRecordWriter('../data/TFRecordFile/train_eb_char.tfrecord')
        val_writer = tf.io.TFRecordWriter('../data/TFRecordFile/val_eb_char.tfrecord')

        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        data = pd.read_csv("../data/OriginalFile/Ebusiness.csv")

        data = shuffle(data)

        k = 1

        for index, row in data.iterrows():
            sen = row["evaluation"].lower().strip()
            label = row["label"].strip()

            if label == "正面":
                label = 1
            else:
                label = 0

            print(k)
            print("sen: ", sen)
            print("lab: ", label)
            print()

            sen2id = tokenizer(sen)["input_ids"]
            if len(sen2id) > 512:
                sen2id = sen2id[:512]

            sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                           sen2id]

            label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

            seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sen': tf.train.FeatureList(feature=sen_feature),
                }),
                context=tf.train.Features(feature={
                    'label': label_feature
                }),

            )

            serialized = seq_example.SerializeToString()

            if np.random.rand() < 0.1:
                val_writer.write(serialized)
                m_samples_val += 1
                if label == 1:
                    m_samples_val_pos += 1
                else:
                    m_samples_val_neg += 1
            else:
                train_writer.write(serialized)
                m_samples_train += 1
                if label == 1:
                    m_samples_train_pos += 1
                else:
                    m_samples_train_neg += 1

            k += 1

        print('\n')

        print("训练样本总量共：%d ,正样本共：%d ,负样本共：%d" % (
            m_samples_train, m_samples_train_pos, m_samples_train_neg))  # 训练样本总量共：3830 ,正样本共：1708 ,负样本共：2122
        print('测试样本总量共：%d ,正样本共：%d ,负样本共：%d ' % (
            m_samples_val, m_samples_val_pos, m_samples_val_neg))  # 测试样本总量共：453 ,正样本共：200 ,负样本共：253

    def toid_fl(self):
        """
        样本不均衡的训练集
        :return:
        """
        m_samples_train = 0
        m_samples_train_pos = 0
        m_samples_train_neg = 0
        m_samples_val = 0
        m_samples_val_pos = 0
        m_samples_val_neg = 0

        train_writer = tf.io.TFRecordWriter('../data/TFRecordFile/train_eb_char_fl.tfrecord')
        val_writer = tf.io.TFRecordWriter('../data/TFRecordFile/val_eb_char_fl.tfrecord')

        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        data = pd.read_csv("../data/OriginalFile/Ebusiness.csv")

        data = shuffle(data)

        k = 1

        for index, row in data.iterrows():
            sen = row["evaluation"].lower().strip()
            label = row["label"].strip()

            if label == "正面":
                label = 1
            else:
                label = 0

            print(k)
            print("sen: ", sen)
            print("lab: ", label)
            print()

            sen2id = tokenizer(sen)["input_ids"]
            if len(sen2id) > 512:
                sen2id = sen2id[:512]

            sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                           sen2id]

            label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

            seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sen': tf.train.FeatureList(feature=sen_feature),
                }),
                context=tf.train.Features(feature={
                    'label': label_feature
                }),

            )

            serialized = seq_example.SerializeToString()

            if np.random.rand() < 0.05:
                val_writer.write(serialized)
                m_samples_val += 1
                if label == 1:
                    m_samples_val_pos += 1
                else:
                    m_samples_val_neg += 1
            else:
                if (label == 1 and np.random.rand() < 0.1) or (label == 0):
                    train_writer.write(serialized)
                    m_samples_train += 1
                    if label == 1:
                        m_samples_train_pos += 1
                    else:
                        m_samples_train_neg += 1

            k += 1

        print('\n')

        print("训练样本总量共：%d ,正样本共：%d ,负样本共：%d" % (
            m_samples_train, m_samples_train_pos, m_samples_train_neg))  # 训练样本总量共：2446 ,正样本共：181 ,负样本共：2265
        print('测试样本总量共：%d ,正样本共：%d ,负样本共：%d ' % (
            m_samples_val, m_samples_val_pos, m_samples_val_neg))  # 测试样本总量共：220 ,正样本共：110 ,负样本共：110


if __name__ == '__main__':
    lang = Lang()
    lang.toid()

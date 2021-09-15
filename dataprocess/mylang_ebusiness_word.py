'''
    Ebusiness 数字化
    基于词
'''
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys
import jieba
from gensim.models import word2vec
import pandas as pd
from sklearn.utils import shuffle

sys.path.append("../")
from utils import load_vocab


class Lang():
    @staticmethod
    def makeworddict():
        model = word2vec.Word2Vec.load('../modelfiles/gm1/gensimmodel')
        vocab_list = [word for word, Vocab in model.wv.vocab.items()]  # 存储 所有的 词语

        word_dict = {" ": 0, "[UNK]": 1}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.

        # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
        # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
        embedding_matrix = np.zeros((len(vocab_list) + 2, model.vector_size))

        ## 3 填充 上述 的字典 和 大矩阵
        with open("../data/OriginalFile/word_dict.txt", "w", encoding="utf-8") as fw:
            fw.write(" \n")
            fw.write("[UNK]\n")

            for i in range(len(vocab_list)):
                # print(i)
                word = vocab_list[i]  # 每个词语
                fw.write(word + "\n")
                embedding_matrix[i + 2] = model.wv[word]  # 词向量矩阵

        np.save("../data/OriginalFile/embedding_matrix.npy", embedding_matrix)

    @staticmethod
    def toid():
        m_samples_train = 0
        m_samples_train_pos = 0
        m_samples_train_neg = 0
        m_samples_val = 0
        m_samples_val_pos = 0
        m_samples_val_neg = 0

        train_writer = tf.io.TFRecordWriter('../data/TFRecordFile/train_eb_word.tfrecord')
        val_writer = tf.io.TFRecordWriter('../data/TFRecordFile/val_eb_word.tfrecord')

        word_dict = load_vocab("../data/OriginalFile/word_dict.txt")
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

            sen = jieba.lcut(sen)

            sen2id = [word_dict[word] if word in word_dict.keys() else word_dict["[UNK]"] for word in sen]
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
            m_samples_train, m_samples_train_pos, m_samples_train_neg))  # 训练样本总量共：3829 ,正样本共：1702 ,负样本共：2127
        print('测试样本总量共：%d ,正样本共：%d ,负样本共：%d ' % (
            m_samples_val, m_samples_val_pos, m_samples_val_neg))  # 测试样本总量共：454 ,正样本共：206 ,负样本共：248

    @staticmethod
    def toid_fl():
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

        train_writer = tf.io.TFRecordWriter('../data/TFRecordFile/train_eb_word_fl.tfrecord')
        val_writer = tf.io.TFRecordWriter('../data/TFRecordFile/val_eb_word_fl.tfrecord')

        word_dict = load_vocab("../data/OriginalFile/word_dict.txt")
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

            sen = jieba.lcut(sen)

            sen2id = [word_dict[word] if word in word_dict.keys() else word_dict["[UNK]"] for word in sen]
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
            m_samples_train, m_samples_train_pos, m_samples_train_neg))  # 训练样本总量共：2410 ,正样本共：172 ,负样本共：2238
        print('测试样本总量共：%d ,正样本共：%d ,负样本共：%d ' % (
            m_samples_val, m_samples_val_pos, m_samples_val_neg))  # 测试样本总量共：250 ,正样本共：113 ,负样本共：137

    @staticmethod
    def train_test_split():
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

        data = pd.read_csv("../data/OriginalFile/Ebusiness.csv")
        data = shuffle(data)

        all_x = open("../data/OriginalFile/all_x.txt", "w", encoding="utf-8")
        all_y = open("../data/OriginalFile/all_y.txt", "w", encoding="utf-8")
        train_x = open("../data/OriginalFile/train_x.txt", "w", encoding="utf-8")
        train_y = open("../data/OriginalFile/train_y.txt", "w", encoding="utf-8")
        test_x = open("../data/OriginalFile/test_x.txt", "w", encoding="utf-8")
        test_y = open("../data/OriginalFile/test_y.txt", "w", encoding="utf-8")

        k = 1

        for index, row in data.iterrows():
            sen = row["evaluation"].lower().replace('\n', '').replace('\r', '').replace(' ', '').strip()
            label = row["label"].strip()

            if label == "正面":
                label = "1"
            else:
                label = "0"

            print(k)
            print("sen: ", sen)
            print("lab: ", label)
            print()

            sen = jieba.lcut(sen)

            all_x.write(" ".join(sen) + "\n")
            all_y.write(label + "\n")

            if np.random.rand() < 0.05:
                m_samples_val += 1
                if label == "1":
                    m_samples_val_pos += 1
                else:
                    m_samples_val_neg += 1

                test_x.write(" ".join(sen) + "\n")
                test_y.write(label + "\n")
            else:
                if (label == "1" and np.random.rand() < 0.1) or (label == "0"):
                    m_samples_train += 1
                    if label == "1":
                        m_samples_train_pos += 1
                    else:
                        m_samples_train_neg += 1

                    train_x.write(" ".join(sen) + "\n")
                    train_y.write(label + "\n")

            k += 1

        print('\n')

        print("训练样本总量共：%d ,正样本共：%d ,负样本共：%d" % (
            m_samples_train, m_samples_train_pos, m_samples_train_neg))  # 训练样本总量共：2424 ,正样本共：167 ,负样本共：2257
        print('测试样本总量共：%d ,正样本共：%d ,负样本共：%d ' % (
            m_samples_val, m_samples_val_pos, m_samples_val_neg))  # 测试样本总量共：212 ,正样本共：94 ,负样本共：118

        all_x.close()
        all_y.close()
        train_x.close()
        train_y.close()
        test_x.close()
        test_y.close()


if __name__ == '__main__':
    Lang.train_test_split()

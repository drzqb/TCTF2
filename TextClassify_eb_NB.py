"""
    朴素贝叶斯方法
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn import metrics


def build_data():
    all_text = []
    with open("data/OriginalFile/all_x.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            all_text.append(line.strip())

    train_texts = []
    with open("data/OriginalFile/train_x.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            train_texts.append(line.strip())

    test_texts = []
    with open("data/OriginalFile/test_x.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            test_texts.append(line.strip())

    stop_words = []
    with open("data/OriginalFile/stopwords.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            stop_words.append(line.strip())

    count_v0 = CountVectorizer(
        # stop_words=stop_words
    )

    counts_all = count_v0.fit_transform(all_text)

    count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)

    counts_train = count_v1.fit_transform(train_texts)
    print("the shape of train is " + repr(counts_train.shape))

    count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)

    counts_test = count_v2.fit_transform(test_texts)
    print("the shape of test is " + repr(counts_test.shape))

    tfidftransformer = TfidfTransformer()
    train_x = tfidftransformer.fit(counts_train).transform(counts_train)
    test_x = tfidftransformer.fit(counts_test).transform(counts_test)

    train_y = []
    with open("data/OriginalFile/train_y.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            train_y.append(int(line.strip()))

    test_y = []
    with open("data/OriginalFile/test_y.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            test_y.append(int(line.strip()))

    return train_x, train_y, test_x, test_y


def train(data):
    x_train, y_train, x_test, y_test = data

    clf = MultinomialNB(alpha=0.01)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('MultinomialNB precision_score:' + str(float(num) / len(preds)))

    clf = GaussianNB()
    clf.fit(x_train.toarray(), y_train)
    preds = clf.predict(x_test.toarray())
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('GaussianNB precision_score:' + str(float(num) / len(preds)))

    clf = BernoulliNB(alpha=0.01)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('BernoulliNB precision_score:' + str(float(num) / len(preds)))


if __name__ == "__main__":
    data = build_data()
    train(data)

import fileutil as utils

encoding = "utf-8"
from collections import defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import gensim
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


# Loading data
POSITIVE_FOLDER = config['DEV']['POSITIVE_FOLDER']
NEGATIVE_FOLDER = config['DEV']['NEGATIVE_FOLDER']
PREDICT_FOLDER = config['DEV']['PREDICT_FOLDER']

positive_X, positive_y, pfnames = utils.resumes2data('hired', POSITIVE_FOLDER)
negative_X, negative_y, nfnames = utils.resumes2data('rejected', NEGATIVE_FOLDER)
predict_X, _, predict_files = utils.resumes2data('predict', PREDICT_FOLDER, file_types=['doc', 'docx', 'pdf', 'txt'])

X = positive_X + negative_X
y = positive_y + negative_y
X = np.array(X)
y = np.array(y)

# save data into files
# utils.resumes2textfile('hired', POSITIVE_FOLDER, './data/hired.txt')
# utils.resumes2textfile('rejected', NEGATIVE_FOLDER, './data/rejected.txt')

# Load Google's pre-trained Word2Vec model.
google_model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin',
                                                               binary=True)
# dictionary of google word to vector
google_w2v = {w: vec for w, vec in zip(google_model.index2word, google_model.vectors)}

# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine
model = Word2Vec(X, size=50, window=5, min_count=1, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.vectors)}


# given a word -> vector mapping and vectorizes texts by taking
# the mean of all the vectors corresponding to individual words.
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# A tf-idf version of the classifier,  that is given a word -> vector mapping and vectorizes texts by taking
# the mean of all the vectors corresponding to individual words.
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
                    ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
                    ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
                          ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
                          ("bernoulli nb", BernoulliNB())])
# SVM - which is supposed to be more or less state of the art
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
                ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
                      ("linear svc", SVC(kernel="linear"))])
etree_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
                        # ("linear svc", SVC(kernel="linear")),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
rftree_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
                         # ("linear svc", SVC(kernel="linear")),
                         ("random forest trees", RandomForestClassifier(n_estimators=200))])

# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_google = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(google_w2v)),
                         ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_google_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(google_w2v)),
                               ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                            ("extra trees", ExtraTreesClassifier(n_estimators=200))])

all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
    ("bern_nb", bern_nb),
    ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("etree_tfidf", etree_tfidf),
    ("rftree_tfidf", rftree_tfidf),
    ("google", etree_google),
    ("google_tfidf", etree_google_tfidf),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf)
]

unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])

print(tabulate(scores, floatfmt=".4f", headers=("model", 'score')))

plt.figure(figsize=(15, 6))
sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])


def benchmark(model, X, y, n_splits):
    kf = KFold(n_splits, shuffle=True)
    kf.get_n_splits(X)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores), len(train_index)


n_splits = [2, 4, 8]
table = []
for name, model in all_models:
    for n in n_splits:
        score, train_size = benchmark(model, X, y, n)
        table.append({'model': name,
                      'accuracy': score,
                      'train_size': train_size})
df = pd.DataFrame(table)

plt.figure(figsize=(15, 6))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model',
                    data=df[df.model.map(lambda x: x in ["mult_nb",
                                                         "mult_nb_tfidf",
                                                         "bern_nb_tfidf",
                                                         "svc_tfidf",
                                                         "etree_tfidf",
                                                         "rftree_tfidf",
                                                         "google_tfidf",
                                                         "etree_google_tfidf",
                                                         "w2v_tfidf"
                                                         ])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="labeled training examples")
fig.set(title="Benchmark (total examples:" + str(len(y)) + ')')
fig.set(ylabel="accuracy")


def predict(model_name='svc_tfidf', n_splits=2):
    for name, m in all_models:
        if name == model_name:
            break
    kf = KFold(n_splits, shuffle=True)
    kf.get_n_splits(X)
    model_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_scores.append(accuracy_score(m.fit(X_train, y_train).predict(X_test), y_test))
    accuracy_score_str = 'accuracy_score: ' + str(np.mean(model_scores))
    print(accuracy_score_str)

    predict_y = m.predict(predict_X)
    unsorted_prediction = [(i, predict_files[i], "yes" if predict_y[i] == 'hired' else 'no') for i in
                           range(len(predict_y))]
    prediction = sorted(unsorted_prediction, key=lambda x: 1 if x[2] == 'yes' else -1)
    # print(tabulate(predict, headers=("resume", 'continue?')))

    f = open('./model/predict_' + model_name + '.txt', 'w')
    f.write(model_name + '\n')
    f.write('n_splits: ' + str(n_splits) + '\n')
    f.write(accuracy_score_str + '\n\n')
    f.write(tabulate(prediction, headers=("resume", 'continue?')))
    f.close()


predict(model_name='mult_nb')
predict(model_name='svc_tfidf')
predict(model_name='rftree_tfidf')

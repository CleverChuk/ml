from __future__ import absolute_import, division, print_function, unicode_literals
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from math import floor
from sklearn.naive_bayes import *
from sklearn.feature_extraction import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.tree import *
import numpy as np
from tensorflow import (keras, estimator)
import xgboost as xgb

# train_file = "./dataset/Train.csv"
# data = pd.read_csv(train_file)
# data = data.drop("netgain", axis=1)
# rows = data.to_dict('records')
# print(rows[0].keys())

MODEL = transformer = None
MAX_FEATURES = 100


def feature_doc(filename):
    X = pd.read_csv(filename)
    Y = X["netgain"]
    X = X.drop("netgain", axis=1)
    X = X.drop('money_back_guarantee', axis=1)

    return X.to_dict('records'), Y


class ItemSelector(BaseEstimator, TransformerMixin):
    """
        This class is used to select particular feature list in the given feature doc
        self.key: the key that identifies the feature in the feature doc
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        out = [x[self.key] for x in X]
        return out


class NpArray(BaseEstimator, TransformerMixin):
    """
        This class is used to reshape nparray
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return np.asarray(X).reshape(-1, 1)


def transform_data_with_dv(filename):
    """
        This function transforms the thread data using both
        the content of the post and the structural information
        to create a vector representation of the ad.
    """
    global transformer
    X, Y = feature_doc(filename)
    transformer = DictVectorizer()

    X_encode = transformer.fit_transform(X, Y)

    return X_encode, Y


def transform_data_with_fu(filename):
    """
        This function transforms the thread data using both
        the content of the post and the structural information
        to create a vector representation of the ad.
    """
    global transformer
    X, Y = feature_doc(filename)
    transformer = FeatureUnion(
        transformer_list=[

            ('realtionship_status', Pipeline([
                ('selector', ItemSelector(key='realtionship_status')),
                ('encoder',
                 TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char'))
            ])),

            ('industry', Pipeline([
                ('selector', ItemSelector(key='industry')),
                ('encoder',
                 TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char'))
            ])),

            # does not have predictive power
            ('genre', Pipeline([
                ('selector', ItemSelector(key='genre')),
                ('encoder',
                 TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char'))
            ])),

            # does not have predictive power
            ('targeted_sex', Pipeline([
                ('selector', ItemSelector(key='targeted_sex')),
                ('encoder',
                 TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char'))
            ])),

            # does not have predictive power
            ('airlocation', Pipeline([
                ('selector', ItemSelector(key='airlocation')),
                ('encoder',
                 TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char'))
            ])),

            ('expensive', Pipeline([
                ('selector', ItemSelector(key='expensive')),
                ('encoder',
                 TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char'))
            ])),

            # better without this
            # ('money_back_guarantee', Pipeline([
            #     ('selector', ItemSelector(key='money_back_guarantee')),
            #     ('encoder', TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2),analyzer='char'))
            # ])),

            ('airtime', Pipeline([
                ('selector', ItemSelector(key='airtime')),
                ('encoder',
                 TfidfVectorizer(max_features=MAX_FEATURES, analyzer='char'))
            ])),

            ('ratings', Pipeline([
                ('selector', ItemSelector(key='ratings')),
                ('np', NpArray()),
            ])),

            ('average_runtime(minutes_per_week)', Pipeline([
                ('selector', ItemSelector(key='average_runtime(minutes_per_week)')),
                ('np', NpArray()),
                ('scaler', MinMaxScaler())
            ])),
        ],
    )

    X_encode = transformer.fit_transform(X, Y)

    return X_encode, Y


def train_baseline_model_gbrt(filename):
    """
        This function trains a gradient boosting classifier on the
        data
    """
    X, Y = transform_data_with_fu(filename)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    gbrt = GradientBoostingClassifier(n_estimators=7500, random_state=0, learning_rate=0.0025)
    gbrt.fit(x_train, y_train)
    training_score = gbrt.score(x_train, y_train) * 100
    test_score = gbrt.score(x_test, y_test) * 100
    print("gbrt")
    print(f"Accuracy on training set: {training_score}")
    print(f"Accuracy on test set: {test_score}")
    with open(f'baseline_gbrt_{floor(test_score)}.pickle', 'wb') as fd:
        pickle.dump(gbrt, fd)

    with open(f'transformer_gbrt_{floor(test_score)}.pickle', 'wb') as fd:
        pickle.dump(transformer, fd)


def train_xgboost(filename):
    """
        This function trains a gradient boosting classifier on the
        data
    """

    def logregobj(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1.0 - preds)
        return grad, hess

    X, Y = transform_data_with_fu(filename)
    Y = Y.to_numpy(dtype='int')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    # specify parameters via map, definition are same as c++ version
    param = {
        'max_depth': 15,
        'eta': 0.0009, 'verbosity': 0,
        'objective': 'binary:logistic',
        'booster': 'gbtree'
    }

    # specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 30

    bst = xgb.train(param, dtrain, num_round, watchlist)
    test_pred = bst.predict(dtest)
    train_pred = bst.predict(dtrain)

    training_score = sum([y_train[i] == (train_pred[i] > 0.5) for i in range(len(y_train))]) / len(
        y_train) * 100
    test_score = sum([y_test[i] == (test_pred[i] > 0.5) for i in range(len(y_test))]) / len(
        y_test) * 100

    print("bst")
    print(f"Accuracy on training set: {training_score}")
    print(f"Accuracy on test set: {test_score}")
    with open(f'xgboost_{floor(test_score)}.pickle', 'wb') as fd:
        pickle.dump(bst, fd)

    with open(f'transformer_xgb_{floor(test_score)}.pickle', 'wb') as fd:
        pickle.dump(transformer, fd)


def train_baseline_model_logreg(filename):
    """
        This function trains a logistic regression classifier on the
        data
    """
    X, Y = transform_data_with_fu(filename)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    logreg = LogisticRegressionCV(
        random_state=0, cv=5, multi_class='multinomial', solver='saga')
    logreg.fit(x_train, y_train)
    training_score = logreg.score(x_train, y_train) * 100
    test_score = logreg.score(x_test, y_test) * 100

    print("Logreg")
    print(f"Accuracy on training set: {training_score}")
    print(f"Accuracy on test set: {test_score}")
    with open(f'baseline_logreg_{floor(test_score)}.pickle', 'wb') as fd:
        pickle.dump(logreg, fd)


def model_search(filename):
    """
        This function trains a logistic regression classifier on the
        data
    """
    X, Y = transform_data_with_fu(filename)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model = MLPClassifier(learning_rate_init=0.0001, hidden_layer_sizes=(200,),
                          learning_rate='adaptive')

    model.fit(x_train, y_train)
    training_score = model.score(x_train, y_train) * 100
    test_score = model.score(x_test, y_test) * 100

    print(model)
    print(f"Accuracy on training set: {training_score}")
    print(f"Accuracy on test set: {test_score}")
    with open(f'model_search_{floor(test_score)}.pickle', 'wb') as fd:
        pickle.dump(model, fd)


def train_nn(filename):
    X, Y = transform_data_with_dv(filename)
    Y = Y.to_numpy(dtype='int')
    X = X.toarray()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model = keras.Sequential(
        [
            keras.layers.Dense(1024, activation='selu', input_shape=(x_train.shape[1],)),
            keras.layers.Dense(512, activation='selu'),
            keras.layers.Dense(256, activation='selu'),
            keras.layers.Dense(128, activation='selu'),
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    # compile the model ,kernel_regularizer=keras.regularizers.l2(0.01)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # fit the model
    model.fit(x_train, y_train, epochs=100, verbose=2,
              validation_data=(x_test, y_test))

    loss, acc = model.evaluate(x_test, y_test)
    model.save("mlp_%d.h5" % (floor(acc * 100)))

    with open(f'transformer_mlp_{floor(acc * 100)}.pickle', 'wb') as fd:
        pickle.dump(transformer, fd)


def load_model_nn():
    global MODEL, transformer

    MODEL = keras.models.load_model('mlp_81.h5')

    with open('transformer_mlp_81.pickle', 'rb') as fd:
        transformer = pickle.load(fd)


def load_model():
    global MODEL, transformer

    with open("baseline_gbrt_82.pickle", 'rb') as fd:
        MODEL = pickle.load(fd)

    with open('transformer_gbrt_82.pickle', 'rb') as fd:
        transformer = pickle.load(fd)


def classify(filename):
    X = pd.read_csv(filename)
    x_test = X.to_dict('records')
    predicted = []

    X_encode = transformer.transform(x_test)
    for i in range(X_encode.shape[0]):
        datum = X_encode[i]
        answer = MODEL.predict(datum)[0]
        predicted.append({'id': x_test[i]['id'], 'netgain': answer})

    df = pd.DataFrame.from_records(predicted)
    df.to_csv('output.csv', index=False)


def classify_xgb(filename):
    X = pd.read_csv(filename)
    x_test = X.to_dict('records')
    predicted = []

    X_encode = transformer.transform(x_test)
    for i in range(X_encode.shape[0]):
        datum = xgb.DMatrix(X_encode[i])
        answer = MODEL.predict(datum)[0]
        predicted.append({'id': x_test[i]['id'], 'netgain': answer >= 0.5})

    df = pd.DataFrame.from_records(predicted)
    df.to_csv('output.csv', index=False)


def classify_nn(filename):
    X = pd.read_csv(filename)
    x_test = X.to_dict('records')
    predicted = []

    X_encode = transformer.transform(x_test)
    for i in range(X_encode.shape[0]):
        datum = X_encode[i].toarray()
        answer = MODEL.predict(datum)[0][0]
        predicted.append({'id': x_test[i]['id'], 'netgain': answer >= 0.5})

    df = pd.DataFrame.from_records(predicted)
    df.to_csv('output.csv', index=False)

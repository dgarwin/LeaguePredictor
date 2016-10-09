import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.mixture import GMM
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PlayerCollection import PlayerCollection
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout, MaxoutDense
from keras.wrappers.scikit_learn import KerasClassifier
from modeling import get_save_results
import numpy as np
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1
from keras.layers.noise import GaussianNoise
from LolApi import LolApi
from keras.layers import Input
from sklearn.feature_selection import SelectKBest
from keras.layers.normalization import BatchNormalization
random_state = 42


def tree_error(X_train, X_test, y_train, y_test):
    n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    class_weight = ['balanced', None]
    for cw in class_weight:
        error_rate = []
        train_error = []
        for ne in n_estimators:
            model = RandomForestClassifier(random_state=random_state, n_estimators=ne, class_weight=cw, n_jobs=4,
                                           max_features=None)
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            score = model.score(X_test, y_test)
            error_rate.append(score)
            train_error.append(train_score)
        plt.plot(n_estimators, error_rate, label=str(cw) + ' test')
        # plt.plot(n_estimators, train_error, label=str(cw) + ' train')
    plt.legend(loc='upper right')
    plt.show()


def random_forest():
    model = RandomForestClassifier(random_state=random_state, n_estimators=35, max_features=None)
    return model


def xgboo():
    model = XGBClassifier(seed=random_state, nthread=4)
    parameters = {'n_estimators': [100, 200, 400, 800],
                  'reg_alpha': [0, 0.1, 1, 10]}
    grid = GridSearchCV(model, parameters)
    return grid


def m(activation='relu', dropout=0.5, layers=3, layer_size=512, layer=Dense, cols=54):
    model = Sequential([
        Activation('linear', input_shape=(cols,)),
    ])
    for i in range(layers):
        model.add(layer(layer_size))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        layer_size /= 2
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def nn(build_fn, net_params={}, batch_size=256):
    sk_params = {'nb_epoch': 80, 'batch_size': batch_size, 'validation_split': 0.2,
                 'callbacks': [EarlyStopping(monitor='val_loss', patience=1, mode='auto')]}
    sk_params.update(net_params)
    return KerasClassifier(build_fn=build_fn, **sk_params)


def load_data(count):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=False)
    print X_train.shape, X_test.shape
    params = {'cols': X_train.shape[1],
              'batch_size': 256,
              'layers': 3,
              'dropout': 0.5,
              'layer_size': 512,
              'layer': Dense}
    # model = nn(m, net_params=params)
    model = xgboo()
    get_save_results(X_train, X_test, y_train, y_test, model, 'XGB', params)

if __name__ == '__main__':
    load_data(15000)

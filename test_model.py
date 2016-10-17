from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from classification import xgboo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from PlayerCollection import PlayerCollection
from classification import nn, m, svm, random_forest, mlp
from keras.layers import Dense
from modeling import training_data_sequence, feature_count_sequence, get_save_results
from sklearn.feature_selection import SelectPercentile
from sklearn.svm import SVC


def sequence():
    xgb = (XGBClassifier(seed=42, nthread=4), 'XGB', False)
    rf = (RandomForestClassifier(random_state=42, n_estimators=320), 'RF', False)
    nn_params = {'cols': 108,
                 'batch_size': 256,
                 'layers': 3,
                 'dropout': 0.5,
                 'layer_size': 512,
                 'layer': Dense}
    mlp = (nn(m, nn_params), 'MLP', True)
    pc = PlayerCollection(size=15000)
    samples = [1000, 2000, 4000, 8000, 14900]
    training_data_sequence([xgb, rf, mlp], pc, samples)


def train_svm(count):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=False)
    model = svm()
    get_save_results(X_train, X_test, y_train, y_test, model, 'SVM')


def train_rf(count):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=False)
    model = random_forest()
    get_save_results(X_train, X_test, y_train, y_test, model, 'RF')


def train_xgb(count):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=False)
    model = xgboo()
    get_save_results(X_train, X_test, y_train, y_test, model, 'XGBoost')


def train_nn(count):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=True)
    model = mlp()
    get_save_results(X_train, X_test, y_train, y_test, model, 'NN')


if __name__ == '__main__':
    # train_svm(15000)
    model = SVC(C=100, gamma=0.0005, random_state=42)
    feature_count_sequence(15000, lambda x: model)
'''
    pc = PlayerCollection(size=15000)
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=True)
    model = nn(m, {'cols': 108, 'layers': 3})
    get_save_results(X_train, X_test, y_train, y_test, model, 'NN')
'''

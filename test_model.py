from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from modeling import get_save_results
from classification import xgboo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from PlayerCollection import PlayerCollection
from classification import nn, m
from keras.layers import Dense
from modeling import training_data_sequence, feature_count_sequence
from sklearn.feature_selection import SelectPercentile, f_classif


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


def load_data(count):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=True)
    print 'Train: ' + str(X_train.shape) + ' ' + str(y_train.shape)
    params = {'cols': X_train.shape[1],
              'batch_size': 256,
              'layers': 3,
              'dropout': 0.5,
              'layer_size': 512,
              'layer': Dense}
    model = nn(m, net_params=params)
    get_save_results(X_train, X_test, y_train, y_test, model, 'NN', params)


if __name__ == '__main__':
    feature_count_sequence(15000)

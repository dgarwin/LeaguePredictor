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
from getgames import load_masteries
from sklearn.feature_selection import SelectKBest
random_state = 42


def load_classification(suffix):
    # Load
    player_collection = PlayerCollection(LolApi())
    players, division_counts = player_collection.load(suffix)
    players = players.tolist()
    masteries = load_masteries(suffix).tolist()

    # Transform divisions
    player_divisions = {}
    division_counts = division_counts.tolist()
    for division, player_ids in division_counts.iteritems():
        for player_id in player_ids:
            if division == 'MASTER' or division == 'DIAMOND':
                division = 'PRO'
            player_divisions[player_id] = {'division': division}

    # Get champion masteries
    if False:
        for player_id in players.keys():
            if player_id not in masteries:
                del players[player_id]
                del player_divisions[player_id]
                continue
            mastery_list = masteries[player_id]
            for i in range(min(3, len(mastery_list))):
                players[player_id]['champion_' + str(i)] = mastery_list[i]['championPoints']

    # Create data frames
    player_divisions = pd.DataFrame(player_divisions).transpose()
    players = pd.DataFrame(players).transpose().fillna(0)
    return players, player_divisions


def preprocess(players, divisions, description):
    players.drop(PlayerCollection.ignore, axis=1).as_matrix()
    divisions = [x[0] for x in divisions.as_matrix()]
    X_train, X_test, y_train, y_test = train_test_split(players, divisions,
                                                        random_state=random_state, stratify=divisions)
    pca = PCA(60)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    if description == 'NN':
        y_train = pd.get_dummies(y_train).as_matrix()
        y_test = pd.get_dummies(y_test).as_matrix()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


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
    parameters = {'n_estimators': [100, 150, 200]}
    grid = GridSearchCV(model, parameters)
    return grid


def mo():
    activation = 'relu'
    model = Sequential([
        Dense(256, input_dim=69),
        Activation(activation),
        Dropout(0.75),
        MaxoutDense(96, W_regularizer=l2(0.05)),
        Dropout(0.75),
        Dense(5),
        Activation('softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def m():
    activation = 'relu'
    model = Sequential([
        Dense(256, input_dim=60),
        Activation(activation),
        Dropout(0.5),
        Dense(128),
        Activation(activation),
        Dropout(0.5),
        Dense(64),
        Activation(activation),
        Dropout(0.5),
        Dense(5),
        Activation('softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def nn(build_fn):
    return KerasClassifier(build_fn=build_fn, nb_epoch=80, batch_size=128, validation_split=0.2,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=1, mode='auto')])


def train_classifier(players, divisions, description):
    X_train, X_test, y_train, y_test = preprocess(players, divisions, description)
    if description == 'RF':
        model = random_forest()
    elif description == 'XGB':
        model = xgboo()
    elif description == 'NN':
        model = nn(m)
    else:
        raise 'Unknown classifier'
    get_save_results(X_train, X_test, y_train, y_test, model, description)

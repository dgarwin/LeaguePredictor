from PlayerCollection import PlayerCollection
import pandas as pd
from LolApi import LolApi
import numpy as np
from sklearn.svm import SVR
from classification import preprocess
from progressbar import ProgressBar
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from modeling import get_save_results


def get_regression_divisions(suffix):
    api = LolApi()
    player_collection = PlayerCollection(api)
    players, _ = player_collection.load(suffix)
    player_ids = players.tolist().keys()
    divisions = {}
    bar = ProgressBar()
    for i in bar(range(0, len(player_ids), 10)):
        if i % (len(player_ids) / 10) == 0:
            np.save('players_regression_' + suffix + '.npy', players)
            np.save('divisions_regression_' + suffix + '.npy', divisions)
        current_divisions = api.solo_divisions_regression(player_ids[i:i+10])
        for player_id, division in current_divisions.iteritems():
            if divisions == -1:
                del players[player_id]
            else:
                divisions[player_id] = {'division': division}
    np.save('players_regression_' + suffix + '.npy', players)
    np.save('divisions_regression_' + suffix + '.npy', divisions)


def load_regression(suffix):
    players = np.load('players_regression_' + suffix + '.npy').tolist()
    divisions = np.load('divisions_regression_' + suffix + '.npy').tolist()
    players = pd.DataFrame(players).transpose().fillna(0)
    divisions = pd.DataFrame(divisions).transpose()
    return players, divisions


def rf():
    parameters = {'n_estimators': [10, 50, 100, 150, 200]}
    model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(model, parameters)
    return grid


def svr():
    model = SVR()
    parameters = {'C': [10, 20, 30, 40, 50], 'gamma': [0.001]}
    grid = GridSearchCV(model, parameters)
    return grid


def train_regressor(players, divisions, description):
    if description == 'RF':
        model = rf()
    elif description == 'SVR':
        model = svr()
    else:
        raise 'Unkown model'
    X_train, X_test, y_train, y_test = preprocess(players, divisions)
    get_save_results(X_train, X_test, y_train, y_test, model, description)
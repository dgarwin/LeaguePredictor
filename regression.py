from PlayerCollection import PlayerCollection
import pandas as pd
from LolApi import LolApi
import numpy as np
from sklearn.svm import SVR
from classification import preprocess


def get_regression_divisions(suffix):
    api = LolApi()
    players, _ = PlayerCollection.load(suffix)
    player_ids = players.tolist().keys()
    divisions = {}
    for i in range(0,len(player_ids), 10):
        current_divisions = api.solo_divisions_regression(player_ids[i:i+10])
        for player_id, division in current_divisions.iteritems():
            if divisions == -1:
                del players[player_id]
            else:
                divisions[player_id] = {'division': division}
    np.save('players_regression_' + suffix + '.npy', players)
    np.save('divisions_regression_' + suffix + '.npy', divisions)


def load_regression(suffix):
    players = np.load('players_regression_' + suffix + '.npy')
    divisions = np.load('divisions_regression_' + suffix + '.npy')
    players = pd.DataFrame(players.tolist()).transpose().fillna(0)
    divisions = pd.DataFrame(divisions).transpose()
    return players, divisions


def svr():
    model = SVR()
    return model


def train_regressor(players, divisions):
    X_train, X_test, y_train, y_test = preprocess(players, divisions)
    model = svr()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print score
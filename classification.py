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

random_state = 42


def load_classification(suffix):
    players, division_counts = PlayerCollection.load(suffix)
    players = pd.DataFrame(players.tolist()).transpose().fillna(0)
    player_divisions = {}
    division_counts = division_counts.tolist()
    for division, player_ids in division_counts.iteritems():
        for player_id in player_ids:
            player_divisions[player_id] = {'division': division}
    player_divisions = pd.DataFrame(player_divisions).transpose()
    return players, player_divisions


def preprocess(players, divisions):
    players.drop(PlayerCollection.ignore, axis=1)
    divisions = [x[0] for x in divisions.as_matrix()]
    X_train, X_test, y_train, y_test = train_test_split(players, divisions,
                                                        random_state=random_state, stratify=divisions)
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


def random_forest(parameters={}):
    model = RandomForestClassifier(random_state=random_state, n_estimators=35, max_features=None)
    grid = GridSearchCV(model, parameters, n_jobs=4)
    return grid


def xgboo(parameters={}):
    model = XGBClassifier(seed=random_state, nthread=4)
    # parameters = {'n_estimators':[100, 150, 200]}
    grid = GridSearchCV(model, parameters)
    return grid


def train_classifier(players, divisions, classifier, parameters):
    X_train, X_test, y_train, y_test = preprocess(players, divisions)
    if classifier == 'RF':
        model = random_forest(parameters)
    elif classifier == 'XGB':
        model = xgboo(parameters)
    else:
        raise 'Unknown classifier'
    model.fit(X_train, y_train)
    print model.score(X_train, y_train)
    if hasattr(model, 'best_params_'):
        print model.best_params_
    print model.score(X_test, y_test)
    predictions = model.predict(X_test)
    print classification_report(y_test, predictions)
    print confusion_matrix(y_test, predictions)

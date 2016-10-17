from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator

from keras.callbacks import EarlyStopping
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


def svm():
    model = SVC(random_state=random_state)
    parameters = {'kernel': ['rbf', 'sigmoid'], 'gamma': [  0.001, 0.0005, 0.0001], 'C': [0.1, 1, 10],
                  'class_weight': [None, 'balanced']}
    return GridSearchCV(model, parameters, n_jobs=8)


def random_forest():
    model = RandomForestClassifier(random_state=random_state)
    parameters = {'n_estimators': [10, 20, 40, 80, 160, 320], 'max_features': ['auto', None],
                  'max_depth': [None, 2, 4, 8], 'class_weight': [None, 'balanced']}
    return GridSearchCV(model, parameters, n_jobs=8)


def xgboo():
    model = XGBClassifier(seed=random_state, nthread=8)
    parameters = {'max_depth': [3, 6, 9], 'n_estimators': [50, 100, 200, 400]}
    grid = GridSearchCV(model, parameters, n_jobs=4, verbose=2)
    return grid


def mlp():
    model = GridSearchNN()
    parameters = {'layers': [1, 2, 3, 4], 'layer_size': [256, 512, 1024], 'layer': [Dense]}
    grid = GridSearchCV(model, parameters)
    return grid


class GridSearchNN(BaseEstimator):
    def __init__(self, activation='relu', dropout='0.5', layers=1, layer_size=512, layer=Dense):
        self.activation = activation
        self.dropout = dropout
        self.layers = layers
        self.layer_size = layer_size
        self.layer = layer

    def fit(self, X, y):
        self.cols = X.shape[1]
        self.model = nn(m, self.__dict__)
        self.model.fit(X, y)

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)


def m(activation='relu', dropout=0.5, layers=3, layer_size=512, layer=Dense, cols=108):
    model = Sequential()
    for i in range(layers):
        if i != 0:
            model.add(layer(layer_size))
        else:
            model.add(layer(layer_size, input_dim=cols))
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


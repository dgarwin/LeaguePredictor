from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout, Reshape
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
# Models for training
random_state = 42


def svm():
    # SVM to grid search
    model = SVC(random_state=random_state)
    parameters = {'kernel': ['rbf', 'sigmoid'], 'gamma': [0.001, 0.0005, 0.0001], 'C': [0.1, 1, 10],
                  'class_weight': [None, 'balanced']}
    return GridSearchCV(model, parameters, n_jobs=8)


def svm_2():
    # SVM to grid search with expanded parameter set
    model = SVC(random_state=random_state)
    parameters = {'kernel': ['rbf', 'sigmoid'], 'gamma': ['auto', 0.1, 0.001, 0.0005, 0.0001],
                  'C': [0.1, 1, 10, 50, 100, 150, 200], 'class_weight': [None, 'balanced']}
    return GridSearchCV(model, parameters, n_jobs=8)


def random_forest():
    # Random Forest to grid search
    model = RandomForestClassifier(random_state=random_state)
    parameters = {'n_estimators': [10, 20, 40, 80, 160, 320], 'max_features': ['auto', None],
                  'max_depth': [None, 2, 4, 8], 'class_weight': [None, 'balanced']}
    return GridSearchCV(model, parameters, n_jobs=8)


def xgboo():
    # Gradient Boosted Trees to grid search
    model = XGBClassifier(seed=random_state, nthread=8)
    parameters = {'max_depth': [3, 6, 9], 'n_estimators': [50, 100, 200, 400]}
    grid = GridSearchCV(model, parameters, n_jobs=4, verbose=2)
    return grid


def mlp():
    # MLP to grid search
    model = GridSearchNN()
    parameters = {'layers': [1, 2, 3, 4], 'layer_size': [256, 512, 1024], 'layer': [Dense]}
    grid = GridSearchCV(model, parameters)
    return grid


class GridSearchNN(BaseEstimator):
    # Wrapper class for grid search to work
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


def conv_net():
    model = Sequential()
    model.add(Conv1D(nb_filter=20, filter_length=1, input_shape=(10, 55)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Reshape((20,)))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return nn(lambda: model)


def m(activation='relu', dropout=0.5, layers=3, layer_size=512, layer=Dense, cols=108):
    # Basic MLP generation function
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
    # Wrapper function for sklearn wrapper class
    sk_params = {'nb_epoch': 80, 'batch_size': batch_size, 'validation_split': 0.2,
                 'callbacks': [EarlyStopping(monitor='val_loss', patience=1, mode='auto')]}
    sk_params.update(net_params)
    return KerasClassifier(build_fn=build_fn, **sk_params)

from PlayerCollection import PlayerCollection
from classification import nn, m, xgboo
from modeling import get_save_results
from keras.layers.core import MaxoutDense, Dense
from keras.regularizers import l1


def fetch_data():
    pc = PlayerCollection(size=15000, load=False)
    pc.get_players(20649224)


def get_best_model():
    pass


def graph_model_depth():
    pass


def graph_model_width():
    pass


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

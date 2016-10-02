from PlayerCollection import PlayerCollection
from classification import nn, m
from modeling import get_save_results
from keras.layers.core import MaxoutDense, Dense
from keras.regularizers import l1


def fetch_data():
    pc = PlayerCollection(size=15000, load=False)
    pc.get_players(20649224)


def load_data(count, description):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data()
    print X_train.shape, X_test.shape
    params = {'cols': X_train.shape[1],
              'batch_size': 256,
              'layers': 3,
              'dropout': 0.5,
              'layer_size': 512,
              'layer': Dense}
    model = nn(m, net_params=params)
    get_save_results(X_train, X_test, y_train, y_test, model, description, params)

if __name__ == '__main__':
    load_data(15000, 'NN')
